"""Offline RL — Conservative Q-Learning (CQL) + Batch Constrained Q-Learning (BCQ)

Supabase decisions 테이블의 과거 매매 데이터로 오프라인 학습.
실제 환경과 상호작용 없이, 축적된 경험 데이터만으로 정책을 학습한다.

CQL: Q-값 과대추정 방지를 위해 보수적 Q-페널티를 적용.
BCQ: VAE로 데이터 분포 내 행동만 생성하여 분포 외 행동을 제한.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.offline_rl")

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

OFFLINE_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "rl_models", "offline")

# PyTorch lazy import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 미설치 — Offline RL 비활성화")

from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM


# ============================================================================
# 1. Offline Dataset Builder
# ============================================================================


class OfflineTransition:
    """단일 전이 (s, a, r, s', done)"""

    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class OfflineDatasetBuilder:
    """Supabase decisions 테이블에서 오프라인 RL 데이터셋 구축

    decisions 테이블 스키마:
        - decision: '매수' | '매도' | '관망'
        - confidence: 0.0 ~ 1.0
        - current_price: 현재가 (bigint)
        - rsi_value: RSI(14)
        - fear_greed_value: FGI
        - sma20_price: SMA20
        - profit_loss: 실현 손익 (decimal)
        - market_data_snapshot: 시장 데이터 JSON 텍스트
        - created_at: 타임스탬프
    """

    # 결정 → 연속 행동 매핑
    DECISION_TO_ACTION = {
        "매수": 1.0,
        "매도": -1.0,
        "관망": 0.0,
    }

    def __init__(self):
        self.encoder = StateEncoder()
        self.transitions: list[OfflineTransition] = []

    def load_from_supabase(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10000,
    ) -> int:
        """Supabase에서 decisions 로드 후 전이 데이터 구축

        Args:
            date_from: 시작 날짜 (ISO format, 예: '2025-01-01')
            date_to: 종료 날짜 (ISO format)
            min_confidence: 최소 신뢰도 필터
            limit: 최대 로드 개수

        Returns:
            구축된 전이 수
        """
        from rl_hybrid.config import config

        if not config.supabase.url or not config.supabase.service_role_key:
            logger.error("Supabase 설정 없음 — .env에 SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY 설정 필요")
            return 0

        decisions = self._fetch_decisions(
            config.supabase.url,
            config.supabase.service_role_key,
            date_from=date_from,
            date_to=date_to,
            min_confidence=min_confidence,
            limit=limit,
        )

        if len(decisions) < 2:
            logger.warning(f"데이터 부족: {len(decisions)}건 (최소 2건 필요)")
            return 0

        self._build_transitions(decisions)
        logger.info(f"오프라인 데이터셋 구축 완료: {len(self.transitions)}개 전이")
        return len(self.transitions)

    def _fetch_decisions(
        self,
        supabase_url: str,
        service_role_key: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10000,
    ) -> list[dict]:
        """Supabase REST API로 decisions 조회"""
        import urllib.request
        import urllib.parse

        # 쿼리 파라미터 구성
        params = {
            "select": "*",
            "order": "created_at.asc",
            "limit": str(limit),
        }

        # 필터 구성 (PostgREST 문법)
        filters = []
        if date_from:
            filters.append(f"created_at=gte.{date_from}")
        if date_to:
            filters.append(f"created_at=lte.{date_to}")
        if min_confidence > 0:
            filters.append(f"confidence=gte.{min_confidence}")

        query = urllib.parse.urlencode(params)
        for f in filters:
            query += f"&{f}"

        url = f"{supabase_url}/rest/v1/decisions?{query}"

        req = urllib.request.Request(
            url,
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                logger.info(f"Supabase에서 {len(data)}건 decisions 로드")
                return data
        except Exception as e:
            logger.error(f"Supabase 조회 실패: {e}")
            return []

    def _build_transitions(self, decisions: list[dict]):
        """연속된 decisions를 (s, a, r, s', done) 전이로 변환"""
        self.transitions = []

        for i in range(len(decisions) - 1):
            curr = decisions[i]
            next_dec = decisions[i + 1]

            state = self._decision_to_state(curr)
            next_state = self._decision_to_state(next_dec)

            if state is None or next_state is None:
                continue

            action = self.DECISION_TO_ACTION.get(curr.get("decision", "관망"), 0.0)
            reward = self._calculate_reward(curr, next_dec)
            done = (i == len(decisions) - 2)

            self.transitions.append(
                OfflineTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

    def _decision_to_state(self, decision: dict) -> Optional[np.ndarray]:
        """decision 레코드를 StateEncoder 호환 상태 벡터로 변환

        market_data_snapshot JSON이 있으면 그 안의 지표 사용,
        없으면 개별 필드(rsi_value, fear_greed_value 등)로 근사.
        """
        price = float(decision.get("current_price", 0) or 0)
        if price <= 0:
            return None

        rsi = float(decision.get("rsi_value", 50) or 50)
        fgi = int(decision.get("fear_greed_value", 50) or 50)
        sma20 = float(decision.get("sma20_price", price) or price)

        # market_data_snapshot에서 추가 지표 추출 시도
        snapshot_data = {}
        snapshot_raw = decision.get("market_data_snapshot")
        if snapshot_raw:
            try:
                if isinstance(snapshot_raw, str):
                    snapshot_data = json.loads(snapshot_raw)
                elif isinstance(snapshot_raw, dict):
                    snapshot_data = snapshot_raw
            except (json.JSONDecodeError, TypeError):
                pass

        # snapshot에서 더 풍부한 지표 추출
        indicators = snapshot_data.get("indicators", {})
        bollinger = indicators.get("bollinger", {})
        macd_data = indicators.get("macd", {})
        stoch = indicators.get("stochastic", {})
        adx_data = indicators.get("adx", {})

        # 시장 데이터 구성 (StateEncoder 호환)
        market_data = {
            "current_price": price,
            "change_rate_24h": float(snapshot_data.get("change_rate_24h", 0) or 0),
            "indicators": {
                "sma_20": sma20,
                "sma_50": float(indicators.get("sma_50", sma20) or sma20),
                "rsi_14": rsi,
                "macd": {
                    "macd": float(macd_data.get("macd", 0) or 0),
                    "signal": float(macd_data.get("signal", 0) or 0),
                    "histogram": float(macd_data.get("histogram", 0) or 0),
                },
                "bollinger": {
                    "upper": float(bollinger.get("upper", price) or price),
                    "middle": float(bollinger.get("middle", price) or price),
                    "lower": float(bollinger.get("lower", price) or price),
                },
                "stochastic": {
                    "k": float(stoch.get("k", 50) or 50),
                    "d": float(stoch.get("d", 50) or 50),
                },
                "adx": {
                    "adx": float(adx_data.get("adx", 25) or 25),
                    "plus_di": float(adx_data.get("plus_di", 20) or 20),
                    "minus_di": float(adx_data.get("minus_di", 20) or 20),
                },
                "atr": float(indicators.get("atr", 0) or 0),
            },
            "indicators_4h": {"rsi_14": rsi},
            "orderbook": {"ratio": float(snapshot_data.get("orderbook_ratio", 1.0) or 1.0)},
            "trade_pressure": {"buy_volume": 1, "sell_volume": 1},
            "eth_btc_analysis": {"eth_btc_z_score": 0},
        }

        # 외부 데이터 (기본값 위주, snapshot에 있으면 사용)
        external_data = {
            "sources": {
                "fear_greed": {"current": {"value": float(fgi)}},
                "news_sentiment": {"sentiment_score": 0},
                "whale_tracker": {"whale_score": {"score": 0}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": 0.0},
                    "top_trader_long_short": {"current_ratio": 1.0},
                    "kimchi_premium": {"premium_pct": 0.0},
                },
                "macro": {"analysis": {"macro_score": 0}},
                "ai_signal": {"ai_composite_signal": {"score": 0}},
                "coinmarketcap": {"btc_dominance": 50},
            },
            "external_signal": {"total_score": 0},
        }

        # 포트폴리오 (결정 시점 근사)
        confidence = float(decision.get("confidence", 0.5) or 0.5)
        dec_type = decision.get("decision", "관망")
        # 매수 결정이면 BTC 보유 중이라고 근사
        position_ratio = 0.5 if dec_type == "매도" else (0.0 if dec_type == "매수" else 0.3)

        portfolio = {
            "krw_balance": price * 10 * (1 - position_ratio),
            "holdings": [
                {
                    "currency": "BTC",
                    "balance": 10 * position_ratio / price if price > 0 else 0,
                    "avg_buy_price": price,
                    "eval_amount": price * 10 * position_ratio,
                    "profit_loss_pct": 0,
                }
            ] if position_ratio > 0 else [],
            "total_eval": price * 10,
        }

        agent_state = {
            "danger_score": 30,
            "opportunity_score": 30,
            "cascade_risk": 20,
            "consecutive_losses": 0,
            "hours_since_last_trade": 24,
            "daily_trade_count": 0,
        }

        try:
            return self.encoder.encode(market_data, external_data, portfolio, agent_state)
        except Exception as e:
            logger.warning(f"상태 인코딩 실패: {e}")
            return None

    def _calculate_reward(self, curr_decision: dict, next_decision: dict) -> float:
        """실제 profit_loss 기반 보상 계산

        profit_loss가 있으면 직접 사용, 없으면 가격 변동으로 추정.
        """
        profit_loss = curr_decision.get("profit_loss")
        if profit_loss is not None:
            pl = float(profit_loss)
            # 보상 스케일링: 큰 수익/손실을 [-2, 2] 범위로
            return float(np.clip(pl / 50000, -2.0, 2.0))

        # profit_loss 없으면 가격 변동 기반 추정
        curr_price = float(curr_decision.get("current_price", 0) or 0)
        next_price = float(next_decision.get("current_price", 0) or 0)

        if curr_price <= 0 or next_price <= 0:
            return 0.0

        price_change = (next_price - curr_price) / curr_price
        decision = curr_decision.get("decision", "관망")

        if decision == "매수":
            return float(np.clip(price_change * 10, -2.0, 2.0))
        elif decision == "매도":
            return float(np.clip(-price_change * 10, -2.0, 2.0))
        else:  # 관망
            return float(np.clip(-abs(price_change) * 2, -0.5, 0.0))

    def load_from_file(self, path: str) -> int:
        """저장된 데이터셋 파일에서 로드"""
        data = np.load(path, allow_pickle=True)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]
        dones = data["dones"]

        self.transitions = []
        for i in range(len(states)):
            self.transitions.append(
                OfflineTransition(
                    state=states[i],
                    action=float(actions[i]),
                    reward=float(rewards[i]),
                    next_state=next_states[i],
                    done=bool(dones[i]),
                )
            )
        logger.info(f"파일에서 {len(self.transitions)}개 전이 로드: {path}")
        return len(self.transitions)

    def save_to_file(self, path: str):
        """데이터셋을 파일로 저장"""
        if not self.transitions:
            logger.warning("저장할 전이 데이터 없음")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        states = np.array([t.state for t in self.transitions])
        actions = np.array([t.action for t in self.transitions])
        rewards = np.array([t.reward for t in self.transitions])
        next_states = np.array([t.next_state for t in self.transitions])
        dones = np.array([t.done for t in self.transitions])

        np.savez_compressed(
            path, states=states, actions=actions, rewards=rewards,
            next_states=next_states, dones=dones,
        )
        logger.info(f"데이터셋 저장: {path} ({len(self.transitions)}개 전이)")

    def get_arrays(self) -> tuple:
        """numpy 배열 튜플 반환: (states, actions, rewards, next_states, dones)"""
        states = np.array([t.state for t in self.transitions], dtype=np.float32)
        actions = np.array([t.action for t in self.transitions], dtype=np.float32).reshape(-1, 1)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([t.next_state for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def train_test_split(self, test_ratio: float = 0.2) -> tuple:
        """훈련/테스트 분할 (시간 순서 유지)

        Returns:
            (train_transitions, test_transitions)
        """
        split_idx = int(len(self.transitions) * (1 - test_ratio))
        return self.transitions[:split_idx], self.transitions[split_idx:]


# ============================================================================
# 2. PyTorch Dataset
# ============================================================================

if TORCH_AVAILABLE:

    class OfflineRLDataset(Dataset):
        """PyTorch Dataset for offline RL transitions"""

        def __init__(self, transitions: list):
            self.states = torch.FloatTensor(
                np.array([t.state for t in transitions])
            )
            self.actions = torch.FloatTensor(
                np.array([t.action for t in transitions]).reshape(-1, 1)
            )
            self.rewards = torch.FloatTensor(
                np.array([t.reward for t in transitions]).reshape(-1, 1)
            )
            self.next_states = torch.FloatTensor(
                np.array([t.next_state for t in transitions])
            )
            self.dones = torch.FloatTensor(
                np.array([t.done for t in transitions], dtype=np.float32).reshape(-1, 1)
            )

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.dones[idx],
            )


# ============================================================================
# 3. CQL Networks
# ============================================================================

if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        """Twin Q-Network for CQL (SAC 스타일)"""

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
            super().__init__()
            hidden_dims = hidden_dims or [256, 128, 64]

            # Q1 네트워크
            q1_layers = []
            in_dim = state_dim + action_dim
            for h_dim in hidden_dims:
                q1_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
                in_dim = h_dim
            q1_layers.append(nn.Linear(in_dim, 1))
            self.q1 = nn.Sequential(*q1_layers)

            # Q2 네트워크 (Twin)
            q2_layers = []
            in_dim = state_dim + action_dim
            for h_dim in hidden_dims:
                q2_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
                in_dim = h_dim
            q2_layers.append(nn.Linear(in_dim, 1))
            self.q2 = nn.Sequential(*q2_layers)

        def forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)

        def q1_forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa)

    class PolicyNetwork(nn.Module):
        """Tanh Gaussian Policy for CQL (SAC 스타일)"""

        LOG_STD_MIN = -20
        LOG_STD_MAX = 2

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
            super().__init__()
            hidden_dims = hidden_dims or [256, 128, 64]

            layers = []
            in_dim = state_dim
            for h_dim in hidden_dims:
                layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
                in_dim = h_dim
            self.backbone = nn.Sequential(*layers)

            self.mean_head = nn.Linear(in_dim, action_dim)
            self.log_std_head = nn.Linear(in_dim, action_dim)

        def forward(self, state):
            x = self.backbone(state)
            mean = self.mean_head(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            return mean, log_std

        def sample(self, state):
            """재매개변수화 트릭으로 행동 샘플링

            Returns:
                action: tanh 적용된 행동 [-1, 1]
                log_prob: 로그 확률
            """
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)

            # Tanh 보정된 로그 확률
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            return action, log_prob

        def deterministic(self, state):
            """결정론적 행동 (평균)"""
            mean, _ = self.forward(state)
            return torch.tanh(mean)


# ============================================================================
# 4. CQL Trainer
# ============================================================================

if TORCH_AVAILABLE:

    class CQLTrainer:
        """Conservative Q-Learning 트레이너

        핵심 아이디어: Q-값을 보수적으로 추정하여 분포 외 행동의 과대 평가를 방지.

        CQL Loss = alpha * (E[logsumexp(Q(s,a'))] - E[Q(s,a_data)])

        a'는 정책에서 샘플링한 행동, a_data는 오프라인 데이터의 실제 행동.
        alpha가 클수록 더 보수적 (OOD 행동에 대한 Q-값을 더 강하게 억제).
        """

        def __init__(
            self,
            state_dim: int = OBSERVATION_DIM,
            action_dim: int = 1,
            hidden_dims: list = None,
            lr: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            alpha: float = 1.0,
            num_random_actions: int = 10,
            auto_alpha_tuning: bool = True,
            target_entropy: float = None,
            device: str = None,
        ):
            """
            Args:
                state_dim: 상태 차원 (42)
                action_dim: 행동 차원 (1)
                hidden_dims: 은닉층 구조
                lr: 학습률
                gamma: 할인 계수
                tau: 타겟 네트워크 소프트 업데이트 비율
                alpha: CQL 정규화 계수 (클수록 보수적)
                num_random_actions: logsumexp 추정용 랜덤 행동 수
                auto_alpha_tuning: SAC 엔트로피 계수 자동 조정
                target_entropy: 목표 엔트로피 (None이면 -action_dim)
                device: 'cuda' 또는 'cpu'
            """
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.gamma = gamma
            self.tau = tau
            self.cql_alpha = alpha
            self.num_random_actions = num_random_actions
            self.action_dim = action_dim
            self.state_dim = state_dim

            hidden_dims = hidden_dims or [256, 128, 64]

            # 네트워크 초기화
            self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())

            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)

            # 옵티마이저
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

            # SAC 엔트로피 자동 조정
            self.auto_alpha_tuning = auto_alpha_tuning
            if auto_alpha_tuning:
                self.target_entropy = target_entropy or -float(action_dim)
                self.log_sac_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optimizer = optim.Adam([self.log_sac_alpha], lr=lr)
                self.sac_alpha = self.log_sac_alpha.exp().item()
            else:
                self.sac_alpha = 0.2

            # 메트릭 추적
            self.metrics_history: list[dict] = []

        def train(
            self,
            transitions: list,
            epochs: int = 100,
            batch_size: int = 256,
            eval_transitions: list = None,
            log_interval: int = 10,
        ) -> dict:
            """오프라인 데이터셋으로 CQL 훈련

            Args:
                transitions: 훈련 전이 리스트
                epochs: 에폭 수
                batch_size: 배치 크기
                eval_transitions: 평가용 전이 (None이면 스킵)
                log_interval: 로그 출력 주기 (에폭)

            Returns:
                최종 메트릭 딕셔너리
            """
            dataset = OfflineRLDataset(transitions)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )

            eval_dataset = None
            if eval_transitions:
                eval_dataset = OfflineRLDataset(eval_transitions)

            logger.info(
                f"CQL 훈련 시작: {len(transitions)}건, {epochs} 에폭, "
                f"alpha={self.cql_alpha}, batch={batch_size}"
            )

            for epoch in range(1, epochs + 1):
                epoch_metrics = self._train_epoch(dataloader)

                if epoch % log_interval == 0:
                    msg = (
                        f"[Epoch {epoch}/{epochs}] "
                        f"q_loss={epoch_metrics['q_loss']:.4f} "
                        f"policy_loss={epoch_metrics['policy_loss']:.4f} "
                        f"cql_penalty={epoch_metrics['cql_penalty']:.4f} "
                        f"q_data_mean={epoch_metrics['q_data_mean']:.4f}"
                    )
                    if eval_dataset:
                        eval_metrics = self._evaluate(eval_dataset)
                        epoch_metrics.update(eval_metrics)
                        msg += f" | eval_return={eval_metrics['eval_offline_return']:.4f}"
                    logger.info(msg)

                self.metrics_history.append({"epoch": epoch, **epoch_metrics})

            final_metrics = self.metrics_history[-1] if self.metrics_history else {}
            logger.info("CQL 훈련 완료")
            return final_metrics

        def _train_epoch(self, dataloader: DataLoader) -> dict:
            """1 에폭 훈련"""
            total_q_loss = 0.0
            total_policy_loss = 0.0
            total_cql_penalty = 0.0
            total_q_data = 0.0
            num_batches = 0

            for states, actions, rewards, next_states, dones in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

                # --- Q-Network 업데이트 (CQL) ---
                q_loss, cql_penalty, q_data_mean = self._update_q(
                    states, actions, rewards, next_states, dones
                )

                # --- Policy 업데이트 ---
                policy_loss = self._update_policy(states)

                # --- 타겟 네트워크 소프트 업데이트 ---
                self._soft_update()

                total_q_loss += q_loss
                total_policy_loss += policy_loss
                total_cql_penalty += cql_penalty
                total_q_data += q_data_mean
                num_batches += 1

            n = max(num_batches, 1)
            return {
                "q_loss": total_q_loss / n,
                "policy_loss": total_policy_loss / n,
                "cql_penalty": total_cql_penalty / n,
                "q_data_mean": total_q_data / n,
                "sac_alpha": self.sac_alpha,
            }

        def _update_q(self, states, actions, rewards, next_states, dones):
            """Q-네트워크 업데이트 with CQL 정규화"""
            with torch.no_grad():
                # 타겟 Q 계산 (SAC 스타일)
                next_actions, next_log_probs = self.policy.sample(next_states)
                target_q1, target_q2 = self.target_q_network(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.sac_alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q

            # 현재 Q 값
            q1, q2 = self.q_network(states, actions)

            # 표준 Bellman 오류
            bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            # === CQL 정규화 ===
            batch_size = states.shape[0]

            # 랜덤 행동에 대한 Q 값 (분포 외 행동 추정)
            random_actions = torch.FloatTensor(
                batch_size * self.num_random_actions, self.action_dim
            ).uniform_(-1, 1).to(self.device)

            repeated_states = states.unsqueeze(1).repeat(
                1, self.num_random_actions, 1
            ).view(batch_size * self.num_random_actions, -1)

            random_q1, random_q2 = self.q_network(repeated_states, random_actions)
            random_q1 = random_q1.view(batch_size, self.num_random_actions, 1)
            random_q2 = random_q2.view(batch_size, self.num_random_actions, 1)

            # 정책 행동에 대한 Q 값
            policy_actions, policy_log_probs = self.policy.sample(states)
            policy_q1, policy_q2 = self.q_network(states, policy_actions)

            # logsumexp(Q(s, a')) — 랜덤 + 정책 행동 결합
            cat_q1 = torch.cat([random_q1, policy_q1.unsqueeze(1)], dim=1)
            cat_q2 = torch.cat([random_q2, policy_q2.unsqueeze(1)], dim=1)

            logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()
            logsumexp_q2 = torch.logsumexp(cat_q2, dim=1).mean()

            # Q(s, a_data) — 데이터의 실제 행동
            data_q1_mean = q1.mean()
            data_q2_mean = q2.mean()

            # CQL 페널티: logsumexp(Q(s, a_random)) - Q(s, a_data)
            cql_loss = (
                (logsumexp_q1 - data_q1_mean) + (logsumexp_q2 - data_q2_mean)
            ) / 2.0

            # 총 Q 손실
            total_q_loss = bellman_loss + self.cql_alpha * cql_loss

            self.q_optimizer.zero_grad()
            total_q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.q_optimizer.step()

            return (
                total_q_loss.item(),
                cql_loss.item(),
                data_q1_mean.item(),
            )

        def _update_policy(self, states):
            """정책 네트워크 업데이트 (SAC 목적함수)"""
            actions, log_probs = self.policy.sample(states)
            q1, q2 = self.q_network(states, actions)
            min_q = torch.min(q1, q2)

            policy_loss = (self.sac_alpha * log_probs - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            # 엔트로피 계수 자동 조정
            if self.auto_alpha_tuning:
                alpha_loss = -(
                    self.log_sac_alpha * (log_probs + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.sac_alpha = self.log_sac_alpha.exp().item()

            return policy_loss.item()

        def _soft_update(self):
            """타겟 네트워크 소프트 업데이트 (Polyak averaging)"""
            for target_param, param in zip(
                self.target_q_network.parameters(),
                self.q_network.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        def _evaluate(self, eval_dataset: "OfflineRLDataset") -> dict:
            """평가 데이터셋에서 오프라인 성능 측정"""
            self.q_network.eval()
            self.policy.eval()

            with torch.no_grad():
                states = eval_dataset.states.to(self.device)
                actions = eval_dataset.actions.to(self.device)
                rewards = eval_dataset.rewards.to(self.device)

                # 데이터 행동의 Q 값
                q1, q2 = self.q_network(states, actions)
                q_data = torch.min(q1, q2).mean().item()

                # 정책 행동의 Q 값
                policy_actions = self.policy.deterministic(states)
                pq1, pq2 = self.q_network(states, policy_actions)
                q_policy = torch.min(pq1, pq2).mean().item()

                # 오프라인 리턴 추정 (데이터 보상 합)
                offline_return = rewards.sum().item()

            self.q_network.train()
            self.policy.train()

            return {
                "eval_q_data": q_data,
                "eval_q_policy": q_policy,
                "eval_q_gap": q_policy - q_data,
                "eval_offline_return": offline_return,
            }

        def predict(self, state: np.ndarray, deterministic: bool = True) -> float:
            """상태 → 행동 추론

            Args:
                state: 정규화된 관측 벡터 (42,)
                deterministic: True면 평균 행동

            Returns:
                행동 값 [-1, 1]
            """
            self.policy.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if deterministic:
                    action = self.policy.deterministic(state_t)
                else:
                    action, _ = self.policy.sample(state_t)
            return float(action.cpu().numpy()[0, 0])

        def save(self, path: str):
            """모델 저장"""
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(
                {
                    "q_network": self.q_network.state_dict(),
                    "target_q_network": self.target_q_network.state_dict(),
                    "policy": self.policy.state_dict(),
                    "q_optimizer": self.q_optimizer.state_dict(),
                    "policy_optimizer": self.policy_optimizer.state_dict(),
                    "sac_alpha": self.sac_alpha,
                    "cql_alpha": self.cql_alpha,
                    "config": {
                        "state_dim": self.state_dim,
                        "action_dim": self.action_dim,
                        "gamma": self.gamma,
                        "tau": self.tau,
                    },
                    "metrics_history": self.metrics_history,
                },
                path,
            )
            logger.info(f"CQL 모델 저장: {path}")

        def load(self, path: str):
            """모델 로드"""
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_q_network.load_state_dict(checkpoint["target_q_network"])
            self.policy.load_state_dict(checkpoint["policy"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            self.sac_alpha = checkpoint.get("sac_alpha", 0.2)
            self.cql_alpha = checkpoint.get("cql_alpha", 1.0)
            self.metrics_history = checkpoint.get("metrics_history", [])

            logger.info(f"CQL 모델 로드: {path}")


# ============================================================================
# 5. BCQ (Batch Constrained Q-Learning)
# ============================================================================

if TORCH_AVAILABLE:

    class VAE(nn.Module):
        """Variational Autoencoder for BCQ action generation

        오프라인 데이터의 행동 분포를 학습하여,
        데이터 분포 내의 행동만 생성하도록 제약.
        """

        def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 16):
            super().__init__()
            self.latent_dim = latent_dim
            self.action_dim = action_dim

            # 인코더: (s, a) → (mean, log_var)
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.mean_layer = nn.Linear(128, latent_dim)
            self.log_var_layer = nn.Linear(128, latent_dim)

            # 디코더: (s, z) → a
            self.decoder = nn.Sequential(
                nn.Linear(state_dim + latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh(),
            )

        def encode(self, state, action):
            x = torch.cat([state, action], dim=-1)
            h = self.encoder(x)
            return self.mean_layer(h), self.log_var_layer(h)

        def decode(self, state, z=None):
            if z is None:
                z = torch.randn(state.shape[0], self.latent_dim).to(state.device)
                z = z.clamp(-0.5, 0.5)
            return self.decoder(torch.cat([state, z], dim=-1))

        def forward(self, state, action):
            mean, log_var = self.encode(state, action)
            std = (0.5 * log_var).exp()
            z = mean + std * torch.randn_like(std)
            recon_action = self.decode(state, z)
            return recon_action, mean, log_var

    class PerturbationNetwork(nn.Module):
        """BCQ Perturbation Model

        VAE가 생성한 행동에 작은 섭동을 가해 미세 조정.
        섭동 범위를 phi로 제한하여 데이터 분포 근처에 머물게 함.
        """

        def __init__(self, state_dim: int, action_dim: int, phi: float = 0.05):
            super().__init__()
            self.phi = phi
            self.network = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh(),
            )

        def forward(self, state, action):
            perturbation = self.network(torch.cat([state, action], dim=-1))
            return (action + self.phi * perturbation).clamp(-1, 1)

    class BCQTrainer:
        """Batch Constrained Q-Learning 트레이너

        VAE로 데이터 분포 내 행동을 생성하고,
        Perturbation 모델로 미세 조정하여 Q-값을 최대화.
        """

        def __init__(
            self,
            state_dim: int = OBSERVATION_DIM,
            action_dim: int = 1,
            latent_dim: int = 16,
            phi: float = 0.05,
            lr: float = 1e-3,
            gamma: float = 0.99,
            tau: float = 0.005,
            num_candidates: int = 10,
            device: str = None,
        ):
            """
            Args:
                state_dim: 상태 차원
                action_dim: 행동 차원
                latent_dim: VAE 잠재 공간 차원
                phi: 섭동 범위 (클수록 탐색 넓음)
                lr: 학습률
                gamma: 할인 계수
                tau: 타겟 네트워크 소프트 업데이트 비율
                num_candidates: 추론 시 후보 행동 수
                device: 연산 장치
            """
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.gamma = gamma
            self.tau = tau
            self.num_candidates = num_candidates
            self.state_dim = state_dim
            self.action_dim = action_dim

            # 네트워크
            self.vae = VAE(state_dim, action_dim, latent_dim).to(self.device)
            self.q_network = QNetwork(state_dim, action_dim).to(self.device)
            self.target_q_network = QNetwork(state_dim, action_dim).to(self.device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.perturbation = PerturbationNetwork(state_dim, action_dim, phi).to(self.device)

            # 옵티마이저
            self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=lr)
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            self.perturbation_optimizer = optim.Adam(self.perturbation.parameters(), lr=lr)

            self.metrics_history: list[dict] = []

        def train(
            self,
            transitions: list,
            epochs: int = 100,
            batch_size: int = 256,
            eval_transitions: list = None,
            log_interval: int = 10,
        ) -> dict:
            """BCQ 훈련

            Args:
                transitions: 훈련 전이
                epochs: 에폭 수
                batch_size: 배치 크기
                eval_transitions: 평가 전이
                log_interval: 로그 주기

            Returns:
                최종 메트릭
            """
            dataset = OfflineRLDataset(transitions)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )

            logger.info(
                f"BCQ 훈련 시작: {len(transitions)}건, {epochs} 에폭, "
                f"phi={self.perturbation.phi}, batch={batch_size}"
            )

            for epoch in range(1, epochs + 1):
                epoch_metrics = self._train_epoch(dataloader)

                if epoch % log_interval == 0:
                    msg = (
                        f"[Epoch {epoch}/{epochs}] "
                        f"vae_loss={epoch_metrics['vae_loss']:.4f} "
                        f"q_loss={epoch_metrics['q_loss']:.4f} "
                        f"perturb_loss={epoch_metrics['perturbation_loss']:.4f}"
                    )
                    logger.info(msg)

                self.metrics_history.append({"epoch": epoch, **epoch_metrics})

            final_metrics = self.metrics_history[-1] if self.metrics_history else {}
            logger.info("BCQ 훈련 완료")
            return final_metrics

        def _train_epoch(self, dataloader: DataLoader) -> dict:
            """1 에폭 훈련"""
            total_vae = 0.0
            total_q = 0.0
            total_perturb = 0.0
            n = 0

            for states, actions, rewards, next_states, dones in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

                # 1. VAE 업데이트
                vae_loss = self._update_vae(states, actions)

                # 2. Q-네트워크 업데이트
                q_loss = self._update_q(states, actions, rewards, next_states, dones)

                # 3. Perturbation 네트워크 업데이트
                perturb_loss = self._update_perturbation(states)

                # 4. 타겟 소프트 업데이트
                self._soft_update()

                total_vae += vae_loss
                total_q += q_loss
                total_perturb += perturb_loss
                n += 1

            n = max(n, 1)
            return {
                "vae_loss": total_vae / n,
                "q_loss": total_q / n,
                "perturbation_loss": total_perturb / n,
            }

        def _update_vae(self, states, actions) -> float:
            """VAE 업데이트: 재구성 손실 + KL 다이버전스"""
            recon_actions, mean, log_var = self.vae(states, actions)

            recon_loss = F.mse_loss(recon_actions, actions)
            kl_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=-1).mean()
            vae_loss = recon_loss + 0.5 * kl_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            return vae_loss.item()

        def _update_q(self, states, actions, rewards, next_states, dones) -> float:
            """Q-네트워크 업데이트: BCQ 타겟 사용"""
            with torch.no_grad():
                # 다수 후보 행동 생성 후 최대 Q 선택
                batch_size = next_states.shape[0]
                repeated_states = next_states.unsqueeze(1).repeat(
                    1, self.num_candidates, 1
                ).view(batch_size * self.num_candidates, -1)

                # VAE로 후보 행동 생성
                sampled_actions = self.vae.decode(repeated_states)
                perturbed_actions = self.perturbation(repeated_states, sampled_actions)

                # 타겟 Q에서 최대 선택
                tq1, tq2 = self.target_q_network(repeated_states, perturbed_actions)
                target_q = torch.min(tq1, tq2)
                target_q = target_q.view(batch_size, self.num_candidates, 1)
                target_q = target_q.max(dim=1)[0]

                target = rewards + (1 - dones) * self.gamma * target_q

            q1, q2 = self.q_network(states, actions)
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.q_optimizer.step()

            return q_loss.item()

        def _update_perturbation(self, states) -> float:
            """Perturbation 네트워크 업데이트: Q-값 최대화"""
            sampled_actions = self.vae.decode(states)
            perturbed_actions = self.perturbation(states, sampled_actions)
            q_val = self.q_network.q1_forward(states, perturbed_actions)
            loss = -q_val.mean()

            self.perturbation_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.perturbation.parameters(), 1.0)
            self.perturbation_optimizer.step()

            return loss.item()

        def _soft_update(self):
            """타겟 네트워크 소프트 업데이트"""
            for tp, p in zip(
                self.target_q_network.parameters(), self.q_network.parameters()
            ):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        def predict(self, state: np.ndarray, deterministic: bool = True) -> float:
            """상태 → 행동 추론 (VAE 생성 + Perturbation + Q 최대화)"""
            self.vae.eval()
            self.q_network.eval()
            self.perturbation.eval()

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # 후보 행동 생성
                repeated = state_t.repeat(self.num_candidates, 1)
                sampled = self.vae.decode(repeated)
                perturbed = self.perturbation(repeated, sampled)

                # Q 최대화로 최선 행동 선택
                q1 = self.q_network.q1_forward(repeated, perturbed)
                best_idx = q1.argmax(dim=0).item()
                action = perturbed[best_idx, 0].item()

            self.vae.train()
            self.q_network.train()
            self.perturbation.train()

            return float(action)

        def save(self, path: str):
            """모델 저장"""
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(
                {
                    "vae": self.vae.state_dict(),
                    "q_network": self.q_network.state_dict(),
                    "target_q_network": self.target_q_network.state_dict(),
                    "perturbation": self.perturbation.state_dict(),
                    "vae_optimizer": self.vae_optimizer.state_dict(),
                    "q_optimizer": self.q_optimizer.state_dict(),
                    "perturbation_optimizer": self.perturbation_optimizer.state_dict(),
                    "config": {
                        "state_dim": self.state_dim,
                        "action_dim": self.action_dim,
                        "gamma": self.gamma,
                        "tau": self.tau,
                    },
                    "metrics_history": self.metrics_history,
                },
                path,
            )
            logger.info(f"BCQ 모델 저장: {path}")

        def load(self, path: str):
            """모델 로드"""
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.vae.load_state_dict(checkpoint["vae"])
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_q_network.load_state_dict(checkpoint["target_q_network"])
            self.perturbation.load_state_dict(checkpoint["perturbation"])
            self.vae_optimizer.load_state_dict(checkpoint["vae_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.perturbation_optimizer.load_state_dict(checkpoint["perturbation_optimizer"])
            self.metrics_history = checkpoint.get("metrics_history", [])
            logger.info(f"BCQ 모델 로드: {path}")


# ============================================================================
# 6. Evaluation & Comparison
# ============================================================================


def evaluate_offline_model(
    trainer,
    test_transitions: list,
    model_name: str = "offline",
) -> dict:
    """오프라인 모델 평가

    Args:
        trainer: CQLTrainer 또는 BCQTrainer
        test_transitions: 테스트 전이
        model_name: 모델 이름

    Returns:
        평가 메트릭
    """
    if not test_transitions:
        return {"error": "테스트 데이터 없음"}

    correct_direction = 0
    total_reward = 0.0
    actions_taken = []

    for t in test_transitions:
        predicted_action = trainer.predict(t.state, deterministic=True)
        actual_action = t.action
        actions_taken.append(predicted_action)

        # 방향 일치 (매수/매도/관망)
        pred_dir = 1 if predicted_action > 0.2 else (-1 if predicted_action < -0.2 else 0)
        actual_dir = 1 if actual_action > 0.2 else (-1 if actual_action < -0.2 else 0)
        if pred_dir == actual_dir:
            correct_direction += 1

        # 행동 방향이 맞았을 때의 보상
        if pred_dir * t.reward > 0:
            total_reward += abs(t.reward)
        else:
            total_reward -= abs(t.reward) * 0.5

    actions_arr = np.array(actions_taken)
    n = len(test_transitions)

    metrics = {
        "model_name": model_name,
        "test_size": n,
        "direction_accuracy": correct_direction / n if n > 0 else 0,
        "total_reward_estimate": total_reward,
        "avg_reward_estimate": total_reward / n if n > 0 else 0,
        "action_mean": float(actions_arr.mean()),
        "action_std": float(actions_arr.std()),
        "buy_ratio": float((actions_arr > 0.2).mean()),
        "sell_ratio": float((actions_arr < -0.2).mean()),
        "hold_ratio": float(((actions_arr >= -0.2) & (actions_arr <= 0.2)).mean()),
    }

    logger.info(
        f"[{model_name}] 평가 결과: "
        f"방향정확도={metrics['direction_accuracy']:.1%} "
        f"평균보상={metrics['avg_reward_estimate']:.4f} "
        f"매수/관망/매도={metrics['buy_ratio']:.1%}/{metrics['hold_ratio']:.1%}/{metrics['sell_ratio']:.1%}"
    )

    return metrics


def compare_with_best_model(
    offline_metrics: dict,
    best_model_dir: str = None,
) -> dict:
    """오프라인 모델과 기존 best 모델 비교

    Args:
        offline_metrics: 오프라인 모델 평가 결과
        best_model_dir: 기존 best 모델 디렉토리

    Returns:
        비교 리포트
    """
    from rl_hybrid.rl.model_registry import ModelRegistry

    registry = ModelRegistry()
    current = registry.get_current_version()

    report = {
        "offline_model": offline_metrics,
        "current_best": None,
        "recommendation": "unknown",
    }

    if current:
        best_metrics = current.get("metrics", {})
        report["current_best"] = {
            "version": current["version_id"],
            "sharpe_ratio": best_metrics.get("sharpe_ratio", "N/A"),
            "total_return_pct": best_metrics.get("total_return_pct", "N/A"),
            "max_drawdown": best_metrics.get("max_drawdown", "N/A"),
        }

        # 비교 판단
        offline_accuracy = offline_metrics.get("direction_accuracy", 0)
        if offline_accuracy >= 0.55:
            report["recommendation"] = "offline_model_promising"
            report["reason"] = f"방향 정확도 {offline_accuracy:.1%}가 55% 임계값 이상"
        else:
            report["recommendation"] = "keep_current"
            report["reason"] = f"방향 정확도 {offline_accuracy:.1%}가 55% 미만"
    else:
        report["recommendation"] = "use_offline"
        report["reason"] = "기존 모델 없음"

    return report


# ============================================================================
# 7. CLI Entry Point
# ============================================================================


def train_offline(
    algorithm: str = "cql",
    cql_alpha: float = 1.0,
    epochs: int = 100,
    batch_size: int = 256,
    min_data_points: int = 50,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_confidence: float = 0.0,
    output_dir: str = None,
    bcq_phi: float = 0.05,
) -> dict:
    """오프라인 RL 훈련 엔트리포인트

    Args:
        algorithm: 'cql' 또는 'bcq'
        cql_alpha: CQL 정규화 계수
        epochs: 훈련 에폭 수
        batch_size: 배치 크기
        min_data_points: 최소 데이터 포인트 수
        date_from: 시작 날짜
        date_to: 종료 날짜
        min_confidence: 최소 신뢰도
        output_dir: 출력 디렉토리
        bcq_phi: BCQ 섭동 범위

    Returns:
        훈련 결과 딕셔너리
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch 미설치 — pip install torch"}

    output_dir = output_dir or OFFLINE_MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 1. 데이터 로드
    logger.info("=== Offline RL 훈련 시작 ===")
    builder = OfflineDatasetBuilder()
    n_transitions = builder.load_from_supabase(
        date_from=date_from,
        date_to=date_to,
        min_confidence=min_confidence,
    )

    if n_transitions < min_data_points:
        msg = f"데이터 부족: {n_transitions}건 (최소 {min_data_points}건 필요)"
        logger.error(msg)
        return {"error": msg, "transitions": n_transitions}

    # 데이터셋 저장 (재사용)
    dataset_path = os.path.join(output_dir, "offline_dataset")
    builder.save_to_file(dataset_path)

    # 2. 훈련/테스트 분할
    train_data, test_data = builder.train_test_split(test_ratio=0.2)
    logger.info(f"훈련: {len(train_data)}건, 테스트: {len(test_data)}건")

    # 3. 모델 훈련
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if algorithm == "cql":
        trainer = CQLTrainer(alpha=cql_alpha)
        train_metrics = trainer.train(
            train_data, epochs=epochs, batch_size=batch_size,
            eval_transitions=test_data,
        )
        model_path = os.path.join(output_dir, f"cql_{timestamp}.pt")
        trainer.save(model_path)

    elif algorithm == "bcq":
        trainer = BCQTrainer(phi=bcq_phi)
        train_metrics = trainer.train(
            train_data, epochs=epochs, batch_size=batch_size,
            eval_transitions=test_data,
        )
        model_path = os.path.join(output_dir, f"bcq_{timestamp}.pt")
        trainer.save(model_path)

    else:
        return {"error": f"미지원 알고리즘: {algorithm}. 'cql' 또는 'bcq' 사용"}

    # 4. 평가
    eval_metrics = evaluate_offline_model(
        trainer, test_data, model_name=f"{algorithm.upper()}"
    )

    # 5. 기존 모델과 비교
    comparison = compare_with_best_model(eval_metrics)

    # 6. ModelRegistry 등록
    from rl_hybrid.rl.model_registry import ModelRegistry

    registry = ModelRegistry()
    version_id = registry.register_model(
        model_path=model_path,
        metrics={
            "direction_accuracy": eval_metrics.get("direction_accuracy", 0),
            "avg_reward_estimate": eval_metrics.get("avg_reward_estimate", 0),
            "total_reward_estimate": eval_metrics.get("total_reward_estimate", 0),
            "action_std": eval_metrics.get("action_std", 0),
        },
        training_config={
            "algorithm": algorithm,
            "cql_alpha": cql_alpha if algorithm == "cql" else None,
            "bcq_phi": bcq_phi if algorithm == "bcq" else None,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_transitions": n_transitions,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "date_from": date_from,
            "date_to": date_to,
        },
        notes=f"Offline RL ({algorithm.upper()}) — {n_transitions}건 decisions 기반",
    )

    # 7. 결과 리포트
    report = {
        "algorithm": algorithm,
        "version_id": version_id,
        "model_path": model_path,
        "dataset": {
            "total": n_transitions,
            "train": len(train_data),
            "test": len(test_data),
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "comparison": comparison,
    }

    # 리포트 저장
    report_path = os.path.join(output_dir, f"report_{algorithm}_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        # numpy/float 직렬화 처리
        def default_serializer(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        json.dump(report, f, indent=2, ensure_ascii=False, default=default_serializer)

    logger.info(f"=== 리포트 저장: {report_path} ===")

    # 콘솔 요약
    print("\n" + "=" * 60)
    print(f"  Offline RL 훈련 완료 ({algorithm.upper()})")
    print("=" * 60)
    print(f"  데이터: {n_transitions}건 (훈련 {len(train_data)} / 테스트 {len(test_data)})")
    print(f"  모델: {model_path}")
    print(f"  버전: {version_id}")
    print(f"  방향 정확도: {eval_metrics.get('direction_accuracy', 0):.1%}")
    print(f"  평균 보상: {eval_metrics.get('avg_reward_estimate', 0):.4f}")
    print(f"  매수/관망/매도: "
          f"{eval_metrics.get('buy_ratio', 0):.1%} / "
          f"{eval_metrics.get('hold_ratio', 0):.1%} / "
          f"{eval_metrics.get('sell_ratio', 0):.1%}")
    print(f"  추천: {comparison.get('recommendation', 'N/A')}")
    if comparison.get("reason"):
        print(f"  사유: {comparison['reason']}")
    print("=" * 60 + "\n")

    return report


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI 엔트리포인트"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Offline RL 훈련 (CQL/BCQ) - Supabase decisions 기반"
    )
    parser.add_argument(
        "--algorithm", type=str, default="cql", choices=["cql", "bcq"],
        help="알고리즘 선택 (기본: cql)",
    )
    parser.add_argument(
        "--cql-alpha", type=float, default=1.0,
        help="CQL 정규화 계수 (기본: 1.0, 클수록 보수적)",
    )
    parser.add_argument(
        "--bcq-phi", type=float, default=0.05,
        help="BCQ 섭동 범위 (기본: 0.05)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="훈련 에폭 수 (기본: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="배치 크기 (기본: 256)",
    )
    parser.add_argument(
        "--min-data-points", type=int, default=50,
        help="최소 데이터 포인트 수 (기본: 50)",
    )
    parser.add_argument(
        "--date-from", type=str, default=None,
        help="시작 날짜 (ISO format, 예: 2025-01-01)",
    )
    parser.add_argument(
        "--date-to", type=str, default=None,
        help="종료 날짜 (ISO format)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="최소 신뢰도 필터 (0.0~1.0, 기본: 0.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"출력 디렉토리 (기본: {OFFLINE_MODEL_DIR})",
    )

    args = parser.parse_args()

    result = train_offline(
        algorithm=args.algorithm,
        cql_alpha=args.cql_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_data_points=args.min_data_points,
        date_from=args.date_from,
        date_to=args.date_to,
        min_confidence=args.min_confidence,
        output_dir=args.output_dir,
        bcq_phi=args.bcq_phi,
    )

    if "error" in result:
        logger.error(f"훈련 실패: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
