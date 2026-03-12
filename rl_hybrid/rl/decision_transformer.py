"""Decision Transformer — GPT-2 기반 오프라인 트레이딩 정책

Decision Transformer (Chen et al., 2021)를 암호화폐 트레이딩에 적용.
오프라인 궤적에서 지도학습으로 훈련하며, return-to-go 조건부로
보수적/공격적 행동을 런타임에 제어할 수 있다.

핵심 아이디어:
  - 시퀀스 모델링: (return-to-go, state, action) 트리플을 인과적 트랜스포머로 처리
  - 조건부 생성: 높은 RTG → 공격적, 낮은 RTG → 보수적 매매
  - 오프라인 학습: RL 환경 상호작용 없이 과거 데이터로만 훈련
"""

import logging
import math
import os
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.decision_transformer")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 미설치 -- Decision Transformer 비활성화")

# 모델 저장 경로
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models", "transformer",
)


# ============================================================
# 1. Decision Transformer Model
# ============================================================

if TORCH_AVAILABLE:

    class CausalSelfAttention(nn.Module):
        """인과적 자기 주의 (미래 토큰 마스킹)"""

        def __init__(self, embed_dim: int, n_heads: int, block_size: int, dropout: float = 0.1):
            super().__init__()
            assert embed_dim % n_heads == 0
            self.n_heads = n_heads
            self.head_dim = embed_dim // n_heads
            self.embed_dim = embed_dim

            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.attn_drop = nn.Dropout(dropout)
            self.proj_drop = nn.Dropout(dropout)

            # 인과적 마스크: 각 토큰은 자기 자신과 이전 토큰만 참조
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

        def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
            B, T, C = x.shape

            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
            q, k, v = qkv.unbind(0)

            # KV 캐시 (추론 시 효율화)
            if kv_cache is not None:
                if "k" in kv_cache and "v" in kv_cache:
                    k = torch.cat([kv_cache["k"], k], dim=2)
                    v = torch.cat([kv_cache["v"], v], dim=2)
                kv_cache["k"] = k
                kv_cache["v"] = v

            # Scaled dot-product attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale

            # 인과적 마스크 적용
            T_k = k.shape[2]
            T_q = q.shape[2]
            # 마스크를 올바르게 슬라이싱 (KV 캐시 사용 시 k가 더 길 수 있음)
            causal_mask = self.mask[:, :, T_k - T_q:T_k, :T_k]
            attn = attn.masked_fill(causal_mask == 0, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            out = (attn @ v).transpose(1, 2).reshape(B, T_q, C)
            out = self.proj_drop(self.proj(out))
            return out

    class TransformerBlock(nn.Module):
        """트랜스포머 블록 (Pre-LN)"""

        def __init__(self, embed_dim: int, n_heads: int, block_size: int, dropout: float = 0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = CausalSelfAttention(embed_dim, n_heads, block_size, dropout)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
            x = x + self.attn(self.ln1(x), kv_cache)
            x = x + self.mlp(self.ln2(x))
            return x

    class DecisionTransformer(nn.Module):
        """Decision Transformer — GPT-2 아키텍처 기반 트레이딩 정책

        입력 시퀀스: (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_T, s_T, a_T)
        각 타임스텝 t에서 3개 토큰(RTG, state, action)을 임베딩하여 처리.
        인과적 어텐션으로 과거만 참조하며, state 토큰 위치에서 action을 예측.
        """

        def __init__(
            self,
            state_dim: int = 42,
            action_dim: int = 1,
            embed_dim: int = 128,
            n_layers: int = 4,
            n_heads: int = 4,
            context_length: int = 100,
            dropout: float = 0.1,
            max_return: float = 0.5,
        ):
            """
            Args:
                state_dim: 관측 벡터 차원 (StateEncoder 출력)
                action_dim: 행동 차원 (연속 [-1, 1])
                embed_dim: 트랜스포머 임베딩 차원
                n_layers: 트랜스포머 레이어 수
                n_heads: 멀티헤드 어텐션 헤드 수
                context_length: 참조할 과거 타임스텝 수 (T)
                dropout: 드롭아웃 비율
                max_return: RTG 정규화 최대값 (수익률 기준)
            """
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.embed_dim = embed_dim
            self.context_length = context_length
            self.max_return = max_return

            # 블록 사이즈 = 타임스텝 * 3 (RTG, state, action)
            block_size = context_length * 3

            # === 임베딩 레이어 ===
            self.embed_return = nn.Linear(1, embed_dim)
            self.embed_state = nn.Linear(state_dim, embed_dim)
            self.embed_action = nn.Linear(action_dim, embed_dim)

            # 학습 가능한 타임스텝 임베딩 (위치 인코딩 대체)
            self.embed_timestep = nn.Embedding(context_length, embed_dim)

            # 각 모달리티(RTG/state/action) 구분 토큰 임베딩
            self.embed_token_type = nn.Embedding(3, embed_dim)  # 0=RTG, 1=state, 2=action

            self.embed_ln = nn.LayerNorm(embed_dim)
            self.embed_drop = nn.Dropout(dropout)

            # === 트랜스포머 블록 ===
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, n_heads, block_size, dropout)
                for _ in range(n_layers)
            ])

            self.ln_f = nn.LayerNorm(embed_dim)

            # === 출력 헤드 (state 위치에서 action 예측) ===
            self.action_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, action_dim),
                nn.Tanh(),  # [-1, 1] 바운드
            )

            self.apply(self._init_weights)
            logger.info(
                f"DecisionTransformer 생성: state_dim={state_dim}, "
                f"action_dim={action_dim}, embed_dim={embed_dim}, "
                f"layers={n_layers}, heads={n_heads}, context={context_length}, "
                f"params={sum(p.numel() for p in self.parameters()):,}"
            )

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        def forward(
            self,
            returns_to_go: torch.Tensor,  # (B, T, 1)
            states: torch.Tensor,          # (B, T, state_dim)
            actions: torch.Tensor,          # (B, T, action_dim)
            timesteps: torch.Tensor,        # (B, T) long
            padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool, True=valid
        ) -> torch.Tensor:
            """
            Args:
                returns_to_go: 남은 수익률 (정규화됨)
                states: 관측 벡터
                actions: 행동 (과거 행동, 현재 스텝은 0으로 패딩)
                timesteps: 타임스텝 인덱스
                padding_mask: True인 위치만 유효 (False=패딩)

            Returns:
                predicted_actions: (B, T, action_dim) — state 위치에서의 action 예측
            """
            B, T, _ = states.shape
            device = states.device

            # 타임스텝 임베딩 (T 컨텍스트 내 상대 위치)
            timestep_emb = self.embed_timestep(
                timesteps.clamp(0, self.context_length - 1)
            )  # (B, T, embed_dim)

            # 토큰 타입 임베딩
            type_ids = torch.arange(3, device=device)  # [0=RTG, 1=state, 2=action]
            token_type_emb = self.embed_token_type(type_ids)  # (3, embed_dim)

            # 각 모달리티 임베딩
            rtg_emb = self.embed_return(returns_to_go) + timestep_emb + token_type_emb[0]
            state_emb = self.embed_state(states) + timestep_emb + token_type_emb[1]
            action_emb = self.embed_action(actions) + timestep_emb + token_type_emb[2]

            # 시퀀스 인터리빙: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # shape: (B, 3*T, embed_dim)
            stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (B, T, 3, E)
            tokens = stacked.reshape(B, 3 * T, self.embed_dim)

            tokens = self.embed_drop(self.embed_ln(tokens))

            # 패딩 마스크 확장 (각 타임스텝의 3개 토큰에 적용)
            if padding_mask is not None:
                # (B, T) → (B, 3*T): 각 타임스텝의 3개 토큰에 동일 마스크
                expanded_mask = padding_mask.unsqueeze(2).expand(-1, -1, 3).reshape(B, 3 * T)
                # False인 위치를 0으로 마스킹
                tokens = tokens * expanded_mask.unsqueeze(-1).float()

            # 트랜스포머 블록 통과
            x = tokens
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)

            # state 위치의 출력만 추출 (인덱스 1, 4, 7, ... = 3*t+1)
            state_indices = torch.arange(1, 3 * T, 3, device=device)
            state_outputs = x[:, state_indices, :]  # (B, T, embed_dim)

            # action 예측
            predicted_actions = self.action_head(state_outputs)  # (B, T, action_dim)
            return predicted_actions

        def get_action(
            self,
            returns_to_go: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            """추론용: 마지막 타임스텝의 action만 반환

            Args:
                모든 텐서는 (1, T, ...) 형태의 단일 시퀀스

            Returns:
                action: (action_dim,) 텐서
            """
            predicted = self.forward(returns_to_go, states, actions, timesteps)
            return predicted[0, -1]  # 마지막 타임스텝의 action


# ============================================================
# 2. Sequence Dataset
# ============================================================

if TORCH_AVAILABLE:

    class TradingSequenceDataset(Dataset):
        """과거 트레이딩 궤적을 Decision Transformer 학습용 시퀀스로 변환

        캔들 데이터 → (return-to-go, state, action) 트리플 시퀀스
        슬라이딩 윈도우로 에피소드를 분할하고, RTG를 다양한 타겟으로 증강.
        """

        def __init__(
            self,
            candles: list[dict],
            context_length: int = 100,
            state_dim: int = 42,
            action_dim: int = 1,
            max_return: float = 0.5,
            stride: int = 10,
            augment_rtg: bool = True,
            n_augmentations: int = 3,
        ):
            """
            Args:
                candles: data_loader.compute_indicators() 출력
                context_length: 시퀀스 길이 (T)
                state_dim: 관측 차원
                action_dim: 행동 차원
                max_return: RTG 정규화 최대값
                stride: 슬라이딩 윈도우 보폭
                augment_rtg: RTG 증강 활성화
                n_augmentations: 추가 RTG 변형 수
            """
            super().__init__()
            self.context_length = context_length
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.max_return = max_return

            # 캔들 → 환경 시뮬레이션으로 궤적 생성
            trajectories = self._build_trajectories(candles)

            # 슬라이딩 윈도우로 고정 길이 시퀀스 추출
            self.sequences = []
            for traj in trajectories:
                traj_len = len(traj["states"])
                if traj_len < context_length:
                    # 짧은 궤적은 패딩하여 하나의 시퀀스로
                    self.sequences.append(self._pad_trajectory(traj))
                else:
                    # 슬라이딩 윈도우
                    for start in range(0, traj_len - context_length + 1, stride):
                        end = start + context_length
                        seq = {
                            "states": traj["states"][start:end],
                            "actions": traj["actions"][start:end],
                            "rewards": traj["rewards"][start:end],
                            "returns_to_go": traj["returns_to_go"][start:end],
                            "timesteps": np.arange(context_length),
                            "mask": np.ones(context_length, dtype=np.float32),
                        }
                        self.sequences.append(seq)

                        # RTG 증강: 랜덤 타겟 조건부
                        if augment_rtg:
                            for _ in range(n_augmentations):
                                aug_seq = seq.copy()
                                # 원래 RTG에 스케일 팩터를 곱하여 다양한 타겟 생성
                                scale = np.random.uniform(0.3, 2.0)
                                aug_seq["returns_to_go"] = seq["returns_to_go"] * scale
                                self.sequences.append(aug_seq)

            logger.info(
                f"TradingSequenceDataset 생성: {len(self.sequences)}개 시퀀스 "
                f"(궤적 {len(trajectories)}개, context={context_length})"
            )

        def _build_trajectories(self, candles: list[dict]) -> list[dict]:
            """캔들 데이터로 최적 행동 궤적 생성

            hindsight optimal policy: 가격 변동 방향을 미리 알고 행동을 결정.
            실제로는 불가능하지만, 오프라인 학습 데이터로는 최적.
            """
            from rl_hybrid.rl.state_encoder import StateEncoder
            from rl_hybrid.rl.reward import RewardCalculator

            encoder = StateEncoder()
            reward_calc = RewardCalculator()

            n = len(candles)
            if n < 2:
                return []

            closes = np.array([c["close"] for c in candles])

            # 미래 수익률 기반 최적 행동 결정
            # price_change[t] = (close[t+1] - close[t]) / close[t]
            price_changes = np.zeros(n)
            price_changes[:-1] = (closes[1:] - closes[:-1]) / closes[:-1]

            # 최적 행동: 오를 때 매수(+1), 내릴 때 매도(-1), 횡보 시 관망(0)
            optimal_actions = np.zeros(n)
            buy_threshold = 0.001   # 0.1% 이상 상승 예상 시 매수
            sell_threshold = -0.001  # 0.1% 이상 하락 예상 시 매도
            optimal_actions[price_changes > buy_threshold] = 1.0
            optimal_actions[price_changes < sell_threshold] = -1.0
            # 강도 조절: 변화폭에 비례
            for i in range(n):
                if optimal_actions[i] != 0:
                    intensity = min(abs(price_changes[i]) / 0.03, 1.0)  # 3%에서 최대
                    optimal_actions[i] *= intensity

            # 보상 계산 (환경 시뮬레이션)
            rewards = np.zeros(n)
            portfolio_value = 10_000_000.0  # 1000만원 시작
            krw_balance = portfolio_value
            btc_balance = 0.0
            reward_calc.reset(portfolio_value)

            prev_action = 0.0
            for i in range(n - 1):
                price = closes[i]
                action = optimal_actions[i]

                prev_value = krw_balance + btc_balance * price

                # 간이 매매 실행
                total_value = prev_value
                target_btc_ratio = (action + 1) / 2
                current_btc_value = btc_balance * price
                target_btc_value = total_value * target_btc_ratio
                diff = target_btc_value - current_btc_value

                if diff > 0 and krw_balance > 0:
                    buy_amount = min(diff, krw_balance)
                    cost = buy_amount * 0.0008  # TRANSACTION_COST
                    btc_bought = (buy_amount - cost) / price
                    krw_balance -= buy_amount
                    btc_balance += btc_bought
                elif diff < 0 and btc_balance > 0:
                    sell_value = min(-diff, current_btc_value)
                    btc_sold = min(sell_value / price, btc_balance)
                    proceeds = btc_sold * price * (1 - 0.0008)
                    btc_balance -= btc_sold
                    krw_balance += proceeds

                next_price = closes[i + 1]
                curr_value = krw_balance + btc_balance * next_price

                reward_info = reward_calc.calculate(
                    prev_portfolio_value=prev_value,
                    curr_portfolio_value=curr_value,
                    action=action,
                    prev_action=prev_action,
                    step=i,
                )
                rewards[i] = reward_info["reward"]
                prev_action = action

            # Return-to-go 계산 (뒤에서부터 누적)
            returns_to_go = np.zeros(n, dtype=np.float32)
            cumulative = 0.0
            gamma = 0.99
            for t in range(n - 1, -1, -1):
                cumulative = rewards[t] + gamma * cumulative
                returns_to_go[t] = cumulative

            # 정규화
            rtg_std = returns_to_go.std()
            if rtg_std > 1e-8:
                returns_to_go = returns_to_go / (rtg_std * 3)  # 대략 [-1, 1] 범위
            returns_to_go = np.clip(returns_to_go, -1.0, 1.0)

            # 상태 벡터 인코딩
            states = np.zeros((n, encoder.obs_dim), dtype=np.float32)
            for i in range(n):
                candle = candles[i]
                price = candle["close"]
                market_data = {
                    "current_price": price,
                    "change_rate_24h": candle.get("change_rate", 0),
                    "indicators": {
                        "sma_20": candle.get("sma_20", price),
                        "sma_50": candle.get("sma_50", price),
                        "rsi_14": candle.get("rsi_14", 50),
                        "macd": {
                            "macd": candle.get("macd", 0),
                            "signal": candle.get("macd_signal", 0),
                            "histogram": candle.get("macd_histogram", 0),
                        },
                        "bollinger": {
                            "upper": candle.get("boll_upper", price),
                            "middle": candle.get("boll_middle", price),
                            "lower": candle.get("boll_lower", price),
                        },
                        "stochastic": {
                            "k": candle.get("stoch_k", 50),
                            "d": candle.get("stoch_d", 50),
                        },
                        "adx": {
                            "adx": candle.get("adx", 25),
                            "plus_di": candle.get("adx_plus_di", 20),
                            "minus_di": candle.get("adx_minus_di", 20),
                        },
                        "atr": candle.get("atr", 0),
                    },
                    "indicators_4h": {"rsi_14": candle.get("rsi_14", 50)},
                    "orderbook": {"ratio": 1.0},
                    "trade_pressure": {"buy_volume": 1, "sell_volume": 1},
                    "eth_btc_analysis": {"eth_btc_z_score": 0},
                }
                external_data = {
                    "sources": {
                        "fear_greed": {"current": {"value": 50.0}},
                        "news_sentiment": {"sentiment_score": 0.0},
                        "whale_tracker": {"whale_score": {"score": 0.0}},
                        "binance_sentiment": {
                            "funding_rate": {"current_rate": 0.0},
                            "top_trader_long_short": {"current_ratio": 1.0},
                            "kimchi_premium": {"premium_pct": 0.0},
                        },
                        "macro": {"analysis": {"macro_score": 0.0}},
                        "ai_signal": {"ai_composite_signal": {"score": 0}},
                        "coinmarketcap": {"btc_dominance": 50},
                    },
                    "external_signal": {"total_score": 0.0},
                    "nvt_signal": 100.0,
                }
                portfolio = {
                    "krw_balance": 10_000_000,
                    "holdings": [],
                    "total_eval": 10_000_000,
                }
                agent_state = {
                    "danger_score": 30,
                    "opportunity_score": 30,
                    "cascade_risk": 20,
                    "consecutive_losses": 0,
                    "hours_since_last_trade": 24,
                    "daily_trade_count": 0,
                }
                states[i] = encoder.encode(market_data, external_data, portfolio, agent_state)

            # 단일 궤적으로 반환 (장기간 데이터를 하나의 에피소드로 취급)
            return [{
                "states": states,
                "actions": optimal_actions.reshape(-1, 1).astype(np.float32),
                "rewards": rewards.astype(np.float32),
                "returns_to_go": returns_to_go.reshape(-1, 1).astype(np.float32),
            }]

        def _pad_trajectory(self, traj: dict) -> dict:
            """짧은 궤적을 context_length에 맞게 좌측 패딩"""
            traj_len = len(traj["states"])
            pad_len = self.context_length - traj_len

            states = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
            actions = np.zeros((self.context_length, self.action_dim), dtype=np.float32)
            rtg = np.zeros((self.context_length, 1), dtype=np.float32)
            rewards = np.zeros(self.context_length, dtype=np.float32)
            mask = np.zeros(self.context_length, dtype=np.float32)

            states[pad_len:] = traj["states"]
            actions[pad_len:] = traj["actions"]
            rtg[pad_len:] = traj["returns_to_go"]
            rewards[pad_len:] = traj["rewards"]
            mask[pad_len:] = 1.0

            return {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "returns_to_go": rtg,
                "timesteps": np.arange(self.context_length),
                "mask": mask,
            }

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> dict:
            seq = self.sequences[idx]
            return {
                "states": torch.FloatTensor(seq["states"]),
                "actions": torch.FloatTensor(seq["actions"]),
                "returns_to_go": torch.FloatTensor(seq["returns_to_go"]),
                "timesteps": torch.LongTensor(seq["timesteps"]),
                "mask": torch.FloatTensor(seq["mask"]),
            }


# ============================================================
# 3. Trainer
# ============================================================

if TORCH_AVAILABLE:

    class DTTrainer:
        """Decision Transformer 훈련기

        오프라인 궤적에서 지도학습으로 훈련.
        Loss = MSE(predicted_action, target_action)
        """

        def __init__(
            self,
            model: "DecisionTransformer",
            dataset: "TradingSequenceDataset",
            lr: float = 1e-4,
            batch_size: int = 64,
            n_epochs: int = 100,
            weight_decay: float = 1e-4,
            warmup_steps: int = 500,
            eval_split: float = 0.1,
            device: str = None,
        ):
            """
            Args:
                model: DecisionTransformer 모델
                dataset: TradingSequenceDataset
                lr: 학습률
                batch_size: 배치 크기
                n_epochs: 에포크 수
                weight_decay: L2 정규화
                warmup_steps: 워밍업 스텝
                eval_split: 평가 데이터 비율
                device: 연산 장치 ("cuda" / "cpu")
            """
            self.model = model
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # 데이터 분할
            n_eval = max(1, int(len(dataset) * eval_split))
            n_train = len(dataset) - n_eval
            self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_eval]
            )

            self.train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )
            self.eval_loader = DataLoader(
                self.eval_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            # 옵티마이저 + 스케줄러
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            total_steps = n_epochs * len(self.train_loader)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=lr * 0.01
            )

            # 워밍업 (선형)
            self.warmup_steps = warmup_steps
            self.global_step = 0

            logger.info(
                f"DTTrainer 생성: train={n_train}, eval={n_eval}, "
                f"batch={batch_size}, epochs={n_epochs}, device={self.device}"
            )

        def train(self) -> dict:
            """전체 훈련 루프 실행

            Returns:
                {"best_eval_loss", "final_train_loss", "epochs_completed", ...}
            """
            best_eval_loss = float("inf")
            best_model_state = None
            train_losses = []
            eval_losses = []

            for epoch in range(self.n_epochs):
                # 훈련
                train_loss = self._train_epoch()
                train_losses.append(train_loss)

                # 평가
                eval_loss = self._eval_epoch()
                eval_losses.append(eval_loss)

                # 최고 모델 저장
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[Epoch {epoch + 1}/{self.n_epochs}] "
                        f"train_loss={train_loss:.6f}, eval_loss={eval_loss:.6f}, "
                        f"best_eval={best_eval_loss:.6f}, lr={lr:.2e}"
                    )

            # 최고 모델 복원
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                self.model.to(self.device)

            return {
                "best_eval_loss": float(best_eval_loss),
                "final_train_loss": float(train_losses[-1]) if train_losses else 0,
                "final_eval_loss": float(eval_losses[-1]) if eval_losses else 0,
                "epochs_completed": self.n_epochs,
                "train_loss_history": train_losses,
                "eval_loss_history": eval_losses,
            }

        def _train_epoch(self) -> float:
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                # 워밍업 학습률
                self.global_step += 1
                if self.global_step <= self.warmup_steps:
                    warmup_factor = self.global_step / self.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup_factor

                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                rtg = batch["returns_to_go"].to(self.device)
                timesteps = batch["timesteps"].to(self.device)
                mask = batch["mask"].to(self.device)

                # 예측
                predicted_actions = self.model(rtg, states, actions, timesteps, mask.bool())

                # 손실: 유효 위치만 MSE
                mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
                loss = F.mse_loss(
                    predicted_actions * mask_expanded,
                    actions * mask_expanded,
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                n_batches += 1

            return total_loss / max(n_batches, 1)

        @torch.no_grad()
        def _eval_epoch(self) -> float:
            self.model.eval()
            total_loss = 0.0
            n_batches = 0

            for batch in self.eval_loader:
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                rtg = batch["returns_to_go"].to(self.device)
                timesteps = batch["timesteps"].to(self.device)
                mask = batch["mask"].to(self.device)

                predicted_actions = self.model(rtg, states, actions, timesteps, mask.bool())

                mask_expanded = mask.unsqueeze(-1)
                loss = F.mse_loss(
                    predicted_actions * mask_expanded,
                    actions * mask_expanded,
                )

                total_loss += loss.item()
                n_batches += 1

            return total_loss / max(n_batches, 1)

        def evaluate_risk_profiles(self) -> dict:
            """다른 RTG 타겟으로 행동 분포 평가

            높은 RTG → 공격적 (큰 양수 action)
            낮은 RTG → 보수적 (0 근처 action)

            Returns:
                {rtg_level: {"mean_action", "std_action", "buy_ratio", "sell_ratio"}}
            """
            self.model.eval()
            results = {}

            for rtg_level, rtg_value in [
                ("conservative", -0.5),
                ("moderate", 0.0),
                ("aggressive", 0.5),
                ("very_aggressive", 1.0),
            ]:
                all_actions = []
                for batch in self.eval_loader:
                    states = batch["states"].to(self.device)
                    actions = batch["actions"].to(self.device)
                    timesteps = batch["timesteps"].to(self.device)
                    mask = batch["mask"].to(self.device)

                    # RTG를 지정된 값으로 오버라이드
                    B, T, _ = states.shape
                    rtg = torch.full((B, T, 1), rtg_value, device=self.device)

                    with torch.no_grad():
                        pred = self.model(rtg, states, actions, timesteps, mask.bool())

                    # 유효 위치의 action만 수집
                    valid = mask.bool()
                    for b in range(B):
                        valid_actions = pred[b, valid[b], :].cpu().numpy()
                        all_actions.extend(valid_actions.flatten().tolist())

                if all_actions:
                    actions_arr = np.array(all_actions)
                    results[rtg_level] = {
                        "mean_action": float(actions_arr.mean()),
                        "std_action": float(actions_arr.std()),
                        "buy_ratio": float((actions_arr > 0.25).mean()),
                        "sell_ratio": float((actions_arr < -0.25).mean()),
                        "hold_ratio": float(
                            ((actions_arr >= -0.25) & (actions_arr <= 0.25)).mean()
                        ),
                    }

            return results


# ============================================================
# 4. Inference (DTPredictor)
# ============================================================

if TORCH_AVAILABLE:

    class DTPredictor:
        """Decision Transformer 추론기 — 실시간 매매용

        롤링 컨텍스트 윈도우를 유지하며, 매 스텝마다
        desired return-to-go를 설정하여 action을 예측.

        DecisionBlender와 호환: output은 연속값 [-1, 1]
        """

        def __init__(
            self,
            model: "DecisionTransformer" = None,
            model_path: str = None,
            device: str = None,
            default_rtg: float = 0.3,
        ):
            """
            Args:
                model: 훈련된 DecisionTransformer (직접 전달)
                model_path: 모델 파일 경로 (model이 None이면 로드)
                device: 연산 장치
                default_rtg: 기본 return-to-go 타겟
            """
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.default_rtg = default_rtg

            if model is not None:
                self.model = model.to(self.device)
            elif model_path and os.path.exists(model_path):
                self.model = self._load_model(model_path)
            else:
                self.model = None
                logger.warning("DTPredictor: 모델 미로드 (경로 없음)")

            # 롤링 컨텍스트 윈도우
            self.context_length = self.model.context_length if self.model else 100
            self.state_dim = self.model.state_dim if self.model else 42
            self.action_dim = self.model.action_dim if self.model else 1

            self.states_buffer: deque = deque(maxlen=self.context_length)
            self.actions_buffer: deque = deque(maxlen=self.context_length)
            self.rtg_buffer: deque = deque(maxlen=self.context_length)

        def _load_model(self, path: str) -> "DecisionTransformer":
            """모델 + 설정 로드"""
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            config = checkpoint.get("config", {})
            model = DecisionTransformer(
                state_dim=config.get("state_dim", 42),
                action_dim=config.get("action_dim", 1),
                embed_dim=config.get("embed_dim", 128),
                n_layers=config.get("n_layers", 4),
                n_heads=config.get("n_heads", 4),
                context_length=config.get("context_length", 100),
                max_return=config.get("max_return", 0.5),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            logger.info(f"DT 모델 로드: {path}")
            return model

        def reset(self):
            """컨텍스트 초기화"""
            self.states_buffer.clear()
            self.actions_buffer.clear()
            self.rtg_buffer.clear()

        def predict(
            self,
            obs: np.ndarray,
            desired_return: float = None,
            deterministic: bool = True,
        ) -> float:
            """현재 관측에서 행동 예측

            Args:
                obs: 정규화된 관측 벡터 (42,)
                desired_return: RTG 타겟 (None이면 default_rtg 사용)
                deterministic: True면 그대로, False면 노이즈 추가

            Returns:
                action: [-1, 1] 연속값
            """
            if self.model is None:
                return 0.0

            rtg = desired_return if desired_return is not None else self.default_rtg

            # 현재 상태를 버퍼에 추가
            self.states_buffer.append(obs.copy())
            self.rtg_buffer.append(rtg)

            # 시퀀스 구성
            T = len(self.states_buffer)

            states = np.zeros((1, T, self.state_dim), dtype=np.float32)
            actions = np.zeros((1, T, self.action_dim), dtype=np.float32)
            rtgs = np.zeros((1, T, 1), dtype=np.float32)
            timesteps = np.arange(T).reshape(1, -1)

            for i in range(T):
                states[0, i] = self.states_buffer[i]
                rtgs[0, i, 0] = self.rtg_buffer[i]
                if i < len(self.actions_buffer):
                    actions[0, i, 0] = self.actions_buffer[i]
                # 현재 스텝의 action은 0 (예측 대상)

            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            rtgs_t = torch.FloatTensor(rtgs).to(self.device)
            timesteps_t = torch.LongTensor(timesteps).to(self.device)

            with torch.no_grad():
                action = self.model.get_action(rtgs_t, states_t, actions_t, timesteps_t)

            action_val = float(action[0].cpu())
            action_val = max(-1.0, min(1.0, action_val))

            if not deterministic:
                noise = np.random.normal(0, 0.05)
                action_val = max(-1.0, min(1.0, action_val + noise))

            # 행동 기록
            self.actions_buffer.append(action_val)

            return action_val

        def set_risk_profile(self, profile: str):
            """리스크 프로파일에 따라 RTG 타겟 설정

            Args:
                profile: "conservative" | "moderate" | "aggressive"
            """
            profile_map = {
                "conservative": -0.2,
                "moderate": 0.2,
                "aggressive": 0.6,
            }
            self.default_rtg = profile_map.get(profile, 0.2)
            logger.info(f"DT 리스크 프로파일 변경: {profile} (RTG={self.default_rtg})")

        def adjust_rtg_for_market(self, danger_score: float, opportunity_score: float):
            """시장 상황에 따라 RTG를 동적 조정

            Args:
                danger_score: 위험도 (0-100)
                opportunity_score: 기회도 (0-100)
            """
            # 위험 높으면 보수적 (낮은 RTG), 기회 높으면 공격적 (높은 RTG)
            market_factor = (opportunity_score - danger_score) / 100.0  # [-1, 1]
            self.default_rtg = 0.2 + market_factor * 0.4  # [-0.2, 0.6]
            self.default_rtg = max(-0.5, min(1.0, self.default_rtg))

        def to_blender_format(self, action: float) -> dict:
            """DecisionBlender 호환 형식으로 변환

            Returns:
                {"action": float, "value": float, "source": "dt"}
            """
            return {
                "action": action,
                "value": abs(action),  # 신뢰도 대용
                "source": "dt",
            }


# ============================================================
# 5. Integration: train_dt() CLI + ModelRegistry 연동
# ============================================================

def train_dt(
    days: int = 180,
    interval: str = "4h",
    context_length: int = 100,
    embed_dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    save_dir: str = None,
) -> dict:
    """Decision Transformer 훈련 CLI 엔트리포인트

    Args:
        days: 훈련 데이터 기간 (일)
        interval: 캔들 간격 ("1h", "4h", "1d")
        context_length: 시퀀스 길이
        embed_dim: 임베딩 차원
        n_layers: 트랜스포머 레이어 수
        n_heads: 어텐션 헤드 수
        n_epochs: 훈련 에포크
        batch_size: 배치 크기
        lr: 학습률
        save_dir: 모델 저장 경로

    Returns:
        훈련 결과 dict
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch가 필요합니다: pip install torch")

    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    from rl_hybrid.rl.state_encoder import OBSERVATION_DIM

    save_dir = save_dir or MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    # DB 로깅: 훈련 시작
    _dt_cycle_id = None
    _dt_start_time = time.time()
    try:
        from rl_hybrid.rl.rl_db_logger import log_training_start
        _dt_cycle_id = log_training_start(
            cycle_type="standalone",
            algorithm="dt",
            module="decision_transformer",
            training_epochs=n_epochs,
            data_days=days,
            interval=interval,
        )
    except Exception:
        pass

    logger.info(f"=== Decision Transformer 훈련 시작 ===")
    logger.info(f"데이터: {days}일 {interval} 캔들")
    logger.info(f"모델: embed={embed_dim}, layers={n_layers}, heads={n_heads}, context={context_length}")

    # 1. 데이터 로드
    logger.info("1/4 데이터 로드 중...")
    loader = HistoricalDataLoader()
    raw_candles = loader.load_candles(days=days, interval=interval)
    candles = loader.compute_indicators(raw_candles)
    logger.info(f"캔들 {len(candles)}개 로드 완료")

    if len(candles) < context_length + 10:
        raise ValueError(
            f"캔들 수({len(candles)})가 context_length({context_length}) + 10보다 작습니다"
        )

    # 2. 데이터셋 생성
    logger.info("2/4 시퀀스 데이터셋 생성 중...")
    dataset = TradingSequenceDataset(
        candles=candles,
        context_length=context_length,
        state_dim=OBSERVATION_DIM,
        action_dim=1,
        stride=max(1, context_length // 10),
    )

    # 3. 모델 생성 + 훈련
    logger.info("3/4 모델 훈련 중...")
    model = DecisionTransformer(
        state_dim=OBSERVATION_DIM,
        action_dim=1,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        context_length=context_length,
    )

    trainer = DTTrainer(
        model=model,
        dataset=dataset,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    results = trainer.train()

    # 리스크 프로파일 평가
    logger.info("리스크 프로파일 평가 중...")
    risk_profiles = trainer.evaluate_risk_profiles()
    results["risk_profiles"] = risk_profiles

    for profile, stats in risk_profiles.items():
        logger.info(
            f"  {profile}: mean_action={stats['mean_action']:.3f}, "
            f"buy={stats['buy_ratio']:.1%}, sell={stats['sell_ratio']:.1%}"
        )

    # 4. 모델 저장
    logger.info("4/4 모델 저장 중...")
    model_path = os.path.join(save_dir, "dt_model.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "state_dim": OBSERVATION_DIM,
            "action_dim": 1,
            "embed_dim": embed_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "context_length": context_length,
            "max_return": 0.5,
        },
        "training_results": {
            "best_eval_loss": results["best_eval_loss"],
            "final_train_loss": results["final_train_loss"],
            "epochs": n_epochs,
        },
        "risk_profiles": risk_profiles,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, model_path)
    logger.info(f"모델 저장: {model_path}")

    # ModelRegistry에 등록
    try:
        from rl_hybrid.rl.model_registry import ModelRegistry
        registry = ModelRegistry()
        version_id = registry.register_model(
            model_path=model_path,
            metrics={
                "model_type": "decision_transformer",
                "best_eval_loss": results["best_eval_loss"],
                "n_epochs": n_epochs,
                "context_length": context_length,
                "data_days": days,
                "data_interval": interval,
                "n_sequences": len(dataset),
            },
            training_config={
                "embed_dim": embed_dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "lr": lr,
                "batch_size": batch_size,
            },
            notes=f"Decision Transformer {embed_dim}d {n_layers}L {n_heads}H",
        )
        results["registry_version"] = version_id
        logger.info(f"ModelRegistry 등록: {version_id}")
    except Exception as e:
        logger.warning(f"ModelRegistry 등록 실패: {e}")

    # DB 로깅: 훈련 완료
    if _dt_cycle_id:
        try:
            from rl_hybrid.rl.rl_db_logger import log_training_complete
            log_training_complete(
                cycle_id=_dt_cycle_id,
                best_eval_loss=results.get("best_eval_loss"),
                n_sequences=len(dataset),
                context_length=context_length,
                model_version=results.get("registry_version"),
                model_path=model_path,
                elapsed_seconds=time.time() - _dt_start_time,
                status="completed",
            )
        except Exception:
            pass

    logger.info(f"=== 훈련 완료: best_eval_loss={results['best_eval_loss']:.6f} ===")
    return results


# ============================================================
# CLI 엔트리포인트
# ============================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Decision Transformer 훈련")
    parser.add_argument("--days", type=int, default=180, help="훈련 데이터 기간 (일)")
    parser.add_argument("--interval", type=str, default="4h", help="캔들 간격")
    parser.add_argument("--context", type=int, default=100, help="컨텍스트 길이")
    parser.add_argument("--embed-dim", type=int, default=128, help="임베딩 차원")
    parser.add_argument("--layers", type=int, default=4, help="트랜스포머 레이어 수")
    parser.add_argument("--heads", type=int, default=4, help="어텐션 헤드 수")
    parser.add_argument("--epochs", type=int, default=100, help="훈련 에포크")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")

    args = parser.parse_args()

    # 프로젝트 루트를 path에 추가
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    results = train_dt(
        days=args.days,
        interval=args.interval,
        context_length=args.context,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(json.dumps({
        k: v for k, v in results.items()
        if k not in ("train_loss_history", "eval_loss_history")
    }, indent=2, default=str))
