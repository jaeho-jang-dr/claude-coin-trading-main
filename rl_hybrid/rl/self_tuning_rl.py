"""Self-Tuning RL — 메타 파라미터 자동 최적화

strategy.md 및 base_agent.py에 정의된 매매 파라미터
(buy_score_threshold, target_profit_pct, stop_loss 등)를
RL 에이전트가 시장 상황에 맞게 자동 조정한다.

아키텍처:
  1. ParameterSpace: 튜닝 대상 파라미터 정의 + 정규화
  2. TuningEnvironment: 파라미터 조정 → 백테스트 → 성과 측정
  3. TuningAgent: PPO로 최적 파라미터 변경량 학습
  4. ParameterTuner: 라이브 통합 (일 1회 실행, 안전장치 포함)
"""

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.self_tuning")

# Gymnasium / SB3 lazy import
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# PyTorch lazy import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 저장 경로
TUNER_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models", "tuner",
)
TUNER_HISTORY_PATH = os.path.join(TUNER_MODEL_DIR, "tuning_history.json")


# ────────────────────────────────────────────────────────
#  ParameterSpace — 튜닝 대상 파라미터 정의
# ────────────────────────────────────────────────────────

@dataclass
class ParameterDef:
    """튜닝 대상 파라미터 정의"""
    name: str
    min_val: float
    max_val: float
    default_val: float
    param_type: str = "float"   # "float", "int", "enum"
    description: str = ""
    # 안전 제약: 1스텝당 최대 변경 비율
    max_change_pct: float = 0.20


# strategy.md 및 base_agent.py에서 추출한 튜닝 가능 파라미터
DEFAULT_PARAMETERS = [
    ParameterDef("buy_score_threshold", 45, 70, 70, "int",
                 "매수 임계 점수 (보수적 70, 보통 55, 공격적 45)"),
    ParameterDef("target_profit_pct", 3.0, 15.0, 15.0, "float",
                 "목표 수익률 (%) — 보수적 15, 보통 10, 공격적 7"),
    ParameterDef("stop_loss_pct", -8.0, -3.0, -5.0, "float",
                 "손절선 (%) — 하이브리드 기준"),
    ParameterDef("forced_stop_pct", -10.0, -5.0, -10.0, "float",
                 "강제 손절선 (%)"),
    ParameterDef("max_trade_ratio", 0.10, 0.50, 0.10, "float",
                 "1회 매매 비율 — 총 자산 대비"),
    ParameterDef("fgi_threshold", 20, 60, 30, "int",
                 "FGI 매수 임계값"),
    ParameterDef("rsi_threshold", 25, 50, 30, "int",
                 "RSI 매수 임계값"),
    ParameterDef("sell_fgi_threshold", 60, 85, 75, "int",
                 "FGI 매도 임계값"),
    ParameterDef("sell_rsi_threshold", 55, 80, 70, "int",
                 "RSI 매도 임계값"),
    ParameterDef("sma_deviation_pct", -8.0, -1.0, -5.0, "float",
                 "SMA 이탈 매수 기준 (%)"),
    ParameterDef("weekend_reduction", 0.0, 0.70, 0.50, "float",
                 "주말 매매 축소 비율"),
    ParameterDef("dca_max_ratio", 0.30, 0.70, 0.50, "float",
                 "DCA 최대 비율"),
]


class ParameterSpace:
    """튜닝 가능 파라미터 공간 관리

    정규화(0~1)와 역정규화, 안전한 변경 적용을 담당한다.
    """

    def __init__(self, params: list[ParameterDef] = None):
        self.params = params or deepcopy(DEFAULT_PARAMETERS)
        self.n_params = len(self.params)
        self._names = [p.name for p in self.params]

    @property
    def names(self) -> list[str]:
        return self._names

    def normalize(self, values: dict[str, float]) -> np.ndarray:
        """원래 값 → [0, 1] 정규화"""
        result = np.zeros(self.n_params, dtype=np.float32)
        for i, p in enumerate(self.params):
            val = values.get(p.name, p.default_val)
            rng = p.max_val - p.min_val
            if rng == 0:
                result[i] = 0.5
            else:
                result[i] = np.clip((val - p.min_val) / rng, 0, 1)
        return result

    def denormalize(self, normalized: np.ndarray) -> dict[str, float]:
        """[0, 1] → 원래 값 역정규화"""
        result = {}
        for i, p in enumerate(self.params):
            val = np.clip(normalized[i], 0, 1)
            raw = p.min_val + val * (p.max_val - p.min_val)
            if p.param_type == "int":
                raw = round(raw)
            else:
                raw = round(raw, 4)
            result[p.name] = raw
        return result

    def get_defaults(self) -> dict[str, float]:
        """기본값 딕셔너리"""
        return {p.name: p.default_val for p in self.params}

    def apply_action(
        self,
        current_values: dict[str, float],
        action: np.ndarray,
    ) -> dict[str, float]:
        """액션(상대 변경량)을 현재 값에 적용

        action: [-1, 1] 범위, 각 파라미터의 상대 변경량
        실제 변경량은 max_change_pct로 제한된다.

        Args:
            current_values: 현재 파라미터 값
            action: np.ndarray shape (n_params,), 값 [-1, 1]

        Returns:
            변경된 파라미터 값 딕셔너리
        """
        new_values = {}
        for i, p in enumerate(self.params):
            current = current_values.get(p.name, p.default_val)
            rng = p.max_val - p.min_val
            if rng == 0:
                new_values[p.name] = current
                continue

            # 상대 변경: action[-1,1] * max_change_pct * range
            delta = float(np.clip(action[i], -1, 1)) * p.max_change_pct * rng
            new_val = current + delta
            new_val = np.clip(new_val, p.min_val, p.max_val)

            if p.param_type == "int":
                new_val = round(float(new_val))
            else:
                new_val = round(float(new_val), 4)

            new_values[p.name] = new_val

        return new_values

    def compute_change_pct(
        self,
        old_values: dict[str, float],
        new_values: dict[str, float],
    ) -> float:
        """두 파라미터 세트 간 최대 변경 비율 (%) 계산"""
        max_pct = 0.0
        for p in self.params:
            old = old_values.get(p.name, p.default_val)
            new = new_values.get(p.name, p.default_val)
            if abs(old) > 1e-8:
                pct = abs(new - old) / abs(old) * 100
                max_pct = max(max_pct, pct)
        return max_pct


# ────────────────────────────────────────────────────────
#  성과 메트릭 계산
# ────────────────────────────────────────────────────────

def compute_performance_metrics(
    portfolio_values: list[float],
    trades: int = 0,
) -> dict:
    """포트폴리오 가치 시퀀스에서 성과 메트릭 계산

    Returns:
        {"return_pct", "mdd", "sharpe", "win_rate", "trade_count",
         "avg_hold_time", "volatility"}
    """
    if len(portfolio_values) < 2:
        return {
            "return_pct": 0.0, "mdd": 0.0, "sharpe": 0.0,
            "win_rate": 0.5, "trade_count": 0, "avg_hold_time": 0.0,
            "volatility": 0.0,
        }

    values = np.array(portfolio_values, dtype=np.float64)
    initial = values[0]

    # 수익률
    total_return = (values[-1] - initial) / initial

    # MDD
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    mdd = float(np.max(drawdowns))

    # 변동성 + Sharpe
    returns = np.diff(values) / values[:-1]
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
    mean_return = float(np.mean(returns))
    risk_free = 0.03 / 365 / 24  # 연 3%
    sharpe = (mean_return - risk_free) / volatility if volatility > 1e-8 else 0.0

    # 승률 (양수 수익 비율)
    positive_returns = np.sum(returns > 0)
    win_rate = float(positive_returns / len(returns)) if len(returns) > 0 else 0.5

    return {
        "return_pct": float(total_return * 100),
        "mdd": float(mdd * 100),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "trade_count": trades,
        "avg_hold_time": float(len(values) / max(trades, 1)),
        "volatility": float(volatility * 100),
    }


# ────────────────────────────────────────────────────────
#  TuningEnvironment — Gymnasium 환경
# ────────────────────────────────────────────────────────

if GYM_AVAILABLE:
    class TuningEnvironment(gym.Env):
        """파라미터 튜닝 Gymnasium 환경

        Observation (obs_dim):
            - 최근 30일 성과 메트릭 (7d): return, MDD, Sharpe, win_rate,
              avg_hold_time, trade_count, volatility
            - 현재 파라미터 값 (정규화, n_params d)
            - 시장 레짐 특성 (6d): fgi, rsi, sma_dev, volatility, trend, volume

        Action:
            - 연속 벡터 (n_params,), 각 파라미터의 상대 변경량 [-1, 1]

        Reward:
            - Sharpe ratio 개선량 (현재 vs 이전 평가 구간)

        Episode:
            - 10 튜닝 라운드, 각 라운드에서 7일 윈도우 평가
        """

        metadata = {"render_modes": ["human"]}

        # 시장 레짐 특성 차원
        REGIME_DIM = 6
        # 성과 메트릭 차원
        METRICS_DIM = 7
        # 에피소드 길이 (튜닝 라운드 수)
        MAX_ROUNDS = 10

        def __init__(
            self,
            candles: list[dict] = None,
            param_space: ParameterSpace = None,
            eval_window_days: int = 7,
            render_mode: str = None,
        ):
            """
            Args:
                candles: 백테스트용 캔들 데이터 (지표 포함)
                param_space: 파라미터 공간 정의
                eval_window_days: 평가 윈도우 (일)
                render_mode: 렌더링 모드
            """
            super().__init__()

            self.param_space = param_space or ParameterSpace()
            self.eval_window_days = eval_window_days
            self.render_mode = render_mode

            # 캔들 데이터 로드
            if candles is None:
                from rl_hybrid.rl.data_loader import HistoricalDataLoader
                loader = HistoricalDataLoader()
                raw = loader.load_candles(days=180, interval="4h")
                candles = loader.compute_indicators(raw)
            self.candles = candles

            # 관측/행동 공간
            n_params = self.param_space.n_params
            obs_dim = self.METRICS_DIM + n_params + self.REGIME_DIM
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_params,), dtype=np.float32,
            )

            # 상태
            self.current_round = 0
            self.current_params = self.param_space.get_defaults()
            self.prev_sharpe = 0.0
            self.eval_start_idx = 0

            # 4h 봉 기준 7일 = 42개
            self.eval_window_steps = eval_window_days * 6

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            self.current_round = 0
            self.current_params = self.param_space.get_defaults()
            self.prev_sharpe = 0.0

            # 랜덤 시작점 (평가 윈도우 * MAX_ROUNDS 확보)
            required = self.eval_window_steps * (self.MAX_ROUNDS + 1) + 24
            max_start = max(24, len(self.candles) - required)
            self.eval_start_idx = self.np_random.integers(24, max_start + 1)

            # 초기 성과 측정
            metrics = self._evaluate_params(self.current_params)
            self.prev_sharpe = metrics["sharpe"]

            obs = self._build_observation(metrics)
            return obs, {"params": self.current_params, "metrics": metrics}

        def step(self, action: np.ndarray):
            # 파라미터 변경 적용
            new_params = self.param_space.apply_action(self.current_params, action)
            self.current_params = new_params

            # 평가 윈도우 전진
            self.eval_start_idx += self.eval_window_steps
            self.current_round += 1

            # 새 파라미터로 백테스트 평가
            metrics = self._evaluate_params(new_params)

            # 보상: Sharpe 개선량
            sharpe_improvement = metrics["sharpe"] - self.prev_sharpe
            # MDD 악화 시 페널티
            mdd_penalty = -0.5 * max(0, metrics["mdd"] - 5.0) / 100
            reward = float(sharpe_improvement * 10 + mdd_penalty)
            self.prev_sharpe = metrics["sharpe"]

            # 종료 조건
            truncated = self.current_round >= self.MAX_ROUNDS
            terminated = False

            # 데이터 범위 초과
            if self.eval_start_idx + self.eval_window_steps >= len(self.candles):
                truncated = True

            obs = self._build_observation(metrics)
            info = {
                "params": dict(self.current_params),
                "metrics": metrics,
                "round": self.current_round,
                "sharpe_improvement": float(sharpe_improvement),
            }

            return obs, reward, terminated, truncated, info

        def _evaluate_params(self, params: dict) -> dict:
            """파라미터로 간이 백테스트 실행

            전체 BitcoinTradingEnv를 실행하는 대신, 캔들 데이터에
            파라미터 기반 간이 시뮬레이션을 수행한다.
            """
            start = self.eval_start_idx
            end = min(start + self.eval_window_steps, len(self.candles) - 1)
            if end <= start:
                return compute_performance_metrics([1.0])

            # 간이 시뮬레이션
            initial_balance = 10_000_000.0
            krw = initial_balance
            btc = 0.0
            portfolio_values = [initial_balance]
            trades = 0
            fee = 0.0008  # 수수료 + 슬리피지

            buy_threshold = params.get("buy_score_threshold", 70)
            target_profit = params.get("target_profit_pct", 15.0)
            stop_loss = params.get("stop_loss_pct", -5.0)
            fgi_th = params.get("fgi_threshold", 30)
            rsi_th = params.get("rsi_threshold", 30)
            sell_rsi_th = params.get("sell_rsi_threshold", 70)
            trade_ratio = params.get("max_trade_ratio", 0.10)

            avg_buy_price = 0.0

            for i in range(start, end):
                candle = self.candles[i]
                price = candle["close"]
                total_val = krw + btc * price

                rsi = candle.get("rsi_14", 50)

                # 매수 시그널 점수 간이 계산
                buy_score = 0
                if rsi <= rsi_th:
                    buy_score += 25
                if rsi <= rsi_th + 5:
                    buy_score += 15

                sma20 = candle.get("sma_20", price)
                if sma20 > 0:
                    sma_dev = (price - sma20) / sma20 * 100
                    sma_threshold = params.get("sma_deviation_pct", -5.0)
                    if sma_dev <= sma_threshold:
                        buy_score += 25

                buy_score += 20  # 뉴스 기본 점수

                # 매수 판단
                if buy_score >= buy_threshold and krw > total_val * 0.05:
                    buy_amount = min(krw * trade_ratio, krw)
                    if buy_amount > 1000:
                        actual = buy_amount * (1 - fee)
                        btc_bought = actual / price
                        # 평단 업데이트
                        if btc > 0:
                            avg_buy_price = (avg_buy_price * btc + price * btc_bought) / (btc + btc_bought)
                        else:
                            avg_buy_price = price
                        btc += btc_bought
                        krw -= buy_amount
                        trades += 1

                # 매도 판단
                if btc > 0 and avg_buy_price > 0:
                    profit_pct = (price - avg_buy_price) / avg_buy_price * 100

                    sell = False
                    if profit_pct >= target_profit:
                        sell = True
                    elif profit_pct <= stop_loss:
                        sell = True
                    elif rsi >= sell_rsi_th:
                        sell = True

                    if sell:
                        proceeds = btc * price * (1 - fee)
                        krw += proceeds
                        btc = 0.0
                        avg_buy_price = 0.0
                        trades += 1

                portfolio_values.append(krw + btc * price)

            return compute_performance_metrics(portfolio_values, trades)

        def _build_observation(self, metrics: dict) -> np.ndarray:
            """관측 벡터 생성"""
            # 1. 성과 메트릭 (7d)
            metrics_vec = np.array([
                metrics.get("return_pct", 0) / 100,
                metrics.get("mdd", 0) / 100,
                metrics.get("sharpe", 0),
                metrics.get("win_rate", 0.5),
                metrics.get("avg_hold_time", 0) / 100,
                metrics.get("trade_count", 0) / 50,
                metrics.get("volatility", 0) / 10,
            ], dtype=np.float32)

            # 2. 현재 파라미터 (정규화)
            params_vec = self.param_space.normalize(self.current_params)

            # 3. 시장 레짐 특성
            regime_vec = self._extract_regime_features()

            return np.concatenate([metrics_vec, params_vec, regime_vec])

        def _extract_regime_features(self) -> np.ndarray:
            """현재 평가 구간의 시장 레짐 특성 추출"""
            idx = min(self.eval_start_idx, len(self.candles) - 1)
            candle = self.candles[idx]

            rsi = candle.get("rsi_14", 50) or 50
            sma20 = candle.get("sma_20", candle["close"])
            price = candle["close"]
            adx = candle.get("adx", 25) or 25
            atr = candle.get("atr", 0) or 0

            sma_dev = (price - sma20) / sma20 if sma20 > 0 else 0

            # 최근 변동성 (간이)
            vol_window = min(10, idx)
            if vol_window > 1:
                prices = [self.candles[idx - j]["close"] for j in range(vol_window)]
                returns = np.diff(prices) / np.array(prices[:-1])
                volatility = float(np.std(returns))
            else:
                volatility = 0.0

            # 추세 (최근 5봉 방향)
            trend_window = min(5, idx)
            if trend_window > 0:
                trend = (price - self.candles[idx - trend_window]["close"]) / self.candles[idx - trend_window]["close"]
            else:
                trend = 0.0

            return np.array([
                rsi / 100,                    # RSI 정규화
                np.clip(sma_dev, -0.2, 0.2),  # SMA 이탈
                adx / 100,                    # ADX 정규화
                np.clip(volatility * 100, 0, 10) / 10,  # 변동성
                np.clip(trend, -0.1, 0.1) * 5,  # 추세
                atr / max(price, 1) * 100,    # ATR 비율
            ], dtype=np.float32)

        def render(self):
            if self.render_mode == "human":
                print(
                    f"Round {self.current_round}/{self.MAX_ROUNDS} | "
                    f"Sharpe: {self.prev_sharpe:.4f} | "
                    f"Params: {self.current_params}"
                )


# ────────────────────────────────────────────────────────
#  TuningAgent — PPO 기반 파라미터 튜닝
# ────────────────────────────────────────────────────────

class TuningAgent:
    """PPO 기반 파라미터 자동 조정 에이전트

    작은 네트워크(64-32)로 빠르게 학습한다.
    """

    def __init__(
        self,
        env=None,
        model_path: str = None,
        param_space: ParameterSpace = None,
    ):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 필요: pip install stable-baselines3")

        self.env = env
        self.param_space = param_space or ParameterSpace()
        self.model_path = model_path or os.path.join(TUNER_MODEL_DIR, "tuner_ppo")
        self.model: Optional[PPO] = None

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        """PPO 모델 생성 (소형 네트워크)"""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=1e-3,
            n_steps=512,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.90,
            clip_range=0.3,
            ent_coef=0.05,       # 탐색 장려 (파라미터 공간 탐색)
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs={
                "net_arch": {
                    "pi": [64, 32],
                    "vf": [64, 32],
                },
            },
        )
        logger.info("TuningAgent PPO 모델 생성 완료")

    def train(self, total_timesteps: int = 50_000):
        """튜닝 에이전트 학습

        Args:
            total_timesteps: 총 학습 스텝
        """
        os.makedirs(TUNER_MODEL_DIR, exist_ok=True)

        logger.info(f"TuningAgent 학습 시작: {total_timesteps} 스텝")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
        self.save()
        logger.info("TuningAgent 학습 완료")

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """관측 → 파라미터 조정 액션

        Returns:
            np.ndarray shape (n_params,), 값 [-1, 1]
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"TuningAgent 저장: {path}")

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = PPO.load(path, env=self.env)
        logger.info(f"TuningAgent 로드: {path}")


# ────────────────────────────────────────────────────────
#  ParameterTuner — 라이브 통합
# ────────────────────────────────────────────────────────

class ParameterTuner:
    """라이브 시스템 파라미터 자동 조정

    일 1회 또는 시장 레짐 변경 시 실행되어 파라미터 조정을 제안한다.

    안전장치:
        - 30% 이상 변경 시 텔레그램 승인 필요
        - 24시간 후 성과 평가 → 악화 시 롤백
        - 모든 변경 이력을 strategy_history 테이블에 기록
    """

    def __init__(
        self,
        agent: TuningAgent = None,
        param_space: ParameterSpace = None,
    ):
        self.param_space = param_space or ParameterSpace()

        # TuningAgent 로드 (학습된 모델이 있으면)
        if agent:
            self.agent = agent
        else:
            model_path = os.path.join(TUNER_MODEL_DIR, "tuner_ppo")
            if os.path.exists(model_path + ".zip") and SB3_AVAILABLE:
                self.agent = TuningAgent(model_path=model_path, param_space=self.param_space)
            else:
                self.agent = None

        # 이력 관리
        self._history: list[dict] = []
        self._load_history()

    def propose_adjustments(
        self,
        current_params: dict[str, float],
        performance_metrics: dict,
        market_regime: dict = None,
    ) -> dict:
        """파라미터 조정 제안

        Args:
            current_params: 현재 파라미터 값
            performance_metrics: 최근 성과 메트릭
            market_regime: 시장 레짐 특성 (rsi, sma_dev, adx, volatility, trend, atr_ratio)

        Returns:
            {
                "proposed_params": dict,
                "changes": dict,          # {param: (old, new)}
                "max_change_pct": float,
                "requires_approval": bool,
                "reason": str,
            }
        """
        if self.agent is None:
            return {
                "proposed_params": current_params,
                "changes": {},
                "max_change_pct": 0.0,
                "requires_approval": False,
                "reason": "TuningAgent 미학습 — 조정 없음",
            }

        # 관측 벡터 구성
        obs = self._build_observation(current_params, performance_metrics, market_regime)

        # 액션 예측
        action = self.agent.predict(obs, deterministic=True)

        # 파라미터 변경 적용
        proposed = self.param_space.apply_action(current_params, action)

        # 변경 내역
        changes = {}
        for name in self.param_space.names:
            old = current_params.get(name, self.param_space.get_defaults()[name])
            new = proposed[name]
            if abs(old - new) > 1e-6:
                changes[name] = (old, new)

        max_change = self.param_space.compute_change_pct(current_params, proposed)
        requires_approval = max_change > 30.0

        reason_parts = []
        if changes:
            for name, (old, new) in list(changes.items())[:3]:
                direction = "증가" if new > old else "감소"
                reason_parts.append(f"{name}: {old} → {new} ({direction})")
        reason = ", ".join(reason_parts) if reason_parts else "변경 없음"

        return {
            "proposed_params": proposed,
            "changes": changes,
            "max_change_pct": max_change,
            "requires_approval": requires_approval,
            "reason": reason,
        }

    def apply_and_record(
        self,
        proposal: dict,
        approved: bool = True,
    ) -> bool:
        """제안된 파라미터를 적용하고 이력 기록

        Args:
            proposal: propose_adjustments() 결과
            approved: 승인 여부

        Returns:
            적용 성공 여부
        """
        if not approved:
            logger.info("파라미터 조정 미승인 — 적용 안함")
            return False

        if not proposal.get("changes"):
            logger.info("변경 사항 없음")
            return False

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "proposed_params": proposal["proposed_params"],
            "changes": {k: list(v) for k, v in proposal["changes"].items()},
            "max_change_pct": proposal["max_change_pct"],
            "approved": approved,
            "reason": proposal["reason"],
            "performance_before": None,  # 나중에 채움
            "performance_after": None,   # 24시간 후 채움
            "rolled_back": False,
        }

        self._history.append(record)
        self._save_history()

        # Supabase strategy_history에 기록
        self._log_to_supabase(record)

        logger.info(f"파라미터 조정 적용: {proposal['reason']}")
        return True

    def check_rollback(
        self,
        current_metrics: dict,
        threshold_sharpe_drop: float = -0.05,
    ) -> Optional[dict]:
        """24시간 후 성과를 평가하여 롤백 필요 여부 판단

        Args:
            current_metrics: 현재 성과 메트릭
            threshold_sharpe_drop: Sharpe 하락 임계값

        Returns:
            롤백할 파라미터 (dict) 또는 None
        """
        if not self._history:
            return None

        latest = self._history[-1]
        if latest.get("rolled_back"):
            return None

        # 24시간 경과 확인
        try:
            from datetime import datetime, timedelta, timezone
            applied_time = datetime.fromisoformat(latest["timestamp"])
            now = datetime.now(timezone.utc)
            if (now - applied_time.replace(tzinfo=timezone.utc)) < timedelta(hours=24):
                return None  # 아직 24시간 미경과
        except Exception:
            pass

        latest["performance_after"] = current_metrics
        sharpe_after = current_metrics.get("sharpe", 0)
        sharpe_before = latest.get("performance_before", {}).get("sharpe", 0)

        if sharpe_after - sharpe_before < threshold_sharpe_drop:
            # 롤백 필요
            latest["rolled_back"] = True
            self._save_history()

            # 이전 파라미터 복원
            rollback_params = {}
            for name, (old_val, _new_val) in latest.get("changes", {}).items():
                rollback_params[name] = old_val

            logger.warning(
                f"성과 악화 (Sharpe {sharpe_before:.4f} → {sharpe_after:.4f}) — "
                f"파라미터 롤백: {rollback_params}"
            )
            return rollback_params

        return None

    def _build_observation(
        self,
        current_params: dict,
        metrics: dict,
        regime: dict = None,
    ) -> np.ndarray:
        """라이브 데이터로 관측 벡터 구성"""
        regime = regime or {}

        metrics_vec = np.array([
            metrics.get("return_pct", 0) / 100,
            metrics.get("mdd", 0) / 100,
            metrics.get("sharpe", 0),
            metrics.get("win_rate", 0.5),
            metrics.get("avg_hold_time", 0) / 100,
            metrics.get("trade_count", 0) / 50,
            metrics.get("volatility", 0) / 10,
        ], dtype=np.float32)

        params_vec = self.param_space.normalize(current_params)

        regime_vec = np.array([
            regime.get("rsi", 50) / 100,
            np.clip(regime.get("sma_dev", 0), -0.2, 0.2),
            regime.get("adx", 25) / 100,
            np.clip(regime.get("volatility", 0) * 100, 0, 10) / 10,
            np.clip(regime.get("trend", 0), -0.1, 0.1) * 5,
            regime.get("atr_ratio", 0.02),
        ], dtype=np.float32)

        return np.concatenate([metrics_vec, params_vec, regime_vec])

    def _load_history(self):
        """이력 파일 로드"""
        if os.path.exists(TUNER_HISTORY_PATH):
            try:
                with open(TUNER_HISTORY_PATH, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = []

    def _save_history(self):
        """이력 파일 저장"""
        os.makedirs(os.path.dirname(TUNER_HISTORY_PATH), exist_ok=True)
        try:
            with open(TUNER_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self._history[-100:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"이력 저장 실패: {e}")

    def _log_to_supabase(self, record: dict):
        """strategy_history 테이블에 파라미터 변경 기록"""
        try:
            import requests as _req
            from rl_hybrid.config import config

            url = config.supabase.url
            key = config.supabase.service_role_key
            if not url or not key:
                return

            headers = {
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }

            payload = {
                "version": f"auto-tune-{time.strftime('%Y%m%d-%H%M')}",
                "content": json.dumps(record["proposed_params"], ensure_ascii=False),
                "change_summary": f"Self-Tuning RL 자동 조정: {record['reason']}",
            }

            _req.post(
                f"{url}/rest/v1/strategy_history",
                headers=headers,
                json=payload,
                timeout=10,
            )
        except Exception as e:
            logger.error(f"Supabase 기록 실패: {e}")

    def send_approval_request(self, proposal: dict) -> bool:
        """30% 이상 변경 시 텔레그램 알림으로 승인 요청

        Returns:
            알림 전송 성공 여부
        """
        try:
            from scripts.notify_telegram import send_message

            changes_text = "\n".join(
                f"  {name}: {old} → {new}"
                for name, (old, new) in proposal.get("changes", {}).items()
            )

            message = (
                "🔧 *Self\\-Tuning RL 파라미터 조정 제안*\n\n"
                f"최대 변경: {proposal['max_change_pct']:.1f}%\n"
                f"변경 내역:\n{changes_text}\n\n"
                f"사유: {proposal['reason']}\n\n"
                "⚠️ 30% 이상 변경 \\— 수동 승인 필요"
            )

            send_message(message)
            return True
        except Exception as e:
            logger.error(f"텔레그램 알림 실패: {e}")
            return False


# ────────────────────────────────────────────────────────
#  파이프라인 통합 함수
# ────────────────────────────────────────────────────────

def run_parameter_tuning(
    current_params: dict = None,
    performance_metrics: dict = None,
    market_regime: dict = None,
    auto_apply: bool = True,
) -> dict:
    """파라미터 튜닝 파이프라인 실행

    run_agents.sh에서 Phase 6.5로 호출한다.

    Args:
        current_params: 현재 에이전트 파라미터 (None이면 기본값)
        performance_metrics: 최근 성과 (None이면 빈 딕트)
        market_regime: 시장 레짐 (None이면 빈 딕트)
        auto_apply: True면 30% 미만 변경 자동 적용

    Returns:
        {"status", "proposal", "applied", "message"}
    """
    param_space = ParameterSpace()
    current_params = current_params or param_space.get_defaults()
    performance_metrics = performance_metrics or {}
    market_regime = market_regime or {}

    tuner = ParameterTuner(param_space=param_space)

    # 롤백 체크
    rollback = tuner.check_rollback(performance_metrics)
    if rollback:
        return {
            "status": "rollback",
            "proposal": None,
            "applied": True,
            "params": rollback,
            "message": f"성과 악화로 파라미터 롤백: {rollback}",
        }

    # 조정 제안
    proposal = tuner.propose_adjustments(
        current_params, performance_metrics, market_regime
    )

    if not proposal.get("changes"):
        return {
            "status": "no_change",
            "proposal": proposal,
            "applied": False,
            "message": "변경 제안 없음",
        }

    # 승인 필요 여부
    if proposal["requires_approval"]:
        tuner.send_approval_request(proposal)
        return {
            "status": "pending_approval",
            "proposal": proposal,
            "applied": False,
            "message": f"30% 이상 변경 → 텔레그램 승인 대기: {proposal['reason']}",
        }

    # 자동 적용
    if auto_apply:
        applied = tuner.apply_and_record(proposal)
        return {
            "status": "applied",
            "proposal": proposal,
            "applied": applied,
            "params": proposal["proposed_params"],
            "message": f"파라미터 자동 조정: {proposal['reason']}",
        }

    return {
        "status": "proposed",
        "proposal": proposal,
        "applied": False,
        "message": f"조정 제안: {proposal['reason']}",
    }


# ────────────────────────────────────────────────────────
#  CLI 진입점
# ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python self_tuning_rl.py train [timesteps]   # 튜닝 에이전트 학습")
        print("  python self_tuning_rl.py tune                # 파라미터 조정 실행")
        print("  python self_tuning_rl.py show                # 파라미터 공간 표시")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "train":
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
        print(f"TuningEnvironment + TuningAgent 학습: {timesteps} 스텝")

        env = TuningEnvironment()
        agent = TuningAgent(env=env)
        agent.train(total_timesteps=timesteps)

        print(f"학습 완료. 모델 저장: {TUNER_MODEL_DIR}")

    elif cmd == "tune":
        result = run_parameter_tuning()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif cmd == "show":
        space = ParameterSpace()
        print(f"튜닝 가능 파라미터 ({space.n_params}개):")
        for p in space.params:
            print(f"  {p.name}: [{p.min_val}, {p.max_val}] (기본 {p.default_val}, {p.param_type})")
            print(f"    {p.description}")
