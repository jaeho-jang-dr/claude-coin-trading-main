"""Hierarchical RL — Meta-Agent (전략 선택) + Execution-Agent (포지션 사이징)

2계층 계층적 강화학습 구조:

  Meta-Agent (상위):
    - 관측: 시장 국면 특성 (변동성, 추세, FGI, danger/opportunity, 전략별 최근 성과)
    - 행동: Discrete(3) — conservative=0, moderate=1, aggressive=2
    - 보상: 메타 기간(4~8시간) 동안의 포트폴리오 수익률
    - 정책: PPO (64-32 소형 네트워크)
    - 결정 주기: N 스텝마다 (기본 6 = 4h봉 6개 ≈ 24시간)

  Execution-Agent (하위):
    - 관측: StateEncoder 42차원 + 현재 전략 one-hot (3차원) = 45차원
    - 행동: Box(-1, 1) — 포지션 비중 (음=매도, 0=관망, 양=매수)
    - 보상: RewardCalculator의 Differential Sharpe
    - 정책: SAC (연속 행동 공간에 적합)
    - 결정 주기: 매 스텝

통합:
  - HierarchicalTradingEnv: 두 레벨을 하나의 환경으로 래핑
  - HierarchicalTrainer: 교대 훈련 + 커리큘럼 학습
  - HierarchicalOrchestrator: 기존 Orchestrator 대체 (폴백 지원)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("rl.hierarchical")

# SB3 lazy import
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 미설치 -- Hierarchical RL 비활성화")

from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM
from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST
from rl_hybrid.rl.data_loader import HistoricalDataLoader

# ── 경로 설정 ──
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_DIR / "data" / "rl_models" / "hierarchical"

# ── 전략 에이전트 인덱스 ──
STRATEGY_NAMES = ["conservative", "moderate", "aggressive"]
NUM_STRATEGIES = len(STRATEGY_NAMES)

# ── 전략별 행동 스케일링 팩터 (보수적→소극적, 공격적→적극적) ──
STRATEGY_ACTION_SCALE = {
    0: 0.5,   # conservative: 행동 크기 50%로 축소
    1: 1.0,   # moderate: 행동 그대로
    2: 1.5,   # aggressive: 행동 크기 150%로 확대 (clip 적용)
}


# =====================================================================
#  Meta-Agent Environment
# =====================================================================

# 메타 관측 차원:
#   volatility_20(1) + trend_strength(1) + fgi(1) + danger_score(1) +
#   opportunity_score(1) + price_change_24h(1) + rsi(1) + funding_rate(1) +
#   kimchi_premium(1) + btc_position_ratio(1) + recent_return_per_strategy(3) +
#   current_strategy_onehot(3) = 16
META_OBS_DIM = 16


class MetaEnvironment(gym.Env):
    """메타 에이전트 환경 — 전략 선택 시뮬레이터

    N 스텝(메타 기간)마다 전략을 선택하고, 해당 기간의 포트폴리오
    수익률을 보상으로 받는다. 내부적으로 BitcoinTradingEnv와 유사한
    가격 시뮬레이션을 수행하지만, 실행 에이전트 없이 전략별 기대
    수익률을 시뮬레이션한다.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: list[dict] = None,
        external_signals: list[dict] | None = None,
        initial_balance: float = 10_000_000,
        meta_period: int = 6,       # N 스텝마다 메타 결정
        lookback: int = 24,
        render_mode: str = None,
    ):
        super().__init__()

        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="4h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.external_signals = external_signals
        self.initial_balance = initial_balance
        self.meta_period = meta_period
        self.lookback = lookback
        self.render_mode = render_mode

        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        if self.end_idx <= self.start_idx:
            self.end_idx = len(candles) - 1
            self.start_idx = min(self.start_idx, max(0, self.end_idx - 1))

        # 관측/행동 공간
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(META_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_STRATEGIES)

        # 내부 상태
        self.current_step = 0
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.current_strategy = 1  # 기본: moderate
        self.total_value_history: list[float] = []
        self.strategy_returns: dict[int, deque] = {
            i: deque(maxlen=20) for i in range(NUM_STRATEGIES)
        }
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = max(self.start_idx, self.end_idx - 200)
        self.current_step = self.np_random.integers(self.start_idx, max_start + 1)

        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.current_strategy = 1
        self.total_value_history = [self.initial_balance]
        self.strategy_returns = {i: deque(maxlen=20) for i in range(NUM_STRATEGIES)}
        self.trade_count = 0

        obs = self._get_meta_observation()
        return obs, {}

    def step(self, action: int):
        """메타 스텝: 전략 선택 후 meta_period 동안 시뮬레이션

        Args:
            action: 전략 인덱스 (0=conservative, 1=moderate, 2=aggressive)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_strategy = int(action)
        scale = STRATEGY_ACTION_SCALE[self.current_strategy]

        start_value = self._portfolio_value()
        steps_taken = 0

        for _ in range(self.meta_period):
            if self.current_step >= self.end_idx:
                break

            candle = self.candles[self.current_step]
            price = candle["close"]

            # 전략에 따른 간단한 시뮬레이션 행동
            simulated_action = self._simulate_strategy_action(candle, scale)
            self._execute_action(simulated_action, price)

            self.current_step += 1
            steps_taken += 1

            curr_value = self._portfolio_value()
            self.total_value_history.append(curr_value)

        end_value = self._portfolio_value()

        # 메타 보상: 기간 수익률
        period_return = (end_value - start_value) / start_value if start_value > 0 else 0
        self.strategy_returns[self.current_strategy].append(period_return)

        # 위험 조정 보상 (Sharpe-like)
        reward = self._compute_meta_reward(period_return, steps_taken)

        truncated = self.current_step >= self.end_idx
        terminated = end_value < self.initial_balance * 0.1

        obs = self._get_meta_observation()
        info = {
            "strategy": STRATEGY_NAMES[self.current_strategy],
            "period_return": period_return,
            "portfolio_value": end_value,
            "steps_taken": steps_taken,
        }

        return obs, float(reward), terminated, truncated, info

    def _simulate_strategy_action(self, candle: dict, scale: float) -> float:
        """전략에 따른 시뮬레이션 행동 생성

        실제 에이전트 대신 기술 지표 기반 간단한 규칙으로 행동을 근사한다.
        scale이 클수록 (공격적) 더 큰 포지션 변경을 허용한다.
        """
        rsi = candle.get("rsi_14", 50)
        change_rate = candle.get("change_rate", 0)

        # RSI 기반 기본 시그널
        if rsi < 30:
            base_action = 0.4
        elif rsi < 40:
            base_action = 0.2
        elif rsi > 70:
            base_action = -0.4
        elif rsi > 60:
            base_action = -0.2
        else:
            base_action = 0.0

        # 모멘텀 보정
        base_action += change_rate * 2

        # 전략 스케일 적용
        action = base_action * scale
        return float(np.clip(action, -1.0, 1.0))

    def _execute_action(self, action: float, price: float):
        """행동 실행 (BitcoinTradingEnv와 동일한 로직)"""
        total_value = self._portfolio_value()
        if total_value <= 0:
            return

        target_btc_ratio = (action + 1) / 2
        current_btc_value = self.btc_balance * price
        current_btc_ratio = current_btc_value / total_value if total_value > 0 else 0
        target_btc_value = total_value * target_btc_ratio
        diff = target_btc_value - current_btc_value

        if diff > 0 and self.krw_balance > 0:
            buy_amount = min(diff, self.krw_balance)
            cost = buy_amount * TRANSACTION_COST
            actual_buy = buy_amount - cost
            btc_bought = actual_buy / price
            self.krw_balance -= buy_amount
            self.btc_balance += btc_bought
            if buy_amount > total_value * 0.01:
                self.trade_count += 1
        elif diff < 0 and self.btc_balance > 0:
            sell_value = min(-diff, current_btc_value)
            btc_sold = sell_value / price
            btc_sold = min(btc_sold, self.btc_balance)
            proceeds = btc_sold * price * (1 - TRANSACTION_COST)
            self.btc_balance -= btc_sold
            self.krw_balance += proceeds
            if sell_value > total_value * 0.01:
                self.trade_count += 1

    def _portfolio_value(self) -> float:
        if self.current_step >= len(self.candles):
            idx = len(self.candles) - 1
        else:
            idx = self.current_step
        price = self.candles[idx]["close"]
        return self.krw_balance + self.btc_balance * price

    def _compute_meta_reward(self, period_return: float, steps: int) -> float:
        """메타 보상: 수익률 + 안정성 보상

        - 양수 수익률: 직접 보상
        - 음수 수익률: 2배 페널티 (비대칭 위험 회피)
        - MDD 페널티
        """
        if steps == 0:
            return 0.0

        reward = period_return * 100  # % 단위로 스케일

        # 비대칭: 손실 페널티 강화
        if period_return < 0:
            reward *= 2.0

        # MDD 페널티
        if len(self.total_value_history) > 1:
            peak = max(self.total_value_history)
            current = self.total_value_history[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0
            if drawdown > 0.05:
                reward -= drawdown * 50

        return float(np.clip(reward, -10.0, 10.0))

    def _get_meta_observation(self) -> np.ndarray:
        """메타 관측 벡터 생성 (16차원)"""
        if self.current_step >= len(self.candles):
            idx = len(self.candles) - 1
        else:
            idx = self.current_step
        candle = self.candles[idx]
        price = candle["close"]

        # 변동성 (20봉 ATR / 가격)
        atr = candle.get("atr", 0)
        volatility = atr / price if price > 0 else 0

        # 추세 강도 (ADX 정규화)
        adx = candle.get("adx", 25)
        trend_strength = adx / 100.0

        # FGI (외부 시그널에서 추출, 없으면 50)
        fgi = 50
        if (
            self.external_signals is not None
            and idx < len(self.external_signals)
        ):
            fgi = self.external_signals[idx].get("fgi_value", 50) or 50
        fgi_norm = fgi / 100.0

        # danger/opportunity는 메타에서 간접 추정
        rsi = candle.get("rsi_14", 50)
        change_24h = candle.get("change_rate", 0)

        # 간접 danger: 급락 + RSI 과매수 + 높은 변동성
        danger_proxy = 0.0
        if change_24h < -0.03:
            danger_proxy += min(abs(change_24h) * 5, 0.5)
        if rsi > 70:
            danger_proxy += (rsi - 70) / 100
        danger_proxy = min(danger_proxy, 1.0)

        # 간접 opportunity: 공포 + RSI 과매도
        opp_proxy = 0.0
        if fgi < 25:
            opp_proxy += (25 - fgi) / 100
        if rsi < 35:
            opp_proxy += (35 - rsi) / 100
        if change_24h > 0.01:
            opp_proxy += min(change_24h * 3, 0.3)
        opp_proxy = min(opp_proxy, 1.0)

        # 포지션 비율
        total_val = self._portfolio_value()
        btc_ratio = (self.btc_balance * price / total_val) if total_val > 0 else 0

        # 펀딩 레이트, 김치 프리미엄
        funding = 0.0
        kimchi = 0.0
        if (
            self.external_signals is not None
            and idx < len(self.external_signals)
        ):
            sig = self.external_signals[idx]
            funding = sig.get("funding_rate", 0.0) or 0.0
            kimchi = sig.get("kimchi_premium_pct", 0.0) or 0.0

        # 전략별 최근 수익률 (정규화)
        strategy_perf = []
        for i in range(NUM_STRATEGIES):
            returns = self.strategy_returns[i]
            if returns:
                avg = sum(returns) / len(returns)
                strategy_perf.append(float(np.clip(avg * 10 + 0.5, 0, 1)))
            else:
                strategy_perf.append(0.5)

        # 현재 전략 one-hot
        strategy_onehot = [0.0] * NUM_STRATEGIES
        strategy_onehot[self.current_strategy] = 1.0

        obs = np.array([
            float(np.clip(volatility * 100, 0, 1)),       # volatility
            float(trend_strength),                          # trend_strength
            float(fgi_norm),                                # fgi
            float(danger_proxy),                            # danger_score proxy
            float(opp_proxy),                               # opportunity_score proxy
            float(np.clip(change_24h * 5 + 0.5, 0, 1)),   # price_change_24h
            float(rsi / 100.0),                             # rsi
            float(np.clip(funding * 10 + 0.5, 0, 1)),     # funding_rate
            float(np.clip(kimchi / 20 + 0.5, 0, 1)),      # kimchi_premium
            float(np.clip(btc_ratio, 0, 1)),               # btc_position_ratio
            strategy_perf[0],                               # conservative perf
            strategy_perf[1],                               # moderate perf
            strategy_perf[2],                               # aggressive perf
            strategy_onehot[0],                             # onehot conservative
            strategy_onehot[1],                             # onehot moderate
            strategy_onehot[2],                             # onehot aggressive
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)


# =====================================================================
#  Execution-Agent Environment
# =====================================================================

# 실행 에이전트 관측 차원: StateEncoder 42 + 전략 one-hot 3 = 45
EXEC_OBS_DIM = OBSERVATION_DIM + NUM_STRATEGIES


class ExecutionEnvironment(gym.Env):
    """실행 에이전트 환경 — 스텝 단위 포지션 사이징

    기존 BitcoinTradingEnv를 확장하여 현재 활성 전략 정보를
    관측에 포함시킨다. 전략에 따라 행동 스케일이 조정된다.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: list[dict] = None,
        external_signals: list[dict] | None = None,
        initial_balance: float = 10_000_000,
        lookback: int = 24,
        render_mode: str = None,
        fixed_strategy: int | None = None,
    ):
        super().__init__()

        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="4h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.external_signals = external_signals
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.render_mode = render_mode
        self.fixed_strategy = fixed_strategy

        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        if self.end_idx <= self.start_idx:
            self.end_idx = len(candles) - 1
            self.start_idx = min(self.start_idx, max(0, self.end_idx - 1))

        # 관측: 42 (StateEncoder) + 3 (전략 one-hot)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(EXEC_OBS_DIM,), dtype=np.float32
        )

        # 행동: 연속 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.encoder = StateEncoder()
        self.reward_calc = RewardCalculator()

        # 상태 변수
        self.current_step = 0
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.current_strategy = 1
        self.total_value_history: list[float] = []
        self.trade_count = 0
        self._cached_external_data: dict | None = None
        self._cached_external_step: int = -1

    def set_strategy(self, strategy_idx: int):
        """메타 에이전트로부터 전략 지시를 받는다."""
        self.current_strategy = int(np.clip(strategy_idx, 0, NUM_STRATEGIES - 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = max(self.start_idx, self.end_idx - 500)
        self.current_step = self.np_random.integers(self.start_idx, max_start + 1)

        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.current_strategy = (
            self.fixed_strategy if self.fixed_strategy is not None else 1
        )
        self.total_value_history = [self.initial_balance]
        self.trade_count = 0
        self._cached_external_data = None
        self._cached_external_step = -1

        self.reward_calc.reset(self.initial_balance)

        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        action_val = float(np.clip(action[0], -1, 1))

        # 전략에 따른 행동 스케일링
        scale = STRATEGY_ACTION_SCALE.get(self.current_strategy, 1.0)
        scaled_action = float(np.clip(action_val * scale, -1, 1))

        candle = self.candles[self.current_step]
        price = candle["close"]
        prev_value = self._portfolio_value(price)

        self._execute_action(scaled_action, price)

        self.current_step += 1
        next_candle = self.candles[self.current_step]
        next_price = next_candle["close"]
        curr_value = self._portfolio_value(next_price)
        self.total_value_history.append(curr_value)

        # Differential Sharpe 보상
        reward_info = self.reward_calc.calculate(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=curr_value,
            action=scaled_action,
            prev_action=self.prev_action,
            step=self.current_step,
        )

        self.prev_action = scaled_action

        terminated = curr_value < self.initial_balance * 0.1
        truncated = self.current_step >= self.end_idx

        if terminated:
            reward_info["reward"] -= 1.0

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "price": next_price,
            "portfolio_value": curr_value,
            "return_pct": (curr_value - self.initial_balance) / self.initial_balance * 100,
            "trade_count": self.trade_count,
            "strategy": STRATEGY_NAMES[self.current_strategy],
            "reward_components": reward_info["components"],
        }

        return obs, reward_info["reward"], terminated, truncated, info

    def _execute_action(self, action: float, price: float):
        """BitcoinTradingEnv와 동일한 실행 로직"""
        total_value = self._portfolio_value(price)
        if total_value <= 0:
            return

        target_btc_ratio = (action + 1) / 2
        current_btc_value = self.btc_balance * price
        current_btc_ratio = current_btc_value / total_value if total_value > 0 else 0
        target_btc_value = total_value * target_btc_ratio
        diff = target_btc_value - current_btc_value

        if diff > 0 and self.krw_balance > 0:
            buy_amount = min(diff, self.krw_balance)
            cost = buy_amount * TRANSACTION_COST
            actual_buy = buy_amount - cost
            btc_bought = actual_buy / price
            self.krw_balance -= buy_amount
            self.btc_balance += btc_bought
            if buy_amount > total_value * 0.01:
                self.trade_count += 1
        elif diff < 0 and self.btc_balance > 0:
            sell_value = min(-diff, current_btc_value)
            btc_sold = sell_value / price
            btc_sold = min(btc_sold, self.btc_balance)
            proceeds = btc_sold * price * (1 - TRANSACTION_COST)
            self.btc_balance -= btc_sold
            self.krw_balance += proceeds
            if sell_value > total_value * 0.01:
                self.trade_count += 1

    def _portfolio_value(self, price: float = None) -> float:
        if price is None:
            idx = min(self.current_step, len(self.candles) - 1)
            price = self.candles[idx]["close"]
        return self.krw_balance + self.btc_balance * price

    # 감성 매핑 (environment.py와 동일)
    _SENTIMENT_MAP = {
        "very_positive": 80, "positive": 50, "slightly_positive": 25,
        "neutral": 0, "slightly_negative": -25, "negative": -50,
        "very_negative": -80,
    }

    def _get_observation(self) -> np.ndarray:
        """42차원 StateEncoder 관측 + 3차원 전략 one-hot"""
        idx = min(self.current_step, len(self.candles) - 1)
        candle = self.candles[idx]
        price = candle["close"]

        # StateEncoder 호환 market_data
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

        external_data = self._build_external_data(idx)

        total_value = self._portfolio_value(price)
        btc_eval = self.btc_balance * price
        holdings = (
            [{"currency": "BTC", "balance": self.btc_balance,
              "avg_buy_price": price, "eval_amount": btc_eval,
              "profit_loss_pct": 0}]
            if self.btc_balance > 0 else []
        )
        portfolio = {
            "krw_balance": self.krw_balance,
            "holdings": holdings,
            "total_eval": total_value,
        }

        agent_state = {
            "danger_score": 30,
            "opportunity_score": 30,
            "cascade_risk": 20,
            "consecutive_losses": 0,
            "hours_since_last_trade": 24,
            "daily_trade_count": self.trade_count,
        }

        base_obs = self.encoder.encode(market_data, external_data, portfolio, agent_state)

        # 전략 one-hot
        strategy_onehot = np.zeros(NUM_STRATEGIES, dtype=np.float32)
        strategy_onehot[self.current_strategy] = 1.0

        return np.concatenate([base_obs, strategy_onehot])

    def _build_external_data(self, idx: int) -> dict:
        """외부 데이터 dict 생성"""
        if self._cached_external_data is not None:
            if self.external_signals is None:
                return self._cached_external_data
            if self._cached_external_step == idx:
                return self._cached_external_data

        fgi = 50
        news_score = 0
        whale = 0
        funding = 0.0
        ls_ratio = 1.0
        kimchi = 0.0
        macro = 0
        fusion = 0

        if self.external_signals is not None and idx < len(self.external_signals):
            sig = self.external_signals[idx]
            fgi = sig.get("fgi_value", 50) or 50
            whale = sig.get("whale_score", 0) or 0
            funding = sig.get("funding_rate", 0.0) or 0.0
            ls_ratio = sig.get("long_short_ratio", 1.0) or 1.0
            kimchi = sig.get("kimchi_premium_pct", 0.0) or 0.0
            macro = sig.get("macro_score", 0) or 0
            fusion = sig.get("fusion_score", 0) or 0

            raw_sentiment = sig.get("news_sentiment", "neutral")
            if isinstance(raw_sentiment, (int, float)):
                news_score = float(raw_sentiment)
            elif isinstance(raw_sentiment, str):
                news_score = self._SENTIMENT_MAP.get(raw_sentiment.lower().strip(), 0)

        result = {
            "sources": {
                "fear_greed": {"current": {"value": float(fgi)}},
                "news_sentiment": {"sentiment_score": float(news_score)},
                "whale_tracker": {"whale_score": {"score": float(whale)}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": float(funding)},
                    "top_trader_long_short": {"current_ratio": float(ls_ratio)},
                    "kimchi_premium": {"premium_pct": float(kimchi)},
                },
                "macro": {"analysis": {"macro_score": float(macro)}},
                "ai_signal": {"ai_composite_signal": {"score": 0}},
                "coinmarketcap": {"btc_dominance": 50},
            },
            "external_signal": {"total_score": float(fusion)},
            "nvt_signal": 100.0,
        }

        self._cached_external_data = result
        self._cached_external_step = idx
        return result

    def get_episode_stats(self) -> dict:
        """에피소드 통계"""
        final = self.total_value_history[-1] if self.total_value_history else self.initial_balance
        stats = self.reward_calc.get_episode_stats(final, self.initial_balance)
        stats["trade_count"] = self.trade_count
        stats["final_value"] = final
        stats["steps"] = len(self.total_value_history) - 1
        return stats


# =====================================================================
#  Hierarchical Trading Environment (통합)
# =====================================================================

class HierarchicalTradingEnv(gym.Env):
    """2계층 통합 환경

    외부에서는 실행 에이전트의 관측/행동 공간을 사용하되,
    내부적으로 meta_period마다 메타 에이전트가 전략을 변경한다.

    훈련 시에는 메타 에이전트를 외부에서 주입하고,
    이 환경은 실행 에이전트의 학습 환경으로 사용된다.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: list[dict] = None,
        external_signals: list[dict] | None = None,
        initial_balance: float = 10_000_000,
        meta_period: int = 6,
        lookback: int = 24,
        render_mode: str = None,
        meta_agent_model=None,
    ):
        super().__init__()

        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="4h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.external_signals = external_signals
        self.meta_period = meta_period
        self.meta_agent_model = meta_agent_model

        # 내부 실행 환경
        self.exec_env = ExecutionEnvironment(
            candles=candles,
            external_signals=external_signals,
            initial_balance=initial_balance,
            lookback=lookback,
            render_mode=render_mode,
        )

        # 내부 메타 환경 (관측 생성용)
        self.meta_env = MetaEnvironment(
            candles=candles,
            external_signals=external_signals,
            initial_balance=initial_balance,
            meta_period=meta_period,
            lookback=lookback,
        )

        self.observation_space = self.exec_env.observation_space
        self.action_space = self.exec_env.action_space

        self.steps_since_meta = 0
        self.meta_rewards: list[float] = []

    def reset(self, seed=None, options=None):
        obs, info = self.exec_env.reset(seed=seed, options=options)
        self.meta_env.current_step = self.exec_env.current_step
        self.meta_env.krw_balance = self.exec_env.krw_balance
        self.meta_env.btc_balance = self.exec_env.btc_balance
        self.steps_since_meta = 0
        self.meta_rewards = []

        # 초기 메타 결정
        self._meta_step()

        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.exec_env.step(action)
        self.steps_since_meta += 1
        self.meta_rewards.append(reward)

        # 메타 에이전트 결정 시점
        if self.steps_since_meta >= self.meta_period and not terminated and not truncated:
            self._meta_step()
            self.steps_since_meta = 0
            self.meta_rewards = []

        info["current_strategy"] = STRATEGY_NAMES[self.exec_env.current_strategy]
        return obs, reward, terminated, truncated, info

    def _meta_step(self):
        """메타 에이전트를 호출하여 전략을 선택한다."""
        if self.meta_agent_model is not None:
            # 메타 환경 상태 동기화
            self.meta_env.current_step = self.exec_env.current_step
            self.meta_env.krw_balance = self.exec_env.krw_balance
            self.meta_env.btc_balance = self.exec_env.btc_balance
            self.meta_env.current_strategy = self.exec_env.current_strategy

            meta_obs = self.meta_env._get_meta_observation()
            strategy, _ = self.meta_agent_model.predict(meta_obs, deterministic=True)
            self.exec_env.set_strategy(int(strategy))
        # meta_agent_model이 없으면 현재 전략 유지


# =====================================================================
#  Meta-Agent
# =====================================================================

class MetaAgent:
    """메타 에이전트 — 전략 선택 PPO

    소형 네트워크(64-32)로 시장 국면에 따른 최적 전략을 학습한다.
    """

    def __init__(self, env: MetaEnvironment = None, model_path: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 필요: pip install stable-baselines3")

        self.env = env
        self.model: Optional[PPO] = None
        self.model_path = model_path or str(MODEL_DIR / "meta_agent_ppo")

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=1e-3,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.95,           # 짧은 호라이즌 (메타 기간 단위)
            gae_lambda=0.90,
            clip_range=0.2,
            ent_coef=0.05,        # 탐색 강화 (3개 중 선택)
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
        logger.info("MetaAgent PPO 모델 생성 완료 (64-32)")

    def train(self, total_timesteps: int = 50_000, eval_env=None):
        os.makedirs(str(MODEL_DIR), exist_ok=True)

        callbacks = []
        if eval_env:
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=str(MODEL_DIR / "best_meta"),
                eval_freq=5000,
                n_eval_episodes=5,
                deterministic=True,
            ))

        logger.info(f"MetaAgent 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        self.save()
        logger.info("MetaAgent 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        """전략 선택 + 신뢰도

        Returns:
            (strategy_idx, confidence)
            strategy_idx: 0=conservative, 1=moderate, 2=aggressive
            confidence: 선택된 전략의 확률 (0~1)
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        strategy_idx = int(action)

        # 행동 확률 추출 (신뢰도)
        obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        confidence = float(probs[strategy_idx])

        return strategy_idx, confidence

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"MetaAgent 저장: {path}")

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = PPO.load(path, env=self.env)
        logger.info(f"MetaAgent 로드: {path}")


# =====================================================================
#  Execution-Agent
# =====================================================================

class ExecutionAgent:
    """실행 에이전트 — 포지션 사이징 SAC

    연속 행동 공간에서 최적의 포지션 비중을 학습한다.
    """

    def __init__(self, env: ExecutionEnvironment = None, model_path: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 필요: pip install stable-baselines3")

        self.env = env
        self.model: Optional[SAC] = None
        self.model_path = model_path or str(MODEL_DIR / "exec_agent_sac")

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            learning_starts=1000,
            verbose=1,
            policy_kwargs={
                "net_arch": [256, 128, 64],
            },
        )
        logger.info("ExecutionAgent SAC 모델 생성 완료")

    def train(self, total_timesteps: int = 100_000, eval_env=None):
        os.makedirs(str(MODEL_DIR), exist_ok=True)

        callbacks = []
        if eval_env:
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=str(MODEL_DIR / "best_exec"),
                eval_freq=10_000,
                n_eval_episodes=5,
                deterministic=True,
            ))

        logger.info(f"ExecutionAgent 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        self.save()
        logger.info("ExecutionAgent 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(action[0])

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"ExecutionAgent 저장: {path}")

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = SAC.load(path, env=self.env)
        logger.info(f"ExecutionAgent 로드: {path}")


# =====================================================================
#  Hierarchical Trainer
# =====================================================================

class _MetaTrainingCallback(BaseCallback if SB3_AVAILABLE else object):
    """메타 에이전트 훈련 중 전략 분포 로깅"""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.log_freq = log_freq
        self.strategy_counts = {i: 0 for i in range(NUM_STRATEGIES)}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            strategy_name = info.get("strategy", "moderate")
            idx = STRATEGY_NAMES.index(strategy_name) if strategy_name in STRATEGY_NAMES else 1
            self.strategy_counts[idx] += 1

        if self.num_timesteps % self.log_freq == 0:
            total = sum(self.strategy_counts.values()) or 1
            dist = {STRATEGY_NAMES[i]: f"{self.strategy_counts[i]/total:.0%}"
                    for i in range(NUM_STRATEGIES)}
            logger.info(f"[Meta Step {self.num_timesteps}] 전략 분포: {dist}")

        return True


class HierarchicalTrainer:
    """계층적 교대 훈련

    Phase 1 (커리큘럼): 고정 전략(conservative)으로 실행 에이전트만 훈련
    Phase 2: 메타 에이전트 훈련 (고정된 실행 에이전트 사용)
    Phase 3: 교대 훈련 (메타 K 에피소드 → 실행 M 에피소드)
    """

    def __init__(
        self,
        candles: list[dict] = None,
        external_signals: list[dict] | None = None,
        initial_balance: float = 10_000_000,
        meta_period: int = 6,
        lookback: int = 24,
    ):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 필요")

        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="4h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.external_signals = external_signals
        self.initial_balance = initial_balance
        self.meta_period = meta_period
        self.lookback = lookback

        # 환경 생성
        self.meta_env = MetaEnvironment(
            candles=candles,
            external_signals=external_signals,
            initial_balance=initial_balance,
            meta_period=meta_period,
            lookback=lookback,
        )

        self.exec_env = ExecutionEnvironment(
            candles=candles,
            external_signals=external_signals,
            initial_balance=initial_balance,
            lookback=lookback,
        )

        # 에이전트 생성
        self.meta_agent = MetaAgent(env=self.meta_env)
        self.exec_agent = ExecutionAgent(env=self.exec_env)

        # 훈련 이력
        self.training_history: list[dict] = []

    def train(
        self,
        curriculum_steps: int = 50_000,
        meta_steps: int = 30_000,
        exec_steps: int = 100_000,
        alternating_rounds: int = 3,
        meta_steps_per_round: int = 10_000,
        exec_steps_per_round: int = 30_000,
    ):
        """전체 훈련 파이프라인

        Args:
            curriculum_steps: Phase 1 (고정 전략) 실행 에이전트 훈련 스텝
            meta_steps: Phase 2 메타 에이전트 초기 훈련 스텝
            exec_steps: Phase 2 실행 에이전트 본 훈련 스텝
            alternating_rounds: Phase 3 교대 훈련 라운드 수
            meta_steps_per_round: Phase 3 라운드당 메타 훈련 스텝
            exec_steps_per_round: Phase 3 라운드당 실행 훈련 스텝
        """
        os.makedirs(str(MODEL_DIR), exist_ok=True)

        # ── Phase 1: 커리큘럼 (고정 conservative 전략) ──
        logger.info("=" * 60)
        logger.info("Phase 1: 커리큘럼 -- 고정 전략(conservative)으로 실행 에이전트 훈련")
        logger.info("=" * 60)

        self.exec_env.fixed_strategy = 0  # conservative
        self.exec_agent = ExecutionAgent(env=self.exec_env)
        self.exec_agent.train(total_timesteps=curriculum_steps)

        phase1_stats = self._evaluate_execution_agent()
        self.training_history.append({
            "phase": "curriculum",
            "steps": curriculum_steps,
            "stats": phase1_stats,
        })
        logger.info(f"Phase 1 결과: {phase1_stats}")

        # ── Phase 2: 메타 에이전트 초기 훈련 ──
        logger.info("=" * 60)
        logger.info("Phase 2: 메타 에이전트 초기 훈련")
        logger.info("=" * 60)

        self.meta_agent.train(total_timesteps=meta_steps)

        # 실행 에이전트: 이제 메타 에이전트가 전략을 바꿀 수 있음
        self.exec_env.fixed_strategy = None
        logger.info("Phase 2: 실행 에이전트 본 훈련 (메타 전략 적용)")
        self.exec_agent.train(total_timesteps=exec_steps)

        phase2_stats = self._evaluate_both()
        self.training_history.append({
            "phase": "initial_training",
            "meta_steps": meta_steps,
            "exec_steps": exec_steps,
            "stats": phase2_stats,
        })
        logger.info(f"Phase 2 결과: {phase2_stats}")

        # ── Phase 3: 교대 훈련 ──
        for round_num in range(1, alternating_rounds + 1):
            logger.info("=" * 60)
            logger.info(f"Phase 3: 교대 훈련 라운드 {round_num}/{alternating_rounds}")
            logger.info("=" * 60)

            # 메타 에이전트 훈련
            logger.info(f"  메타 에이전트 훈련: {meta_steps_per_round} 스텝")
            self.meta_agent.train(total_timesteps=meta_steps_per_round)

            # 실행 에이전트 훈련 (메타가 전략을 바꾸며)
            logger.info(f"  실행 에이전트 훈련: {exec_steps_per_round} 스텝")
            self._train_exec_with_meta(exec_steps_per_round)

            round_stats = self._evaluate_both()
            self.training_history.append({
                "phase": f"alternating_round_{round_num}",
                "meta_steps": meta_steps_per_round,
                "exec_steps": exec_steps_per_round,
                "stats": round_stats,
            })
            logger.info(f"  라운드 {round_num} 결과: {round_stats}")

        # 최종 저장
        self.meta_agent.save()
        self.exec_agent.save()

        # 훈련 이력 저장
        history_path = MODEL_DIR / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"훈련 완료. 모델: {MODEL_DIR}")
        return self.training_history

    def _train_exec_with_meta(self, total_timesteps: int):
        """메타 에이전트가 전략을 결정하면서 실행 에이전트를 훈련한다.

        HierarchicalTradingEnv를 사용하여 실행 에이전트가
        메타 에이전트의 전략 지시에 따라 학습한다.
        """
        hier_env = HierarchicalTradingEnv(
            candles=self.candles,
            external_signals=self.external_signals,
            initial_balance=self.initial_balance,
            meta_period=self.meta_period,
            lookback=self.lookback,
            meta_agent_model=self.meta_agent.model,
        )

        # 기존 실행 에이전트의 모델을 새 환경에 연결
        self.exec_agent.model.set_env(hier_env)
        self.exec_agent.model.learn(total_timesteps=total_timesteps)

    def _evaluate_execution_agent(self, n_episodes: int = 10) -> dict:
        """실행 에이전트 단독 평가"""
        returns = []
        trades = []
        mdds = []

        for _ in range(n_episodes):
            obs, _ = self.exec_env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.exec_agent.predict(obs)
                obs, reward, terminated, truncated, info = self.exec_env.step(
                    np.array([action], dtype=np.float32)
                )
                total_reward += reward
                done = terminated or truncated

            stats = self.exec_env.get_episode_stats()
            returns.append(stats.get("total_return_pct", 0))
            trades.append(stats.get("trade_count", 0))
            mdds.append(stats.get("max_drawdown", 0))

        return {
            "avg_return_pct": float(np.mean(returns)),
            "std_return_pct": float(np.std(returns)),
            "avg_trades": float(np.mean(trades)),
            "avg_mdd": float(np.mean(mdds)),
        }

    def _evaluate_both(self, n_episodes: int = 10) -> dict:
        """계층적 시스템 통합 평가"""
        hier_env = HierarchicalTradingEnv(
            candles=self.candles,
            external_signals=self.external_signals,
            initial_balance=self.initial_balance,
            meta_period=self.meta_period,
            lookback=self.lookback,
            meta_agent_model=self.meta_agent.model,
        )

        returns = []
        trades = []
        strategy_counts = {s: 0 for s in STRATEGY_NAMES}

        for _ in range(n_episodes):
            obs, _ = hier_env.reset()
            done = False

            while not done:
                action = self.exec_agent.predict(obs)
                obs, reward, terminated, truncated, info = hier_env.step(
                    np.array([action], dtype=np.float32)
                )
                strategy = info.get("current_strategy", "moderate")
                strategy_counts[strategy] += 1
                done = terminated or truncated

            stats = hier_env.exec_env.get_episode_stats()
            returns.append(stats.get("total_return_pct", 0))
            trades.append(stats.get("trade_count", 0))

        total_counts = sum(strategy_counts.values()) or 1
        strategy_dist = {k: round(v / total_counts, 3) for k, v in strategy_counts.items()}

        return {
            "avg_return_pct": float(np.mean(returns)),
            "std_return_pct": float(np.std(returns)),
            "avg_trades": float(np.mean(trades)),
            "strategy_distribution": strategy_dist,
        }

    def evaluate_comparison(self, n_episodes: int = 10) -> dict:
        """Hierarchical vs Flat RL vs Rule-based 비교 평가

        Returns:
            {"hierarchical": {...}, "flat_rl": {...}, "rule_based": {...}}
        """
        results = {}

        # 1. Hierarchical
        results["hierarchical"] = self._evaluate_both(n_episodes)

        # 2. Flat RL (고정 moderate 전략)
        self.exec_env.fixed_strategy = 1
        results["flat_rl"] = self._evaluate_execution_agent(n_episodes)
        self.exec_env.fixed_strategy = None

        # 3. Rule-based (전략별 고정 성능)
        rule_returns = []
        for strategy_idx in range(NUM_STRATEGIES):
            self.exec_env.fixed_strategy = strategy_idx
            stats = self._evaluate_execution_agent(n_episodes // NUM_STRATEGIES or 3)
            rule_returns.append(stats["avg_return_pct"])
            self.exec_env.fixed_strategy = None

        results["rule_based"] = {
            "conservative_return_pct": rule_returns[0],
            "moderate_return_pct": rule_returns[1],
            "aggressive_return_pct": rule_returns[2],
            "avg_return_pct": float(np.mean(rule_returns)),
        }

        # 비교 요약
        hier_ret = results["hierarchical"]["avg_return_pct"]
        flat_ret = results["flat_rl"]["avg_return_pct"]
        rule_ret = results["rule_based"]["avg_return_pct"]

        results["summary"] = {
            "hierarchical_vs_flat": round(hier_ret - flat_ret, 3),
            "hierarchical_vs_rule": round(hier_ret - rule_ret, 3),
            "best_system": max(
                [("hierarchical", hier_ret), ("flat_rl", flat_ret), ("rule_based", rule_ret)],
                key=lambda x: x[1]
            )[0],
        }

        logger.info(f"비교 평가 결과: {results['summary']}")
        return results


# =====================================================================
#  HierarchicalOrchestrator — 기존 Orchestrator 대체
# =====================================================================

class HierarchicalOrchestrator:
    """계층적 RL 기반 오케스트레이터

    기존 규칙 기반 Orchestrator를 메타 에이전트로 대체한다.
    메타 에이전트의 신뢰도가 낮으면 규칙 기반으로 폴백한다.
    """

    # 폴백 임계값: 메타 에이전트 신뢰도가 이보다 낮으면 규칙 기반 사용
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(
        self,
        meta_model_path: str = None,
        fallback_to_rules: bool = True,
    ):
        self.meta_agent: Optional[MetaAgent] = None
        self.fallback_to_rules = fallback_to_rules
        self._rule_based_orchestrator = None

        meta_path = meta_model_path or str(MODEL_DIR / "meta_agent_ppo")
        if os.path.exists(meta_path + ".zip"):
            try:
                self.meta_agent = MetaAgent(model_path=meta_path)
                logger.info(f"HierarchicalOrchestrator: 메타 에이전트 로드 완료")
            except Exception as e:
                logger.warning(f"메타 에이전트 로드 실패: {e} -- 규칙 기반 폴백")
        else:
            logger.warning(f"메타 에이전트 없음: {meta_path} -- 규칙 기반 폴백")

    def select_strategy(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
    ) -> dict:
        """전략 선택 + 근거

        Returns:
            {
                "strategy": "conservative"|"moderate"|"aggressive",
                "confidence": float,
                "source": "meta_agent"|"rule_based",
                "reason": str,
                "meta_obs": list (디버깅용),
            }
        """
        # 메타 에이전트 사용 가능 시
        if self.meta_agent is not None:
            meta_obs = self._build_meta_observation(market_data, external_data, portfolio)
            strategy_idx, confidence = self.meta_agent.predict(meta_obs)
            strategy_name = STRATEGY_NAMES[strategy_idx]

            # 신뢰도 기반 폴백 판단
            if confidence < self.CONFIDENCE_THRESHOLD and self.fallback_to_rules:
                rule_result = self._rule_based_select(market_data, external_data, portfolio)
                logger.info(
                    f"메타 에이전트 신뢰도 부족 ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD}) "
                    f"→ 규칙 기반 폴백: {rule_result['strategy']}"
                )
                rule_result["meta_suggestion"] = strategy_name
                rule_result["meta_confidence"] = confidence
                return rule_result

            return {
                "strategy": strategy_name,
                "confidence": confidence,
                "source": "meta_agent",
                "reason": (
                    f"MetaAgent PPO 선택: {strategy_name} "
                    f"(신뢰도 {confidence:.2f})"
                ),
                "meta_obs": meta_obs.tolist(),
            }

        # 폴백: 규칙 기반
        return self._rule_based_select(market_data, external_data, portfolio)

    def _build_meta_observation(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
    ) -> np.ndarray:
        """라이브 데이터로부터 메타 관측 벡터 구축 (16차원)"""
        indicators = market_data.get("indicators", {})
        ticker = market_data.get("ticker", {})
        sources = external_data.get("sources", external_data)

        price = ticker.get("trade_price", 0) or market_data.get("current_price", 0)
        atr = indicators.get("atr", 0) or 0
        volatility = atr / price if price > 0 else 0

        adx_data = indicators.get("adx", {})
        adx_val = adx_data.get("adx", 25) if isinstance(adx_data, dict) else float(adx_data or 25)
        trend_strength = adx_val / 100.0

        # FGI
        fgi_data = sources.get("fear_greed", {})
        if isinstance(fgi_data.get("current"), dict):
            fgi = fgi_data["current"].get("value", 50)
        else:
            fgi = fgi_data.get("value", 50)
        fgi = fgi or 50
        fgi_norm = float(fgi) / 100.0

        rsi = float(indicators.get("rsi_14", 50) or 50)
        price_change_24h = float(ticker.get("signed_change_rate", 0))

        # Danger / opportunity (Orchestrator 방식)
        danger = 0.0
        if price_change_24h < -0.03:
            danger += min(abs(price_change_24h) * 5, 0.5)
        if rsi > 70:
            danger += (rsi - 70) / 100
        danger = min(danger, 1.0)

        opp = 0.0
        if fgi < 25:
            opp += (25 - float(fgi)) / 100
        if rsi < 35:
            opp += (35 - rsi) / 100
        if price_change_24h > 0.01:
            opp += min(price_change_24h * 3, 0.3)
        opp = min(opp, 1.0)

        # 포지션 비율
        btc_ratio = 0.0
        total_eval = float(portfolio.get("total_eval", 1) or 1)
        for h in portfolio.get("holdings", []):
            if h.get("currency") == "BTC":
                btc_ratio = float(h.get("eval_amount", 0)) / total_eval
                break

        # 바이낸스 데이터
        binance = sources.get("binance_sentiment", {})
        funding_data = binance.get("funding_rate", {})
        funding = float(funding_data.get("current_rate", 0) if isinstance(funding_data, dict) else 0)
        kimchi_data = binance.get("kimchi_premium", {})
        kimchi = float(kimchi_data.get("premium_pct", 0) if isinstance(kimchi_data, dict) else 0)

        # 전략별 성과: DB에서 로드 가능하나, 여기서는 기본값
        strategy_perf = [0.5, 0.5, 0.5]

        # 현재 전략: 기본 moderate
        strategy_onehot = [0.0, 1.0, 0.0]

        obs = np.array([
            float(np.clip(volatility * 100, 0, 1)),
            float(trend_strength),
            float(fgi_norm),
            float(danger),
            float(opp),
            float(np.clip(price_change_24h * 5 + 0.5, 0, 1)),
            float(rsi / 100.0),
            float(np.clip(funding * 10 + 0.5, 0, 1)),
            float(np.clip(kimchi / 20 + 0.5, 0, 1)),
            float(np.clip(btc_ratio, 0, 1)),
            strategy_perf[0],
            strategy_perf[1],
            strategy_perf[2],
            strategy_onehot[0],
            strategy_onehot[1],
            strategy_onehot[2],
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _rule_based_select(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
    ) -> dict:
        """규칙 기반 전략 선택 (기존 Orchestrator 로직 경량 버전)"""
        indicators = market_data.get("indicators", {})
        ticker = market_data.get("ticker", {})
        sources = external_data.get("sources", external_data)

        rsi = float(indicators.get("rsi_14", 50) or 50)
        price_change = float(ticker.get("signed_change_rate", 0)) * 100

        fgi_data = sources.get("fear_greed", {})
        if isinstance(fgi_data.get("current"), dict):
            fgi = fgi_data["current"].get("value", 50)
        else:
            fgi = fgi_data.get("value", 50)
        fgi = fgi or 50

        # 간단한 danger/opportunity 계산
        danger = 0
        if price_change < -3:
            danger += min(int(abs(price_change) * 5), 25)
        if rsi > 70:
            danger += 15

        opportunity = 0
        if fgi <= 25:
            opportunity += 25 - fgi
        if rsi < 35:
            opportunity += int((35 - rsi) * 1.3)
        if price_change > 1:
            opportunity += min(int(price_change * 5), 15)

        # 전환 규칙
        if danger >= 70:
            strategy = "conservative"
            reason = f"위험도 극심 (danger={danger}) → 보수적"
        elif danger >= 50:
            strategy = "conservative"
            reason = f"위험도 높음 (danger={danger}) → 보수적"
        elif opportunity >= 60 and danger < 30:
            strategy = "aggressive"
            reason = f"기회 높음 (opp={opportunity}, danger={danger}) → 공격적"
        elif opportunity >= 40 and danger < 35:
            strategy = "moderate"
            reason = f"기회 보통 (opp={opportunity}, danger={danger}) → 보통"
        else:
            strategy = "moderate"
            reason = f"기본 (danger={danger}, opp={opportunity}) → 보통"

        return {
            "strategy": strategy,
            "confidence": 0.6,
            "source": "rule_based",
            "reason": reason,
            "meta_obs": [],
        }

    def log_decision_to_db(
        self,
        strategy_result: dict,
        market_state: dict,
    ) -> bool:
        """메타 에이전트 결정을 agent_switches 테이블에 기록

        Returns:
            True if successful
        """
        try:
            import requests
            from dotenv import load_dotenv
            load_dotenv(PROJECT_DIR / ".env")

            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
            if not supabase_url or not supabase_key:
                return False

            row = {
                "from_agent": strategy_result.get("meta_suggestion", "unknown"),
                "to_agent": strategy_result["strategy"],
                "reason": strategy_result["reason"],
                "source": strategy_result["source"],
                "meta_confidence": strategy_result.get("confidence", 0),
                "danger_score": market_state.get("danger_score", 0),
                "opportunity_score": market_state.get("opportunity_score", 0),
                "fgi": market_state.get("fgi", 50),
                "rsi": market_state.get("rsi", 50),
                "price_change_24h": market_state.get("price_change_24h", 0),
                "switched_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            }

            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
            }

            resp = requests.post(
                f"{supabase_url}/rest/v1/agent_switches",
                headers=headers,
                json=row,
                timeout=10,
            )

            if resp.ok:
                logger.info(f"메타 결정 DB 기록: {strategy_result['strategy']}")
                return True
            else:
                logger.warning(f"DB 기록 실패: {resp.status_code}")
                return False

        except Exception as e:
            logger.warning(f"DB 기록 예외: {e}")
            return False
