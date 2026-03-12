"""Multi-Agent Consensus RL — 다중 에이전트 합의 기반 트레이딩 시스템

스캘핑(1h) + 스윙(4h-1d) RL 에이전트가 독립적으로 시장을 분석하고,
합의 엔진이 시장 국면에 따라 동적 가중치로 최종 행동을 결정한다.

구성:
  - ScalpingAgent: SAC 기반 단기 트레이더 (1h, 30차원 관측)
  - SwingAgent: PPO 기반 중기 트레이더 (4h-1d, 50차원 관측)
  - ConsensusEngine: 동적 가중 + 거부권 + 신뢰도 결합
  - WeightLearner: 가중치 최적화 메타-RL (PPO, 32-16 네트워크)
  - MultiAgentTradingEnv: 다중 타임프레임 환경 래퍼
  - MultiAgentTrainer: 3단계 훈련 파이프라인
  - MultiAgentPredictor: DecisionBlender 호환 추론 인터페이스
"""

import logging
import os
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM, FEATURE_SPEC
from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST
from rl_hybrid.rl.data_loader import HistoricalDataLoader

logger = logging.getLogger("rl.multi_agent")

# SB3 lazy import
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    import torch
    import torch.nn as nn
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 미설치 -- Multi-Agent RL 비활성화")

# 모델 저장 경로
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models", "multi_agent",
)


# ============================================================
#  관측 공간 정의
# ============================================================

# 스캘핑 에이전트 관측 (30차원): 단기 가격/기술/거래량/포트폴리오 중심
SCALPING_FEATURES = [
    "price_change_1h", "price_change_4h", "price_change_24h",
    "price_vs_sma20", "price_vs_boll_upper", "price_vs_boll_lower",
    "rsi_14", "macd_histogram", "stochastic_k", "stochastic_d",
    "atr_pct", "bollinger_width",
    "volume_change_ratio", "orderbook_ratio", "trade_pressure_ratio",
    "whale_buy_ratio",
    "fgi", "funding_rate", "long_short_ratio", "kimchi_premium",
    "ai_composite_signal",
    "danger_score", "opportunity_score", "cascade_risk",
    "fusion_score",
    "position_ratio", "unrealized_pnl", "cash_ratio",
    "hours_since_last_trade", "daily_trade_count",
]
SCALPING_DIM = len(SCALPING_FEATURES)  # 30

# 스윙 에이전트 관측 (50차원): 전체 42차원 + 매크로 확장 8차원
SWING_EXTRA_FEATURES = [
    "trend_strength",       # ADX 정규화
    "trend_direction",      # +DI - -DI 정규화
    "fgi_trend",            # FGI 7일 변화율
    "macro_momentum",       # S&P500 변화 방향
    "btc_dominance_delta",  # BTC 도미넌스 변화
    "funding_pressure",     # 펀딩비 누적 방향
    "volume_trend",         # 거래량 추세 (5봉 SMA vs 20봉 SMA)
    "price_momentum",       # 가격 모멘텀 (12봉 EMA vs 26봉 EMA)
]
SWING_DIM = OBSERVATION_DIM + len(SWING_EXTRA_FEATURES)  # 42 + 8 = 50

# 가중치 학습기 관측 (12차원)
WEIGHT_LEARNER_DIM = 12

# feature name → index in FEATURE_SPEC
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_SPEC.keys())}


# ============================================================
#  스캘핑 보상 함수
# ============================================================

class ScalpingRewardCalculator:
    """스캘핑 전용 보상: 거래별 PnL + 장기 보유 페널티"""

    def __init__(self, hold_penalty_threshold: int = 4):
        """
        Args:
            hold_penalty_threshold: 이 스텝 이상 보유 시 페널티 (1h봉 기준 4시간)
        """
        self.hold_penalty_threshold = hold_penalty_threshold
        self.steps_in_position = 0
        self.entry_value = 0.0
        self.peak_value = 0.0
        self.max_drawdown = 0.0

    def reset(self, initial_value: float):
        self.steps_in_position = 0
        self.entry_value = initial_value
        self.peak_value = initial_value
        self.max_drawdown = 0.0

    def calculate(
        self,
        prev_value: float,
        curr_value: float,
        action: float,
        prev_action: float,
        btc_ratio: float,
    ) -> dict:
        raw_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0.0

        # 포지션 보유 시간 추적
        if btc_ratio > 0.05:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0

        # PnL 보상 (스케일링)
        pnl_reward = raw_return * 20.0

        # 장기 보유 페널티 (4스텝 = 4시간 이상)
        hold_penalty = 0.0
        if self.steps_in_position > self.hold_penalty_threshold:
            excess = self.steps_in_position - self.hold_penalty_threshold
            hold_penalty = -0.02 * excess  # 시간당 -0.02

        # MDD 페널티
        self.peak_value = max(self.peak_value, curr_value)
        drawdown = (self.peak_value - curr_value) / self.peak_value if self.peak_value > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        mdd_penalty = -drawdown * 1.5 if drawdown > 0.03 else 0.0

        # 거래 보상: 빠른 수익 실현 보너스
        trade_bonus = 0.0
        action_change = abs(action - prev_action)
        if action_change > 0.1 and raw_return > 0.002:
            trade_bonus = 0.15

        total = pnl_reward + hold_penalty + mdd_penalty + trade_bonus

        return {
            "reward": float(np.clip(total, -2.0, 2.0)),
            "components": {
                "raw_return": float(raw_return),
                "pnl_reward": float(pnl_reward),
                "hold_penalty": float(hold_penalty),
                "mdd_penalty": float(mdd_penalty),
                "trade_bonus": float(trade_bonus),
                "steps_in_position": self.steps_in_position,
            },
        }


# ============================================================
#  스윙 보상 함수
# ============================================================

class SwingRewardCalculator(RewardCalculator):
    """스윙 전용 보상: Differential Sharpe + 트렌드 추종 보너스"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_trend_aligned = False

    def calculate(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: float,
        prev_action: float,
        step: int,
        trend_strength: float = 0.0,
        trend_direction: float = 0.0,
    ) -> dict:
        base_result = super().calculate(
            prev_portfolio_value, curr_portfolio_value,
            action, prev_action, step,
        )

        # 트렌드 추종 보너스: 강한 트렌드 방향으로 포지션 유지 시
        trend_bonus = 0.0
        if abs(trend_strength) > 0.5:  # ADX 정규화 기준 강한 트렌드
            # action과 트렌드 방향이 일치하면 보너스
            alignment = action * trend_direction
            if alignment > 0.3:
                trend_bonus = 0.05 * abs(trend_strength)
                self.prev_trend_aligned = True
            elif self.prev_trend_aligned and alignment < -0.1:
                # 트렌드 전환 감지 시 빠른 대응 보상
                trend_bonus = 0.03
                self.prev_trend_aligned = False

        base_result["reward"] += trend_bonus
        base_result["components"]["trend_bonus"] = float(trend_bonus)
        return base_result


# ============================================================
#  스캘핑 환경
# ============================================================

class ScalpingTradingEnv(gym.Env):
    """스캘핑 에이전트 전용 환경 (1h 캔들, 30차원 관측)"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: list[dict] = None,
        initial_balance: float = 10_000_000,
        max_steps: int = None,
        lookback: int = 24,
        external_signals: list[dict] | None = None,
    ):
        super().__init__()

        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=90, interval="1h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.external_signals = external_signals
        self.initial_balance = initial_balance
        self.lookback = lookback

        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        if max_steps:
            self.end_idx = min(self.start_idx + max_steps, self.end_idx)
        if self.end_idx <= self.start_idx:
            self.end_idx = len(candles) - 1
            self.start_idx = min(self.start_idx, self.end_idx - 1)
        self.start_idx = max(0, self.start_idx)

        # 30차원 관측 공간
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(SCALPING_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self.encoder = StateEncoder()
        self.reward_calc = ScalpingRewardCalculator()

        # 스캘핑 특성 인덱스 (전체 42차원에서 30차원 추출)
        self._scalping_indices = np.array(
            [_FEATURE_INDEX[f] for f in SCALPING_FEATURES], dtype=np.int32,
        )

        self.current_step = 0
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.total_value_history = []
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and options.get("start_idx"):
            self.current_step = options["start_idx"]
        else:
            max_start = max(self.start_idx, self.end_idx - 200)
            self.current_step = self.np_random.integers(self.start_idx, max_start + 1)

        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.total_value_history = [self.initial_balance]
        self.trade_count = 0

        self.reward_calc.reset(self.initial_balance)
        obs = self._get_observation()
        return obs, self._get_info()

    def step(self, action: np.ndarray):
        action_val = float(np.clip(action[0], -1, 1))
        candle = self.candles[self.current_step]
        price = candle["close"]
        prev_value = self._portfolio_value(price)

        self._execute_action(action_val, price)
        self.current_step += 1
        next_price = self.candles[self.current_step]["close"]
        curr_value = self._portfolio_value(next_price)
        self.total_value_history.append(curr_value)

        btc_ratio = (self.btc_balance * next_price / curr_value) if curr_value > 0 else 0
        reward_info = self.reward_calc.calculate(
            prev_value, curr_value, action_val, self.prev_action, btc_ratio,
        )
        self.prev_action = action_val

        terminated = curr_value < self.initial_balance * 0.1
        truncated = self.current_step >= self.end_idx

        if terminated:
            reward_info["reward"] -= 1.0

        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_info["components"]
        return obs, reward_info["reward"], terminated, truncated, info

    def _execute_action(self, action: float, price: float):
        total_value = self._portfolio_value(price)
        target_btc_ratio = (action + 1) / 2
        current_btc_value = self.btc_balance * price
        target_btc_value = total_value * target_btc_ratio
        diff = target_btc_value - current_btc_value

        if diff > 0 and self.krw_balance > 0:
            buy_amount = min(diff, self.krw_balance)
            cost = buy_amount * TRANSACTION_COST
            btc_bought = (buy_amount - cost) / price
            self.krw_balance -= buy_amount
            self.btc_balance += btc_bought
            if buy_amount > total_value * 0.01:
                self.trade_count += 1
        elif diff < 0 and self.btc_balance > 0:
            sell_value = min(-diff, current_btc_value)
            btc_sold = min(sell_value / price, self.btc_balance)
            proceeds = btc_sold * price * (1 - TRANSACTION_COST)
            self.btc_balance -= btc_sold
            self.krw_balance += proceeds
            if sell_value > total_value * 0.01:
                self.trade_count += 1

    def _portfolio_value(self, price: float) -> float:
        return self.krw_balance + self.btc_balance * price

    def _get_observation(self) -> np.ndarray:
        """전체 42차원 인코딩 후 30차원 서브셋 추출"""
        candle = self.candles[self.current_step]
        price = candle["close"]

        market_data = self._build_market_data(candle, price)
        external_data = self._build_external_data()
        portfolio = self._build_portfolio(price)
        agent_state = {
            "danger_score": 30, "opportunity_score": 30,
            "cascade_risk": 20, "consecutive_losses": 0,
            "hours_since_last_trade": 24, "daily_trade_count": self.trade_count,
        }

        full_obs = self.encoder.encode(market_data, external_data, portfolio, agent_state)
        return full_obs[self._scalping_indices]

    def _build_market_data(self, candle: dict, price: float) -> dict:
        return {
            "current_price": price,
            "change_rate_24h": candle.get("change_rate", 0),
            "indicators": {
                "sma_20": candle.get("sma_20", price),
                "sma_50": candle.get("sma_50", price),
                "rsi_14": candle.get("rsi_14", 50),
                "macd": {"macd": candle.get("macd", 0), "signal": candle.get("macd_signal", 0),
                         "histogram": candle.get("macd_histogram", 0)},
                "bollinger": {"upper": candle.get("boll_upper", price),
                              "middle": candle.get("boll_middle", price),
                              "lower": candle.get("boll_lower", price)},
                "stochastic": {"k": candle.get("stoch_k", 50), "d": candle.get("stoch_d", 50)},
                "adx": {"adx": candle.get("adx", 25), "plus_di": candle.get("adx_plus_di", 20),
                         "minus_di": candle.get("adx_minus_di", 20)},
                "atr": candle.get("atr", 0),
            },
            "indicators_4h": {"rsi_14": candle.get("rsi_14", 50)},
            "orderbook": {"ratio": 1.0},
            "trade_pressure": {"buy_volume": 1, "sell_volume": 1},
            "eth_btc_analysis": {"eth_btc_z_score": 0},
        }

    def _build_external_data(self) -> dict:
        fgi, news, whale, funding = 50, 0, 0, 0.0
        ls_ratio, kimchi, macro, fusion = 1.0, 0.0, 0, 0

        if (self.external_signals is not None
                and self.current_step < len(self.external_signals)):
            sig = self.external_signals[self.current_step]
            fgi = sig.get("fgi_value", 50) or 50
            whale = sig.get("whale_score", 0) or 0
            funding = sig.get("funding_rate", 0.0) or 0.0
            ls_ratio = sig.get("long_short_ratio", 1.0) or 1.0
            kimchi = sig.get("kimchi_premium_pct", 0.0) or 0.0
            macro = sig.get("macro_score", 0) or 0
            fusion = sig.get("fusion_score", 0) or 0

        return {
            "sources": {
                "fear_greed": {"current": {"value": float(fgi)}},
                "news_sentiment": {"sentiment_score": float(news)},
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

    def _build_portfolio(self, price: float) -> dict:
        total = self._portfolio_value(price)
        btc_eval = self.btc_balance * price
        holdings = []
        if self.btc_balance > 0:
            holdings.append({
                "currency": "BTC", "balance": self.btc_balance,
                "avg_buy_price": price, "eval_amount": btc_eval,
                "profit_loss_pct": 0,
            })
        return {"krw_balance": self.krw_balance, "holdings": holdings, "total_eval": total}

    def _get_info(self) -> dict:
        candle = self.candles[self.current_step]
        price = candle["close"]
        total = self._portfolio_value(price)
        return {
            "step": self.current_step,
            "price": price,
            "portfolio_value": total,
            "return_pct": (total - self.initial_balance) / self.initial_balance * 100,
            "trade_count": self.trade_count,
        }

    def get_episode_stats(self) -> dict:
        final_value = self.total_value_history[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        returns = []
        for i in range(1, len(self.total_value_history)):
            r = (self.total_value_history[i] - self.total_value_history[i - 1]) / self.total_value_history[i - 1]
            returns.append(r)
        returns = np.array(returns) if returns else np.array([0.0])

        peak = self.initial_balance
        max_dd = 0.0
        for v in self.total_value_history:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        std = returns.std() if len(returns) > 1 else 1e-8
        sharpe = float((returns.mean()) / std) if std > 1e-8 else 0.0

        return {
            "total_return_pct": float(total_return * 100),
            "sharpe_ratio": sharpe,
            "max_drawdown": float(max_dd),
            "trade_count": self.trade_count,
            "final_value": float(final_value),
            "steps": len(self.total_value_history) - 1,
        }


# ============================================================
#  스윙 환경
# ============================================================

class SwingTradingEnv(gym.Env):
    """스윙 에이전트 전용 환경 (4h 캔들, 50차원 관측)"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: list[dict] = None,
        initial_balance: float = 10_000_000,
        max_steps: int = None,
        lookback: int = 42,
        external_signals: list[dict] | None = None,
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

        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        if max_steps:
            self.end_idx = min(self.start_idx + max_steps, self.end_idx)
        if self.end_idx <= self.start_idx:
            self.end_idx = len(candles) - 1
            self.start_idx = min(self.start_idx, self.end_idx - 1)
        self.start_idx = max(0, self.start_idx)

        # 50차원 관측 공간 (42 기본 + 8 매크로 확장)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(SWING_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self.encoder = StateEncoder()
        self.reward_calc = SwingRewardCalculator(
            window_size=20,
            risk_free_rate=0.03 / 365 / 6,  # 4h봉 기준 조정
            max_drawdown_penalty=2.0,
        )

        self.current_step = 0
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.total_value_history = []
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and options.get("start_idx"):
            self.current_step = options["start_idx"]
        else:
            max_start = max(self.start_idx, self.end_idx - 500)
            self.current_step = self.np_random.integers(self.start_idx, max_start + 1)

        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.total_value_history = [self.initial_balance]
        self.trade_count = 0
        self.reward_calc.reset(self.initial_balance)

        obs = self._get_observation()
        return obs, self._get_info()

    def step(self, action: np.ndarray):
        action_val = float(np.clip(action[0], -1, 1))
        candle = self.candles[self.current_step]
        price = candle["close"]
        prev_value = self._portfolio_value(price)

        self._execute_action(action_val, price)
        self.current_step += 1
        next_candle = self.candles[self.current_step]
        next_price = next_candle["close"]
        curr_value = self._portfolio_value(next_price)
        self.total_value_history.append(curr_value)

        # 트렌드 정보 추출
        adx = next_candle.get("adx", 25)
        plus_di = next_candle.get("adx_plus_di", 20)
        minus_di = next_candle.get("adx_minus_di", 20)
        trend_strength = min(adx / 50.0, 1.0)
        trend_direction = 1.0 if plus_di > minus_di else -1.0

        reward_info = self.reward_calc.calculate(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=curr_value,
            action=action_val,
            prev_action=self.prev_action,
            step=self.current_step,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
        )
        self.prev_action = action_val

        terminated = curr_value < self.initial_balance * 0.1
        truncated = self.current_step >= self.end_idx
        if terminated:
            reward_info["reward"] -= 1.0

        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_info["components"]
        return obs, reward_info["reward"], terminated, truncated, info

    def _execute_action(self, action: float, price: float):
        """BitcoinTradingEnv와 동일한 실행 로직"""
        total_value = self._portfolio_value(price)
        target_btc_ratio = (action + 1) / 2
        current_btc_value = self.btc_balance * price
        target_btc_value = total_value * target_btc_ratio
        diff = target_btc_value - current_btc_value

        if diff > 0 and self.krw_balance > 0:
            buy_amount = min(diff, self.krw_balance)
            cost = buy_amount * TRANSACTION_COST
            btc_bought = (buy_amount - cost) / price
            self.krw_balance -= buy_amount
            self.btc_balance += btc_bought
            if buy_amount > total_value * 0.01:
                self.trade_count += 1
        elif diff < 0 and self.btc_balance > 0:
            sell_value = min(-diff, current_btc_value)
            btc_sold = min(sell_value / price, self.btc_balance)
            proceeds = btc_sold * price * (1 - TRANSACTION_COST)
            self.btc_balance -= btc_sold
            self.krw_balance += proceeds
            if sell_value > total_value * 0.01:
                self.trade_count += 1

    def _portfolio_value(self, price: float) -> float:
        return self.krw_balance + self.btc_balance * price

    def _get_observation(self) -> np.ndarray:
        """42차원 기본 + 8차원 매크로 확장 = 50차원"""
        candle = self.candles[self.current_step]
        price = candle["close"]

        # 기본 42차원
        market_data = self._build_market_data(candle, price)
        external_data = self._build_external_data()
        portfolio = self._build_portfolio(price)
        agent_state = {
            "danger_score": 30, "opportunity_score": 30,
            "cascade_risk": 20, "consecutive_losses": 0,
            "hours_since_last_trade": 24, "daily_trade_count": self.trade_count,
        }
        base_obs = self.encoder.encode(market_data, external_data, portfolio, agent_state)

        # 확장 8차원
        adx = candle.get("adx", 25)
        plus_di = candle.get("adx_plus_di", 20)
        minus_di = candle.get("adx_minus_di", 20)
        ema12 = candle.get("ema_12", price)
        ema26 = candle.get("ema_26", price)

        extra = np.array([
            min(adx / 50.0, 1.0),                              # trend_strength
            np.clip((plus_di - minus_di) / 30.0, -1, 1),       # trend_direction
            self._get_fgi_trend(),                               # fgi_trend
            self._get_macro_momentum(),                          # macro_momentum
            0.5,                                                 # btc_dominance_delta (default)
            self._get_funding_pressure(),                        # funding_pressure
            self._get_volume_trend(candle),                      # volume_trend
            np.clip((ema12 - ema26) / ema26, -0.05, 0.05) * 10, # price_momentum
        ], dtype=np.float32)

        # 정규화 [0, 1]
        extra = np.clip((extra + 1) / 2, 0, 1)

        return np.concatenate([base_obs, extra])

    def _get_fgi_trend(self) -> float:
        """FGI 변화 추세 (외부 시그널 기반)"""
        if self.external_signals is None:
            return 0.0
        idx = min(self.current_step, len(self.external_signals) - 1)
        curr_fgi = self.external_signals[idx].get("fgi_value", 50) or 50
        # 7스텝 전 (4h * 7 = ~28시간)
        prev_idx = max(0, idx - 7)
        prev_fgi = self.external_signals[prev_idx].get("fgi_value", 50) or 50
        return np.clip((curr_fgi - prev_fgi) / 30.0, -1, 1)

    def _get_macro_momentum(self) -> float:
        """매크로 모멘텀 (외부 시그널)"""
        if self.external_signals is None:
            return 0.0
        idx = min(self.current_step, len(self.external_signals) - 1)
        macro = self.external_signals[idx].get("macro_score", 0) or 0
        return np.clip(macro / 30.0, -1, 1)

    def _get_funding_pressure(self) -> float:
        """펀딩비 누적 압력"""
        if self.external_signals is None:
            return 0.0
        idx = min(self.current_step, len(self.external_signals) - 1)
        funding = self.external_signals[idx].get("funding_rate", 0.0) or 0.0
        return np.clip(funding * 100, -1, 1)

    def _get_volume_trend(self, candle: dict) -> float:
        """거래량 추세"""
        vol = candle.get("volume", 0)
        vol_sma = candle.get("volume_sma20", vol) or vol
        if vol_sma > 0:
            return np.clip((vol / vol_sma - 1.0), -1, 1)
        return 0.0

    def _build_market_data(self, candle: dict, price: float) -> dict:
        return {
            "current_price": price,
            "change_rate_24h": candle.get("change_rate", 0),
            "indicators": {
                "sma_20": candle.get("sma_20", price),
                "sma_50": candle.get("sma_50", price),
                "rsi_14": candle.get("rsi_14", 50),
                "macd": {"macd": candle.get("macd", 0), "signal": candle.get("macd_signal", 0),
                         "histogram": candle.get("macd_histogram", 0)},
                "bollinger": {"upper": candle.get("boll_upper", price),
                              "middle": candle.get("boll_middle", price),
                              "lower": candle.get("boll_lower", price)},
                "stochastic": {"k": candle.get("stoch_k", 50), "d": candle.get("stoch_d", 50)},
                "adx": {"adx": candle.get("adx", 25), "plus_di": candle.get("adx_plus_di", 20),
                         "minus_di": candle.get("adx_minus_di", 20)},
                "atr": candle.get("atr", 0),
            },
            "indicators_4h": {"rsi_14": candle.get("rsi_14", 50)},
            "orderbook": {"ratio": 1.0},
            "trade_pressure": {"buy_volume": 1, "sell_volume": 1},
            "eth_btc_analysis": {"eth_btc_z_score": 0},
        }

    def _build_external_data(self) -> dict:
        fgi, news, whale, funding = 50, 0, 0, 0.0
        ls_ratio, kimchi, macro, fusion = 1.0, 0.0, 0, 0

        if (self.external_signals is not None
                and self.current_step < len(self.external_signals)):
            sig = self.external_signals[self.current_step]
            fgi = sig.get("fgi_value", 50) or 50
            whale = sig.get("whale_score", 0) or 0
            funding = sig.get("funding_rate", 0.0) or 0.0
            ls_ratio = sig.get("long_short_ratio", 1.0) or 1.0
            kimchi = sig.get("kimchi_premium_pct", 0.0) or 0.0
            macro = sig.get("macro_score", 0) or 0
            fusion = sig.get("fusion_score", 0) or 0

        return {
            "sources": {
                "fear_greed": {"current": {"value": float(fgi)}},
                "news_sentiment": {"sentiment_score": float(news)},
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

    def _build_portfolio(self, price: float) -> dict:
        total = self._portfolio_value(price)
        btc_eval = self.btc_balance * price
        holdings = []
        if self.btc_balance > 0:
            holdings.append({
                "currency": "BTC", "balance": self.btc_balance,
                "avg_buy_price": price, "eval_amount": btc_eval,
                "profit_loss_pct": 0,
            })
        return {"krw_balance": self.krw_balance, "holdings": holdings, "total_eval": total}

    def _get_info(self) -> dict:
        candle = self.candles[self.current_step]
        price = candle["close"]
        total = self._portfolio_value(price)
        return {
            "step": self.current_step,
            "price": price,
            "portfolio_value": total,
            "return_pct": (total - self.initial_balance) / self.initial_balance * 100,
            "trade_count": self.trade_count,
        }

    def get_episode_stats(self) -> dict:
        final_value = self.total_value_history[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        returns = []
        for i in range(1, len(self.total_value_history)):
            r = (self.total_value_history[i] - self.total_value_history[i - 1]) / self.total_value_history[i - 1]
            returns.append(r)
        returns = np.array(returns) if returns else np.array([0.0])

        peak = self.initial_balance
        max_dd = 0.0
        for v in self.total_value_history:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        std = returns.std() if len(returns) > 1 else 1e-8
        sharpe = float(returns.mean() / std) if std > 1e-8 else 0.0

        return {
            "total_return_pct": float(total_return * 100),
            "sharpe_ratio": sharpe,
            "max_drawdown": float(max_dd),
            "trade_count": self.trade_count,
            "final_value": float(final_value),
            "steps": len(self.total_value_history) - 1,
        }


# ============================================================
#  개별 에이전트 래퍼
# ============================================================

class ScalpingAgent:
    """SAC 기반 스캘핑 에이전트 (1h 캔들, 30차원 관측)

    단기 변동성 포착에 특화. 4시간 이상 보유 시 페널티.
    """

    def __init__(self, env=None, model_path: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3가 필요합니다")

        self.env = env
        self.model: Optional[SAC] = None
        self.model_path = model_path or os.path.join(MODEL_DIR, "scalping_sac")
        self.recent_accuracy: deque = deque(maxlen=50)

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
            policy_kwargs={"net_arch": [128, 64]},
        )
        logger.info("ScalpingAgent SAC 모델 생성 완료 (128-64)")

    def train(self, total_timesteps: int = 200_000, eval_env=None, save_freq: int = 20_000):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        callbacks = [_MultiAgentMetricsCallback(agent_name="scalping")]
        if eval_env:
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(MODEL_DIR, "best_scalping"),
                eval_freq=save_freq,
                n_eval_episodes=5,
                deterministic=True,
            ))

        logger.info(f"ScalpingAgent 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        self.save()
        logger.info("ScalpingAgent 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[float, float]:
        """관측 → (행동, 신뢰도)"""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action_val = float(action[0])
        confidence = min(abs(action_val), 1.0)
        return action_val, confidence

    def record_outcome(self, correct: bool):
        self.recent_accuracy.append(1.0 if correct else 0.0)

    def get_accuracy(self) -> float:
        if not self.recent_accuracy:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = SAC.load(path, env=self.env)
        logger.info(f"ScalpingAgent 로드: {path}")


class SwingAgent:
    """PPO 기반 스윙 에이전트 (4h 캔들, 50차원 관측)

    중기 트렌드 추종에 특화. 트렌드 방향 포지션 유지 보너스.
    """

    def __init__(self, env=None, model_path: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3가 필요합니다")

        self.env = env
        self.model: Optional[PPO] = None
        self.model_path = model_path or os.path.join(MODEL_DIR, "swing_ppo")
        self.recent_accuracy: deque = deque(maxlen=50)

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs={
                "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
            },
        )
        logger.info("SwingAgent PPO 모델 생성 완료 (256-128-64)")

    def train(self, total_timesteps: int = 500_000, eval_env=None, save_freq: int = 50_000):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        callbacks = [_MultiAgentMetricsCallback(agent_name="swing")]
        if eval_env:
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(MODEL_DIR, "best_swing"),
                eval_freq=save_freq,
                n_eval_episodes=5,
                deterministic=True,
            ))

        logger.info(f"SwingAgent 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        self.save()
        logger.info("SwingAgent 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[float, float]:
        """관측 → (행동, 신뢰도)"""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action_val = float(action[0])
        confidence = min(abs(action_val), 1.0)
        return action_val, confidence

    def record_outcome(self, correct: bool):
        self.recent_accuracy.append(1.0 if correct else 0.0)

    def get_accuracy(self) -> float:
        if not self.recent_accuracy:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = PPO.load(path, env=self.env)
        logger.info(f"SwingAgent 로드: {path}")


# ============================================================
#  합의 엔진
# ============================================================

@dataclass
class ConsensusResult:
    """합의 결정 결과"""
    action: float                   # [-1, 1] 최종 행동
    confidence: float               # [0, 1] 합의 신뢰도
    decision: str                   # "buy" | "sell" | "hold"

    # 개별 에이전트 결과
    scalping_action: float = 0.0
    scalping_confidence: float = 0.0
    swing_action: float = 0.0
    swing_confidence: float = 0.0

    # 가중치
    scalping_weight: float = 0.5
    swing_weight: float = 0.5

    # 메타 정보
    regime: str = "neutral"         # "high_vol" | "trending" | "choppy" | "neutral"
    vetoed: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "decision": self.decision,
            "agents": {
                "scalping": {"action": self.scalping_action, "confidence": self.scalping_confidence,
                             "weight": self.scalping_weight},
                "swing": {"action": self.swing_action, "confidence": self.swing_confidence,
                           "weight": self.swing_weight},
            },
            "regime": self.regime,
            "vetoed": self.vetoed,
        }


class ConsensusEngine:
    """다중 에이전트 합의 엔진

    시장 국면에 따라 동적 가중치를 조정하고,
    에이전트 간 충돌 시 거부권(veto)으로 불확실성을 관리한다.
    """

    # 거부권 임계값
    VETO_THRESHOLD = 0.7

    # 기본 가중치
    DEFAULT_SCALP_WEIGHT = 0.4
    DEFAULT_SWING_WEIGHT = 0.6

    def __init__(self):
        self.atr_history: deque = deque(maxlen=100)
        self.atr_mean = 0.0
        self.atr_std = 1.0

    def update_atr_stats(self, atr_pct: float):
        """ATR 통계 업데이트 (변동성 국면 판단용)"""
        self.atr_history.append(atr_pct)
        if len(self.atr_history) >= 10:
            arr = np.array(self.atr_history)
            self.atr_mean = arr.mean()
            self.atr_std = max(arr.std(), 1e-6)

    def compute_consensus(
        self,
        scalp_action: float,
        scalp_confidence: float,
        swing_action: float,
        swing_confidence: float,
        atr_pct: float = 0.0,
        adx: float = 25.0,
        learned_weights: tuple[float, float] | None = None,
    ) -> ConsensusResult:
        """합의 결정 계산

        Args:
            scalp_action: 스캘핑 에이전트 행동 [-1, 1]
            scalp_confidence: 스캘핑 신뢰도 [0, 1]
            swing_action: 스윙 에이전트 행동 [-1, 1]
            swing_confidence: 스윙 신뢰도 [0, 1]
            atr_pct: ATR / 가격 * 100 (변동성)
            adx: ADX 값 (추세 강도)
            learned_weights: WeightLearner 출력 (있으면 우선 사용)

        Returns:
            ConsensusResult
        """
        self.update_atr_stats(atr_pct)

        # 1. 시장 국면 판단
        regime = self._detect_regime(atr_pct, adx)

        # 2. 거부권 검사
        vetoed = self._check_veto(scalp_action, swing_action)

        if vetoed:
            # 충돌 = 불확실성 → 관망
            return ConsensusResult(
                action=0.0, confidence=0.3, decision="hold",
                scalping_action=scalp_action, scalping_confidence=scalp_confidence,
                swing_action=swing_action, swing_confidence=swing_confidence,
                scalping_weight=0.5, swing_weight=0.5,
                regime=regime, vetoed=True,
            )

        # 3. 가중치 결정
        if learned_weights is not None:
            w_scalp, w_swing = learned_weights
        else:
            w_scalp, w_swing = self._regime_weights(regime)

        # 4. 신뢰도 가중 결합
        numerator = (w_scalp * scalp_action * scalp_confidence
                     + w_swing * swing_action * swing_confidence)
        denominator = w_scalp * scalp_confidence + w_swing * swing_confidence

        if denominator < 1e-6:
            final_action = 0.0
        else:
            final_action = numerator / denominator

        final_action = float(np.clip(final_action, -1, 1))

        # 5. 합의 신뢰도
        # 방향 일치 시 신뢰도 증가, 불일치 시 감소
        direction_agreement = 1.0 if np.sign(scalp_action) == np.sign(swing_action) else 0.5
        confidence = min(abs(final_action) * direction_agreement, 1.0)

        # 6. 이산 결정
        if final_action > 0.25:
            decision = "buy"
        elif final_action < -0.25:
            decision = "sell"
        else:
            decision = "hold"

        return ConsensusResult(
            action=final_action,
            confidence=confidence,
            decision=decision,
            scalping_action=scalp_action,
            scalping_confidence=scalp_confidence,
            swing_action=swing_action,
            swing_confidence=swing_confidence,
            scalping_weight=w_scalp,
            swing_weight=w_swing,
            regime=regime,
            vetoed=False,
        )

    def _detect_regime(self, atr_pct: float, adx: float) -> str:
        """시장 국면 탐지

        - high_vol: ATR > 평균 + 2σ → 고변동성
        - trending: ADX > 25 → 추세장
        - choppy: ADX < 20 + ATR < 평균 → 횡보장
        - neutral: 나머지
        """
        high_vol = (len(self.atr_history) >= 10
                    and atr_pct > self.atr_mean + 2 * self.atr_std)

        if high_vol:
            return "high_vol"
        elif adx > 25:
            return "trending"
        elif adx < 20 and atr_pct < self.atr_mean:
            return "choppy"
        else:
            return "neutral"

    def _regime_weights(self, regime: str) -> tuple[float, float]:
        """국면별 기본 가중치"""
        regime_map = {
            "high_vol": (0.6, 0.4),   # 고변동 → 스캘핑 우세
            "trending": (0.3, 0.7),   # 추세 → 스윙 우세
            "choppy":   (0.5, 0.5),   # 횡보 → 균등
            "neutral":  (0.4, 0.6),   # 중립 → 스윙 약간 우세
        }
        return regime_map.get(regime, (0.4, 0.6))

    def _check_veto(self, scalp_action: float, swing_action: float) -> bool:
        """거부권: 한쪽 강매수 + 다른쪽 강매도 → 충돌"""
        if (scalp_action > self.VETO_THRESHOLD and swing_action < -self.VETO_THRESHOLD):
            return True
        if (scalp_action < -self.VETO_THRESHOLD and swing_action > self.VETO_THRESHOLD):
            return True
        return False


# ============================================================
#  가중치 학습기 환경 + 에이전트
# ============================================================

class WeightLearnerEnv(gym.Env):
    """가중치 학습기 환경

    관측: [volatility_regime, trend_strength, scalp_accuracy, swing_accuracy,
           recent_scalp_return, recent_swing_return, atr_z, adx_norm,
           fgi_norm, scalp_action, swing_action, consensus_return]
    행동: [w_scalp, w_swing] (softmax 정규화)
    보상: 합의 포트폴리오 수익률
    """

    def __init__(
        self,
        scalping_returns: list[float] = None,
        swing_returns: list[float] = None,
        market_features: list[dict] = None,
        episode_length: int = 100,
    ):
        super().__init__()

        self.scalping_returns = scalping_returns or [0.0] * 200
        self.swing_returns = swing_returns or [0.0] * 200
        self.market_features = market_features or [{}] * 200
        self.episode_length = min(episode_length, len(self.scalping_returns))

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(WEIGHT_LEARNER_DIM,), dtype=np.float32,
        )
        # 2차원 행동 → softmax로 가중치
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=np.float32,
        )

        self.current_step = 0
        self.portfolio_value = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.scalping_returns) - self.episode_length - 1)
        self.current_step = self.np_random.integers(0, max(1, max_start))
        self.portfolio_value = 1.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # softmax 정규화
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / exp_a.sum()
        w_scalp, w_swing = float(weights[0]), float(weights[1])

        idx = min(self.current_step, len(self.scalping_returns) - 1)
        scalp_ret = self.scalping_returns[idx]
        swing_ret = self.swing_returns[idx]

        # 가중 수익
        consensus_return = w_scalp * scalp_ret + w_swing * swing_ret
        self.portfolio_value *= (1 + consensus_return)

        reward = float(consensus_return * 100)  # 스케일링

        self.current_step += 1
        terminated = False
        truncated = (self.current_step - (self.current_step - self.episode_length)) >= self.episode_length

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        idx = min(self.current_step, len(self.market_features) - 1)
        feat = self.market_features[idx] if idx < len(self.market_features) else {}

        # 최근 10스텝 수익률 통계
        start = max(0, idx - 10)
        recent_scalp = self.scalping_returns[start:idx + 1] or [0.0]
        recent_swing = self.swing_returns[start:idx + 1] or [0.0]

        scalp_acc = sum(1 for r in recent_scalp if r > 0) / max(len(recent_scalp), 1)
        swing_acc = sum(1 for r in recent_swing if r > 0) / max(len(recent_swing), 1)

        obs = np.array([
            feat.get("atr_z", 0.0),
            feat.get("adx_norm", 0.5),
            scalp_acc,
            swing_acc,
            np.mean(recent_scalp),
            np.mean(recent_swing),
            feat.get("volatility_regime", 0.0),
            feat.get("trend_strength", 0.5),
            feat.get("fgi_norm", 0.5),
            self.scalping_returns[idx] if idx < len(self.scalping_returns) else 0.0,
            self.swing_returns[idx] if idx < len(self.swing_returns) else 0.0,
            float(self.portfolio_value - 1.0),
        ], dtype=np.float32)

        return np.clip(obs, -5.0, 5.0)


class WeightLearner:
    """가중치 최적화 메타-RL 에이전트 (PPO, 32-16 네트워크)"""

    def __init__(self, env=None, model_path: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3가 필요합니다")

        self.env = env
        self.model: Optional[PPO] = None
        self.model_path = model_path or os.path.join(MODEL_DIR, "weight_learner")

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
            batch_size=32,
            n_epochs=5,
            gamma=0.99,
            verbose=0,
            policy_kwargs={"net_arch": {"pi": [32, 16], "vf": [32, 16]}},
        )
        logger.info("WeightLearner PPO 생성 (32-16)")

    def train(self, total_timesteps: int = 50_000):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        logger.info(f"WeightLearner 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(total_timesteps=total_timesteps)
        self.save()
        logger.info("WeightLearner 훈련 완료")

    def predict_weights(self, obs: np.ndarray) -> tuple[float, float]:
        """관측 → 최적 가중치 (w_scalp, w_swing)"""
        action, _ = self.model.predict(obs, deterministic=True)
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / exp_a.sum()
        return float(weights[0]), float(weights[1])

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = PPO.load(path, env=self.env)
        logger.info(f"WeightLearner 로드: {path}")


# ============================================================
#  다중 에이전트 통합 환경
# ============================================================

class MultiAgentTradingEnv(gym.Env):
    """다중 에이전트 통합 환경

    두 타임프레임(1h, 4h) 데이터를 제공하고,
    각 에이전트의 행동을 합의 → 단일 실행으로 변환한다.
    에이전트별 기여도(attribution)를 추적한다.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles_1h: list[dict] = None,
        candles_4h: list[dict] = None,
        initial_balance: float = 10_000_000,
        external_signals_1h: list[dict] | None = None,
        external_signals_4h: list[dict] | None = None,
    ):
        super().__init__()

        self.candles_1h = candles_1h or []
        self.candles_4h = candles_4h or []
        self.external_signals_1h = external_signals_1h
        self.external_signals_4h = external_signals_4h
        self.initial_balance = initial_balance

        # 4h 기준 스텝 (1h는 4배속)
        self.lookback = 42
        self.start_idx = self.lookback
        self.end_idx = len(self.candles_4h) - 1 if self.candles_4h else 0

        # 관측: scalping_dim + swing_dim (에이전트 래퍼가 직접 사용)
        # 이 환경은 에이전트에게 직접 관측을 제공하는 dispatch 역할
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(SCALPING_DIM + SWING_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self.consensus_engine = ConsensusEngine()

        # 상태
        self.current_step_4h = 0
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.total_value_history = []
        self.trade_count = 0

        # 에이전트별 기여도 추적
        self.attribution = {"scalping": [], "swing": []}

    def get_scalping_obs(self, scalping_env: ScalpingTradingEnv) -> np.ndarray:
        """스캘핑 에이전트용 관측 추출"""
        # 1h 캔들 중 현재 4h에 해당하는 위치의 관측 반환
        step_1h = min(self.current_step_4h * 4, len(self.candles_1h) - 1)
        scalping_env.current_step = step_1h
        scalping_env.krw_balance = self.krw_balance
        scalping_env.btc_balance = self.btc_balance
        scalping_env.trade_count = self.trade_count
        return scalping_env._get_observation()

    def get_swing_obs(self, swing_env: SwingTradingEnv) -> np.ndarray:
        """스윙 에이전트용 관측 추출"""
        swing_env.current_step = self.current_step_4h
        swing_env.krw_balance = self.krw_balance
        swing_env.btc_balance = self.btc_balance
        swing_env.trade_count = self.trade_count
        return swing_env._get_observation()

    def get_market_regime_features(self) -> dict:
        """현재 시장 국면 특성"""
        if not self.candles_4h or self.current_step_4h >= len(self.candles_4h):
            return {"atr_pct": 0.0, "adx": 25.0}
        candle = self.candles_4h[self.current_step_4h]
        price = candle["close"]
        atr = candle.get("atr", 0)
        atr_pct = (atr / price * 100) if price > 0 else 0
        adx = candle.get("adx", 25)
        return {"atr_pct": atr_pct, "adx": adx}

    def execute_consensus(self, consensus: ConsensusResult) -> tuple[float, dict]:
        """합의 결정 실행 → 보상 계산

        Returns:
            (reward, info)
        """
        if not self.candles_4h or self.current_step_4h >= self.end_idx:
            return 0.0, {"terminated": True}

        candle = self.candles_4h[self.current_step_4h]
        price = candle["close"]
        prev_value = self._portfolio_value(price)

        # 행동 실행
        self._execute_action(consensus.action, price)

        # 다음 스텝
        self.current_step_4h += 1
        if self.current_step_4h < len(self.candles_4h):
            next_price = self.candles_4h[self.current_step_4h]["close"]
        else:
            next_price = price
        curr_value = self._portfolio_value(next_price)
        self.total_value_history.append(curr_value)

        # 수익률 기반 보상
        raw_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
        reward = raw_return * 10

        # 기여도 추적
        scalp_contrib = consensus.scalping_weight * consensus.scalping_action
        swing_contrib = consensus.swing_weight * consensus.swing_action
        self.attribution["scalping"].append(scalp_contrib * raw_return)
        self.attribution["swing"].append(swing_contrib * raw_return)

        info = {
            "step": self.current_step_4h,
            "price": next_price,
            "portfolio_value": curr_value,
            "return_pct": (curr_value - self.initial_balance) / self.initial_balance * 100,
            "raw_return": raw_return,
        }
        return reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(self.start_idx, self.end_idx - 200)
        self.current_step_4h = self.np_random.integers(
            self.start_idx, max(self.start_idx + 1, max_start),
        )
        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.total_value_history = [self.initial_balance]
        self.trade_count = 0
        self.attribution = {"scalping": [], "swing": []}

        obs = np.zeros(SCALPING_DIM + SWING_DIM, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        # 이 환경은 직접 사용하기보다 MultiAgentTrainer에서 조율
        raise NotImplementedError(
            "MultiAgentTradingEnv는 execute_consensus()를 통해 사용합니다."
        )

    def _execute_action(self, action: float, price: float):
        total_value = self._portfolio_value(price)
        target_btc_ratio = (action + 1) / 2
        current_btc_value = self.btc_balance * price
        target_btc_value = total_value * target_btc_ratio
        diff = target_btc_value - current_btc_value

        if diff > 0 and self.krw_balance > 0:
            buy_amount = min(diff, self.krw_balance)
            cost = buy_amount * TRANSACTION_COST
            btc_bought = (buy_amount - cost) / price
            self.krw_balance -= buy_amount
            self.btc_balance += btc_bought
            if buy_amount > total_value * 0.01:
                self.trade_count += 1
        elif diff < 0 and self.btc_balance > 0:
            sell_value = min(-diff, current_btc_value)
            btc_sold = min(sell_value / price, self.btc_balance)
            proceeds = btc_sold * price * (1 - TRANSACTION_COST)
            self.btc_balance -= btc_sold
            self.krw_balance += proceeds
            if sell_value > total_value * 0.01:
                self.trade_count += 1

    def _portfolio_value(self, price: float) -> float:
        return self.krw_balance + self.btc_balance * price

    def get_attribution_summary(self) -> dict:
        """에이전트별 기여도 요약"""
        result = {}
        for agent in ["scalping", "swing"]:
            contribs = self.attribution[agent]
            if contribs:
                arr = np.array(contribs)
                result[agent] = {
                    "total_contribution": float(arr.sum()),
                    "avg_contribution": float(arr.mean()),
                    "positive_count": int((arr > 0).sum()),
                    "negative_count": int((arr < 0).sum()),
                }
            else:
                result[agent] = {
                    "total_contribution": 0.0, "avg_contribution": 0.0,
                    "positive_count": 0, "negative_count": 0,
                }
        return result


# ============================================================
#  훈련 콜백
# ============================================================

class _MultiAgentMetricsCallback(BaseCallback if SB3_AVAILABLE else object):
    """에이전트별 훈련 메트릭 로깅"""

    def __init__(self, agent_name: str = "", log_freq: int = 2000, verbose: int = 0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.agent_name = agent_name
        self.log_freq = log_freq
        self.episode_returns = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])
            elif info.get("return_pct") is not None:
                self.episode_returns.append(info["return_pct"])

        if self.num_timesteps % self.log_freq == 0 and self.episode_returns:
            avg = np.mean(self.episode_returns[-10:])
            logger.info(f"[{self.agent_name} Step {self.num_timesteps}] avg_return={avg:.2f}%")
        return True


# ============================================================
#  훈련 파이프라인
# ============================================================

class MultiAgentTrainer:
    """3단계 훈련 파이프라인

    Phase 1: 스캘핑/스윙 에이전트 독립 훈련 (병렬 가능)
    Phase 2: 에이전트 가중치 고정 → WeightLearner 훈련
    Phase 3: 전체 fine-tuning (선택사항)
    """

    def __init__(
        self,
        scalping_steps: int = 200_000,
        swing_steps: int = 500_000,
        weight_learner_steps: int = 50_000,
        initial_balance: float = 10_000_000,
    ):
        self.scalping_steps = scalping_steps
        self.swing_steps = swing_steps
        self.weight_learner_steps = weight_learner_steps
        self.initial_balance = initial_balance

        self.scalping_agent: Optional[ScalpingAgent] = None
        self.swing_agent: Optional[SwingAgent] = None
        self.weight_learner: Optional[WeightLearner] = None
        self.consensus_engine = ConsensusEngine()

    def train(
        self,
        scalping_days: int = 90,
        swing_days: int = 180,
        joint_finetune: bool = False,
    ):
        """전체 파이프라인 실행"""
        logger.info("=" * 60)
        logger.info("  Multi-Agent Consensus RL 훈련 시작")
        logger.info("=" * 60)

        # DB 로깅: 훈련 시작
        _ma_cycle_id = None
        _ma_start_time = time.time()
        try:
            from rl_hybrid.rl.rl_db_logger import log_training_start
            _ma_cycle_id = log_training_start(
                cycle_type="standalone",
                algorithm="multi_agent",
                module="multi_agent_consensus",
                training_steps=self.scalping_steps + self.swing_steps,
                data_days=swing_days,
            )
        except Exception:
            pass

        try:
            # Phase 1: 독립 훈련
            self._train_phase1(scalping_days, swing_days)

            # Phase 2: 가중치 학습
            self._train_phase2(swing_days)

            # Phase 3: Joint fine-tuning (선택)
            if joint_finetune:
                self._train_phase3(swing_days)

            # 평가
            self._evaluate_all(swing_days)

            # 저장
            self._save_all()

            logger.info("=" * 60)
            logger.info("  Multi-Agent Consensus RL 훈련 완료")
            logger.info("=" * 60)

            # DB 로깅: 훈련 완료
            if _ma_cycle_id:
                try:
                    from rl_hybrid.rl.rl_db_logger import log_training_complete
                    log_training_complete(
                        cycle_id=_ma_cycle_id,
                        elapsed_seconds=time.time() - _ma_start_time,
                        status="completed",
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Multi-Agent 훈련 실패: {e}", exc_info=True)
            if _ma_cycle_id:
                try:
                    from rl_hybrid.rl.rl_db_logger import log_training_complete
                    log_training_complete(
                        cycle_id=_ma_cycle_id,
                        elapsed_seconds=time.time() - _ma_start_time,
                        status="failed",
                        error_message=str(e)[:500],
                    )
                except Exception:
                    pass
            raise

    def _train_phase1(self, scalping_days: int, swing_days: int):
        """Phase 1: 개별 에이전트 훈련"""
        logger.info("\n--- Phase 1: 개별 에이전트 훈련 ---")

        loader = HistoricalDataLoader()

        # 스캘핑 데이터 (1h, 90일)
        logger.info("[스캘핑] 1h 캔들 로드 중...")
        raw_1h = loader.load_candles(days=scalping_days, interval="1h")
        candles_1h = loader.compute_indicators(raw_1h)

        signals_1h = None
        try:
            raw_sig = loader.load_external_signals(days=scalping_days)
            if raw_sig:
                signals_1h = loader.align_external_to_candles(candles_1h, raw_sig)
        except Exception as e:
            logger.warning(f"스캘핑 외부 시그널 로드 실패: {e}")

        split_1h = int(len(candles_1h) * 0.8)
        train_env_scalp = ScalpingTradingEnv(
            candles=candles_1h[:split_1h],
            initial_balance=self.initial_balance,
            external_signals=signals_1h[:split_1h] if signals_1h else None,
        )
        eval_env_scalp = ScalpingTradingEnv(
            candles=candles_1h[split_1h:],
            initial_balance=self.initial_balance,
            external_signals=signals_1h[split_1h:] if signals_1h else None,
        )

        self.scalping_agent = ScalpingAgent(env=train_env_scalp)
        self.scalping_agent.train(
            total_timesteps=self.scalping_steps,
            eval_env=eval_env_scalp,
        )

        # 스윙 데이터 (4h, 180일)
        logger.info("[스윙] 4h 캔들 로드 중...")
        raw_4h = loader.load_candles(days=swing_days, interval="4h")
        candles_4h = loader.compute_indicators(raw_4h)

        signals_4h = None
        try:
            raw_sig = loader.load_external_signals(days=swing_days)
            if raw_sig:
                signals_4h = loader.align_external_to_candles(candles_4h, raw_sig)
        except Exception as e:
            logger.warning(f"스윙 외부 시그널 로드 실패: {e}")

        split_4h = int(len(candles_4h) * 0.8)
        train_env_swing = SwingTradingEnv(
            candles=candles_4h[:split_4h],
            initial_balance=self.initial_balance,
            external_signals=signals_4h[:split_4h] if signals_4h else None,
        )
        eval_env_swing = SwingTradingEnv(
            candles=candles_4h[split_4h:],
            initial_balance=self.initial_balance,
            external_signals=signals_4h[split_4h:] if signals_4h else None,
        )

        self.swing_agent = SwingAgent(env=train_env_swing)
        self.swing_agent.train(
            total_timesteps=self.swing_steps,
            eval_env=eval_env_swing,
        )

    def _train_phase2(self, swing_days: int):
        """Phase 2: WeightLearner 훈련

        Phase 1에서 훈련된 에이전트로 시뮬레이션하여
        각 에이전트의 스텝별 수익률을 수집하고, 최적 가중치를 학습한다.
        """
        logger.info("\n--- Phase 2: WeightLearner 훈련 ---")

        loader = HistoricalDataLoader()
        raw_4h = loader.load_candles(days=swing_days, interval="4h")
        candles_4h = loader.compute_indicators(raw_4h)
        raw_1h = loader.load_candles(days=swing_days, interval="1h")
        candles_1h = loader.compute_indicators(raw_1h)

        signals_4h = None
        signals_1h = None
        try:
            raw_sig = loader.load_external_signals(days=swing_days)
            if raw_sig:
                signals_4h = loader.align_external_to_candles(candles_4h, raw_sig)
                signals_1h = loader.align_external_to_candles(candles_1h, raw_sig)
        except Exception:
            pass

        # 시뮬레이션으로 스텝별 수익률 수집
        scalp_returns, swing_returns, market_feats = self._simulate_agents(
            candles_1h, candles_4h, signals_1h, signals_4h,
        )

        # WeightLearner 환경 생성
        wl_env = WeightLearnerEnv(
            scalping_returns=scalp_returns,
            swing_returns=swing_returns,
            market_features=market_feats,
            episode_length=min(100, len(scalp_returns)),
        )

        self.weight_learner = WeightLearner(env=wl_env)
        self.weight_learner.train(total_timesteps=self.weight_learner_steps)

    def _simulate_agents(
        self,
        candles_1h: list[dict],
        candles_4h: list[dict],
        signals_1h, signals_4h,
    ) -> tuple[list[float], list[float], list[dict]]:
        """에이전트 시뮬레이션 → 스텝별 수익률 수집"""
        logger.info("에이전트 시뮬레이션 (가중치 학습용 데이터 수집)...")

        # 스캘핑 시뮬레이션
        scalp_env = ScalpingTradingEnv(
            candles=candles_1h, initial_balance=self.initial_balance,
            external_signals=signals_1h,
        )
        scalp_returns = []
        obs, _ = scalp_env.reset(options={"start_idx": scalp_env.start_idx})
        for i in range(min(len(candles_1h) - scalp_env.start_idx - 2, 2000)):
            action, _ = self.scalping_agent.predict(obs)
            try:
                obs, reward, terminated, truncated, info = scalp_env.step(np.array([action]))
            except (IndexError, KeyError):
                break
            ret = info.get("reward_components", {}).get("raw_return", 0.0)
            scalp_returns.append(ret)
            if terminated or truncated:
                break

        # 스윙 시뮬레이션
        swing_env = SwingTradingEnv(
            candles=candles_4h, initial_balance=self.initial_balance,
            external_signals=signals_4h,
        )
        swing_returns = []
        obs, _ = swing_env.reset(options={"start_idx": swing_env.start_idx})
        for i in range(min(len(candles_4h) - swing_env.start_idx - 2, 2000)):
            action, _ = self.swing_agent.predict(obs)
            try:
                obs, reward, terminated, truncated, info = swing_env.step(np.array([action]))
            except (IndexError, KeyError):
                break
            ret = info.get("reward_components", {}).get("raw_return", 0.0)
            swing_returns.append(ret)
            if terminated or truncated:
                break

        # 길이 맞추기 (짧은 쪽에 맞춤)
        min_len = min(len(scalp_returns), len(swing_returns))
        if min_len == 0:
            min_len = 50
            scalp_returns = [0.0] * min_len
            swing_returns = [0.0] * min_len

        scalp_returns = scalp_returns[:min_len]
        swing_returns = swing_returns[:min_len]

        # 시장 특성 수집
        market_feats = []
        for i in range(min_len):
            idx_4h = min(i, len(candles_4h) - 1)
            c = candles_4h[idx_4h]
            price = c["close"]
            atr = c.get("atr", 0)
            adx = c.get("adx", 25)
            market_feats.append({
                "atr_z": float(np.clip(atr / price * 100 if price > 0 else 0, -3, 3)),
                "adx_norm": float(min(adx / 50.0, 1.0)),
                "volatility_regime": 1.0 if atr / price * 100 > 2.0 else 0.0 if price > 0 else 0.0,
                "trend_strength": float(min(adx / 50.0, 1.0)),
                "fgi_norm": 0.5,
            })

        logger.info(f"시뮬레이션 완료: {min_len} 스텝")
        return scalp_returns, swing_returns, market_feats

    def _train_phase3(self, swing_days: int):
        """Phase 3: Joint fine-tuning (선택사항)

        합의 기반 통합 환경에서 모든 에이전트를 동시에 미세조정한다.
        현재 구현: 각 에이전트를 합의 보상으로 추가 훈련.
        """
        logger.info("\n--- Phase 3: Joint Fine-Tuning ---")
        logger.info("Joint fine-tuning은 Phase 1/2 모델 기반으로 추가 학습합니다.")

        # 스캘핑 에이전트 미세조정 (짧은 스텝)
        if self.scalping_agent and self.scalping_agent.model:
            self.scalping_agent.model.learn(total_timesteps=self.scalping_steps // 5)
            self.scalping_agent.save()
            logger.info("ScalpingAgent fine-tuning 완료")

        # 스윙 에이전트 미세조정
        if self.swing_agent and self.swing_agent.model:
            self.swing_agent.model.learn(total_timesteps=self.swing_steps // 5)
            self.swing_agent.save()
            logger.info("SwingAgent fine-tuning 완료")

    def _evaluate_all(self, swing_days: int):
        """개별 에이전트, 합의, 기존 단일 에이전트 비교 평가"""
        logger.info("\n--- 평가: 개별 vs 합의 비교 ---")

        loader = HistoricalDataLoader()
        raw_4h = loader.load_candles(days=swing_days, interval="4h")
        candles_4h = loader.compute_indicators(raw_4h)
        raw_1h = loader.load_candles(days=swing_days, interval="1h")
        candles_1h = loader.compute_indicators(raw_1h)

        signals_4h = None
        signals_1h = None
        try:
            raw_sig = loader.load_external_signals(days=swing_days)
            if raw_sig:
                signals_4h = loader.align_external_to_candles(candles_4h, raw_sig)
                signals_1h = loader.align_external_to_candles(candles_1h, raw_sig)
        except Exception:
            pass

        # 평가 데이터 (후반 20%)
        split_4h = int(len(candles_4h) * 0.8)
        split_1h = int(len(candles_1h) * 0.8)
        eval_4h = candles_4h[split_4h:]
        eval_1h = candles_1h[split_1h:]
        eval_sig_4h = signals_4h[split_4h:] if signals_4h else None
        eval_sig_1h = signals_1h[split_1h:] if signals_1h else None

        results = {}

        # 스캘핑 단독
        scalp_stats = self._eval_agent(
            self.scalping_agent, ScalpingTradingEnv,
            eval_1h, eval_sig_1h, episodes=5,
        )
        results["scalping_solo"] = scalp_stats
        logger.info(f"[스캘핑 단독] 수익률={scalp_stats['avg_return']:.2f}%, "
                     f"샤프={scalp_stats['avg_sharpe']:.3f}")

        # 스윙 단독
        swing_stats = self._eval_agent(
            self.swing_agent, SwingTradingEnv,
            eval_4h, eval_sig_4h, episodes=5,
        )
        results["swing_solo"] = swing_stats
        logger.info(f"[스윙 단독] 수익률={swing_stats['avg_return']:.2f}%, "
                     f"샤프={swing_stats['avg_sharpe']:.3f}")

        # 합의 (가중치 학습기 사용)
        consensus_stats = self._eval_consensus(
            eval_1h, eval_4h, eval_sig_1h, eval_sig_4h, episodes=5,
        )
        results["consensus"] = consensus_stats
        logger.info(f"[합의] 수익률={consensus_stats['avg_return']:.2f}%, "
                     f"샤프={consensus_stats['avg_sharpe']:.3f}")

        # 비교 테이블
        logger.info(f"\n{'=' * 60}")
        logger.info("  Multi-Agent 평가 비교")
        logger.info(f"{'=' * 60}")
        logger.info(f"{'모드':>15} | {'수익률':>10} | {'샤프':>8} | {'MDD':>10}")
        logger.info("-" * 50)
        for name, stats in results.items():
            logger.info(
                f"{name:>15} | {stats['avg_return']:>9.2f}% | "
                f"{stats['avg_sharpe']:>8.3f} | {stats['avg_mdd']:>9.2%}"
            )

        # 결과 저장
        info_path = os.path.join(MODEL_DIR, "evaluation_results.json")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"평가 결과 저장: {info_path}")

    def _eval_agent(self, agent, env_class, candles, signals, episodes=5) -> dict:
        """개별 에이전트 평가"""
        all_returns, all_sharpes, all_mdds = [], [], []

        for _ in range(episodes):
            env = env_class(
                candles=candles, initial_balance=self.initial_balance,
                external_signals=signals,
            )
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(np.array([action]))
                done = terminated or truncated
            stats = env.get_episode_stats()
            all_returns.append(stats["total_return_pct"])
            all_sharpes.append(stats["sharpe_ratio"])
            all_mdds.append(stats["max_drawdown"])

        return {
            "avg_return": float(np.mean(all_returns)),
            "avg_sharpe": float(np.mean(all_sharpes)),
            "avg_mdd": float(np.mean(all_mdds)),
        }

    def _eval_consensus(self, candles_1h, candles_4h, signals_1h, signals_4h,
                         episodes=5) -> dict:
        """합의 기반 평가"""
        all_returns, all_sharpes, all_mdds = [], [], []

        for _ in range(episodes):
            scalp_env = ScalpingTradingEnv(
                candles=candles_1h, initial_balance=self.initial_balance,
                external_signals=signals_1h,
            )
            swing_env = SwingTradingEnv(
                candles=candles_4h, initial_balance=self.initial_balance,
                external_signals=signals_4h,
            )

            # 스윙 기준 에피소드
            obs_sw, _ = swing_env.reset()
            # 스캘핑 환경도 동기화
            scalp_env.current_step = min(
                swing_env.current_step * 4, len(candles_1h) - 1
            )
            scalp_env.krw_balance = swing_env.krw_balance
            scalp_env.btc_balance = swing_env.btc_balance
            obs_sc = scalp_env._get_observation()

            done = False
            while not done:
                # 각 에이전트 예측
                scalp_action, scalp_conf = self.scalping_agent.predict(obs_sc)
                swing_action, swing_conf = self.swing_agent.predict(obs_sw)

                # 시장 국면
                candle = candles_4h[min(swing_env.current_step, len(candles_4h) - 1)]
                price = candle["close"]
                atr_pct = (candle.get("atr", 0) / price * 100) if price > 0 else 0
                adx = candle.get("adx", 25)

                # 가중치 학습기 (사용 가능하면)
                learned_weights = None
                if self.weight_learner and self.weight_learner.model:
                    wl_obs = np.zeros(WEIGHT_LEARNER_DIM, dtype=np.float32)
                    wl_obs[0] = float(np.clip(atr_pct, -3, 3))
                    wl_obs[1] = float(min(adx / 50.0, 1.0))
                    wl_obs[2] = self.scalping_agent.get_accuracy()
                    wl_obs[3] = self.swing_agent.get_accuracy()
                    wl_obs[9] = scalp_action
                    wl_obs[10] = swing_action
                    learned_weights = self.weight_learner.predict_weights(wl_obs)

                # 합의
                consensus = self.consensus_engine.compute_consensus(
                    scalp_action, scalp_conf,
                    swing_action, swing_conf,
                    atr_pct=atr_pct, adx=adx,
                    learned_weights=learned_weights,
                )

                # 스윙 환경에서 실행
                obs_sw, reward, terminated, truncated, info = swing_env.step(
                    np.array([consensus.action])
                )
                done = terminated or truncated

                # 스캘핑 환경 동기화
                scalp_env.current_step = min(
                    swing_env.current_step * 4, len(candles_1h) - 1
                )
                scalp_env.krw_balance = swing_env.krw_balance
                scalp_env.btc_balance = swing_env.btc_balance
                if scalp_env.current_step < len(candles_1h):
                    obs_sc = scalp_env._get_observation()

            stats = swing_env.get_episode_stats()
            all_returns.append(stats["total_return_pct"])
            all_sharpes.append(stats["sharpe_ratio"])
            all_mdds.append(stats["max_drawdown"])

        return {
            "avg_return": float(np.mean(all_returns)),
            "avg_sharpe": float(np.mean(all_sharpes)),
            "avg_mdd": float(np.mean(all_mdds)),
        }

    def _save_all(self):
        """모든 모델 저장"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.scalping_agent:
            self.scalping_agent.save()
        if self.swing_agent:
            self.swing_agent.save()
        if self.weight_learner:
            self.weight_learner.save()

        # 메타 정보
        meta = {
            "type": "multi_agent_consensus",
            "agents": {
                "scalping": {"algo": "SAC", "obs_dim": SCALPING_DIM, "net": [128, 64]},
                "swing": {"algo": "PPO", "obs_dim": SWING_DIM, "net": [256, 128, 64]},
                "weight_learner": {"algo": "PPO", "obs_dim": WEIGHT_LEARNER_DIM, "net": [32, 16]},
            },
            "scalping_steps": self.scalping_steps,
            "swing_steps": self.swing_steps,
            "weight_learner_steps": self.weight_learner_steps,
        }
        meta_path = os.path.join(MODEL_DIR, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Multi-Agent 모델 저장 완료: {MODEL_DIR}")


# ============================================================
#  추론 인터페이스 (DecisionBlender 호환)
# ============================================================

class MultiAgentPredictor:
    """다중 에이전트 합의 추론기 — DecisionBlender 호환

    기존 RL 시그널을 대체하거나 보충할 수 있다.
    predict() 출력이 DecisionBlender._rl_to_continuous() 호환.
    """

    def __init__(self, model_dir: str = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3가 필요합니다")

        self.model_dir = model_dir or MODEL_DIR
        self.scalping_agent: Optional[ScalpingAgent] = None
        self.swing_agent: Optional[SwingAgent] = None
        self.weight_learner: Optional[WeightLearner] = None
        self.consensus_engine = ConsensusEngine()
        self.encoder = StateEncoder()

        self._scalping_indices = np.array(
            [_FEATURE_INDEX[f] for f in SCALPING_FEATURES], dtype=np.int32,
        )

        self._loaded = False

    def load(self) -> bool:
        """저장된 모델 로드"""
        scalp_path = os.path.join(self.model_dir, "scalping_sac")
        swing_path = os.path.join(self.model_dir, "swing_ppo")
        wl_path = os.path.join(self.model_dir, "weight_learner")

        try:
            if os.path.exists(scalp_path + ".zip"):
                self.scalping_agent = ScalpingAgent(model_path=scalp_path)
                logger.info("ScalpingAgent 로드 완료")

            if os.path.exists(swing_path + ".zip"):
                self.swing_agent = SwingAgent(model_path=swing_path)
                logger.info("SwingAgent 로드 완료")

            if os.path.exists(wl_path + ".zip"):
                wl_env = WeightLearnerEnv()  # 더미 환경
                self.weight_learner = WeightLearner(env=wl_env, model_path=wl_path)
                logger.info("WeightLearner 로드 완료")

            self._loaded = (self.scalping_agent is not None
                            and self.swing_agent is not None)
            return self._loaded

        except Exception as e:
            logger.error(f"Multi-Agent 모델 로드 실패: {e}")
            return False

    def predict(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict = None,
    ) -> dict:
        """DecisionBlender 호환 예측

        Returns:
            {"action": float, "value": float, "consensus": ConsensusResult.to_dict()}
        """
        if not self._loaded:
            if not self.load():
                return {"action": 0.0, "value": 0.0}

        agent_state = agent_state or {}

        # 기본 42차원 관측
        full_obs = self.encoder.encode(market_data, external_data, portfolio, agent_state)

        # 스캘핑 관측 (30차원 서브셋)
        scalp_obs = full_obs[self._scalping_indices]

        # 스윙 관측 (42 + 8 확장)
        indicators = market_data.get("indicators", {})
        adx_data = indicators.get("adx", {})
        adx = float(adx_data.get("adx", 25) if isinstance(adx_data, dict) else 25)
        plus_di = float(adx_data.get("plus_di", 20) if isinstance(adx_data, dict) else 20)
        minus_di = float(adx_data.get("minus_di", 20) if isinstance(adx_data, dict) else 20)

        price = float(market_data.get("current_price", 0))
        atr = float(indicators.get("atr", 0) or 0)
        atr_pct = (atr / price * 100) if price > 0 else 0

        extra = np.array([
            min(adx / 50.0, 1.0),
            np.clip((plus_di - minus_di) / 30.0, -1, 1),
            0.0,  # fgi_trend (실시간에서는 외부에서 주입)
            0.0,  # macro_momentum
            0.5,  # btc_dominance_delta
            0.0,  # funding_pressure
            0.0,  # volume_trend
            0.0,  # price_momentum
        ], dtype=np.float32)
        extra = np.clip((extra + 1) / 2, 0, 1)
        swing_obs = np.concatenate([full_obs, extra])

        # 에이전트 예측
        scalp_action, scalp_conf = self.scalping_agent.predict(scalp_obs)
        swing_action, swing_conf = self.swing_agent.predict(swing_obs)

        # 가중치 학습기
        learned_weights = None
        if self.weight_learner and self.weight_learner.model:
            wl_obs = np.zeros(WEIGHT_LEARNER_DIM, dtype=np.float32)
            wl_obs[0] = float(np.clip(atr_pct, -3, 3))
            wl_obs[1] = float(min(adx / 50.0, 1.0))
            wl_obs[2] = self.scalping_agent.get_accuracy()
            wl_obs[3] = self.swing_agent.get_accuracy()
            wl_obs[9] = scalp_action
            wl_obs[10] = swing_action
            learned_weights = self.weight_learner.predict_weights(wl_obs)

        # 합의
        consensus = self.consensus_engine.compute_consensus(
            scalp_action, scalp_conf,
            swing_action, swing_conf,
            atr_pct=atr_pct, adx=adx,
            learned_weights=learned_weights,
        )

        logger.info(
            f"[MultiAgent] regime={consensus.regime} | "
            f"scalp={scalp_action:+.2f}(c={scalp_conf:.2f}) | "
            f"swing={swing_action:+.2f}(c={swing_conf:.2f}) | "
            f"consensus={consensus.action:+.2f} [{consensus.decision}] "
            f"{'VETO' if consensus.vetoed else ''}"
        )

        return {
            "action": consensus.action,
            "value": consensus.confidence,
            "consensus": consensus.to_dict(),
        }

    def record_outcome(self, reward: float):
        """실제 결과를 기록하여 에이전트 정확도 업데이트"""
        if self.scalping_agent:
            self.scalping_agent.record_outcome(reward > 0)
        if self.swing_agent:
            self.swing_agent.record_outcome(reward > 0)


# ============================================================
#  설정 (config.py 확장용)
# ============================================================

@dataclass
class MultiAgentConfig:
    """Multi-Agent Consensus 설정"""
    enabled: bool = True
    model_dir: str = MODEL_DIR
    scalping_steps: int = 200_000
    swing_steps: int = 500_000
    weight_learner_steps: int = 50_000
    veto_threshold: float = 0.7
    default_scalp_weight: float = 0.4
    default_swing_weight: float = 0.6


# ============================================================
#  CLI 엔트리포인트
# ============================================================

def main():
    """CLI: python -m rl_hybrid.rl.multi_agent_consensus [--train|--eval|--predict]"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Multi-Agent Consensus RL 트레이딩")
    parser.add_argument("--train", action="store_true", help="전체 훈련 파이프라인 실행")
    parser.add_argument("--eval", action="store_true", help="저장된 모델 평가")
    parser.add_argument("--joint", action="store_true", help="Phase 3 joint fine-tuning 포함")
    parser.add_argument("--scalping-days", type=int, default=90, help="스캘핑 훈련 기간 (일)")
    parser.add_argument("--swing-days", type=int, default=180, help="스윙 훈련 기간 (일)")
    parser.add_argument("--scalping-steps", type=int, default=200_000, help="스캘핑 훈련 스텝")
    parser.add_argument("--swing-steps", type=int, default=500_000, help="스윙 훈련 스텝")
    parser.add_argument("--balance", type=float, default=10_000_000, help="초기 잔고")

    args = parser.parse_args()

    if args.train:
        trainer = MultiAgentTrainer(
            scalping_steps=args.scalping_steps,
            swing_steps=args.swing_steps,
            initial_balance=args.balance,
        )
        trainer.train(
            scalping_days=args.scalping_days,
            swing_days=args.swing_days,
            joint_finetune=args.joint,
        )
    elif args.eval:
        predictor = MultiAgentPredictor()
        if predictor.load():
            logger.info("Multi-Agent 모델 로드 성공 -- 추론 테스트")
            # 더미 데이터로 추론 테스트
            result = predictor.predict(
                market_data={"current_price": 100_000_000, "change_rate_24h": 0.01,
                             "indicators": {"sma_20": 99_000_000, "sma_50": 98_000_000,
                                            "rsi_14": 55, "macd": {}, "bollinger": {},
                                            "stochastic": {}, "adx": {"adx": 30}, "atr": 500000},
                             "indicators_4h": {}, "orderbook": {}, "trade_pressure": {},
                             "eth_btc_analysis": {}},
                external_data={"sources": {}, "external_signal": {}},
                portfolio={"krw_balance": 5_000_000, "holdings": [], "total_eval": 5_000_000},
            )
            logger.info(f"추론 결과: {json.dumps(result, indent=2, default=str)}")
        else:
            logger.error("모델 로드 실패 -- 먼저 --train을 실행하세요")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
