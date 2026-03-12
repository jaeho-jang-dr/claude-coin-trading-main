"""보상 함수 v8 — v6 안정성 + v7 트렌드 팔로잉 하이브리드

v6 장점: Sharpe + PnL 직접 반영, 안정적 MDD 관리, 정책 붕괴 방지
v7 장점: 트렌드 팔로잉, 수익 포지션 홀딩 보상

v8 개선:
  1. Sharpe(30%) + Trend(15%) + PnL(30%) + 나머지(25%) — 균형잡힌 가중치
  2. 수익 거래 보너스 스케일 확대 (0.15 → 0.25)
  3. 트렌드 팔로잉 + 홀딩 보상 동시 적용
  4. 비활동 페널티 강화 (6스텝부터)
  5. 손실 거래 페널티 추가 (손절 학습 유도)
  6. 정책 붕괴 방지: 포지션 변동 보상 + 강한 비활동 패널티

v8.2 수정 (2026-03-13): 상수 행동 정책 붕괴 근본 해결
  - 핵심 변경: action_history 기반 행동 다양성 보상 추가
  - 상수 행동(std < 0.03) → 강한 페널티 (-0.15/step)
  - 다양한 행동(std > 0.1) → 보너스 (+0.08)
  - 비활동 페널티 더 빠르게 시작 (12→6 스텝)
  - 비활동 강도 2배 증가 (-0.06→-0.12)
  - 홀딩 보상 축소 (0.04→0.02) — 상수행동 유인 감소
"""

import numpy as np
from collections import deque

UPBIT_FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0003
TRANSACTION_COST = UPBIT_FEE_RATE + SLIPPAGE_RATE


class RewardCalculatorV8:
    """Sharpe + Trend + PnL + Action Diversity 하이브리드 보상"""

    def __init__(
        self,
        window_size: int = 20,
        trend_window: int = 10,
        action_window: int = 10,
        risk_free_rate: float = 0.18 / 365 / 24,  # 연 18% 목표
        max_drawdown_penalty: float = 2.5,
    ):
        self.window_size = window_size
        self.trend_window = trend_window
        self.action_window = action_window
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty

        self.returns_history: deque[float] = deque(maxlen=window_size)
        self.price_history: deque[float] = deque(maxlen=trend_window)
        self.action_history: deque[float] = deque(maxlen=action_window)
        self.peak_value = 0.0
        self.total_trades = 0
        self.steps_since_trade = 0
        self.consecutive_losses = 0

    def reset(self, initial_value: float):
        self.returns_history.clear()
        self.price_history.clear()
        self.action_history.clear()
        self.peak_value = initial_value
        self.total_trades = 0
        self.steps_since_trade = 0
        self.consecutive_losses = 0

    def calculate(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: float,
        prev_action: float,
        step: int,
        price: float = None,
    ) -> dict:
        if price is not None:
            self.price_history.append(price)
        self.action_history.append(action)

        # 1. 수익률
        raw_return = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(raw_return)

        # 2. Differential Sharpe
        sharpe_reward = self._compute_sharpe_reward(raw_return)

        # 3. 트렌드 팔로잉 보상
        trend_reward = self._compute_trend_reward(action)

        # 4. PnL 직접 보상
        pnl_reward = float(np.clip(raw_return * 50, -1.0, 1.0))

        # 5. 적응형 MDD 페널티
        self.peak_value = max(self.peak_value, curr_portfolio_value)
        drawdown = (self.peak_value - curr_portfolio_value) / self.peak_value

        if len(self.returns_history) >= 5:
            volatility = np.std(list(self.returns_history))
            dd_threshold = max(0.03, min(0.10, volatility * 10))
        else:
            dd_threshold = 0.05

        mdd_penalty = -drawdown * self.max_drawdown_penalty if drawdown > dd_threshold else 0

        # 6. 거래 관련 보상/페널티
        action_change = abs(action - prev_action)
        profit_bonus = 0.0
        trade_incentive = 0.0
        holding_bonus = 0.0
        loss_penalty = 0.0

        if action_change > 0.05:
            self.total_trades += 1
            self.steps_since_trade = 0

            if raw_return > 0.001:
                profit_bonus = 0.25
                self.consecutive_losses = 0
            elif raw_return < -0.001:
                loss_penalty = -0.05
                self.consecutive_losses += 1
                if self.consecutive_losses >= 3:
                    loss_penalty = -0.15

            # 거래 자체 인센티브 (정책 붕괴 방지)
            trade_incentive = 0.12
        else:
            self.steps_since_trade += 1
            # 수익 중인 포지션 유지 → 홀딩 보상 (축소: 상수행동 유인 방지)
            if raw_return > 0 and abs(action) > 0.3:
                holding_bonus = 0.02

        # 7. 비활동 페널티 (6스텝부터 — v8.2 강화)
        inactivity_penalty = 0.0
        if self.steps_since_trade > 6:
            inactivity_penalty = -0.12 * min(self.steps_since_trade - 6, 24) / 24

        # 8. 행동 다양성 보상 (v8.2 핵심 — 상수행동 정책 붕괴 방지)
        diversity_reward = self._compute_diversity_reward()

        # 종합: Sharpe(25%) + Trend(10%) + PnL(25%) + Diversity(15%) + 나머지(25%)
        total_reward = (
            sharpe_reward * 0.25
            + trend_reward * 0.10
            + pnl_reward * 0.25
            + diversity_reward * 0.15
            + mdd_penalty
            + profit_bonus
            + trade_incentive
            + holding_bonus
            + loss_penalty
            + inactivity_penalty
        )

        return {
            "reward": float(np.clip(total_reward, -2.0, 2.0)),
            "components": {
                "raw_return": float(raw_return),
                "sharpe_reward": float(sharpe_reward),
                "trend_reward": float(trend_reward),
                "pnl_reward": float(pnl_reward),
                "diversity_reward": float(diversity_reward),
                "mdd_penalty": float(mdd_penalty),
                "profit_bonus": float(profit_bonus),
                "trade_incentive": float(trade_incentive),
                "holding_bonus": float(holding_bonus),
                "loss_penalty": float(loss_penalty),
                "inactivity_penalty": float(inactivity_penalty),
                "drawdown": float(drawdown),
                "total_trades": self.total_trades,
            },
        }

    def _compute_diversity_reward(self) -> float:
        """행동 다양성 보상: 상수 행동 정책을 강하게 벌하고, 다양한 행동을 보상"""
        if len(self.action_history) < 5:
            return 0.0

        actions = np.array(self.action_history)
        action_std = actions.std()
        action_range = actions.max() - actions.min()

        # 상수 행동 감지 (std < 0.03) → 강한 페널티
        if action_std < 0.03:
            return -1.0  # 매우 강한 페널티

        # 낮은 다양성 (std 0.03~0.08) → 약한 페널티
        if action_std < 0.08:
            return -0.3

        # 적당한 다양성 (std 0.08~0.15) → 소폭 보상
        if action_std < 0.15:
            return 0.3

        # 높은 다양성 (std > 0.15) → 보너스 (단, 범위가 넓어야)
        if action_range > 0.5:
            return 0.6
        return 0.4

    def _compute_sharpe_reward(self, latest_return: float) -> float:
        """Differential Sharpe Ratio"""
        if len(self.returns_history) < 3:
            return latest_return * 15

        returns = np.array(self.returns_history)
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return < 1e-8:
            return mean_return * 15

        sharpe = (mean_return - self.risk_free_rate) / std_return
        return float(np.clip(sharpe * 0.3, -2.0, 2.0))

    def _compute_trend_reward(self, action: float) -> float:
        """트렌드 팔로잉: 추세와 같은 방향 포지션 시 보너스"""
        if len(self.price_history) < 5:
            return 0.0

        prices = np.array(self.price_history)
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_direction = np.sign(slope)
        trend_strength = min(abs(slope) / (prices.mean() * 0.001), 1.0)

        alignment = action * trend_direction
        return float(alignment * trend_strength * 0.15)

    def get_episode_stats(self, final_value: float, initial_value: float) -> dict:
        total_return = (final_value - initial_value) / initial_value
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])

        return {
            "total_return_pct": float(total_return * 100),
            "total_trades": self.total_trades,
            "avg_return": float(returns.mean()) if len(returns) > 0 else 0,
            "std_return": float(returns.std()) if len(returns) > 1 else 0,
            "max_drawdown": float((self.peak_value - final_value) / self.peak_value),
            "sharpe_ratio": float(
                (returns.mean() - self.risk_free_rate) / returns.std()
                if len(returns) > 1 and returns.std() > 1e-8
                else 0
            ),
        }
