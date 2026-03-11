"""보상 함수 v7 — 트렌드 팔로잉 + 적응형 보상

v6 문제점:
  - 상승장 추종력 부족 (90일 기준 B&H 대비 -5%p)
  - 샤프 보상만으로는 트렌드 방향성 학습 어려움

v7 개선:
  1. 트렌드 팔로잉 보상: 추세 방향 포지션 유지 시 보너스
  2. 포지션 사이즈 보상: 확신도에 비례한 포지션 크기 장려
  3. 적응형 MDD 페널티: 시장 변동성 기반 동적 임계값
  4. 홀딩 보상: 수익 포지션 유지 보너스 (과다 거래 방지)
"""

import numpy as np
from collections import deque

UPBIT_FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0003
TRANSACTION_COST = UPBIT_FEE_RATE + SLIPPAGE_RATE


class RewardCalculatorV7:
    """트렌드 팔로잉 + 적응형 보상"""

    def __init__(
        self,
        window_size: int = 20,
        trend_window: int = 10,
        risk_free_rate: float = 0.18 / 365 / 24,
        max_drawdown_penalty: float = 2.0,
    ):
        self.window_size = window_size
        self.trend_window = trend_window
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty

        self.returns_history: deque[float] = deque(maxlen=window_size)
        self.price_history: deque[float] = deque(maxlen=trend_window)
        self.peak_value = 0.0
        self.total_trades = 0
        self.prev_portfolio_value = 0.0

    def reset(self, initial_value: float):
        self.returns_history.clear()
        self.price_history.clear()
        self.peak_value = initial_value
        self.total_trades = 0
        self.prev_portfolio_value = initial_value

    def calculate(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: float,
        prev_action: float,
        step: int,
        price: float = None,
    ) -> dict:
        # 가격 히스토리 추적
        if price is not None:
            self.price_history.append(price)

        # 1. 수익률
        raw_return = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(raw_return)

        # 2. Differential Sharpe 보상
        sharpe_reward = self._compute_sharpe_reward(raw_return)

        # 3. 트렌드 팔로잉 보상
        trend_reward = self._compute_trend_reward(action)

        # 4. 적응형 MDD 페널티
        self.peak_value = max(self.peak_value, curr_portfolio_value)
        drawdown = (self.peak_value - curr_portfolio_value) / self.peak_value
        volatility = np.std(list(self.returns_history)) if len(self.returns_history) > 3 else 0.01
        dd_threshold = max(0.03, min(0.10, volatility * 10))  # 변동성 기반 동적 임계값
        mdd_penalty = -drawdown * self.max_drawdown_penalty if drawdown > dd_threshold else 0

        # 5. 수익 포지션 홀딩 보상 (과다 거래 방지)
        action_change = abs(action - prev_action)
        holding_bonus = 0.0
        profit_bonus = 0.0

        if action_change > 0.05:
            self.total_trades += 1
            if raw_return > 0.001:
                profit_bonus = 0.1
        elif raw_return > 0 and abs(action) > 0.3:
            # 수익 중인 포지션 유지 → 보너스
            holding_bonus = 0.05

        # 6. PnL 직접 보상 (트렌드 추종 강화)
        pnl_reward = raw_return * 5.0  # 수익률 직접 반영

        # 종합
        total_reward = (
            sharpe_reward * 0.3
            + trend_reward * 0.2
            + pnl_reward * 0.3
            + mdd_penalty
            + profit_bonus
            + holding_bonus
        )

        return {
            "reward": float(np.clip(total_reward, -2.0, 2.0)),
            "components": {
                "raw_return": float(raw_return),
                "sharpe_reward": float(sharpe_reward),
                "trend_reward": float(trend_reward),
                "pnl_reward": float(pnl_reward),
                "mdd_penalty": float(mdd_penalty),
                "profit_bonus": float(profit_bonus),
                "holding_bonus": float(holding_bonus),
                "drawdown": float(drawdown),
                "dd_threshold": float(dd_threshold),
                "total_trades": self.total_trades,
            },
        }

    def _compute_sharpe_reward(self, latest_return: float) -> float:
        """Differential Sharpe Ratio"""
        if len(self.returns_history) < 3:
            return latest_return * 10

        returns = np.array(self.returns_history)
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return < 1e-8:
            return mean_return * 10

        sharpe = (mean_return - self.risk_free_rate) / std_return
        return float(np.clip(sharpe * 0.15, -1.5, 1.5))

    def _compute_trend_reward(self, action: float) -> float:
        """트렌드 팔로잉 보상: 추세와 같은 방향 포지션 시 보너스"""
        if len(self.price_history) < 5:
            return 0.0

        prices = np.array(self.price_history)
        # 단기 트렌드 방향 (선형 회귀 기울기)
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_direction = np.sign(slope)  # +1: 상승, -1: 하락

        # 트렌드 강도 (정규화)
        trend_strength = min(abs(slope) / (prices.mean() * 0.001), 1.0)

        # 행동과 트렌드 방향 일치도
        alignment = action * trend_direction  # [-1, 1]

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
