"""보상 함수 v8 — 방향성 보상 + 수익성 중심

v7 문제: 모델이 거래하지만 방향 판단(사야 할 때 vs 팔아야 할 때)을 못 함
v8 개선:
  - 방향성 보상: BTC 비중 × 가격 변화 → 올바른 포지션에 보상
  - PnL 스케일 100 (v7의 50에서 2배)
  - 수익 거래 보너스 +0.5 / 손실 거래 -0.2 (비대칭 강화)
  - Sharpe 스케일 유지 0.25
"""

import numpy as np
from collections import deque

UPBIT_FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0003
TRANSACTION_COST = UPBIT_FEE_RATE + SLIPPAGE_RATE


class RewardCalculator:
    """방향성 + PnL + Sharpe 하이브리드 보상 (v8)"""

    def __init__(
        self,
        window_size: int = 20,
        risk_free_rate: float = 0.03 / 365 / 24,
        max_drawdown_penalty: float = 3.0,
        pnl_scale: float = 100.0,
    ):
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.pnl_scale = pnl_scale

        self.returns_history: deque[float] = deque(maxlen=window_size)
        self.peak_value = 0.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.steps_since_last_trade = 0
        self.prev_trade_value = 0.0

    def reset(self, initial_value: float):
        self.returns_history.clear()
        self.peak_value = initial_value
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.steps_since_last_trade = 0
        self.prev_trade_value = initial_value

    def calculate(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: float,
        prev_action: float,
        step: int,
        btc_ratio: float = 0.0,
        price_change: float = 0.0,
    ) -> dict:
        # 1. 수익률
        raw_return = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(raw_return)

        # 2. PnL 직접 보상 (스케일 2배 강화)
        pnl_reward = np.clip(raw_return * self.pnl_scale, -1.5, 1.5)

        # 3. 방향성 보상 — 올바른 포지션에 있으면 보상
        #   BTC 많이 보유 + 가격 상승 = 좋음
        #   현금 많이 보유 + 가격 하락 = 좋음
        direction_reward = 0.0
        if abs(price_change) > 0.001:  # 0.1% 이상 변동 시
            # btc_ratio: 0~1, price_change: 양수=상승, 음수=하락
            # 올바른 방향: btc_ratio * price_change > 0 (BTC 보유 중 상승)
            #            또는 (1-btc_ratio) * (-price_change) > 0 (현금 보유 중 하락)
            alignment = (btc_ratio - 0.5) * price_change  # 양수면 올바른 방향
            direction_reward = np.clip(alignment * 200, -0.5, 0.5)

        # 4. Sharpe 보상
        sharpe_reward = self._compute_sharpe_reward(raw_return)

        # 5. MDD 페널티 (3% 초과)
        self.peak_value = max(self.peak_value, curr_portfolio_value)
        drawdown = (self.peak_value - curr_portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        mdd_penalty = -drawdown * self.max_drawdown_penalty if drawdown > 0.03 else 0.0

        # 6. 거래 수익성 보상 (비대칭 강화)
        action_change = abs(action - prev_action)
        trade_pnl_bonus = 0.0
        self.steps_since_last_trade += 1

        if action_change > 0.05:
            self.total_trades += 1
            trade_return = (curr_portfolio_value - self.prev_trade_value) / self.prev_trade_value
            if trade_return > 0.002:
                trade_pnl_bonus = 0.5   # 수익 거래 강화 (v7: 0.3)
            elif trade_return < -0.002:
                trade_pnl_bonus = -0.2  # 손실 거래 강화 (v7: -0.15)
            self.prev_trade_value = curr_portfolio_value
            self.steps_since_last_trade = 0

        # 종합
        total_reward = pnl_reward + direction_reward + sharpe_reward + mdd_penalty + trade_pnl_bonus

        return {
            "reward": float(total_reward),
            "components": {
                "raw_return": float(raw_return),
                "pnl_reward": float(pnl_reward),
                "direction_reward": float(direction_reward),
                "sharpe_reward": float(sharpe_reward),
                "mdd_penalty": float(mdd_penalty),
                "trade_pnl_bonus": float(trade_pnl_bonus),
                "drawdown": float(drawdown),
                "total_trades": self.total_trades,
            },
        }

    def _compute_sharpe_reward(self, latest_return: float) -> float:
        n = len(self.returns_history)
        if n < 3:
            return latest_return * 15

        total = sum(self.returns_history)
        total_sq = sum(r * r for r in self.returns_history)
        mean_return = total / n
        variance = total_sq / n - mean_return * mean_return
        std_return = variance ** 0.5 if variance > 0 else 0.0

        if std_return < 1e-8:
            return mean_return * 15

        sharpe = (mean_return - self.risk_free_rate) / std_return
        scaled = sharpe * 0.25
        if scaled > 1.5:
            return 1.5
        if scaled < -1.5:
            return -1.5
        return float(scaled)

    def get_episode_stats(self, final_value: float, initial_value: float) -> dict:
        total_return = (final_value - initial_value) / initial_value
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        return {
            "total_return_pct": float(total_return * 100),
            "total_trades": self.total_trades,
            "avg_return": float(returns.mean()) if len(returns) > 0 else 0,
            "std_return": float(returns.std()) if len(returns) > 1 else 0,
            "max_drawdown": float(max(self.max_drawdown, (self.peak_value - final_value) / self.peak_value if self.peak_value > 0 else 0)),
            "sharpe_ratio": float(
                (returns.mean() - self.risk_free_rate) / returns.std()
                if len(returns) > 1 and returns.std() > 1e-8 else 0
            ),
        }
