"""보상 함수 v6 — Differential Sharpe + PnL 하이브리드

v1 Sharpe 기반이 가장 학습 가능했음 (explained_variance 0.42, 68 trades, +1.43%)
v4/v5 순수 PnL은 노이즈가 심해 value function 학습 불가 (explained_variance 0)

v6: v1 구조 유지 + 개선
  - Sharpe 보상 스케일 강화 (0.1 → 0.15)
  - 극단 포지션 페널티 제거 (확신 있는 거래 허용)
  - 거래 비용 이중 차감 제거 (환경에서만 적용)
  - 수익 실현 보너스 추가 (이익 거래 장려)
"""

import numpy as np
from collections import deque

UPBIT_FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0003
TRANSACTION_COST = UPBIT_FEE_RATE + SLIPPAGE_RATE


class RewardCalculator:
    """Differential Sharpe + PnL 하이브리드 보상"""

    def __init__(
        self,
        window_size: int = 20,
        risk_free_rate: float = 0.03 / 365 / 24,  # 연 3% (현실적 목표)
        max_drawdown_penalty: float = 2.0,
        overtrade_penalty: float = 0.05,  # 과매매 페널티
    ):
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.overtrade_penalty = overtrade_penalty

        self.returns_history: deque[float] = deque(maxlen=window_size)
        self.peak_value = 0.0
        self.total_trades = 0
        self.steps_since_last_trade = 0

    def reset(self, initial_value: float):
        self.returns_history.clear()
        self.peak_value = initial_value
        self.total_trades = 0
        self.steps_since_last_trade = 0

    def calculate(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: float,
        prev_action: float,
        step: int,
    ) -> dict:
        # 1. 수익률
        raw_return = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(raw_return)

        # 2. Differential Sharpe 보상 (윈도우 기반)
        sharpe_reward = self._compute_sharpe_reward(raw_return)

        # 3. MDD 페널티 (5% 초과 시)
        self.peak_value = max(self.peak_value, curr_portfolio_value)
        drawdown = (self.peak_value - curr_portfolio_value) / self.peak_value
        mdd_penalty = -drawdown * self.max_drawdown_penalty if drawdown > 0.05 else 0

        # 4. 수익 실현 보너스 — 포지션 변경 후 이익이면 보상
        action_change = abs(action - prev_action)
        profit_bonus = 0.0
        trade_penalty = 0.0
        self.steps_since_last_trade += 1

        if action_change > 0.05:
            self.total_trades += 1
            if raw_return > 0.001:  # 0.1% 이상 수익
                profit_bonus = 0.1

            # 과매매 페널티: 최근 거래 후 4스텝 이내 재거래 시 페널티
            if self.steps_since_last_trade < 4:
                trade_penalty = -self.overtrade_penalty
            self.steps_since_last_trade = 0

        # 종합
        total_reward = sharpe_reward + mdd_penalty + profit_bonus + trade_penalty

        return {
            "reward": float(total_reward),
            "components": {
                "raw_return": float(raw_return),
                "sharpe_reward": float(sharpe_reward),
                "mdd_penalty": float(mdd_penalty),
                "profit_bonus": float(profit_bonus),
                "drawdown": float(drawdown),
                "trade_penalty": float(trade_penalty),
                "total_trades": self.total_trades,
            },
        }

    def _compute_sharpe_reward(self, latest_return: float) -> float:
        """Differential Sharpe Ratio

        윈도우 내 평균/분산 기반 증분 샤프.
        초기 3스텝: 수익률 직접 스케일링 (안정적 학습 시작)
        """
        if len(self.returns_history) < 3:
            return latest_return * 10  # 초기: 단순 스케일링

        returns = np.array(self.returns_history)
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return < 1e-8:
            return mean_return * 10

        sharpe = (mean_return - self.risk_free_rate) / std_return
        return float(np.clip(sharpe * 0.15, -1.5, 1.5))

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
