"""보상 함수 v7 — 수익성 중심 + 정책 붕괴 방지

v6 문제: Sharpe 보상이 anti-collapse 보너스에 묻혀 수익성 학습 불가
v7 개선:
  - PnL 직접 보상 강화 (수익률 × 스케일러)
  - 수익 거래 보너스 / 손실 거래 페널티 (비대칭)
  - MDD 페널티 조기 발동 (3%)
  - Sharpe 스케일 상향 (0.15 → 0.25)
  - 과매매 판단 완화 (2스텝 → 적응형)
"""

import numpy as np
from collections import deque

UPBIT_FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0003
TRANSACTION_COST = UPBIT_FEE_RATE + SLIPPAGE_RATE


class RewardCalculator:
    """PnL 중심 + Sharpe 하이브리드 보상 (v7)"""

    def __init__(
        self,
        window_size: int = 20,
        risk_free_rate: float = 0.03 / 365 / 24,
        max_drawdown_penalty: float = 3.0,
        pnl_scale: float = 50.0,
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
        self.prev_trade_value = 0.0  # 직전 거래 시점 포트폴리오 값

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
    ) -> dict:
        # 1. 수익률 (스텝 간)
        raw_return = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(raw_return)

        # 2. PnL 직접 보상 — 포트폴리오 가치 변화를 직접 반영
        pnl_reward = np.clip(raw_return * self.pnl_scale, -1.0, 1.0)

        # 3. Differential Sharpe 보상
        sharpe_reward = self._compute_sharpe_reward(raw_return)

        # 4. MDD 페널티 (3% 초과 시, v6의 5%에서 강화)
        self.peak_value = max(self.peak_value, curr_portfolio_value)
        drawdown = (self.peak_value - curr_portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        mdd_penalty = 0.0
        if drawdown > 0.03:
            mdd_penalty = -drawdown * self.max_drawdown_penalty

        # 5. 거래 수익성 보상 — 거래 시 이전 거래 대비 성과
        action_change = abs(action - prev_action)
        trade_pnl_bonus = 0.0
        self.steps_since_last_trade += 1

        if action_change > 0.05:
            self.total_trades += 1
            # 이전 거래 시점 대비 수익률
            trade_return = (curr_portfolio_value - self.prev_trade_value) / self.prev_trade_value
            if trade_return > 0.002:   # 0.2% 이상 수익
                trade_pnl_bonus = 0.3   # 수익 거래 강한 보상
            elif trade_return < -0.002:  # 0.2% 이상 손실
                trade_pnl_bonus = -0.15  # 손실 거래 페널티
            self.prev_trade_value = curr_portfolio_value
            self.steps_since_last_trade = 0

        # 종합: PnL + Sharpe + MDD + 거래수익성
        total_reward = pnl_reward + sharpe_reward + mdd_penalty + trade_pnl_bonus

        return {
            "reward": float(total_reward),
            "components": {
                "raw_return": float(raw_return),
                "pnl_reward": float(pnl_reward),
                "sharpe_reward": float(sharpe_reward),
                "mdd_penalty": float(mdd_penalty),
                "trade_pnl_bonus": float(trade_pnl_bonus),
                "drawdown": float(drawdown),
                "total_trades": self.total_trades,
            },
        }

    def _compute_sharpe_reward(self, latest_return: float) -> float:
        """Differential Sharpe Ratio — v7에서 스케일 강화."""
        n = len(self.returns_history)
        if n < 3:
            return latest_return * 15  # 초기: 수익률 직접 스케일링

        total = sum(self.returns_history)
        total_sq = sum(r * r for r in self.returns_history)
        mean_return = total / n
        variance = total_sq / n - mean_return * mean_return
        std_return = variance ** 0.5 if variance > 0 else 0.0

        if std_return < 1e-8:
            return mean_return * 15

        sharpe = (mean_return - self.risk_free_rate) / std_return
        scaled = sharpe * 0.25  # v6의 0.15에서 강화
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
                if len(returns) > 1 and returns.std() > 1e-8
                else 0
            ),
        }
