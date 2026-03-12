"""Multi-Objective RL 보상 시스템

5가지 목표를 동시에 추적하고, Pareto 최적 모델을 발견하며,
적응적 가중치 조절을 통해 제약 조건을 만족하는 정책을 학습한다.

목표:
  1. Profit: 순수 PnL 수익률
  2. Risk: MDD 최소화 (음의 MDD)
  3. Efficiency: 거래당 수익 (과매매 억제)
  4. Sharpe: 위험 조정 수익률 (기존 v6 호환)
  5. Tail Risk: CVaR 5% (극단 손실 관리)

기존 RewardCalculator와 동일한 인터페이스를 제공하여 drop-in 교체 가능.
Envelope MORL: 가중치 벡터를 관측 공간에 concat하여
단일 네트워크로 다양한 정책을 생성할 수 있다.
"""

import copy
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST

logger = logging.getLogger("rl.multi_objective")

# --- 기본 상수 ---
NUM_OBJECTIVES = 5
OBJECTIVE_NAMES = ["profit", "risk", "efficiency", "sharpe", "tail_risk"]

# 기본 가중치 (균등)
DEFAULT_WEIGHTS = {
    "profit": 0.2,
    "risk": 0.2,
    "efficiency": 0.2,
    "sharpe": 0.2,
    "tail_risk": 0.2,
}

# 제약 임계값
DEFAULT_CONSTRAINTS = {
    "max_mdd": 0.10,  # MDD 10% 이하
    "max_trades_per_day": 20.0,  # 일 20회 이하
}

# CVaR 설정
CVAR_ALPHA = 0.05  # 하위 5%
CVAR_WINDOW = 200  # 롤링 윈도우


# ========================================================================
# 1. Multi-Objective Reward Calculator
# ========================================================================


class MultiObjectiveReward:
    """다중 목표 보상 계산기

    5가지 독립적 보상 스트림을 각각 [-1, 1]로 정규화하고,
    가중 합산(scalarization)으로 단일 스칼라 보상을 생성한다.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        window_size: int = 20,
        risk_free_rate: float = 0.03 / 365 / 24,
        cvar_alpha: float = CVAR_ALPHA,
        cvar_window: int = CVAR_WINDOW,
    ):
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self._validate_weights()

        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.cvar_alpha = cvar_alpha
        self.cvar_window = cvar_window

        # 내부 상태
        self.returns_history: deque[float] = deque(maxlen=window_size)
        self.all_returns: deque[float] = deque(maxlen=cvar_window)
        self.peak_value = 0.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.total_profit = 0.0  # 누적 수익률
        self.steps_since_last_trade = 0
        self.episode_steps = 0

        # 목표별 누적 보상 (에피소드 통계용)
        self.objective_sums: dict[str, float] = {n: 0.0 for n in OBJECTIVE_NAMES}

    def _validate_weights(self):
        for name in OBJECTIVE_NAMES:
            if name not in self.weights:
                self.weights[name] = DEFAULT_WEIGHTS[name]
        total = sum(self.weights[n] for n in OBJECTIVE_NAMES)
        if total > 0:
            for n in OBJECTIVE_NAMES:
                self.weights[n] /= total

    def set_weights(self, weights: dict[str, float]):
        """런타임 가중치 변경 (Envelope MORL용)"""
        self.weights = dict(weights)
        self._validate_weights()

    def get_weights_vector(self) -> np.ndarray:
        """현재 가중치를 numpy 배열로 반환 (관측 공간 concat용)"""
        return np.array(
            [self.weights[n] for n in OBJECTIVE_NAMES], dtype=np.float32
        )

    def reset(self, initial_value: float):
        self.returns_history.clear()
        self.all_returns.clear()
        self.peak_value = initial_value
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.total_profit = 0.0
        self.steps_since_last_trade = 0
        self.episode_steps = 0
        self.objective_sums = {n: 0.0 for n in OBJECTIVE_NAMES}

    def calculate(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: float,
        prev_action: float,
        step: int,
    ) -> dict:
        """보상 계산 — RewardCalculator와 동일 인터페이스

        Returns:
            {"reward": float, "components": dict, "objectives": dict}
        """
        self.episode_steps += 1

        # 1. 기본 수익률
        raw_return = (
            (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            if prev_portfolio_value > 0
            else 0.0
        )
        self.returns_history.append(raw_return)
        self.all_returns.append(raw_return)
        self.total_profit += raw_return

        # 2. MDD 업데이트
        self.peak_value = max(self.peak_value, curr_portfolio_value)
        drawdown = (
            (self.peak_value - curr_portfolio_value) / self.peak_value
            if self.peak_value > 0
            else 0.0
        )
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # 3. 거래 감지
        action_change = abs(action - prev_action)
        traded = action_change > 0.05
        self.steps_since_last_trade += 1
        if traded:
            self.total_trades += 1
            self.steps_since_last_trade = 0

        # --- 5가지 목표별 보상 계산 ---
        objectives = {}

        # (a) Profit objective: 스텝 수익률 [-1, 1]
        profit_raw = raw_return * 100  # 0.01 → 1.0 스케일
        objectives["profit"] = float(np.clip(profit_raw, -1.0, 1.0))

        # (b) Risk objective: -drawdown [-1, 0]
        # drawdown이 0이면 0, 10%이면 -1
        risk_raw = -drawdown * 10
        objectives["risk"] = float(np.clip(risk_raw, -1.0, 0.0))

        # (c) Efficiency objective: 거래당 수익
        efficiency_raw = self._compute_efficiency(raw_return, traded)
        objectives["efficiency"] = float(np.clip(efficiency_raw, -1.0, 1.0))

        # (d) Sharpe objective: Differential Sharpe
        sharpe_raw = self._compute_sharpe_reward(raw_return)
        objectives["sharpe"] = float(np.clip(sharpe_raw, -1.0, 1.0))

        # (e) Tail risk objective: CVaR penalty
        tail_raw = self._compute_cvar_reward()
        objectives["tail_risk"] = float(np.clip(tail_raw, -1.0, 0.0))

        # --- 가중 합산 ---
        total_reward = sum(
            self.weights[name] * objectives[name] for name in OBJECTIVE_NAMES
        )

        # 누적 통계
        for name in OBJECTIVE_NAMES:
            self.objective_sums[name] += objectives[name]

        return {
            "reward": float(total_reward),
            "objectives": objectives,
            "components": {
                "raw_return": float(raw_return),
                "sharpe_reward": objectives["sharpe"],
                "mdd_penalty": objectives["risk"],
                "profit_bonus": max(0.0, objectives["profit"]),
                "drawdown": float(drawdown),
                "trade_penalty": min(0.0, objectives["efficiency"]),
                "total_trades": self.total_trades,
                "cvar_5pct": float(tail_raw),
            },
        }

    def _compute_efficiency(self, raw_return: float, traded: bool) -> float:
        """거래 효율성 보상

        거래 없이 보유 → 소폭 양의 보상 (hold 장려)
        수익 거래 → 큰 양의 보상
        손실 거래 → 음의 보상
        과매매 (4스텝 이내 재거래) → 추가 페널티
        """
        if not traded:
            # 관망 시: 시장이 하락하면 약간 보상 (빠져나간 것)
            # 시장이 상승하면 약간 페널티 (기회 비용) — 하지만 아주 작게
            return raw_return * 5  # 스케일 작게

        # 거래 발생
        efficiency = raw_return * 50  # 수익 거래 = 양, 손실 = 음

        # 과매매 페널티: 최근 거래 후 4스텝 미만
        if self.steps_since_last_trade < 4 and self.total_trades > 1:
            efficiency -= 0.3

        return efficiency

    def _compute_sharpe_reward(self, latest_return: float) -> float:
        """Differential Sharpe Ratio (기존 v6 호환)"""
        n = len(self.returns_history)
        if n < 3:
            return latest_return * 10

        total = sum(self.returns_history)
        total_sq = sum(r * r for r in self.returns_history)
        mean_return = total / n
        variance = total_sq / n - mean_return * mean_return
        std_return = variance**0.5 if variance > 0 else 0.0

        if std_return < 1e-8:
            return mean_return * 10

        sharpe = (mean_return - self.risk_free_rate) / std_return
        return float(sharpe * 0.15)

    def _compute_cvar_reward(self) -> float:
        """CVaR(5%) 보상 — 극단 손실 페널티

        하위 5% 수익률의 평균을 계산하고, 이를 페널티로 변환.
        수익률이 전부 양수면 페널티 0.
        """
        n = len(self.all_returns)
        if n < 10:
            return 0.0

        sorted_returns = sorted(self.all_returns)
        cutoff = max(1, int(n * self.cvar_alpha))
        tail_returns = sorted_returns[:cutoff]
        cvar = sum(tail_returns) / len(tail_returns)

        # cvar이 음수면 페널티 (스케일: -0.01 → -0.5)
        return float(cvar * 50)

    def get_episode_stats(self, final_value: float, initial_value: float) -> dict:
        """에피소드 통계 — 기존 RewardCalculator 호환"""
        total_return = (
            (final_value - initial_value) / initial_value
            if initial_value > 0
            else 0.0
        )
        returns = (
            np.array(list(self.returns_history))
            if self.returns_history
            else np.array([0.0])
        )
        std = float(returns.std()) if len(returns) > 1 else 0.0

        stats = {
            "total_return_pct": float(total_return * 100),
            "total_trades": self.total_trades,
            "avg_return": float(returns.mean()),
            "std_return": std,
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(
                (returns.mean() - self.risk_free_rate) / std
                if std > 1e-8
                else 0.0
            ),
        }

        # 다중 목표 추가 통계
        stats["objective_means"] = {
            name: self.objective_sums[name] / max(1, self.episode_steps)
            for name in OBJECTIVE_NAMES
        }
        stats["cvar_5pct"] = float(self._compute_cvar_reward())
        stats["return_per_trade"] = (
            float(total_return / self.total_trades)
            if self.total_trades > 0
            else 0.0
        )

        return stats

    def get_objective_scores(self) -> list[float]:
        """현재 에피소드의 목표별 평균 점수 (Pareto 평가용)"""
        return [
            self.objective_sums[name] / max(1, self.episode_steps)
            for name in OBJECTIVE_NAMES
        ]


# ========================================================================
# 2. Pareto Frontier
# ========================================================================


@dataclass
class ParetoSolution:
    """Pareto 해 하나를 표현"""

    model_path: str
    scores: list[float]  # [profit, risk, efficiency, sharpe, tail_risk]
    metadata: dict = field(default_factory=dict)

    def dominates(self, other: "ParetoSolution") -> bool:
        """self가 other를 지배하는지 (모든 목표에서 같거나 나음, 하나 이상 나음)"""
        dominated = False
        for s, o in zip(self.scores, other.scores):
            if s < o:
                return False
            if s > o:
                dominated = True
        return dominated


class ParetoFrontier:
    """Pareto 최적 해 집합 관리

    훈련 중 비지배 해를 유지하고, 시각화와 모델 선택을 지원한다.
    """

    def __init__(self, max_solutions: int = 10, save_dir: str = None):
        self.max_solutions = max_solutions
        self.solutions: list[ParetoSolution] = []
        self.save_dir = save_dir or os.path.join(
            "data", "rl_models", "pareto"
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, solution: ParetoSolution) -> bool:
        """새 해를 추가하고 지배되는 해를 제거

        Returns:
            True if solution was added to the frontier
        """
        # 기존 해에 의해 지배되면 추가하지 않음
        for existing in self.solutions:
            if existing.dominates(solution):
                return False

        # 새 해에 의해 지배되는 기존 해 제거
        self.solutions = [
            s for s in self.solutions if not solution.dominates(s)
        ]

        self.solutions.append(solution)

        # 최대 수 초과 시 가장 약한 해 제거 (합산 점수 기준)
        if len(self.solutions) > self.max_solutions:
            self.solutions.sort(key=lambda s: sum(s.scores), reverse=True)
            self.solutions = self.solutions[: self.max_solutions]

        return True

    def get_best(self, objective_idx: int = 0) -> Optional[ParetoSolution]:
        """특정 목표 기준 최적 해 반환"""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda s: s.scores[objective_idx])

    def get_balanced(self) -> Optional[ParetoSolution]:
        """균형 잡힌 해 반환 (합산 점수 최대)"""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda s: sum(s.scores))

    def save(self):
        """Pareto frontier를 JSON으로 저장"""
        data = []
        for sol in self.solutions:
            data.append(
                {
                    "model_path": sol.model_path,
                    "scores": {
                        name: score
                        for name, score in zip(OBJECTIVE_NAMES, sol.scores)
                    },
                    "metadata": sol.metadata,
                }
            )

        path = os.path.join(self.save_dir, "pareto_frontier.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Pareto frontier 저장: {path} ({len(data)}개 해)")

    def load(self) -> bool:
        """저장된 Pareto frontier 로드"""
        path = os.path.join(self.save_dir, "pareto_frontier.json")
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.solutions = []
        for item in data:
            scores = [item["scores"].get(name, 0.0) for name in OBJECTIVE_NAMES]
            self.solutions.append(
                ParetoSolution(
                    model_path=item["model_path"],
                    scores=scores,
                    metadata=item.get("metadata", {}),
                )
            )
        logger.info(f"Pareto frontier 로드: {len(self.solutions)}개 해")
        return True

    def plot(self, filename: str = "pareto_front.png"):
        """Pareto front 시각화 (profit vs risk, profit vs sharpe)

        2D scatter plot으로 주요 목표 쌍의 트레이드오프를 보여준다.
        """
        if not PLT_AVAILABLE:
            logger.warning("matplotlib 미설치 — Pareto 시각화 생략")
            return

        if len(self.solutions) < 2:
            logger.info("Pareto 해가 2개 미만 — 시각화 생략")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        scores_array = np.array([s.scores for s in self.solutions])

        # (a) Profit vs Risk
        ax = axes[0]
        ax.scatter(scores_array[:, 0], scores_array[:, 1], c="steelblue", s=60)
        ax.set_xlabel("Profit")
        ax.set_ylabel("Risk (neg MDD)")
        ax.set_title("Profit vs Risk")
        ax.grid(True, alpha=0.3)

        # (b) Profit vs Sharpe
        ax = axes[1]
        ax.scatter(scores_array[:, 0], scores_array[:, 3], c="darkorange", s=60)
        ax.set_xlabel("Profit")
        ax.set_ylabel("Sharpe")
        ax.set_title("Profit vs Sharpe")
        ax.grid(True, alpha=0.3)

        # (c) Efficiency vs Tail Risk
        ax = axes[2]
        ax.scatter(scores_array[:, 2], scores_array[:, 4], c="seagreen", s=60)
        ax.set_xlabel("Efficiency")
        ax.set_ylabel("Tail Risk (CVaR)")
        ax.set_title("Efficiency vs Tail Risk")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Pareto front 저장: {path}")


# ========================================================================
# 3. Adaptive Weight Scheduler
# ========================================================================


class AdaptiveWeightScheduler:
    """적응적 가중치 조절기

    훈련 진행 중 성과 갭과 제약 위반에 따라 목표 가중치를 동적으로 조절한다.
    """

    def __init__(
        self,
        initial_weights: dict[str, float] | None = None,
        constraints: dict[str, float] | None = None,
        adjustment_rate: float = 0.1,
        constraint_multiplier: float = 2.0,
        update_interval: int = 1000,  # 매 N스텝마다 가중치 업데이트
    ):
        self.base_weights = dict(initial_weights or DEFAULT_WEIGHTS)
        self.current_weights = dict(self.base_weights)
        self.constraints = dict(constraints or DEFAULT_CONSTRAINTS)
        self.adjustment_rate = adjustment_rate
        self.constraint_multiplier = constraint_multiplier
        self.update_interval = update_interval

        # 모니터링 버퍼
        self._mdd_history: deque[float] = deque(maxlen=50)
        self._trade_counts: deque[float] = deque(maxlen=50)
        self._return_history: deque[float] = deque(maxlen=50)
        self._step_count = 0

    def update(
        self,
        mdd: float,
        trades_per_day: float,
        episode_return: float,
        objective_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """에피소드 종료 시 가중치 업데이트

        Args:
            mdd: 에피소드 MDD
            trades_per_day: 일 평균 거래 수
            episode_return: 에피소드 수익률
            objective_scores: 목표별 평균 점수

        Returns:
            업데이트된 가중치
        """
        self._mdd_history.append(mdd)
        self._trade_counts.append(trades_per_day)
        self._return_history.append(episode_return)
        self._step_count += 1

        if self._step_count % self.update_interval != 0 and self._step_count > 1:
            return dict(self.current_weights)

        # 최근 평균으로 판단
        avg_mdd = sum(self._mdd_history) / len(self._mdd_history)
        avg_trades = sum(self._trade_counts) / len(self._trade_counts)
        avg_return = sum(self._return_history) / len(self._return_history)

        new_weights = dict(self.base_weights)

        # --- 제약 기반 조절 ---

        # MDD 제약 위반 → risk 가중치 강화
        if avg_mdd > self.constraints["max_mdd"]:
            violation_ratio = avg_mdd / self.constraints["max_mdd"]
            new_weights["risk"] *= self.constraint_multiplier * min(
                violation_ratio, 3.0
            )
            logger.debug(
                f"MDD 제약 위반: {avg_mdd:.2%} > {self.constraints['max_mdd']:.2%} "
                f"→ risk 가중치 {new_weights['risk']:.3f}"
            )

        # 과매매 제약 위반 → efficiency 가중치 강화
        if avg_trades > self.constraints["max_trades_per_day"]:
            violation_ratio = avg_trades / self.constraints["max_trades_per_day"]
            new_weights["efficiency"] *= self.constraint_multiplier * min(
                violation_ratio, 3.0
            )
            logger.debug(
                f"거래 빈도 제약 위반: {avg_trades:.1f} > "
                f"{self.constraints['max_trades_per_day']:.0f} "
                f"→ efficiency 가중치 {new_weights['efficiency']:.3f}"
            )

        # --- 성과 갭 기반 조절 ---

        # 수익률이 낮으면 profit 가중치 증가
        if avg_return < -0.01:  # -1% 미만
            new_weights["profit"] *= 1.0 + self.adjustment_rate * 2
        elif avg_return < 0:
            new_weights["profit"] *= 1.0 + self.adjustment_rate

        # 목표별 점수 기반 미세 조절
        if objective_scores:
            for name in OBJECTIVE_NAMES:
                score = objective_scores.get(name, 0.0)
                # 점수가 매우 낮은 목표의 가중치를 올림
                if score < -0.5:
                    new_weights[name] *= 1.0 + self.adjustment_rate

        # 정규화
        total = sum(new_weights[n] for n in OBJECTIVE_NAMES)
        if total > 0:
            for n in OBJECTIVE_NAMES:
                new_weights[n] /= total

        # 점진적 변화 (현재 가중치와 새 가중치의 EMA)
        alpha = 0.3
        for n in OBJECTIVE_NAMES:
            self.current_weights[n] = (
                alpha * new_weights[n] + (1 - alpha) * self.current_weights[n]
            )

        # 최종 정규화
        total = sum(self.current_weights[n] for n in OBJECTIVE_NAMES)
        if total > 0:
            for n in OBJECTIVE_NAMES:
                self.current_weights[n] /= total

        return dict(self.current_weights)

    def get_weights(self) -> dict[str, float]:
        return dict(self.current_weights)

    def get_weights_vector(self) -> np.ndarray:
        return np.array(
            [self.current_weights[n] for n in OBJECTIVE_NAMES], dtype=np.float32
        )


# ========================================================================
# 4. Multi-Objective Environment Wrapper
# ========================================================================


if GYM_AVAILABLE:

    class MultiObjectiveEnv(gym.Wrapper):
        """다중 목표 보상을 사용하는 환경 래퍼

        기존 BitcoinTradingEnv를 감싸고, RewardCalculator를
        MultiObjectiveReward로 교체한다.

        Envelope MORL 모드에서는 관측 공간에 가중치 벡터(5차원)를
        concat하여 가중치 조건부 정책을 학습한다.
        """

        def __init__(
            self,
            env: gym.Env,
            weights: dict[str, float] | None = None,
            envelope_morl: bool = False,
            weight_scheduler: AdaptiveWeightScheduler | None = None,
            randomize_weights: bool = False,
        ):
            """
            Args:
                env: BitcoinTradingEnv 인스턴스
                weights: 목표별 가중치 (None이면 균등)
                envelope_morl: True면 가중치를 관측에 concat
                weight_scheduler: 적응적 가중치 조절기
                randomize_weights: True면 에피소드마다 가중치를 랜덤 샘플링
                    (envelope_morl과 함께 사용)
            """
            super().__init__(env)

            self.mo_reward = MultiObjectiveReward(weights=weights)
            self.envelope_morl = envelope_morl
            self.weight_scheduler = weight_scheduler
            self.randomize_weights = randomize_weights

            # 기존 환경의 reward_calc를 교체
            if hasattr(self.env, "reward_calc"):
                self.env.reward_calc = self.mo_reward

            # Envelope MORL: 관측 공간 확장 (+5차원)
            if self.envelope_morl:
                base_shape = self.observation_space.shape[0]
                self.observation_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(base_shape + NUM_OBJECTIVES,),
                    dtype=np.float32,
                )

            # 에피소드 통계
            self._episode_count = 0
            self._prev_action = 0.0

        def reset(self, seed=None, options=None):
            obs, info = self.env.reset(seed=seed, options=options)

            initial_value = (
                self.env.initial_balance
                if hasattr(self.env, "initial_balance")
                else 10_000_000
            )
            self.mo_reward.reset(initial_value)
            self._prev_action = 0.0
            self._episode_count += 1

            # Envelope MORL: 에피소드마다 가중치 랜덤화
            if self.randomize_weights and self.envelope_morl:
                random_w = np.random.dirichlet(np.ones(NUM_OBJECTIVES))
                w_dict = {
                    name: float(w)
                    for name, w in zip(OBJECTIVE_NAMES, random_w)
                }
                self.mo_reward.set_weights(w_dict)

            if self.envelope_morl:
                obs = self._append_weights(obs)

            return obs, info

        def step(self, action):
            obs, _, terminated, truncated, info = self.env.step(action)

            # 다중 목표 보상 재계산
            action_val = float(np.clip(action[0], -1, 1))
            candle = self.env.candles[self.env.current_step]
            price = candle["close"]
            curr_value = self.env._portfolio_value(price)
            prev_value = (
                self.env.total_value_history[-2]
                if len(self.env.total_value_history) >= 2
                else self.env.initial_balance
            )

            reward_info = self.mo_reward.calculate(
                prev_portfolio_value=prev_value,
                curr_portfolio_value=curr_value,
                action=action_val,
                prev_action=self._prev_action,
                step=self.env.current_step,
            )

            self._prev_action = action_val
            reward = reward_info["reward"]

            info["reward_components"] = reward_info["components"]
            info["objectives"] = reward_info["objectives"]

            if self.envelope_morl:
                obs = self._append_weights(obs)

            # 파산 페널티 (원본 환경과 동일)
            if terminated:
                reward -= 1.0

            return obs, reward, terminated, truncated, info

        def _append_weights(self, obs: np.ndarray) -> np.ndarray:
            """관측에 가중치 벡터를 concat"""
            weights = self.mo_reward.get_weights_vector()
            return np.concatenate([obs, weights]).astype(np.float32)

        def get_episode_stats(self) -> dict:
            """에피소드 통계 (다중 목표 포함)"""
            if hasattr(self.env, "get_episode_stats"):
                base_stats = self.env.get_episode_stats()
            else:
                base_stats = {}

            # 다중 목표 통계 추가
            final_value = (
                self.env.total_value_history[-1]
                if hasattr(self.env, "total_value_history")
                and self.env.total_value_history
                else self.env.initial_balance
            )
            initial_value = (
                self.env.initial_balance
                if hasattr(self.env, "initial_balance")
                else 10_000_000
            )
            mo_stats = self.mo_reward.get_episode_stats(final_value, initial_value)

            # base_stats에 mo 데이터 병합
            base_stats.update(mo_stats)
            base_stats["weights"] = dict(self.mo_reward.weights)

            return base_stats

        def set_objective_weights(self, weights: dict[str, float]):
            """런타임에 가중치 변경 (추론 시 모드 전환용)"""
            self.mo_reward.set_weights(weights)

else:
    # gymnasium 미설치 시 placeholder
    class MultiObjectiveEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium이 필요합니다: pip install gymnasium")


# ========================================================================
# 5. Multi-Objective Training Utilities
# ========================================================================


class MultiObjectiveCallback:
    """SB3 콜백 — 다중 목표 훈련 모니터링 + Pareto 업데이트

    stable-baselines3의 BaseCallback을 상속하여 사용한다.
    """

    def __init__(
        self,
        pareto: ParetoFrontier,
        scheduler: AdaptiveWeightScheduler | None = None,
        eval_env=None,
        eval_freq: int = 10000,
        model_save_dir: str = None,
    ):
        self.pareto = pareto
        self.scheduler = scheduler
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.model_save_dir = model_save_dir or os.path.join(
            "data", "rl_models", "pareto", "checkpoints"
        )
        os.makedirs(self.model_save_dir, exist_ok=True)
        self._n_calls = 0

    def on_training_step(self, model, env, n_steps: int):
        """훈련 스텝마다 호출 (SB3 콜백 외부에서 수동 호출 가능)"""
        self._n_calls += 1

        if self._n_calls % self.eval_freq != 0:
            return

        if self.eval_env is None:
            return

        # 평가 에피소드 실행
        scores = self._evaluate_model(model)
        if scores is None:
            return

        # 모델 저장
        model_path = os.path.join(
            self.model_save_dir, f"model_step_{self._n_calls}"
        )
        model.save(model_path)

        # Pareto frontier 업데이트
        solution = ParetoSolution(
            model_path=model_path,
            scores=scores,
            metadata={"step": self._n_calls},
        )
        added = self.pareto.add(solution)
        if added:
            logger.info(
                f"[Step {self._n_calls}] Pareto 해 추가: "
                f"profit={scores[0]:.4f}, risk={scores[1]:.4f}, "
                f"efficiency={scores[2]:.4f}, sharpe={scores[3]:.4f}, "
                f"tail_risk={scores[4]:.4f}"
            )
            self.pareto.save()

        # 적응적 가중치 업데이트
        if self.scheduler and isinstance(self.eval_env, MultiObjectiveEnv):
            stats = self.eval_env.get_episode_stats()
            trades_per_day = stats.get("total_trades", 0) / max(
                1, stats.get("steps", 1) / 6
            )  # 4h봉 기준 6스텝=1일
            new_weights = self.scheduler.update(
                mdd=stats.get("max_drawdown", 0),
                trades_per_day=trades_per_day,
                episode_return=stats.get("total_return_pct", 0) / 100,
                objective_scores=stats.get("objective_means"),
            )
            self.eval_env.set_objective_weights(new_weights)
            logger.debug(f"가중치 업데이트: {new_weights}")

    def _evaluate_model(self, model) -> list[float] | None:
        """모델을 평가하여 목표별 점수 반환"""
        try:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

            stats = self.eval_env.get_episode_stats()
            obj_means = stats.get("objective_means", {})
            return [obj_means.get(name, 0.0) for name in OBJECTIVE_NAMES]
        except Exception as e:
            logger.warning(f"평가 실패: {e}")
            return None

    def finalize(self):
        """훈련 종료 시 Pareto front 저장 + 시각화"""
        self.pareto.save()
        self.pareto.plot()
        logger.info(
            f"훈련 종료. Pareto frontier: {len(self.pareto.solutions)}개 해"
        )


# ========================================================================
# 6. SB3 BaseCallback 통합
# ========================================================================

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class MultiObjectiveSB3Callback(BaseCallback):
        """SB3 콜백으로 MultiObjectiveCallback을 래핑

        SB3의 learn() 메서드에 직접 전달할 수 있다.
        """

        def __init__(
            self,
            mo_callback: MultiObjectiveCallback,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.mo_callback = mo_callback

        def _on_step(self) -> bool:
            self.mo_callback.on_training_step(
                model=self.model,
                env=self.training_env,
                n_steps=self.n_calls,
            )
            return True

        def _on_training_end(self) -> None:
            self.mo_callback.finalize()

    SB3_CALLBACK_AVAILABLE = True
except ImportError:
    SB3_CALLBACK_AVAILABLE = False


# ========================================================================
# 7. Convenience Functions
# ========================================================================


def create_multi_objective_env(
    base_env,
    weights: dict[str, float] | None = None,
    envelope_morl: bool = False,
    adaptive_weights: bool = False,
    constraints: dict[str, float] | None = None,
) -> "MultiObjectiveEnv":
    """다중 목표 환경을 쉽게 생성하는 팩토리 함수

    Args:
        base_env: BitcoinTradingEnv 인스턴스
        weights: 초기 가중치
        envelope_morl: Envelope MORL 모드
        adaptive_weights: 적응적 가중치 사용 여부
        constraints: 제약 임계값

    Returns:
        MultiObjectiveEnv
    """
    scheduler = None
    if adaptive_weights:
        scheduler = AdaptiveWeightScheduler(
            initial_weights=weights,
            constraints=constraints,
        )

    return MultiObjectiveEnv(
        env=base_env,
        weights=weights,
        envelope_morl=envelope_morl,
        weight_scheduler=scheduler,
        randomize_weights=envelope_morl,  # envelope일 때 자동 랜덤화
    )


def create_training_pipeline(
    train_env,
    eval_env,
    weights: dict[str, float] | None = None,
    envelope_morl: bool = False,
    adaptive_weights: bool = True,
    pareto_max_k: int = 10,
    eval_freq: int = 10000,
) -> tuple:
    """다중 목표 훈련 파이프라인 생성

    Returns:
        (mo_train_env, mo_eval_env, sb3_callback_or_None)
    """
    scheduler = (
        AdaptiveWeightScheduler(initial_weights=weights)
        if adaptive_weights
        else None
    )

    mo_train_env = MultiObjectiveEnv(
        env=train_env,
        weights=weights,
        envelope_morl=envelope_morl,
        weight_scheduler=scheduler,
        randomize_weights=envelope_morl,
    )
    mo_eval_env = MultiObjectiveEnv(
        env=eval_env,
        weights=weights,
        envelope_morl=envelope_morl,
    )

    pareto = ParetoFrontier(max_solutions=pareto_max_k)
    pareto.load()  # 기존 frontier 로드 시도

    mo_callback = MultiObjectiveCallback(
        pareto=pareto,
        scheduler=scheduler,
        eval_env=mo_eval_env,
        eval_freq=eval_freq,
    )

    sb3_callback = None
    if SB3_CALLBACK_AVAILABLE:
        sb3_callback = MultiObjectiveSB3Callback(mo_callback)

    return mo_train_env, mo_eval_env, sb3_callback


# ========================================================================
# 8. Configuration Extension
# ========================================================================


@dataclass
class MultiObjectiveConfig:
    """다중 목표 RL 설정 — config.py 확장"""

    # 목표 가중치
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # 제약
    max_mdd: float = 0.10
    max_trades_per_day: float = 20.0

    # CVaR
    cvar_alpha: float = CVAR_ALPHA
    cvar_window: int = CVAR_WINDOW

    # Pareto
    pareto_max_k: int = 10
    pareto_save_dir: str = "data/rl_models/pareto"

    # Envelope MORL
    envelope_morl: bool = False
    randomize_weights: bool = False

    # Adaptive weights
    adaptive_weights: bool = True
    weight_adjustment_rate: float = 0.1
    weight_update_interval: int = 1000

    # Evaluation
    eval_freq: int = 10000
