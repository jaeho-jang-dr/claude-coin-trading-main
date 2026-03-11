"""Trajectory Buffer — 분산 학습용 경험 데이터 수집/직렬화

RL Worker가 로컬 환경에서 수집한 롤아웃(trajectory)을 구조화하고,
ZeroMQ로 메인 브레인에 전송 가능한 형태로 직렬화한다.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.trajectory")


@dataclass
class Transition:
    """단일 전이 (s, a, r, s', done)"""
    obs: np.ndarray           # 관측 (42d or 42+768d)
    action: np.ndarray        # 행동 (1d)
    reward: float
    next_obs: np.ndarray
    done: bool
    value: float = 0.0        # V(s) - PPO value estimate
    log_prob: float = 0.0     # log π(a|s)
    info: dict = field(default_factory=dict)


class TrajectoryBuffer:
    """롤아웃 단위 경험 버퍼

    한 에피소드 또는 n_steps 분량의 전이를 수집하고,
    PPO 학습에 필요한 형태로 변환한다.
    """

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self.transitions: list[Transition] = []
        self.episode_rewards: list[float] = []
        self._current_episode_reward = 0.0

    def add(self, transition: Transition):
        """전이 추가"""
        self.transitions.append(transition)
        self._current_episode_reward += transition.reward

        if transition.done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        # 크기 제한
        if len(self.transitions) > self.max_size:
            self.transitions = self.transitions[-self.max_size:]

    def is_ready(self, min_steps: int = 2048) -> bool:
        """충분한 데이터가 수집되었는지"""
        return len(self.transitions) >= min_steps

    def get_batch(self, n_steps: int = None) -> dict:
        """PPO 학습용 배치 데이터 추출

        Returns:
            {
                "obs": np.ndarray (N, obs_dim),
                "actions": np.ndarray (N, 1),
                "rewards": np.ndarray (N,),
                "dones": np.ndarray (N,),
                "values": np.ndarray (N,),
                "log_probs": np.ndarray (N,),
                "returns": np.ndarray (N,),   # GAE 계산 후
                "advantages": np.ndarray (N,), # GAE 계산 후
            }
        """
        n = n_steps or len(self.transitions)
        batch = self.transitions[:n]

        obs = np.array([t.obs for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        values = np.array([t.value for t in batch], dtype=np.float32)
        log_probs = np.array([t.log_prob for t in batch], dtype=np.float32)

        # GAE (Generalized Advantage Estimation) 계산
        returns, advantages = self._compute_gae(rewards, values, dones)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "n_steps": n,
            "avg_reward": float(rewards.mean()),
            "episode_rewards": self.episode_rewards.copy(),
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation

        Returns:
            (returns, advantages)
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        returns = advantages + values
        return returns, advantages

    def clear(self):
        """버퍼 초기화"""
        self.transitions.clear()
        self.episode_rewards.clear()
        self._current_episode_reward = 0.0

    def serialize(self) -> dict:
        """ZMQ 전송용 직렬화 (numpy → list)"""
        batch = self.get_batch()
        return {
            "obs": batch["obs"].tolist(),
            "actions": batch["actions"].tolist(),
            "rewards": batch["rewards"].tolist(),
            "dones": batch["dones"].tolist(),
            "values": batch["values"].tolist(),
            "log_probs": batch["log_probs"].tolist(),
            "returns": batch["returns"].tolist(),
            "advantages": batch["advantages"].tolist(),
            "n_steps": batch["n_steps"],
            "avg_reward": batch["avg_reward"],
            "episode_rewards": batch["episode_rewards"],
        }

    @staticmethod
    def deserialize(data: dict) -> dict:
        """수신된 직렬 데이터 → numpy 배치로 복원"""
        return {
            "obs": np.array(data["obs"], dtype=np.float32),
            "actions": np.array(data["actions"], dtype=np.float32),
            "rewards": np.array(data["rewards"], dtype=np.float32),
            "dones": np.array(data["dones"], dtype=np.float32),
            "values": np.array(data["values"], dtype=np.float32),
            "log_probs": np.array(data["log_probs"], dtype=np.float32),
            "returns": np.array(data["returns"], dtype=np.float32),
            "advantages": np.array(data["advantages"], dtype=np.float32),
            "n_steps": data["n_steps"],
            "avg_reward": data["avg_reward"],
            "episode_rewards": data.get("episode_rewards", []),
        }

    def stats(self) -> dict:
        """현재 버퍼 통계"""
        return {
            "buffer_size": len(self.transitions),
            "episodes_completed": len(self.episode_rewards),
            "avg_episode_reward": (
                float(np.mean(self.episode_rewards))
                if self.episode_rewards else 0.0
            ),
        }
