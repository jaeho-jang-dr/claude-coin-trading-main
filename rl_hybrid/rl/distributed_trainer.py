"""분산 PPO 트레이너 — 메인 브레인에서 글로벌 가중치 업데이트

RL Worker들로부터 trajectory를 수집하고, 글로벌 PPO 모델을 업데이트한 뒤,
새 가중치를 워커들에게 배포한다.

아키텍처:
    RL Worker 1 ─┐
    RL Worker 2 ─┤── trajectory ──→ DistributedTrainer ──→ updated weights
    RL Worker N ─┘     (ZMQ)         (Main Brain)            (ZMQ broadcast)
"""

import logging
import os
import time
import threading
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.distributed_trainer")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 미설치 -- 분산 RL 훈련 비활성화")

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models",
)


class ActorCritic(nn.Module if TORCH_AVAILABLE else object):
    """PPO Actor-Critic 네트워크 (직접 구현)

    SB3 의존 없이 가중치 전송이 용이한 순수 PyTorch 구현.
    분산 환경에서 가중치 직렬화/역직렬화를 최적화한다.
    """

    def __init__(self, obs_dim: int = 42, action_dim: int = 1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 공유 특성 추출기
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Actor (정책) — 평균 + 로그 표준편차
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # [-1, 1] 범위
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        value = self.critic(features)
        return action_mean, action_std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """관측 → (action, log_prob, value)"""
        mean, std, value = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """배치 평가 — (log_probs, values, entropy)"""
        mean, std, values = self.forward(obs)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy


class DistributedTrainer:
    """분산 PPO 글로벌 트레이너

    메인 브레인에서 실행되며:
    1. 워커들의 trajectory를 수집
    2. 글로벌 모델을 PPO 업데이트
    3. 새 가중치를 직렬화하여 반환 (배포용)
    """

    def __init__(
        self,
        obs_dim: int = 42,
        action_dim: int = 1,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        min_trajectories_for_update: int = 2,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다: pip install torch")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.min_trajectories = min_trajectories_for_update

        # 글로벌 모델
        self.model = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        # Trajectory 수집 큐
        self.trajectory_queue: deque[dict] = deque(maxlen=32)
        self._lock = threading.Lock()

        # 학습 통계
        self.update_count = 0
        self.total_steps_collected = 0
        self.training_stats: list[dict] = []

        logger.info(
            f"DistributedTrainer 초기화: obs={obs_dim}, action={action_dim}, "
            f"params={sum(p.numel() for p in self.model.parameters()):,}"
        )

    def receive_trajectory(self, trajectory_data: dict, worker_id: str = ""):
        """워커로부터 trajectory 수신

        Args:
            trajectory_data: TrajectoryBuffer.serialize() 출력
            worker_id: 워커 식별자
        """
        with self._lock:
            trajectory_data["worker_id"] = worker_id
            trajectory_data["received_at"] = time.time()
            self.trajectory_queue.append(trajectory_data)
            self.total_steps_collected += trajectory_data["n_steps"]

        logger.info(
            f"Trajectory 수신: worker={worker_id}, "
            f"steps={trajectory_data['n_steps']}, "
            f"avg_reward={trajectory_data['avg_reward']:.4f}, "
            f"queue_size={len(self.trajectory_queue)}"
        )

    def should_update(self) -> bool:
        """업데이트 조건 충족 여부"""
        return len(self.trajectory_queue) >= self.min_trajectories

    def update(self) -> Optional[dict]:
        """글로벌 PPO 업데이트 실행

        수집된 trajectory들을 병합하여 PPO 업데이트를 수행한다.

        Returns:
            학습 통계 dict, trajectory 부족 시 None
        """
        if not self.should_update():
            return None

        with self._lock:
            trajectories = list(self.trajectory_queue)
            self.trajectory_queue.clear()

        # trajectory 병합
        merged = self._merge_trajectories(trajectories)
        if merged["obs"].shape[0] < self.batch_size:
            logger.warning("데이터 부족 -- 업데이트 스킵")
            return None

        # PPO 업데이트
        stats = self._ppo_update(merged)
        self.update_count += 1
        self.training_stats.append(stats)

        # 주기적 저장
        if self.update_count % 10 == 0:
            self.save_model()

        logger.info(
            f"글로벌 업데이트 #{self.update_count}: "
            f"policy_loss={stats['policy_loss']:.4f}, "
            f"value_loss={stats['value_loss']:.4f}, "
            f"entropy={stats['entropy']:.4f}, "
            f"total_steps={self.total_steps_collected}"
        )

        return stats

    def _merge_trajectories(self, trajectories: list[dict]) -> dict:
        """여러 워커의 trajectory를 하나로 병합"""
        from rl_hybrid.rl.trajectory import TrajectoryBuffer

        all_obs, all_actions, all_rewards = [], [], []
        all_dones, all_values, all_log_probs = [], [], []
        all_returns, all_advantages = [], []

        for traj_data in trajectories:
            batch = TrajectoryBuffer.deserialize(traj_data)
            all_obs.append(batch["obs"])
            all_actions.append(batch["actions"])
            all_rewards.append(batch["rewards"])
            all_dones.append(batch["dones"])
            all_values.append(batch["values"])
            all_log_probs.append(batch["log_probs"])
            all_returns.append(batch["returns"])
            all_advantages.append(batch["advantages"])

        merged = {
            "obs": np.concatenate(all_obs),
            "actions": np.concatenate(all_actions),
            "rewards": np.concatenate(all_rewards),
            "dones": np.concatenate(all_dones),
            "values": np.concatenate(all_values),
            "log_probs": np.concatenate(all_log_probs),
            "returns": np.concatenate(all_returns),
            "advantages": np.concatenate(all_advantages),
        }

        # Advantage 정규화
        adv = merged["advantages"]
        merged["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        logger.info(
            f"Trajectory 병합: {len(trajectories)}개 워커, "
            f"총 {merged['obs'].shape[0]} 스텝"
        )
        return merged

    def _ppo_update(self, batch: dict) -> dict:
        """PPO Clipped Surrogate 업데이트"""
        obs = torch.FloatTensor(batch["obs"])
        actions = torch.FloatTensor(batch["actions"])
        old_log_probs = torch.FloatTensor(batch["log_probs"])
        returns = torch.FloatTensor(batch["returns"])
        advantages = torch.FloatTensor(batch["advantages"])

        n_samples = obs.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # 미니배치 셔플
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # 현재 정책 평가
                new_log_probs, values, entropy = self.model.evaluate(mb_obs, mb_actions)

                # 정책 손실 (Clipped Surrogate)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # 엔트로피 보너스
                entropy_loss = -entropy.mean()

                # 총 손실
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_epochs": self.n_epochs,
            "n_samples": n_samples,
            "n_updates": n_updates,
            "update_count": self.update_count,
            "total_steps": self.total_steps_collected,
        }

    def get_weights_serializable(self) -> dict:
        """ZMQ 전송용 가중치 직렬화

        Returns:
            {param_name: list[list[float]]}
        """
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().tolist()
        return weights

    def load_weights_from_dict(self, weights: dict):
        """직렬화된 가중치를 모델에 적용"""
        state_dict = {}
        for name, values in weights.items():
            state_dict[name] = torch.FloatTensor(np.array(values))
        self.model.load_state_dict(state_dict)
        logger.info(f"가중치 로드 완료: {len(weights)}개 파라미터")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[float, float, float]:
        """관측 → (action, log_prob, value) 추론

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = self.model.get_action(obs_t, deterministic)
            return (
                float(action.squeeze().item()),
                float(log_prob.squeeze().item()),
                float(value.squeeze().item()),
            )

    def save_model(self, path: str = None):
        """모델 저장"""
        path = path or os.path.join(MODEL_DIR, "distributed_ppo_global.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "total_steps": self.total_steps_collected,
        }, path)
        logger.info(f"글로벌 모델 저장: {path}")

    def load_model(self, path: str = None):
        """모델 로드"""
        path = path or os.path.join(MODEL_DIR, "distributed_ppo_global.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False)
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    logger.warning(f"모델 차원 불일치 -- 새 모델로 시작: {e}")
                    return
                raise
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.update_count = checkpoint.get("update_count", 0)
            self.total_steps_collected = checkpoint.get("total_steps", 0)
            logger.info(f"글로벌 모델 로드: {path} (update #{self.update_count})")

    def get_stats(self) -> dict:
        """훈련 통계 요약"""
        recent = self.training_stats[-10:] if self.training_stats else []
        return {
            "update_count": self.update_count,
            "total_steps": self.total_steps_collected,
            "queue_size": len(self.trajectory_queue),
            "recent_policy_loss": float(np.mean([s["policy_loss"] for s in recent])) if recent else 0,
            "recent_value_loss": float(np.mean([s["value_loss"] for s in recent])) if recent else 0,
            "recent_entropy": float(np.mean([s["entropy"] for s in recent])) if recent else 0,
        }
