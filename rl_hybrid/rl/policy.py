"""RL 정책 래퍼 — PPO / SAC / TD3 모델 훈련/추론/저장

Phase 2: 단일 머신 훈련
Phase 3: 분산 훈련 (메인 노드에서 글로벌 가중치 업데이트)
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.policy")

# SB3 lazy import (설치 안 되어있을 수 있음)
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 미설치 — RL 훈련 비활성화")


# 모델 저장 경로
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models",
)


class TradingMetricsCallback(BaseCallback if SB3_AVAILABLE else object):
    """훈련 중 트레이딩 메트릭 로깅 콜백"""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_returns = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        # 에피소드 종료 시 통계 수집
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])

            if info.get("return_pct") is not None and info.get("_final_info", False):
                self.episode_returns.append(info["return_pct"])
                self.episode_trades.append(info.get("trade_count", 0))

        if self.num_timesteps % self.log_freq == 0 and self.episode_returns:
            avg_return = np.mean(self.episode_returns[-10:])
            avg_trades = np.mean(self.episode_trades[-10:]) if self.episode_trades else 0
            logger.info(
                f"[Step {self.num_timesteps}] "
                f"avg_return={avg_return:.2f}% | "
                f"avg_trades={avg_trades:.1f}"
            )

        return True


class PPOTrader:
    """PPO 기반 트레이딩 정책"""

    def __init__(self, env=None, model_path: str = None):
        """
        Args:
            env: Gymnasium 환경 (훈련 시)
            model_path: 기존 모델 로드 경로 (추론 시)
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3가 필요합니다: pip install stable-baselines3"
            )

        self.env = env
        self.model: Optional[PPO] = None
        self.model_path = model_path or os.path.join(MODEL_DIR, "ppo_btc_latest")

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        """새 PPO 모델 생성"""
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
            ent_coef=0.01,       # 탐색 장려
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=None,
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 128, 64],
                    "vf": [256, 128, 64],
                },
            },
        )
        logger.info("PPO 모델 생성 완료")

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_env=None,
        save_freq: int = 10_000,
        callbacks: list = None,
    ):
        """모델 훈련

        Args:
            total_timesteps: 총 훈련 스텝
            eval_env: 평가용 환경 (별도 데이터)
            save_freq: 모델 저장 주기 (스텝)
            callbacks: 추가 SB3 콜백 리스트
        """
        os.makedirs(MODEL_DIR, exist_ok=True)

        cb_list = [TradingMetricsCallback(log_freq=2000)]
        if callbacks:
            cb_list.extend(callbacks)
        callbacks = cb_list

        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(MODEL_DIR, "best"),
                eval_freq=save_freq,
                n_eval_episodes=5,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        logger.info(f"PPO 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        self.save()
        logger.info("PPO 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        """관측 → 행동 추론

        Args:
            obs: 정규화된 관측 벡터
            deterministic: True면 가장 확률 높은 행동 선택

        Returns:
            action 값 (-1 ~ 1)
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(action[0])

    def save(self, path: str = None):
        """모델 저장"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"모델 저장: {path}")

    def load(self, path: str = None):
        """모델 로드"""
        path = path or self.model_path
        self.model = PPO.load(path, env=self.env)
        logger.info(f"모델 로드: {path}")

    def get_weights(self) -> dict:
        """모델 가중치 추출 (분산 학습용)

        Returns:
            {param_name: numpy_array} 딕셔너리
        """
        params = {}
        for name, param in self.model.policy.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params

    def set_weights(self, weights: dict):
        """외부 가중치 적용 (분산 학습용)"""
        import torch
        for name, param in self.model.policy.named_parameters():
            if name in weights:
                param.data = torch.tensor(weights[name], dtype=param.dtype)
        logger.info(f"가중치 업데이트: {len(weights)}개 파라미터")


class SACTrader:
    """SAC (Soft Actor-Critic) 기반 트레이딩 정책

    오프폴리시 알고리즘으로, 엔트로피 최대화를 통해 탐색과 활용의 균형을 자동 조정한다.
    연속 행동 공간 Box(-1, 1, (1,))에 적합하며, PPO 대비 샘플 효율이 높다.
    """

    def __init__(self, env=None, model_path: str = None):
        """
        Args:
            env: Gymnasium 환경 (훈련 시)
            model_path: 기존 모델 로드 경로 (추론 시)
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3가 필요합니다: pip install stable-baselines3"
            )

        self.env = env
        self.model: Optional[SAC] = None
        self.model_path = model_path or os.path.join(MODEL_DIR, "sac_btc_latest")

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        """새 SAC 모델 생성"""
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",        # 엔트로피 계수 자동 조정
            learning_starts=1000,
            verbose=1,
            tensorboard_log=None,
            policy_kwargs={
                "net_arch": [256, 128, 64],
            },
        )
        logger.info("SAC 모델 생성 완료")

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_env=None,
        save_freq: int = 10_000,
        callbacks: list = None,
    ):
        """모델 훈련

        Args:
            total_timesteps: 총 훈련 스텝
            eval_env: 평가용 환경 (별도 데이터)
            save_freq: 모델 저장 주기 (스텝)
            callbacks: 추가 SB3 콜백 리스트
        """
        os.makedirs(MODEL_DIR, exist_ok=True)

        cb_list = [TradingMetricsCallback(log_freq=2000)]
        if callbacks:
            cb_list.extend(callbacks)
        callbacks = cb_list

        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(MODEL_DIR, "best_sac"),
                eval_freq=save_freq,
                n_eval_episodes=5,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        logger.info(f"SAC 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        self.save()
        logger.info("SAC 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        """관측 → 행동 추론

        Args:
            obs: 정규화된 관측 벡터
            deterministic: True면 평균 행동 선택 (탐색 노이즈 없음)

        Returns:
            action 값 (-1 ~ 1)
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(action[0])

    def save(self, path: str = None):
        """모델 저장"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"모델 저장: {path}")

    def load(self, path: str = None):
        """모델 로드"""
        path = path or self.model_path
        self.model = SAC.load(path, env=self.env)
        logger.info(f"모델 로드: {path}")

    def get_weights(self) -> dict:
        """모델 가중치 추출 (분산 학습용)"""
        params = {}
        for name, param in self.model.policy.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params

    def set_weights(self, weights: dict):
        """외부 가중치 적용 (분산 학습용)"""
        import torch
        for name, param in self.model.policy.named_parameters():
            if name in weights:
                param.data = torch.tensor(weights[name], dtype=param.dtype)
        logger.info(f"가중치 업데이트: {len(weights)}개 파라미터")


class TD3Trader:
    """TD3 (Twin Delayed DDPG) 기반 트레이딩 정책

    DDPG의 개선판으로, 이중 Q-네트워크와 지연된 정책 업데이트로 과추정을 방지한다.
    연속 행동 공간 Box(-1, 1, (1,))에 적합하며, 결정론적 정책으로 안정적인 추론이 가능하다.
    """

    def __init__(self, env=None, model_path: str = None):
        """
        Args:
            env: Gymnasium 환경 (훈련 시)
            model_path: 기존 모델 로드 경로 (추론 시)
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3가 필요합니다: pip install stable-baselines3"
            )

        self.env = env
        self.model: Optional[TD3] = None
        self.model_path = model_path or os.path.join(MODEL_DIR, "td3_btc_latest")

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)
        elif env:
            self._create_model()

    def _create_model(self):
        """새 TD3 모델 생성"""
        # 행동 공간 차원에 맞는 탐색 노이즈
        n_actions = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else 1
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )

        self.model = TD3(
            "MlpPolicy",
            self.env,
            learning_rate=1e-3,
            buffer_size=100_000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            policy_delay=2,          # 정책 업데이트 지연 (TD3 핵심)
            action_noise=action_noise,
            learning_starts=1000,
            verbose=1,
            tensorboard_log=None,
            policy_kwargs={
                "net_arch": [256, 128, 64],
            },
        )
        logger.info("TD3 모델 생성 완료")

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_env=None,
        save_freq: int = 10_000,
        callbacks: list = None,
    ):
        """모델 훈련

        Args:
            total_timesteps: 총 훈련 스텝
            eval_env: 평가용 환경 (별도 데이터)
            save_freq: 모델 저장 주기 (스텝)
            callbacks: 추가 SB3 콜백 리스트
        """
        os.makedirs(MODEL_DIR, exist_ok=True)

        cb_list = [TradingMetricsCallback(log_freq=2000)]
        if callbacks:
            cb_list.extend(callbacks)
        callbacks = cb_list

        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(MODEL_DIR, "best_td3"),
                eval_freq=save_freq,
                n_eval_episodes=5,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        logger.info(f"TD3 훈련 시작: {total_timesteps} 스텝")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        self.save()
        logger.info("TD3 훈련 완료")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        """관측 → 행동 추론

        Args:
            obs: 정규화된 관측 벡터
            deterministic: True면 노이즈 없이 행동 선택

        Returns:
            action 값 (-1 ~ 1)
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(action[0])

    def save(self, path: str = None):
        """모델 저장"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"모델 저장: {path}")

    def load(self, path: str = None):
        """모델 로드"""
        path = path or self.model_path
        self.model = TD3.load(path, env=self.env)
        logger.info(f"모델 로드: {path}")

    def get_weights(self) -> dict:
        """모델 가중치 추출 (분산 학습용)"""
        params = {}
        for name, param in self.model.policy.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params

    def set_weights(self, weights: dict):
        """외부 가중치 적용 (분산 학습용)"""
        import torch
        for name, param in self.model.policy.named_parameters():
            if name in weights:
                param.data = torch.tensor(weights[name], dtype=param.dtype)
        logger.info(f"가중치 업데이트: {len(weights)}개 파라미터")
