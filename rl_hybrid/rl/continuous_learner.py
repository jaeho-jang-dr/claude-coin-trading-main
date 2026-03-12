"""지속 학습 엔진 — 백그라운드 주기적 재학습

시스템 런칭 후에도 실시간 매매 로그와 새로운 데이터를 바탕으로
모델이 주기적으로 Fine-Tuning 되는 루프.

학습 주기:
  1. Supabase에서 최근 매매 결과 수집 (DataCollector)
  2. 최근 Upbit 캔들로 환경 갱신 (HistoricalDataLoader)
  3. PPO 증분 학습 (DistributedTrainer)
  4. 평가 → 성능 향상 시 모델 등록 (ModelRegistry)
  5. 성능 저하 시 자동 롤백
"""

import logging
import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.config import config, SystemConfig

logger = logging.getLogger("rl.continuous_learner")

try:
    import torch
    from rl_hybrid.rl.distributed_trainer import DistributedTrainer
    from rl_hybrid.rl.environment import BitcoinTradingEnv
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    from rl_hybrid.rl.data_collector import LiveDataCollector
    from rl_hybrid.rl.model_registry import ModelRegistry
    from rl_hybrid.rl.trajectory import TrajectoryBuffer, Transition
    AVAILABLE = True
except ImportError as e:
    AVAILABLE = False
    logger.warning(f"ContinuousLearner 의존성 부족: {e}")


class ContinuousLearner:
    """지속 학습 엔진"""

    def __init__(
        self,
        trainer: DistributedTrainer = None,
        retrain_interval_hours: float = 6,
        min_new_decisions: int = 5,
        eval_episodes: int = 5,
        incremental_steps: int = 20_000,
        data_days: int = 30,
    ):
        """
        Args:
            trainer: 글로벌 트레이너 (Main Brain에서 공유)
            retrain_interval_hours: 재학습 주기 (시간)
            min_new_decisions: 최소 새 매매 데이터 수 (이 이하면 스킵)
            eval_episodes: 평가 에피소드 수
            incremental_steps: 증분 학습 스텝 수
            data_days: 훈련 데이터 기간 (일)
        """
        if not AVAILABLE:
            raise ImportError("PyTorch 및 의존 모듈이 필요합니다")

        # MORL 활성화 시 obs_dim = 42 + 5(weight vector) = 47
        self._sys_config = SystemConfig()
        obs_dim = 42
        if self._sys_config.multi_objective_rl.envelope_morl or self._sys_config.multi_objective_rl.adaptive_weights:
            obs_dim = 47
        self.trainer = trainer or DistributedTrainer(obs_dim=obs_dim)
        self.registry = ModelRegistry()
        self.collector = LiveDataCollector()
        self.loader = HistoricalDataLoader()

        self.retrain_interval = retrain_interval_hours * 3600
        self.min_new_decisions = min_new_decisions
        self.eval_episodes = eval_episodes
        self.incremental_steps = incremental_steps
        self.data_days = data_days

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_retrain = 0
        self._last_decision_count = 0

        # 기존 모델 로드
        self.trainer.load_model()

    def start_background(self):
        """백그라운드 학습 루프 시작"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"지속 학습 시작: interval={self.retrain_interval/3600:.1f}h, "
            f"steps={self.incremental_steps}"
        )

    def stop(self):
        """학습 루프 중지"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("지속 학습 중지")

    def _learning_loop(self):
        """주기적 재학습 루프"""
        while self._running:
            try:
                now = time.time()
                if now - self._last_retrain >= self.retrain_interval:
                    self._retrain_cycle()
                    self._last_retrain = now
            except Exception as e:
                logger.error(f"학습 루프 에러: {e}", exc_info=True)

            time.sleep(60)  # 1분마다 체크

    def _retrain_cycle(self):
        """재학습 사이클 실행"""
        logger.info("=== 지속 학습 사이클 시작 ===")
        start = time.time()

        # DB 로깅: 훈련 시작
        cycle_id = None
        try:
            from rl_hybrid.rl.rl_db_logger import log_training_start
            cycle_id = log_training_start(
                cycle_type="continuous",
                algorithm="ppo",
                module="continuous_learner",
                training_steps=self.incremental_steps,
                data_days=self.data_days,
                obs_dim=self.trainer.model.obs_dim if hasattr(self.trainer.model, 'obs_dim') else 42,
                morl_enabled=self._sys_config.multi_objective_rl.envelope_morl,
            )
        except Exception as e:
            logger.debug(f"DB 훈련 시작 기록 실패: {e}")

        cfg = SystemConfig()

        # 0. Self-Tuning 롤백 체크
        if cfg.self_tuning.enabled:
            try:
                from rl_hybrid.rl.self_tuning_rl import ParameterTuner
                tuner = ParameterTuner()
                tuner.check_rollback(
                    current_metrics={},
                    threshold_sharpe_drop=cfg.self_tuning.rollback_sharpe_drop,
                )
                logger.debug("Self-tuning 롤백 체크 완료")
            except Exception as e:
                logger.debug(f"Self-tuning 롤백 체크: {e}")

        # 1. 새 데이터 확인
        stats = self.collector.get_training_stats(days=7)
        if not stats.get("available"):
            logger.info("학습 데이터 없음 -- 스킵")
            return

        new_count = stats["count"]
        if (new_count - self._last_decision_count) < self.min_new_decisions:
            logger.info(f"새 데이터 부족: {new_count - self._last_decision_count}건 -- 스킵")
            if cycle_id:
                try:
                    from rl_hybrid.rl.rl_db_logger import log_training_complete
                    log_training_complete(cycle_id=cycle_id, status="skipped",
                                         error_message="데이터 부족", elapsed_seconds=time.time() - start)
                except Exception:
                    pass
            return

        # 2. 현재 모델 평가 (베이스라인)
        baseline_stats = self._evaluate_current_model()
        logger.info(
            f"베이스라인: return={baseline_stats['avg_return']:.2f}%, "
            f"sharpe={baseline_stats['avg_sharpe']:.3f}"
        )

        # 3. 최신 캔들 데이터로 환경 갱신
        candles = self._load_fresh_data()
        if not candles:
            logger.warning("최신 캔들 로드 실패")
            if cycle_id:
                try:
                    from rl_hybrid.rl.rl_db_logger import log_training_complete
                    log_training_complete(cycle_id=cycle_id, status="failed",
                                         error_message="캔들 데이터 로드 실패", elapsed_seconds=time.time() - start)
                except Exception:
                    pass
            return

        # 4. 증분 학습
        self._incremental_train(candles)

        # 5. 학습 후 평가
        new_stats = self._evaluate_current_model()
        logger.info(
            f"학습 후: return={new_stats['avg_return']:.2f}%, "
            f"sharpe={new_stats['avg_sharpe']:.3f}"
        )

        # 6. 성능 비교 + 모델 등록/롤백
        improved = new_stats["avg_sharpe"] > baseline_stats["avg_sharpe"] - 0.1

        if improved:
            # 모델 저장 + 레지스트리 등록
            model_path = os.path.join(
                config.project_root, "data", "rl_models", "distributed_ppo_global.pt"
            )
            self.trainer.save_model(model_path)

            version_id = self.registry.register_model(
                model_path=model_path,
                metrics={
                    "sharpe_ratio": new_stats["avg_sharpe"],
                    "total_return_pct": new_stats["avg_return"],
                    "max_drawdown": new_stats["avg_mdd"],
                    "eval_episodes": self.eval_episodes,
                },
                training_config={
                    "algorithm": "ppo",
                    "incremental_steps": self.incremental_steps,
                    "data_days": self.data_days,
                    "live_decisions_used": new_count,
                },
                notes=f"지속 학습 사이클 (live {new_count}건)",
            )
            logger.info(f"모델 등록: {version_id}")
        else:
            # 성능 저하 → 롤백
            logger.warning(
                f"성능 저하 감지: {baseline_stats['avg_sharpe']:.3f} → {new_stats['avg_sharpe']:.3f}"
            )
            rollback_version = self.registry.rollback()
            if rollback_version:
                model_path = self.registry.get_model_path(rollback_version)
                if model_path and os.path.exists(model_path):
                    self.trainer.load_model(model_path)
                    logger.info(f"롤백 완료: → {rollback_version}")

        self._last_decision_count = new_count

        # Multi-Agent Weight Learner 증분 업데이트
        if cfg.multi_agent.enabled:
            try:
                from rl_hybrid.rl.multi_agent_consensus import MultiAgentTrainer
                trainer = MultiAgentTrainer(
                    weight_learner_steps=10_000,
                )
                trainer._train_phase2(swing_days=30)
                logger.info("Weight learner 증분 업데이트 완료")
            except Exception as e:
                logger.warning(f"Weight learner 업데이트 스킵: {e}")

        elapsed = time.time() - start

        # DB 로깅: 훈련 완료
        if cycle_id:
            try:
                from rl_hybrid.rl.rl_db_logger import log_training_complete
                log_training_complete(
                    cycle_id=cycle_id,
                    avg_return_pct=new_stats.get("avg_return"),
                    avg_sharpe=new_stats.get("avg_sharpe"),
                    avg_mdd=new_stats.get("avg_mdd"),
                    baseline_sharpe=baseline_stats.get("avg_sharpe"),
                    improved=improved,
                    elapsed_seconds=elapsed,
                    status="completed",
                )
            except Exception as e:
                logger.debug(f"DB 훈련 완료 기록 실패: {e}")

        logger.info(f"=== 지속 학습 완료: {elapsed:.1f}초 ===")

    def _load_fresh_data(self) -> list[dict]:
        """최신 캔들 데이터 로드 + 지표 계산"""
        try:
            raw = self.loader.load_candles(days=self.data_days, interval="1h")
            return self.loader.compute_indicators(raw)
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []

    def _incremental_train(self, candles: list[dict]):
        """증분 PPO 학습

        최신 시장 데이터로 환경을 생성하고,
        n_steps만큼 롤아웃하여 글로벌 모델을 업데이트한다.
        Multi-Objective RL이 활성화되면 환경을 MORL 래퍼로 감싼다.
        """
        env = BitcoinTradingEnv(candles=candles, initial_balance=10_000_000)

        # Multi-Objective 환경 래핑
        cfg = self._sys_config
        if cfg.multi_objective_rl.envelope_morl or cfg.multi_objective_rl.adaptive_weights:
            try:
                from rl_hybrid.rl.multi_objective_reward import MultiObjectiveEnv
                env = MultiObjectiveEnv(
                    env,
                    envelope_morl=cfg.multi_objective_rl.envelope_morl,
                )
                logger.info("MORL 환경 래퍼 적용")
            except Exception as e:
                logger.warning(f"MORL 환경 래핑 실패 (기본 환경 사용): {e}")
        buffer = TrajectoryBuffer(max_size=self.incremental_steps * 2)

        obs, info = env.reset()
        steps_done = 0

        self.trainer.model.eval()

        with torch.no_grad():
            while steps_done < self.incremental_steps:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                action, log_prob, value = self.trainer.model.get_action(obs_t)
                action_np = float(action.squeeze().item())
                action_clipped = np.clip(action_np, -1.0, 1.0)

                next_obs, reward, terminated, truncated, step_info = env.step(
                    np.array([action_clipped])
                )
                done = terminated or truncated

                buffer.add(Transition(
                    obs=obs.copy(),
                    action=np.array([action_clipped]),
                    reward=reward,
                    next_obs=next_obs.copy(),
                    done=done,
                    value=float(value.squeeze().item()),
                    log_prob=float(log_prob.squeeze().item()),
                ))

                obs = next_obs
                steps_done += 1

                if done:
                    obs, info = env.reset()

        # 수집된 데이터로 업데이트
        if buffer.is_ready(min(self.incremental_steps, 64)):
            self.trainer.receive_trajectory(
                buffer.serialize(), worker_id="continuous_learner"
            )
            stats = self.trainer.update()
            if stats:
                logger.info(
                    f"증분 학습: {steps_done}스텝, "
                    f"policy={stats['policy_loss']:.4f}, "
                    f"value={stats['value_loss']:.4f}"
                )

    def _evaluate_current_model(self) -> dict:
        """현재 모델을 최신 데이터로 평가"""
        candles = self._load_fresh_data()
        if not candles:
            return {"avg_return": 0, "avg_sharpe": 0, "avg_mdd": 0}

        # 평가용 데이터: 마지막 20%
        split = int(len(candles) * 0.8)
        eval_candles = candles[split:]

        if len(eval_candles) < 50:
            return {"avg_return": 0, "avg_sharpe": 0, "avg_mdd": 0}

        returns, sharpes, mdds = [], [], []

        for ep in range(self.eval_episodes):
            env = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=10_000_000
            )
            # MORL 환경 래핑 (학습 환경과 동일하게)
            cfg = self._sys_config
            if cfg.multi_objective_rl.envelope_morl or cfg.multi_objective_rl.adaptive_weights:
                try:
                    from rl_hybrid.rl.multi_objective_reward import MultiObjectiveEnv
                    env = MultiObjectiveEnv(env, envelope_morl=cfg.multi_objective_rl.envelope_morl)
                except Exception:
                    pass
            obs, _ = env.reset(seed=ep)
            done = False

            while not done:
                action, _, _ = self.trainer.predict(obs)
                obs, reward, terminated, truncated, info = env.step(
                    np.array([action])
                )
                done = terminated or truncated

            stats = env.get_episode_stats()
            returns.append(stats["total_return_pct"])
            sharpes.append(stats["sharpe_ratio"])
            mdds.append(stats["max_drawdown"])

        return {
            "avg_return": float(np.mean(returns)),
            "avg_sharpe": float(np.mean(sharpes)),
            "avg_mdd": float(np.mean(mdds)),
        }

    def force_retrain(self):
        """수동 재학습 트리거"""
        self._retrain_cycle()

    def get_status(self) -> dict:
        """학습 상태 조회"""
        current = self.registry.get_current_version()
        return {
            "running": self._running,
            "last_retrain": datetime.fromtimestamp(self._last_retrain).isoformat()
                if self._last_retrain > 0 else "never",
            "next_retrain_in": max(0, self.retrain_interval - (time.time() - self._last_retrain)),
            "current_model": current["version_id"] if current else None,
            "total_versions": len(self.registry.registry.get("versions", [])),
            "trainer_stats": self.trainer.get_stats(),
        }
