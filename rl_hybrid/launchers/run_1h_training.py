"""1시간 자율 RL 훈련 세션

모든 RL 모듈을 순차 실행하고, 결과를 DB에 기록한다.
시간 초과 시 현재 단계를 완료한 뒤 중단한다.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "logs", "training_1h.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("rl.1h_training")

DEADLINE = None  # 전역 마감 시각


def remaining():
    return max(0, DEADLINE - time.time())


def run_stage(name, func, min_minutes=2):
    """단계 실행 — 시간 부족하면 스킵"""
    if remaining() < min_minutes * 60:
        logger.warning(f"[TIME] [{name}] 시간 부족 ({remaining():.0f}s) -- 스킵")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f">> [{name}] 시작 (남은 시간: {remaining()/60:.1f}분)")
    logger.info(f"{'='*60}")
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        logger.info(f"[OK] [{name}] 완료 ({elapsed:.1f}초)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[FAIL] [{name}] 실패 ({elapsed:.1f}초): {e}")
        return False


# ============================================================
# Stage 1: PPO 증분 학습 (Continuous Learner)
# ============================================================
def stage_continuous_learning():
    from rl_hybrid.rl.continuous_learner import ContinuousLearner
    cl = ContinuousLearner(incremental_steps=30_000)
    cl.force_retrain()


# ============================================================
# Stage 2: SB3 PPO 본격 훈련 (100K 스텝)
# ============================================================
def stage_sb3_ppo_training():
    from rl_hybrid.rl.train import train
    steps = 100_000
    if remaining() < 600:  # 10분 미만이면 축소
        steps = 50_000
    train(
        days=90,
        total_steps=steps,
        balance=10_000_000,
        interval="4h",
        algo="ppo",
    )


# ============================================================
# Stage 3: Weekly Retrain (PPO/SAC/TD3 비교, 축소)
# ============================================================
def stage_weekly_retrain():
    from rl_hybrid.rl.weekly_retrain import weekly_retrain
    steps = 100_000
    if remaining() < 900:  # 15분 미만이면 축소
        steps = 50_000
    weekly_retrain(days=90, total_steps=steps)


# ============================================================
# Stage 4: Offline RL (CQL)
# ============================================================
def stage_offline_rl():
    from rl_hybrid.rl.offline_rl import train_offline
    train_offline(min_data_points=30, epochs=30)


# ============================================================
# Stage 5: Decision Transformer
# ============================================================
def stage_decision_transformer():
    from rl_hybrid.rl.decision_transformer import train_dt
    epochs = 30
    if remaining() < 600:
        epochs = 15
    train_dt(n_epochs=epochs, days=90)


# ============================================================
# Stage 6: Multi-Agent Consensus
# ============================================================
def stage_multi_agent():
    from rl_hybrid.rl.multi_agent_consensus import MultiAgentTrainer
    trainer = MultiAgentTrainer(
        scalping_steps=30_000,
        swing_steps=30_000,
        weight_learner_steps=10_000,
    )
    trainer.train(scalping_days=30, swing_days=90)


# ============================================================
# Stage 7: Self-Tuning 체크
# ============================================================
def stage_self_tuning():
    from rl_hybrid.rl.self_tuning_rl import ParameterTuner
    tuner = ParameterTuner()
    tuner.check_rollback(
        current_metrics={
            "sharpe_ratio": 0.0,
            "total_return_pct": 0.0,
        },
    )


# ============================================================
# Main
# ============================================================
def main():
    global DEADLINE

    duration_minutes = 60
    if len(sys.argv) > 1:
        try:
            duration_minutes = int(sys.argv[1])
        except ValueError:
            pass

    DEADLINE = time.time() + duration_minutes * 60
    logger.info(f"[START] 자율 RL 훈련 시작 -- {duration_minutes}분 세션")
    logger.info(f"   마감: {time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")

    stages = [
        ("1. PPO 증분학습 (Continuous)", stage_continuous_learning, 2),
        ("2. SB3 PPO 본격훈련 (100K)", stage_sb3_ppo_training, 3),
        ("3. Weekly Retrain (PPO/SAC/TD3)", stage_weekly_retrain, 5),
        ("4. Offline RL (CQL)", stage_offline_rl, 2),
        ("5. Decision Transformer", stage_decision_transformer, 2),
        ("6. Multi-Agent Consensus", stage_multi_agent, 3),
        ("7. Self-Tuning 체크", stage_self_tuning, 1),
    ]

    results = {}
    total_start = time.time()

    for name, func, min_min in stages:
        success = run_stage(name, func, min_min)
        results[name] = "[OK]" if success else "[FAIL]"

    # 남은 시간이 5분 이상이면 추가 PPO 라운드
    round_num = 2
    while remaining() > 300:
        name = f"8-{round_num}. PPO 추가 라운드"
        success = run_stage(name, stage_continuous_learning, 2)
        results[name] = "[OK]" if success else "[FAIL]"
        round_num += 1

    total_elapsed = time.time() - total_start

    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info(f"[END] 자율 훈련 완료 -- 총 {total_elapsed/60:.1f}분")
    logger.info(f"{'='*60}")
    for name, status in results.items():
        logger.info(f"  {status} {name}")

    # DB에서 이번 세션 결과 조회
    try:
        from rl_hybrid.rl.rl_db_logger import get_recent_training_cycles
        cycles = get_recent_training_cycles(limit=20)
        logger.info(f"\n[DB] DB 기록된 훈련 사이클: {len(cycles)}건")
        for c in cycles:
            logger.info(
                f"  [{c['algorithm']}/{c['module']}] {c['status']} | "
                f"sharpe={c.get('avg_sharpe', 'N/A')} | "
                f"return={c.get('avg_return_pct', 'N/A')}"
            )
    except Exception:
        pass

    # 텔레그램 알림
    try:
        from scripts.notify_telegram import send_message
        msg = f"*RL 자율 훈련 완료*\n\n"
        msg += f"소요: {total_elapsed/60:.1f}분\n"
        for name, status in results.items():
            msg += f"{status} {name}\n"
        send_message(msg)
    except Exception:
        pass


if __name__ == "__main__":
    main()
