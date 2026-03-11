"""지속 학습 단독 실행 (Main Brain 없이 독립 운영 가능)

사용법:
    python start_continuous_learning.py                    # 6시간 간격 자동 학습
    python start_continuous_learning.py --interval 2       # 2시간 간격
    python start_continuous_learning.py --force             # 즉시 1회 학습
    python start_continuous_learning.py --status            # 현재 상태 조회
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("continuous_learning")


def main():
    parser = argparse.ArgumentParser(description="RL 지속 학습 엔진")
    parser.add_argument("--interval", type=float, default=6, help="재학습 간격(시간)")
    parser.add_argument("--steps", type=int, default=20000, help="증분 학습 스텝")
    parser.add_argument("--days", type=int, default=30, help="훈련 데이터 기간(일)")
    parser.add_argument("--force", action="store_true", help="즉시 1회 학습 실행")
    parser.add_argument("--status", action="store_true", help="현재 상태만 조회")
    args = parser.parse_args()

    from rl_hybrid.rl.continuous_learner import ContinuousLearner
    from rl_hybrid.rl.model_registry import ModelRegistry

    if args.status:
        registry = ModelRegistry()
        versions = registry.list_versions()
        print(json.dumps({
            "current": registry.get_current_version(),
            "versions": versions,
            "total": len(versions),
        }, indent=2, ensure_ascii=False, default=str))
        return

    learner = ContinuousLearner(
        retrain_interval_hours=args.interval,
        incremental_steps=args.steps,
        data_days=args.days,
    )

    if args.force:
        logger.info("즉시 재학습 실행")
        learner.force_retrain()
        print(json.dumps(learner.get_status(), indent=2, ensure_ascii=False, default=str))
        return

    try:
        learner.start_background()
        logger.info(f"지속 학습 루프 시작 (interval={args.interval}h)")

        while True:
            time.sleep(60)
            status = learner.get_status()
            next_in = status.get("next_retrain_in", 0)
            if next_in > 0:
                logger.debug(f"다음 재학습: {next_in/3600:.1f}시간 후")
    except KeyboardInterrupt:
        learner.stop()
        logger.info("종료")


if __name__ == "__main__":
    main()
