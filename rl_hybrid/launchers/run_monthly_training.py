"""월간 고급 RL 훈련 — Windows 작업 스케줄러용 1회 실행 스크립트

매월 1일 실행. Multi-Agent, Decision Transformer, Offline RL 순차 훈련.

사용법:
    python rl_hybrid/launchers/run_monthly_training.py              # 전체 실행
    python rl_hybrid/launchers/run_monthly_training.py --module dt  # DT만 실행
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
logger = logging.getLogger("monthly_training")


def train_multi_agent() -> dict:
    """Multi-Agent (Scalping+Swing) 훈련"""
    try:
        from rl_hybrid.rl.multi_agent_consensus import MultiAgentTrainer
        trainer = MultiAgentTrainer()
        trainer.train(scalping_days=90, swing_days=180)
        return {"module": "multi_agent", "status": "success"}
    except Exception as e:
        logger.error(f"Multi-Agent 훈련 실패: {e}")
        return {"module": "multi_agent", "status": "failed", "error": str(e)}


def train_decision_transformer() -> dict:
    """Decision Transformer 재훈련"""
    try:
        from rl_hybrid.rl.decision_transformer import train_dt
        result = train_dt(days=180, interval="4h", n_epochs=100)
        return {"module": "decision_transformer", "status": "success", "detail": result}
    except Exception as e:
        logger.error(f"Decision Transformer 훈련 실패: {e}")
        return {"module": "decision_transformer", "status": "failed", "error": str(e)}


def train_offline_rl() -> dict:
    """Offline RL (CQL) 훈련"""
    try:
        from rl_hybrid.rl.offline_rl import train_offline
        result = train_offline(algorithm="cql", epochs=100)
        return {"module": "offline_rl", "status": "success", "detail": result}
    except Exception as e:
        logger.error(f"Offline RL 훈련 실패: {e}")
        return {"module": "offline_rl", "status": "failed", "error": str(e)}


MODULES = {
    "multi_agent": train_multi_agent,
    "dt": train_decision_transformer,
    "offline_rl": train_offline_rl,
}


def main():
    parser = argparse.ArgumentParser(description="월간 고급 RL 훈련 (1회 실행)")
    parser.add_argument(
        "--module",
        choices=list(MODULES.keys()) + ["all"],
        default="all",
        help="훈련할 모듈 (기본: all)",
    )
    args = parser.parse_args()

    start = time.time()
    results = []

    if args.module == "all":
        targets = list(MODULES.items())
    else:
        targets = [(args.module, MODULES[args.module])]

    for name, func in targets:
        logger.info(f"=== {name} 훈련 시작 ===")
        result = func()
        results.append(result)
        logger.info(f"=== {name} 완료: {result['status']} ===")

    elapsed = int(time.time() - start)
    summary = {
        "total_modules": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "elapsed_seconds": elapsed,
        "results": results,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # 텔레그램 알림 (선택)
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        import requests

        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        user_id = os.environ.get("TELEGRAM_USER_ID")
        if bot_token and user_id:
            msg = f"[Monthly RL Training] {summary['success']}/{summary['total_modules']} 성공 ({elapsed}초)"
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": user_id, "text": msg},
                timeout=10,
            )
    except Exception:
        pass

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
