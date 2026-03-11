"""주간 자동 재학습 — 매주 최신 데이터로 RL 모델 재훈련

매주 1회 cron으로 실행하여:
1. 현재 best 모델을 최신 데이터로 fine-tuning
2. 새 모델을 scratch에서 훈련
3. 3개 알고리즘(PPO/SAC/TD3) 비교
4. 기존 best 대비 개선된 경우에만 교체
5. 결과를 텔레그램으로 알림

사용법:
    python -m rl_hybrid.rl.weekly_retrain
    python -m rl_hybrid.rl.weekly_retrain --days 90 --steps 200000
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.rl.train import prepare_data, evaluate, get_trader_class
from rl_hybrid.rl.environment import BitcoinTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.weekly_retrain")

KST = timezone(timedelta(hours=9))
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_DIR / "data" / "rl_models"
BEST_DIR = MODEL_DIR / "best"
BEST_MODEL = BEST_DIR / "best_model"
INFO_PATH = BEST_DIR / "model_info.json"
HISTORY_DIR = MODEL_DIR / "retrain_history"


def load_current_best_info() -> dict:
    """현재 best 모델 정보 로드"""
    if INFO_PATH.exists():
        try:
            with open(INFO_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"algorithm": "ppo", "avg_return_pct": 0, "avg_sharpe": 0}


def evaluate_model(trader, eval_candles, eval_signals, balance, episodes=10) -> dict:
    """모델 평가 후 통계 반환"""
    eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=balance,
        external_signals=eval_signals,
    )
    stats = evaluate(trader, eval_env, episodes=episodes)
    return {
        "avg_return": float(np.mean([s["total_return_pct"] for s in stats])),
        "avg_sharpe": float(np.mean([s["sharpe_ratio"] for s in stats])),
        "avg_mdd": float(np.mean([s["max_drawdown"] for s in stats])),
        "avg_trades": float(np.mean([s["trade_count"] for s in stats])),
    }


def notify_telegram(message: str):
    """텔레그램 알림 전송"""
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/notify_telegram.py", "info", "RL 주간 재학습", message],
            cwd=str(PROJECT_DIR), check=False, timeout=15,
        )
    except Exception as e:
        logger.warning(f"텔레그램 알림 실패: {e}")


def weekly_retrain(days: int = 90, total_steps: int = 200_000, balance: float = 10_000_000):
    """주간 재학습 메인 로직"""
    from rl_hybrid.rl.policy import SB3_AVAILABLE
    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 미설치")
        return

    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== 주간 재학습 시작 ({timestamp}) ===")

    # 데이터 준비
    train_candles, eval_candles, train_signals, eval_signals = prepare_data(days)

    # 현재 best 모델 정보
    current_info = load_current_best_info()
    current_algo = current_info.get("algorithm", "ppo")
    logger.info(f"현재 best: {current_algo.upper()} (수익률={current_info.get('avg_return_pct', '?')}%)")

    candidates = {}

    # 후보 1: 현재 best fine-tuning (같은 알고리즘)
    if (BEST_MODEL.parent / "best_model.zip").exists():
        try:
            logger.info(f"\n--- 후보 1: {current_algo.upper()} fine-tuning ---")
            TraderClass = get_trader_class(current_algo)
            train_env = BitcoinTradingEnv(
                candles=train_candles, initial_balance=balance,
                external_signals=train_signals,
            )
            trader_ft = TraderClass(env=train_env)
            trader_ft.load(str(BEST_MODEL))
            trader_ft.model.set_env(train_env)

            eval_env_ft = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=balance,
                external_signals=eval_signals,
            )
            trader_ft.train(
                total_timesteps=total_steps,
                eval_env=eval_env_ft,
                save_freq=total_steps // 5,
            )

            stats = evaluate_model(trader_ft, eval_candles, eval_signals, balance)
            candidates[f"{current_algo}_finetune"] = {"trader": trader_ft, **stats}
            logger.info(f"  fine-tune 결과: 수익률={stats['avg_return']:.2f}%, 샤프={stats['avg_sharpe']:.3f}")
        except Exception as e:
            logger.warning(f"fine-tuning 실패: {e}")

    # 후보 2-4: 알고리즘 scratch 훈련
    # SAC/TD3는 월 1회(1일)만, PPO는 매주
    day_of_month = datetime.now(KST).day
    if day_of_month <= 7:
        algos_to_train = ["ppo", "sac", "td3"]
        logger.info("월초 — PPO + SAC + TD3 전체 비교")
    else:
        algos_to_train = ["ppo"]
        logger.info("주간 — PPO만 훈련 (SAC/TD3는 월초에만)")

    for algo in algos_to_train:
        try:
            logger.info(f"\n--- 후보: {algo.upper()} scratch ---")
            TraderClass = get_trader_class(algo)
            train_env = BitcoinTradingEnv(
                candles=train_candles, initial_balance=balance,
                external_signals=train_signals,
            )
            eval_env = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=balance,
                external_signals=eval_signals,
            )
            trader = TraderClass(env=train_env)
            trader.train(
                total_timesteps=total_steps,
                eval_env=eval_env,
                save_freq=total_steps // 5,
            )

            stats = evaluate_model(trader, eval_candles, eval_signals, balance)
            candidates[f"{algo}_scratch"] = {"trader": trader, **stats}
            logger.info(f"  {algo} scratch 결과: 수익률={stats['avg_return']:.2f}%, 샤프={stats['avg_sharpe']:.3f}")
        except Exception as e:
            logger.warning(f"{algo} scratch 훈련 실패: {e}")

    if not candidates:
        logger.error("모든 후보 훈련 실패")
        notify_telegram("주간 재학습 실패: 모든 후보 훈련 실패")
        return

    # 현재 best 평가 (동일 eval 데이터로)
    current_stats = None
    if (BEST_MODEL.parent / "best_model.zip").exists():
        try:
            TraderClass = get_trader_class(current_algo)
            eval_env = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=balance,
                external_signals=eval_signals,
            )
            trader_current = TraderClass(env=eval_env, model_path=str(BEST_MODEL))
            current_stats = evaluate_model(trader_current, eval_candles, eval_signals, balance)
            candidates["current_best"] = {"trader": None, **current_stats}
            logger.info(f"\n현재 best 평가: 수익률={current_stats['avg_return']:.2f}%, 샤프={current_stats['avg_sharpe']:.3f}")
        except Exception as e:
            logger.warning(f"현재 best 평가 실패: {e}")

    # 비교 테이블
    logger.info(f"\n{'='*60}")
    logger.info("  주간 재학습 비교 결과")
    logger.info(f"{'='*60}")
    logger.info(f"{'후보':>20} | {'수익률':>10} | {'샤프':>8} | {'MDD':>10} | {'거래수':>8}")
    logger.info("-" * 65)
    for name, r in sorted(candidates.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True):
        logger.info(
            f"{name:>20} | {r['avg_return']:>9.2f}% | {r['avg_sharpe']:>8.3f} | "
            f"{r['avg_mdd']:>9.2%} | {r['avg_trades']:>8.1f}"
        )

    # 최적 후보 선정 (current_best 제외한 새 후보 중)
    new_candidates = {k: v for k, v in candidates.items() if k != "current_best"}
    if not new_candidates:
        logger.info("새 후보 없음, 현재 best 유지")
        return

    best_name = max(new_candidates, key=lambda k: (new_candidates[k]["avg_sharpe"], new_candidates[k]["avg_return"]))
    best_result = new_candidates[best_name]

    # 교체 판단: 현재 best 대비 샤프 or 수익률이 개선되어야 함
    should_replace = True
    if current_stats:
        sharpe_better = best_result["avg_sharpe"] > current_stats["avg_sharpe"] - 0.05
        return_better = best_result["avg_return"] > current_stats["avg_return"]
        should_replace = sharpe_better and return_better
        if not should_replace:
            logger.info(f"\n새 최적({best_name})이 현재 best보다 나쁨 → 교체 안 함")

    if should_replace and best_result.get("trader"):
        # 기존 best 백업
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = HISTORY_DIR / f"best_{timestamp}.zip"
        if (BEST_MODEL.parent / "best_model.zip").exists():
            shutil.copy2(BEST_MODEL.parent / "best_model.zip", backup_path)
            logger.info(f"기존 best 백업: {backup_path}")

        # 새 best 저장
        best_result["trader"].save(str(BEST_MODEL))
        shutil.copy2(BEST_MODEL.parent / "best_model.zip",
                      MODEL_DIR / "ppo_btc_latest.zip")

        # 알고리즘 정보에서 algo 추출
        algo_name = best_name.split("_")[0]
        model_info = {
            "algorithm": algo_name,
            "avg_return_pct": round(best_result["avg_return"], 4),
            "avg_sharpe": round(best_result["avg_sharpe"], 4),
            "avg_mdd": round(best_result["avg_mdd"], 6),
            "training_steps": total_steps,
            "training_days": days,
            "retrained_at": datetime.now(KST).isoformat(),
            "candidate_name": best_name,
        }
        with open(INFO_PATH, "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"\n새 best 모델 교체 완료: {best_name} ({algo_name.upper()})")
        msg = (
            f"주간 재학습 완료 — 모델 교체\n"
            f"후보: {best_name}\n"
            f"수익률: {best_result['avg_return']:.2f}%\n"
            f"샤프: {best_result['avg_sharpe']:.3f}\n"
            f"MDD: {best_result['avg_mdd']:.2%}"
        )
    else:
        logger.info("\n현재 best 모델 유지")
        msg = (
            f"주간 재학습 완료 — 현재 모델 유지\n"
            f"최적 후보({best_name}): 수익률={best_result['avg_return']:.2f}%\n"
            f"현재 best: 수익률={current_stats['avg_return']:.2f}%" if current_stats
            else f"주간 재학습 완료 — 모델 유지"
        )

    notify_telegram(msg)
    logger.info(f"\n=== 주간 재학습 완료 ===")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(description="RL 주간 자동 재학습")
    parser.add_argument("--days", type=int, default=90, help="훈련 데이터 기간 (일)")
    parser.add_argument("--steps", type=int, default=200_000, help="총 훈련 스텝")
    parser.add_argument("--balance", type=float, default=10_000_000, help="초기 잔고")
    args = parser.parse_args()

    weekly_retrain(args.days, args.steps, args.balance)
