"""주간 심층 재학습 스크립트 (Track B)

매주 일요일 새벽 3시 Windows 작업 스케줄러로 실행.
4시간봉 180일 데이터로 500K 스텝 SB3 PPO 훈련 후,
기존 best 모델 대비 성능이 향상되면 교체, 아니면 롤백.

사용법:
    python scripts/weekly_retrain.py              # 기본 실행
    python scripts/weekly_retrain.py --force       # 성능 비교 없이 강제 저장
    python scripts/weekly_retrain.py --dry-run     # 훈련만, 모델 교체 안 함

Windows 작업 스케줄러 등록:
    python scripts/setup_weekly_retrain.py
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "logs", "weekly_retrain.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("weekly_retrain")

# === 설정 ===
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "rl_models")
BEST_MODEL = os.path.join(MODEL_DIR, "best", "best_model.zip")
LATEST_MODEL = os.path.join(MODEL_DIR, "ppo_btc_latest.zip")
BACKUP_DIR = os.path.join(MODEL_DIR, "backups")
HISTORY_FILE = os.path.join(MODEL_DIR, "retrain_history.json")

# 훈련 파라미터 (최적 설정 확정)
TRAIN_DAYS = 180
TRAIN_STEPS = 500_000
TRAIN_INTERVAL = "4h"
INITIAL_BALANCE = 10_000_000
EVAL_EPISODES = 10


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def evaluate_model(model_path: str, eval_candles: list) -> dict:
    """모델 평가 → 수익률, Sharpe, MDD 반환"""
    from rl_hybrid.rl.policy import PPOTrader
    from rl_hybrid.rl.environment import BitcoinTradingEnv

    env = BitcoinTradingEnv(candles=eval_candles, initial_balance=INITIAL_BALANCE)
    trader = PPOTrader(env=env, model_path=model_path.replace(".zip", ""))

    all_stats = []
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = trader.predict(obs)
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
        all_stats.append(env.get_episode_stats())

    return {
        "avg_return": float(np.mean([s["total_return_pct"] for s in all_stats])),
        "avg_sharpe": float(np.mean([s["sharpe_ratio"] for s in all_stats])),
        "avg_mdd": float(np.mean([s["max_drawdown"] for s in all_stats])),
        "avg_trades": float(np.mean([s["trade_count"] for s in all_stats])),
    }


def backup_current_best():
    """현재 best 모델을 타임스탬프 백업"""
    if not os.path.exists(BEST_MODEL):
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"best_model_{ts}.zip")
    shutil.copy2(BEST_MODEL, backup_path)
    logger.info(f"기존 best 백업: {backup_path}")

    # 최근 5개만 유지
    backups = sorted(
        [f for f in os.listdir(BACKUP_DIR) if f.endswith(".zip")],
        reverse=True,
    )
    for old in backups[5:]:
        os.remove(os.path.join(BACKUP_DIR, old))
        logger.info(f"오래된 백업 삭제: {old}")

    return backup_path


def train_new_model():
    """SB3 PPO 500K 훈련 실행"""
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    from rl_hybrid.rl.environment import BitcoinTradingEnv
    from rl_hybrid.rl.policy import PPOTrader

    # 데이터 준비
    loader = HistoricalDataLoader()
    logger.info(f"Upbit에서 {TRAIN_DAYS}일 {TRAIN_INTERVAL} 캔들 로드 중...")
    raw = loader.load_candles(days=TRAIN_DAYS, interval=TRAIN_INTERVAL)
    candles = loader.compute_indicators(raw)

    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    logger.info(
        f"데이터: 훈련={len(train_candles)}봉, 평가={len(eval_candles)}봉"
    )

    # 훈련
    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=INITIAL_BALANCE)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=INITIAL_BALANCE)

    trader = PPOTrader(env=train_env)
    trader.train(
        total_timesteps=TRAIN_STEPS,
        eval_env=eval_env,
        save_freq=TRAIN_STEPS // 10,
    )

    return eval_candles


def run_retrain(force: bool = False, dry_run: bool = False):
    """주간 재학습 메인 로직"""
    start = time.time()
    logger.info("=" * 60)
    logger.info(f"주간 심층 재학습 시작: {datetime.now().isoformat()}")
    logger.info(f"설정: {TRAIN_DAYS}일 {TRAIN_INTERVAL} {TRAIN_STEPS:,} 스텝")
    logger.info("=" * 60)

    # 1. 기존 best 모델 평가 (베이스라인)
    baseline = None
    if os.path.exists(BEST_MODEL) and not force:
        logger.info("\n[Step 1] 기존 best 모델 평가...")
        from rl_hybrid.rl.data_loader import HistoricalDataLoader

        loader = HistoricalDataLoader()
        raw = loader.load_candles(days=TRAIN_DAYS, interval=TRAIN_INTERVAL)
        candles = loader.compute_indicators(raw)
        eval_candles = candles[int(len(candles) * 0.8):]

        baseline = evaluate_model(BEST_MODEL, eval_candles)
        logger.info(
            f"  베이스라인: return={baseline['avg_return']:+.2f}% "
            f"sharpe={baseline['avg_sharpe']:.3f} mdd={baseline['avg_mdd']:.2%}"
        )
    else:
        logger.info("\n[Step 1] 기존 best 모델 없음 또는 강제 모드 -- 스킵")

    # 2. 새 모델 훈련
    logger.info("\n[Step 2] 새 모델 훈련...")
    eval_candles = train_new_model()

    # 3. 새 모델 평가
    logger.info("\n[Step 3] 새 모델 평가...")
    new_stats = evaluate_model(LATEST_MODEL, eval_candles)
    logger.info(
        f"  신규 모델: return={new_stats['avg_return']:+.2f}% "
        f"sharpe={new_stats['avg_sharpe']:.3f} mdd={new_stats['avg_mdd']:.2%} "
        f"trades={new_stats['avg_trades']:.0f}"
    )

    # 4. 성능 비교 + 교체 판단
    should_replace = force
    if baseline and not force:
        # Sharpe 향상 또는 동등 Sharpe에서 MDD 개선
        sharpe_improved = new_stats["avg_sharpe"] > baseline["avg_sharpe"] - 0.05
        mdd_improved = new_stats["avg_mdd"] < baseline["avg_mdd"] + 0.01
        should_replace = sharpe_improved and mdd_improved

        logger.info(
            f"\n[Step 4] 성능 비교: "
            f"sharpe {baseline['avg_sharpe']:.3f} → {new_stats['avg_sharpe']:.3f} "
            f"({'개선' if sharpe_improved else '저하'}), "
            f"mdd {baseline['avg_mdd']:.2%} → {new_stats['avg_mdd']:.2%} "
            f"({'개선' if mdd_improved else '저하'})"
        )
    elif not baseline:
        should_replace = True

    # 5. 모델 교체 또는 롤백
    if dry_run:
        logger.info("\n[Step 5] DRY-RUN 모드 -- 모델 교체 안 함")
    elif should_replace:
        backup_current_best()
        os.makedirs(os.path.dirname(BEST_MODEL), exist_ok=True)
        shutil.copy2(LATEST_MODEL, BEST_MODEL)
        logger.info(f"\n[Step 5] 모델 교체 완료 → {BEST_MODEL}")
    else:
        logger.info("\n[Step 5] 성능 저하 -- 기존 best 유지 (롤백)")

    # 6. 이력 기록
    elapsed = time.time() - start
    record = {
        "timestamp": datetime.now().isoformat(),
        "train_days": TRAIN_DAYS,
        "train_steps": TRAIN_STEPS,
        "interval": TRAIN_INTERVAL,
        "new_stats": new_stats,
        "baseline_stats": baseline,
        "replaced": should_replace and not dry_run,
        "elapsed_sec": round(elapsed),
    }

    history = load_history()
    history.append(record)
    save_history(history)

    # 7. 텔레그램 알림
    try:
        _send_telegram_report(record)
    except Exception as e:
        logger.warning(f"텔레그램 알림 실패: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"주간 재학습 완료: {elapsed:.0f}초")
    logger.info(f"결과: {'모델 교체' if record['replaced'] else '기존 유지'}")
    logger.info(f"{'=' * 60}")

    return record


def _send_telegram_report(record: dict):
    """텔레그램으로 재학습 결과 보고"""
    import subprocess

    stats = record["new_stats"]
    baseline = record.get("baseline_stats")
    replaced = record["replaced"]

    lines = [
        "🧠 *주간 RL 재학습 완료*",
        "",
        f"⏱ 소요: {record['elapsed_sec']}초",
        f"📊 설정: {record['interval']} {record['train_days']}일 {record['train_steps']:,}스텝",
        "",
        "*신규 모델:*",
        f"  수익률: {stats['avg_return']:+.2f}%",
        f"  Sharpe: {stats['avg_sharpe']:.3f}",
        f"  MDD: {stats['avg_mdd']:.2%}",
        f"  거래수: {stats['avg_trades']:.0f}",
    ]

    if baseline:
        lines += [
            "",
            "*기존 모델:*",
            f"  수익률: {baseline['avg_return']:+.2f}%",
            f"  Sharpe: {baseline['avg_sharpe']:.3f}",
            f"  MDD: {baseline['avg_mdd']:.2%}",
        ]

    lines.append(f"\n{'✅ 모델 교체됨' if replaced else '⏸ 기존 모델 유지'}")

    message = "\n".join(lines)

    subprocess.run(
        [sys.executable, "scripts/notify_telegram.py", "--message", message],
        capture_output=True,
        timeout=15,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="주간 RL 심층 재학습")
    parser.add_argument("--force", action="store_true", help="성능 비교 없이 강제 교체")
    parser.add_argument("--dry-run", action="store_true", help="훈련만, 모델 교체 안 함")
    args = parser.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    run_retrain(force=args.force, dry_run=args.dry_run)
