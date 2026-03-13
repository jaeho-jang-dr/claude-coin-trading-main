#!/usr/bin/env python3
"""RL 모델 최적화 실험 — 보상함수 × 하이퍼파라미터 탐색

실험 조합:
  - Reward: v6 (현재), v7 (트렌드), v8 (하이브리드)
  - Entropy: 0.01, 0.02, 0.03
  - Learning Rate: 3e-4, 5e-4
  - 총 18개 조합 중 상위 후보 9개 실험

사용법:
    python scripts/train_experiment.py                 # 9개 실험 실행
    python scripts/train_experiment.py --quick          # 3개 빠른 실험 (v8만)
    python scripts/train_experiment.py --steps 200000   # 스텝 수 조정
    python scripts/train_experiment.py --days 365       # 데이터 기간 조정
"""

import argparse
import io
import json
import logging
import os
import sys
import time

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_experiment")

# ── 실험 구성 ──

FULL_EXPERIMENTS = [
    # v8 (하이브리드) — 최우선 테스트
    {"reward": "v8", "lr": 5e-4, "ent_coef": 0.02, "n_steps": 2048, "label": "v8_lr5e4_ent0.02"},
    {"reward": "v8", "lr": 3e-4, "ent_coef": 0.02, "n_steps": 2048, "label": "v8_lr3e4_ent0.02"},
    {"reward": "v8", "lr": 5e-4, "ent_coef": 0.03, "n_steps": 2048, "label": "v8_lr5e4_ent0.03"},
    # v7 (트렌드) — 비교군
    {"reward": "v7", "lr": 5e-4, "ent_coef": 0.02, "n_steps": 2048, "label": "v7_lr5e4_ent0.02"},
    {"reward": "v7", "lr": 3e-4, "ent_coef": 0.015, "n_steps": 2048, "label": "v7_lr3e4_ent0.015"},
    # v6 (현재 베이스라인) — 대조군
    {"reward": "v6", "lr": 5e-4, "ent_coef": 0.02, "n_steps": 2048, "label": "v6_lr5e4_ent0.02"},
    {"reward": "v6", "lr": 5e-4, "ent_coef": 0.01, "n_steps": 2048, "label": "v6_lr5e4_ent0.01_baseline"},
    # 큰 배치 실험
    {"reward": "v8", "lr": 3e-4, "ent_coef": 0.02, "n_steps": 4096, "label": "v8_lr3e4_ent0.02_ns4096"},
    # 높은 entropy 실험
    {"reward": "v8", "lr": 5e-4, "ent_coef": 0.05, "n_steps": 2048, "label": "v8_lr5e4_ent0.05"},
]

QUICK_EXPERIMENTS = [
    {"reward": "v8", "lr": 5e-4, "ent_coef": 0.02, "n_steps": 2048, "label": "v8_lr5e4_ent0.02"},
    {"reward": "v8", "lr": 3e-4, "ent_coef": 0.03, "n_steps": 2048, "label": "v8_lr3e4_ent0.03"},
    {"reward": "v6", "lr": 5e-4, "ent_coef": 0.01, "n_steps": 2048, "label": "v6_baseline"},
]


def linear_schedule(initial_lr: float):
    def _schedule(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return _schedule


def prepare_data(days: int):
    loader = HistoricalDataLoader()
    logger.info(f"Upbit에서 {days}일 1h 캔들 로드 중...")
    raw = loader.load_candles(days=days, interval="1h")
    candles = loader.compute_indicators(raw)

    split = int(len(candles) * 0.8)
    train = candles[:split]
    eval_ = candles[split:]
    logger.info(f"데이터 준비: 훈련={len(train)}봉, 평가={len(eval_)}봉")
    return train, eval_


def run_experiment(config: dict, train_candles, eval_candles, total_steps: int, out_dir: Path):
    """단일 실험 실행"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    label = config["label"]
    logger.info(f"\n{'='*60}")
    logger.info(f"실험: {label}")
    logger.info(f"  reward={config['reward']}, lr={config['lr']}, "
                f"ent_coef={config['ent_coef']}, n_steps={config['n_steps']}")
    logger.info(f"  총 스텝: {total_steps:,}")
    logger.info(f"{'='*60}")

    exp_dir = out_dir / label
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 환경 생성
    train_env = BitcoinTradingEnv(
        candles=train_candles,
        initial_balance=10_000_000,
        reward_version=config["reward"],
    )
    eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=10_000_000,
        reward_version=config["reward"],
    )

    # PPO 모델
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(config["lr"]),
        n_steps=config["n_steps"],
        batch_size=min(128, config["n_steps"]),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 128, 64]},
        verbose=0,
        seed=42,
    )

    # 평가 콜백
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(exp_dir),
        log_path=str(exp_dir),
        eval_freq=max(total_steps // 10, 2048),
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )

    # 학습
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=eval_callback)
    except Exception as e:
        logger.error(f"실험 {label} 학습 실패: {e}")
        return None
    elapsed = time.time() - t0

    # 평가
    metrics = evaluate_model(model, eval_env, n_episodes=5)
    metrics["label"] = label
    metrics["config"] = config
    metrics["elapsed_sec"] = round(elapsed, 1)
    metrics["total_steps"] = total_steps

    # 모델 저장
    model.save(str(exp_dir / "model"))

    # 결과 저장
    with open(exp_dir / "result.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"  결과: return={metrics['avg_return_pct']:+.2f}%, "
                f"sharpe={metrics['avg_sharpe']:.3f}, "
                f"mdd={metrics['avg_mdd']:.2f}%, "
                f"trades={metrics['avg_trades']:.0f}, "
                f"time={elapsed:.0f}s")

    train_env.close()
    eval_env.close()
    return metrics


def evaluate_model(model, env, n_episodes=5):
    """모델 평가 — N 에피소드 평균"""
    returns, sharpes, mdds, trades_list = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        stats = env.reward_calc.get_episode_stats(
            env._portfolio_value(env.candles[env.current_step]["close"]),
            env.initial_balance,
        )
        returns.append(stats["total_return_pct"])
        sharpes.append(stats["sharpe_ratio"])
        mdds.append(stats["max_drawdown"] * 100)
        trades_list.append(stats["total_trades"])

    return {
        "avg_return_pct": round(float(np.mean(returns)), 3),
        "std_return_pct": round(float(np.std(returns)), 3),
        "avg_sharpe": round(float(np.mean(sharpes)), 4),
        "avg_mdd": round(float(np.mean(mdds)), 3),
        "avg_trades": round(float(np.mean(trades_list)), 1),
        "all_returns": [round(r, 3) for r in returns],
    }


def print_leaderboard(results: list):
    """실험 결과 리더보드"""
    # Sharpe 기준 정렬
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda r: r["avg_sharpe"], reverse=True)

    print(f"\n{'='*80}")
    print(f"{'실험 리더보드':^80}")
    print(f"{'='*80}")
    print(f"{'#':>2} {'Label':<30} {'Return%':>8} {'Sharpe':>8} {'MDD%':>7} {'Trades':>7} {'Time':>6}")
    print(f"{'-'*80}")

    for i, r in enumerate(valid, 1):
        marker = " ★" if i == 1 else ""
        print(f"{i:>2} {r['label']:<30} {r['avg_return_pct']:>+7.2f}% "
              f"{r['avg_sharpe']:>8.4f} {r['avg_mdd']:>6.2f}% "
              f"{r['avg_trades']:>6.0f} {r['elapsed_sec']:>5.0f}s{marker}")

    print(f"{'='*80}")

    if valid:
        best = valid[0]
        print(f"\n[BEST] 최고 모델: {best['label']}")
        print(f"   Return: {best['avg_return_pct']:+.2f}% | Sharpe: {best['avg_sharpe']:.4f} "
              f"| MDD: {best['avg_mdd']:.2f}% | Trades: {best['avg_trades']:.0f}")


def main():
    parser = argparse.ArgumentParser(description="RL 모델 최적화 실험")
    parser.add_argument("--quick", action="store_true", help="빠른 실험 (3개만)")
    parser.add_argument("--steps", type=int, default=150_000, help="실험당 스텝 수")
    parser.add_argument("--days", type=int, default=180, help="데이터 기간 (일)")
    parser.add_argument("--index", type=int, nargs="+", help="실험 인덱스 (0-based, 복수 가능)")
    parser.add_argument("--out-dir", type=str, help="출력 디렉토리 (병렬 실행 시 공유)")
    args = parser.parse_args()

    all_experiments = QUICK_EXPERIMENTS if args.quick else FULL_EXPERIMENTS
    if args.index:
        experiments = [all_experiments[i] for i in args.index if i < len(all_experiments)]
    else:
        experiments = all_experiments

    # 출력 디렉토리
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/training_results") / f"experiment_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 준비 (한 번만)
    train_candles, eval_candles = prepare_data(args.days)

    # 실험 실행
    results = []
    total = len(experiments)
    for i, config in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{total}] 실험 시작: {config['label']}")
        metrics = run_experiment(config, train_candles, eval_candles, args.steps, out_dir)
        results.append(metrics)

    # 리더보드
    print_leaderboard(results)

    # 전체 결과 저장
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump([r for r in results if r], f, indent=2, ensure_ascii=False)
    logger.info(f"\n전체 결과 저장: {summary_path}")

    # 최고 모델을 ppo_btc_latest.zip으로 복사
    valid = [r for r in results if r is not None]
    if valid:
        best = max(valid, key=lambda r: r["avg_sharpe"])
        best_model = out_dir / best["label"] / "model.zip"
        if best_model.exists():
            import shutil
            dest = Path("data/rl_models/ppo_btc_latest.zip")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_model, dest)
            logger.info(f"[BEST] 최고 모델 -> {dest}")


if __name__ == "__main__":
    main()
