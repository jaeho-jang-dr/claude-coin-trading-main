"""RL 강화학습 v3 — v6 보상함수 유지 + 하이퍼파라미터 그리드 서치

v2 교훈:
  - PnL 직접 보상은 노이즈 증폭 → 제거
  - 엔트로피 0.02는 과도 → 0.005~0.015 범위 탐색
  - v6 Sharpe 보상이 가장 안정적 → 유지

v3 전략:
  1. v6 보상함수 그대로 사용
  2. 핵심 하이퍼파라미터 3개 그리드 서치 (LR, ent_coef, n_steps)
  3. 커리큘럼 학습 유지 (90→180일)
  4. best 모델 자동 선택
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.policy import MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_v3")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "training_results",
)


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def evaluate(model, env, episodes=10):
    all_stats = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr
        all_stats.append(env.get_episode_stats())
    return {
        "return_pct": float(np.mean([s["total_return_pct"] for s in all_stats])),
        "sharpe": float(np.mean([s["sharpe_ratio"] for s in all_stats])),
        "mdd": float(np.mean([s["max_drawdown"] for s in all_stats])),
        "trades": float(np.mean([s["trade_count"] for s in all_stats])),
    }


def train_one_config(config, train_candles, eval_candles, balance, steps):
    """단일 하이퍼파라미터 조합으로 학습"""
    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(config["lr"]),
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        policy_kwargs={
            "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
        },
    )

    save_dir = os.path.join(MODEL_DIR, f"grid_{config['name']}")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=steps // 5,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    start = time.time()
    model.learn(total_timesteps=steps, callback=eval_cb)
    elapsed = time.time() - start

    result = evaluate(model, eval_env, episodes=10)
    result["elapsed_sec"] = elapsed
    result["config"] = config

    return model, result


def main():
    balance = 10_000_000
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  v3: 하이퍼파라미터 그리드 서치 (v6 보상 유지)")
    logger.info("=" * 60)

    # 데이터 준비
    loader = HistoricalDataLoader()
    candles_180 = loader.compute_indicators(loader.load_candles(days=180, interval="1h"))
    split = int(len(candles_180) * 0.8)
    train_c = candles_180[:split]
    eval_c = candles_180[split:]

    # 그리드 서치 설정
    grid = {
        "lr": [1e-4, 3e-4, 5e-4],
        "ent_coef": [0.005, 0.01, 0.015],
        "n_steps": [1024, 2048],
    }

    configs = []
    for lr, ent, ns in product(grid["lr"], grid["ent_coef"], grid["n_steps"]):
        configs.append({
            "name": f"lr{lr:.0e}_ent{ent}_ns{ns}",
            "lr": lr,
            "ent_coef": ent,
            "n_steps": ns,
            "batch_size": min(128, ns),
            "n_epochs": 10,
        })

    logger.info(f"총 {len(configs)}개 조합 테스트 (각 100K 스텝)")

    # Phase 1: 그리드 서치
    results = []
    for i, cfg in enumerate(configs):
        logger.info(f"\n[{i+1}/{len(configs)}] {cfg['name']}")
        try:
            _, result = train_one_config(cfg, train_c, eval_c, balance, steps=100_000)
            results.append(result)
            logger.info(
                f"  수익률: {result['return_pct']:.2f}% | "
                f"샤프: {result['sharpe']:.3f} | MDD: {result['mdd']:.2%} | "
                f"거래: {result['trades']:.0f} | {result['elapsed_sec']:.0f}초"
            )
        except Exception as e:
            logger.error(f"  실패: {e}")
            results.append({"config": cfg, "error": str(e)})

    # 최적 조합 선택 (샤프 기준)
    valid = [r for r in results if "error" not in r]
    if not valid:
        logger.error("모든 조합 실패")
        return

    # 정렬: 샤프 > 수익률 > 낮은 MDD
    valid.sort(key=lambda r: (r["sharpe"], r["return_pct"], -r["mdd"]), reverse=True)

    best = valid[0]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"최적 조합: {best['config']['name']}")
    logger.info(
        f"  수익률: {best['return_pct']:.2f}% | 샤프: {best['sharpe']:.3f} | "
        f"MDD: {best['mdd']:.2%} | 거래: {best['trades']:.0f}"
    )

    # Phase 2: 최적 조합으로 커리큘럼 본학습
    logger.info(f"\n{'=' * 60}")
    logger.info("Phase 2: 최적 조합으로 커리큘럼 본학습")
    logger.info("=" * 60)

    best_cfg = best["config"]

    # Stage A: 90일 워밍업 (50K)
    logger.info("\n[Stage A] 90일 워밍업 (50K)")
    candles_90 = loader.compute_indicators(loader.load_candles(days=90, interval="1h"))
    sp = int(len(candles_90) * 0.8)
    train_env_a = BitcoinTradingEnv(candles=candles_90[:sp], initial_balance=balance)
    eval_env_a = BitcoinTradingEnv(candles=candles_90[sp:], initial_balance=balance)

    model = PPO(
        "MlpPolicy",
        train_env_a,
        learning_rate=linear_schedule(best_cfg["lr"]),
        n_steps=best_cfg["n_steps"],
        batch_size=best_cfg["batch_size"],
        n_epochs=best_cfg["n_epochs"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=best_cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs={
            "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
        },
    )

    eval_cb_a = EvalCallback(
        eval_env_a,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v3"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    model.learn(total_timesteps=50_000, callback=eval_cb_a)
    sa = evaluate(model, eval_env_a, episodes=10)
    logger.info(f"[Stage A] 수익률: {sa['return_pct']:.2f}% | 샤프: {sa['sharpe']:.3f}")

    # Stage B: 180일 본학습 (200K)
    logger.info("\n[Stage B] 180일 본학습 (200K)")
    train_env_b = BitcoinTradingEnv(candles=train_c, initial_balance=balance)
    eval_env_b = BitcoinTradingEnv(candles=eval_c, initial_balance=balance)
    model.set_env(train_env_b)

    eval_cb_b = EvalCallback(
        eval_env_b,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v3"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    model.learn(total_timesteps=200_000, callback=eval_cb_b, reset_num_timesteps=False)
    sb = evaluate(model, eval_env_b, episodes=10)
    logger.info(f"[Stage B] 수익률: {sb['return_pct']:.2f}% | 샤프: {sb['sharpe']:.3f}")

    # Stage C: 4h 미세조정 (50K)
    logger.info("\n[Stage C] 4h 미세조정 (50K)")
    candles_4h = loader.compute_indicators(loader.load_candles(days=180, interval="4h"))
    sp4 = int(len(candles_4h) * 0.8)
    train_env_c = BitcoinTradingEnv(candles=candles_4h[:sp4], initial_balance=balance)
    eval_env_c = BitcoinTradingEnv(candles=candles_4h[sp4:], initial_balance=balance)
    model.set_env(train_env_c)
    model.learning_rate = linear_schedule(best_cfg["lr"] * 0.3)

    eval_cb_c = EvalCallback(
        eval_env_c,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v3"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    model.learn(total_timesteps=50_000, callback=eval_cb_c, reset_num_timesteps=False)
    sc = evaluate(model, eval_env_c, episodes=10)
    logger.info(f"[Stage C] 수익률: {sc['return_pct']:.2f}% | 샤프: {sc['sharpe']:.3f}")

    model.save(os.path.join(MODEL_DIR, "v3_final"))
    model.save(os.path.join(MODEL_DIR, "ppo_btc_latest"))

    # Phase 3: 최종 비교
    logger.info(f"\n{'=' * 60}")
    logger.info("Phase 3: 최종 비교 (v3 vs v6 vs B&H)")
    logger.info("=" * 60)

    v6_path = os.path.join(MODEL_DIR, "best", "best_model")
    final_comparison = {}

    for days, label in [(90, "90일"), (180, "180일")]:
        candles = loader.compute_indicators(loader.load_candles(days=days, interval="1h"))
        sp = int(len(candles) * 0.8)
        ec = candles[sp:]

        # v3
        e3 = BitcoinTradingEnv(candles=ec, initial_balance=balance)
        v3_r = evaluate(model, e3, episodes=10)
        logger.info(f"[v3 {label}] 수익률: {v3_r['return_pct']:.2f}% | 샤프: {v3_r['sharpe']:.3f} | MDD: {v3_r['mdd']:.2%}")

        # v6
        e6 = BitcoinTradingEnv(candles=ec, initial_balance=balance)
        v6m = PPO.load(v6_path, env=e6)
        v6_r = evaluate(v6m, e6, episodes=10)
        logger.info(f"[v6 {label}] 수익률: {v6_r['return_pct']:.2f}% | 샤프: {v6_r['sharpe']:.3f} | MDD: {v6_r['mdd']:.2%}")

        # B&H
        bnh = {"return_pct": (ec[-1]["close"] - ec[0]["close"]) / ec[0]["close"] * 100}
        prices = [c["close"] for c in ec]
        peak = prices[0]
        mdd = 0
        for p in prices:
            peak = max(peak, p)
            mdd = max(mdd, (peak - p) / peak)
        bnh["mdd"] = mdd
        logger.info(f"[B&H {label}] 수익률: {bnh['return_pct']:.2f}% | MDD: {bnh['mdd']:.2%}")

        logger.info(f"  v3 vs v6: {v3_r['return_pct'] - v6_r['return_pct']:+.2f}%p")
        logger.info(f"  v3 vs B&H: {v3_r['return_pct'] - bnh['return_pct']:+.2f}%p")

        final_comparison[label] = {
            "v3": v3_r, "v6": v6_r, "bnh": bnh,
            "v3_vs_v6": v3_r["return_pct"] - v6_r["return_pct"],
            "v3_vs_bnh": v3_r["return_pct"] - bnh["return_pct"],
        }

    total_elapsed = time.time() - total_start

    # 결과 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "grid_search": [
            {k: v for k, v in r.items() if k != "config" or True}
            for r in results
        ],
        "best_config": best_cfg,
        "curriculum": {"stage_a": sa, "stage_b": sb, "stage_c": sc},
        "final_comparison": final_comparison,
        "total_elapsed_sec": total_elapsed,
    }

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    path = os.path.join(RESULTS_DIR, f"v3_grid_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  v3 완료! 총 {total_elapsed/60:.1f}분")
    logger.info(f"  최적 조합: {best_cfg['name']}")
    logger.info(f"  결과: {path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
