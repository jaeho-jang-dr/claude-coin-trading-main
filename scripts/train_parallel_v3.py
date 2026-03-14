"""병렬 RL 훈련 V3 — 최종 최적화

발견:
  - 300K 스텝이 최적 (이상은 과적합)
  - 깊은 네트워크 [512,256,128]이 유망
  - 균형 전략 (ent=0.10, γ=0.995)이 최고

최종: 300K~400K 스텝 + 학습률 감쇠 + 다양한 네트워크/파라미터 조합
"""

import io
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "rl_models" / "v5"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_DIR / "logs"


def log(msg: str, tag: str = "MAIN"):
    ts = datetime.now(KST).strftime("%H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "train_parallel_v3.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_data():
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    loader = HistoricalDataLoader()
    raw = loader.load_candles(days=180, interval="4h")
    return loader.compute_indicators(raw)


def evaluate_model(model, candles, n_episodes=20):
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
    import numpy as np

    eval_env = BitcoinTradingEnvV2(
        candles=candles, crash_injection=False,
        discrete_actions=True, reward_clip=2.0,
    )
    profits, trades, rewards = [], [], []
    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        done, ep_r = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_r += reward
            done = terminated or truncated
        profits.append(info.get("profit_pct", 0))
        trades.append(info.get("trade_count", 0))
        rewards.append(ep_r)
    eval_env.close()

    return {
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "mean_trades": float(np.mean(trades)),
        "mean_reward": float(np.mean(rewards)),
        "max_profit": float(np.max(profits)),
        "min_profit": float(np.min(profits)),
        "win_rate": float(np.mean([1 if p > 0 else 0 for p in profits]) * 100),
        "median_profit": float(np.median(profits)),
    }


def lr_schedule(initial_lr: float):
    """학습률 선형 감쇠 스케줄."""
    def schedule(progress_remaining: float) -> float:
        return initial_lr * max(0.1, progress_remaining)
    return schedule


def train_variant(vid: int, candles: list[dict]) -> dict:
    from stable_baselines3 import PPO
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
    from stable_baselines3.common.callbacks import EvalCallback
    import numpy as np

    variants = {
        1: {
            "name": "최적조합 A (깊은+감쇠+300K)",
            "steps": 300_000,
            "ent_coef": 0.10,
            "lr": 3e-4,
            "gamma": 0.995,
            "net_arch": [512, 256, 128],
            "crash_prob": 0.02,
        },
        2: {
            "name": "최적조합 B (깊은+감쇠+400K)",
            "steps": 400_000,
            "ent_coef": 0.10,
            "lr": 3e-4,
            "gamma": 0.995,
            "net_arch": [512, 256, 128],
            "crash_prob": 0.02,
        },
        3: {
            "name": "넓은 네트워크 [256,256,256]",
            "steps": 350_000,
            "ent_coef": 0.10,
            "lr": 2e-4,
            "gamma": 0.995,
            "net_arch": [256, 256, 256],
            "crash_prob": 0.02,
        },
        4: {
            "name": "보수적+깊은 (ent=0.06)",
            "steps": 350_000,
            "ent_coef": 0.06,
            "lr": 2e-4,
            "gamma": 0.995,
            "net_arch": [512, 256, 128],
            "crash_prob": 0.02,
        },
        5: {
            "name": "고탐험+빠른학습 (ent=0.15)",
            "steps": 300_000,
            "ent_coef": 0.15,
            "lr": 4e-4,
            "gamma": 0.995,
            "net_arch": [256, 256],
            "crash_prob": 0.02,
        },
        6: {
            "name": "폭락+깊은 (crash=3%)",
            "steps": 350_000,
            "ent_coef": 0.10,
            "lr": 2e-4,
            "gamma": 0.995,
            "net_arch": [512, 256, 128],
            "crash_prob": 0.03,
        },
    }

    v = variants[vid]
    log(f"시작: {v['name']}, {v['steps']:,} 스텝", f"V{vid}")
    t0 = time.time()

    # 환경
    env = BitcoinTradingEnvV2(
        candles=candles,
        discrete_actions=True, reward_clip=2.0,
        anti_collapse=True, dca_reward=True,
        crash_injection=True, crash_prob=v["crash_prob"],
        regime_aware=True,
    )

    # 평가 환경 (Early Stopping용)
    eval_candles = candles[int(len(candles) * 0.8):]
    eval_env = BitcoinTradingEnvV2(
        candles=eval_candles,
        discrete_actions=True, reward_clip=2.0,
        crash_injection=False, anti_collapse=False,
    )

    # 모델 (학습률 감쇠 적용)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr_schedule(v["lr"]),
        ent_coef=v["ent_coef"],
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=v["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": v["net_arch"]},
        verbose=0,
    )

    # Early Stopping + Best Model 저장
    best_dir = str(MODEL_DIR / f"best_v{vid}")
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=max(v["steps"] // 15, 10000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=v["steps"], callback=eval_cb, progress_bar=False)

    # Best model 로드 (EvalCallback이 저장한 최고 모델)
    best_path = Path(best_dir) / "best_model.zip"
    if best_path.exists():
        best_model = PPO.load(str(best_path))
        log(f"Best model 로드 완료", f"V{vid}")
    else:
        best_model = model

    # 최종 모델도 저장
    model.save(str(MODEL_DIR / f"final_v{vid}"))
    best_model.save(str(MODEL_DIR / f"variant{vid}"))

    # 두 모델 평가
    metrics_best = evaluate_model(best_model, candles)
    metrics_final = evaluate_model(model, candles)

    # 더 좋은 쪽 선택
    if metrics_best["mean_profit"] >= metrics_final["mean_profit"]:
        metrics = metrics_best
        source = "best"
    else:
        metrics = metrics_final
        source = "final"

    elapsed = time.time() - t0
    log(f"완료: {v['name']} ({source}) — {elapsed:.0f}초, "
        f"수익률 {metrics['mean_profit']:+.2f}% (±{metrics['std_profit']:.1f}), "
        f"거래 {metrics['mean_trades']:.0f}회, "
        f"승률 {metrics['win_rate']:.0f}%, "
        f"최대 {metrics['max_profit']:+.2f}%, "
        f"중앙 {metrics['median_profit']:+.2f}%", f"V{vid}")

    env.close()
    eval_env.close()
    return {
        "variant": vid, "name": v["name"], "elapsed": elapsed,
        "steps": v["steps"], "metrics": metrics, "source": source,
        "model_path": str(MODEL_DIR / f"variant{vid}"),
    }


def main():
    log("╔═══════════════════════════════════════════════════════╗")
    log("║   최종 최적화 훈련 — 300K~400K + LR감쇠 + EarlyStopping ║")
    log("╚═══════════════════════════════════════════════════════╝")
    start_time = time.time()

    candles = load_data()
    log(f"데이터: {len(candles)}봉")
    log("─" * 55)

    results = {}
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(train_variant, i, candles): i for i in range(1, 7)}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                results[vid] = future.result()
            except Exception as e:
                log(f"V{vid} 실패: {e}", f"V{vid}")
                results[vid] = {"variant": vid, "error": str(e)}

    elapsed = time.time() - start_time
    log("")
    log("╔═══════════════════════════════════════════════════════╗")
    log("║               최종 훈련 결과 요약                      ║")
    log("╚═══════════════════════════════════════════════════════╝")
    log(f"  총 소요: {elapsed / 60:.1f}분")

    best_vid, best_profit = None, -999
    for vid in sorted(results.keys()):
        r = results[vid]
        if "error" in r:
            log(f"  V{vid}: 실패"); continue
        m = r["metrics"]
        if m["mean_profit"] > best_profit:
            best_profit = m["mean_profit"]
            best_vid = vid
        log(f"  V{vid} ({r['name']}, {r['source']}): "
            f"수익률 {m['mean_profit']:+.2f}% (±{m['std_profit']:.1f}), "
            f"거래 {m['mean_trades']:.0f}회, 승률 {m['win_rate']:.0f}%, "
            f"최대 {m['max_profit']:+.2f}%")

    if best_vid is not None:
        from stable_baselines3 import PPO
        best = PPO.load(str(MODEL_DIR / f"variant{best_vid}"))
        best.save(str(MODEL_DIR / "final_v5"))
        best.save(str(PROJECT_DIR / "data" / "rl_models" / "ppo_btc_best.zip"))
        log("")
        log(f"  ★ 최고 모델: V{best_vid} (수익률 {best_profit:+.2f}%)")
        log(f"  ★ 저장: data/rl_models/v5/final_v5 + ppo_btc_best.zip")

    # DB 기록
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env")
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if url and key:
            best_m = results.get(best_vid, {}).get("metrics", {})
            row = {
                "model_type": "ppo_v5_optimized",
                "training_steps": sum(r.get("steps", 0) for r in results.values()),
                "reward_mean": best_m.get("mean_reward", 0),
                "reward_std": best_m.get("std_profit", 0),
                "notes": json.dumps({
                    "method": "optimized_lr_decay_early_stop",
                    "best_variant": best_vid,
                    "elapsed_min": round(elapsed / 60, 1),
                    "results": {str(k): v.get("metrics", {}) for k, v in results.items() if "metrics" in v},
                }, ensure_ascii=False, default=str),
            }
            requests.post(
                f"{url}/rest/v1/rl_training_log", json=row,
                headers={"apikey": key, "Authorization": f"Bearer {key}",
                          "Content-Type": "application/json", "Prefer": "return=minimal"},
                timeout=10,
            )
            log("DB 기록 완료")
    except Exception as e:
        log(f"DB 기록 실패: {e}")

    log("═══ 최종 훈련 완료 ═══")


if __name__ == "__main__":
    main()
