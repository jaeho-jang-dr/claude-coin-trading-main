"""병렬 RL 훈련 V2 — Phase 6 기반 심화 훈련

Phase 6 (균형 전략, -0.02%)이 최고였으므로,
해당 설정을 기반으로 변형 6개를 병렬 훈련한다.
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
MODEL_DIR = PROJECT_DIR / "data" / "rl_models" / "v4"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_DIR / "logs"


def log(msg: str, phase: str = "MAIN"):
    ts = datetime.now(KST).strftime("%H:%M:%S")
    line = f"[{ts}] [{phase}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "train_parallel_v2.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_data():
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    loader = HistoricalDataLoader()
    raw = loader.load_candles(days=180, interval="4h")
    return loader.compute_indicators(raw)


def create_env(candles, **kwargs):
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
    defaults = {
        "discrete_actions": True,
        "reward_clip": 2.0,
        "anti_collapse": True,
        "dca_reward": True,
        "crash_injection": True,
        "crash_prob": 0.02,
        "regime_aware": True,
    }
    defaults.update(kwargs)
    return BitcoinTradingEnvV2(candles=candles, **defaults)


def evaluate_model(model, candles, n_episodes=15):
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
    import numpy as np

    eval_env = BitcoinTradingEnvV2(
        candles=candles, crash_injection=False,
        discrete_actions=True, reward_clip=2.0,
    )

    profits, trades, rewards = [], [], []
    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        profits.append(info.get("profit_pct", 0))
        trades.append(info.get("trade_count", 0))
        rewards.append(ep_reward)
    eval_env.close()

    return {
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "mean_trades": float(np.mean(trades)),
        "mean_reward": float(np.mean(rewards)),
        "max_profit": float(np.max(profits)),
        "min_profit": float(np.min(profits)),
        "win_rate": float(np.mean([1 if p > 0 else 0 for p in profits]) * 100),
    }


def train_variant(variant_id: int, candles: list[dict]) -> dict:
    """Phase 6 기반 변형을 훈련한다."""
    from stable_baselines3 import PPO
    import numpy as np

    variants = {
        1: {
            "name": "원본 P6 장기 (1M 스텝)",
            "steps": 1_000_000,
            "ent_coef": 0.10,
            "lr": 2e-4,
            "gamma": 0.995,
            "crash_prob": 0.02,
            "net_arch": [256, 256],
        },
        2: {
            "name": "낮은 탐험 (0.05 엔트로피)",
            "steps": 800_000,
            "ent_coef": 0.05,
            "lr": 1e-4,
            "gamma": 0.995,
            "crash_prob": 0.02,
            "net_arch": [256, 256],
        },
        3: {
            "name": "깊은 네트워크 [512,256,128]",
            "steps": 800_000,
            "ent_coef": 0.08,
            "lr": 1.5e-4,
            "gamma": 0.995,
            "crash_prob": 0.02,
            "net_arch": [512, 256, 128],
        },
        4: {
            "name": "높은 감마 (0.999, 장기 시야)",
            "steps": 800_000,
            "ent_coef": 0.08,
            "lr": 2e-4,
            "gamma": 0.999,
            "crash_prob": 0.02,
            "net_arch": [256, 256],
        },
        5: {
            "name": "폭락 강화 (5% 확률)",
            "steps": 800_000,
            "ent_coef": 0.10,
            "lr": 2e-4,
            "gamma": 0.995,
            "crash_prob": 0.05,
            "net_arch": [256, 256],
        },
        6: {
            "name": "안정 학습 (큰 배치, 느린 학습)",
            "steps": 1_000_000,
            "ent_coef": 0.08,
            "lr": 1e-4,
            "gamma": 0.995,
            "crash_prob": 0.02,
            "net_arch": [256, 256],
            "n_steps": 4096,
            "batch_size": 256,
        },
    }

    v = variants[variant_id]
    log(f"시작: {v['name']}, {v['steps']:,} 스텝", f"V{variant_id}")
    t0 = time.time()

    env = create_env(candles, crash_prob=v["crash_prob"])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=v["lr"],
        ent_coef=v["ent_coef"],
        n_steps=v.get("n_steps", 2048),
        batch_size=v.get("batch_size", 128),
        n_epochs=10,
        gamma=v["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": v["net_arch"]},
        verbose=0,
    )

    model.learn(total_timesteps=v["steps"], progress_bar=False)

    save_path = str(MODEL_DIR / f"variant{variant_id}")
    model.save(save_path)

    metrics = evaluate_model(model, candles)
    elapsed = time.time() - t0

    log(f"완료: {v['name']} — {elapsed:.0f}초, "
        f"수익률 {metrics['mean_profit']:+.2f}% (±{metrics['std_profit']:.1f}), "
        f"거래 {metrics['mean_trades']:.0f}회, "
        f"승률 {metrics['win_rate']:.0f}%, "
        f"최대 {metrics['max_profit']:+.2f}%", f"V{variant_id}")

    env.close()
    return {
        "variant": variant_id,
        "name": v["name"],
        "elapsed": elapsed,
        "steps": v["steps"],
        "metrics": metrics,
        "model_path": save_path,
        "config": {k: v for k, v in v.items() if k != "name"},
    }


def main():
    log("╔══════════════════════════════════════════════════════╗")
    log("║   심화 병렬 훈련 — Phase 6 기반 6개 변형 (V2.2/v7)   ║")
    log("╚══════════════════════════════════════════════════════╝")
    start_time = time.time()

    candles = load_data()
    log(f"데이터: {len(candles)}봉 (180일 4h)")
    total_steps = 1_000_000 + 800_000 * 4 + 1_000_000
    log(f"총 스텝: {total_steps:,} (6개 병렬)")
    log("─" * 55)

    results = {}
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(train_variant, i, candles): i
            for i in range(1, 7)
        }
        for future in as_completed(futures):
            vid = futures[future]
            try:
                results[vid] = future.result()
            except Exception as e:
                log(f"Variant {vid} 실패: {e}", f"V{vid}")
                results[vid] = {"variant": vid, "error": str(e)}

    elapsed = time.time() - start_time
    log("")
    log("╔══════════════════════════════════════════════════════╗")
    log("║              심화 훈련 결과 요약                      ║")
    log("╚══════════════════════════════════════════════════════╝")
    log(f"  총 소요: {elapsed / 60:.1f}분 (병렬)")

    best_vid, best_profit = None, -999
    for vid in sorted(results.keys()):
        r = results[vid]
        if "error" in r:
            log(f"  V{vid}: 실패 — {r['error']}")
            continue
        m = r["metrics"]
        if m["mean_profit"] > best_profit:
            best_profit = m["mean_profit"]
            best_vid = vid
        star = " ★" if vid == best_vid else ""
        log(f"  V{vid} ({r['name']}): "
            f"수익률 {m['mean_profit']:+.2f}% (±{m['std_profit']:.1f}), "
            f"거래 {m['mean_trades']:.0f}회, "
            f"승률 {m['win_rate']:.0f}%, "
            f"최대 {m['max_profit']:+.2f}%{star}")

    if best_vid is not None:
        from stable_baselines3 import PPO
        best = PPO.load(str(MODEL_DIR / f"variant{best_vid}"))
        best.save(str(MODEL_DIR / "final_v4"))
        best.save(str(PROJECT_DIR / "data" / "rl_models" / "ppo_btc_v4.zip"))
        log("")
        log(f"  ★ 최고 모델: Variant {best_vid} (수익률 {best_profit:+.2f}%)")
        log(f"  ★ 저장: data/rl_models/v4/final_v4")

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
                "model_type": "ppo_v4_deep_parallel",
                "training_steps": total_steps,
                "reward_mean": best_m.get("mean_reward", 0),
                "reward_std": best_m.get("std_profit", 0),
                "notes": json.dumps({
                    "method": "deep_parallel_6variant",
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

    log("═══ 심화 훈련 완료 ═══")


if __name__ == "__main__":
    main()
