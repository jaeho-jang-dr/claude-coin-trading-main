"""병렬 RL 훈련 — 6개 Phase를 동시 실행

각 Phase가 독립적으로 처음부터 훈련하고,
최종적으로 모든 모델을 평가하여 최고 성능 모델을 선택한다.
"""

import io
import sys
import os
import json
import time
import copy
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "rl_models" / "v3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, phase: str = "MAIN"):
    ts = datetime.now(KST).strftime("%H:%M:%S")
    line = f"[{ts}] [{phase}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "train_parallel.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_data(days: int = 180, interval: str = "4h") -> list[dict]:
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    loader = HistoricalDataLoader()
    raw = loader.load_candles(days=days, interval=interval)
    return loader.compute_indicators(raw)


def create_env(candles, **kwargs):
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
    defaults = {
        "discrete_actions": True,
        "reward_clip": 2.0,
        "anti_collapse": True,
        "dca_reward": True,
    }
    defaults.update(kwargs)
    return BitcoinTradingEnvV2(candles=candles, **defaults)


def create_model(env, **kwargs):
    from stable_baselines3 import PPO
    defaults = {
        "learning_rate": 3e-4,
        "ent_coef": 0.12,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "verbose": 0,
    }
    defaults.update(kwargs)
    return PPO("MlpPolicy", env, **defaults)


def evaluate_model(model, candles, n_episodes=10, **env_kwargs):
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
    import numpy as np

    kwargs = {"crash_injection": False, "discrete_actions": True, "reward_clip": 2.0}
    kwargs.update(env_kwargs)
    eval_env = BitcoinTradingEnvV2(candles=candles, **kwargs)

    profits = []
    trades = []
    rewards = []
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
        "mean_trades": float(np.mean(trades)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_profit": float(np.max(profits)),
        "min_profit": float(np.min(profits)),
    }


# ═══════════════════════════════════════════════════
# 6개 독립 훈련 워커
# ═══════════════════════════════════════════════════

def train_phase(phase_num: int, candles_data: list[dict], steps: int) -> dict:
    """단일 Phase를 독립적으로 훈련한다. (별도 프로세스에서 실행)"""
    import numpy as np

    phase_names = {
        1: "정책 기초 (높은 탐험)",
        2: "폭락 대응",
        3: "외부 데이터 활용",
        4: "보수적 전략 (위기 회피)",
        5: "DCA/손절 타이밍",
        6: "균형 전략 (최종)",
    }
    name = phase_names.get(phase_num, f"Phase {phase_num}")
    log(f"시작: {name}, {steps:,} 스텝", f"P{phase_num}")

    t0 = time.time()

    # Phase별 환경/모델 설정
    if phase_num == 1:
        # 높은 엔트로피로 기초 탐험
        env = create_env(candles_data, crash_injection=False)
        model = create_model(env, ent_coef=0.15, learning_rate=3e-4)

    elif phase_num == 2:
        # 폭락 시나리오 집중
        env = create_env(candles_data, crash_injection=True, crash_prob=0.05)
        model = create_model(env, ent_coef=0.10, learning_rate=2e-4)

    elif phase_num == 3:
        # 외부 데이터 활용 (현실적 시뮬레이션)
        env = create_env(candles_data, crash_injection=True, crash_prob=0.02)
        model = create_model(env, ent_coef=0.10, learning_rate=2e-4)

    elif phase_num == 4:
        # 보수적: 낮은 엔트로피, MDD 회피
        env = create_env(candles_data, crash_injection=True, crash_prob=0.04)
        model = create_model(env, ent_coef=0.05, learning_rate=1e-4, gamma=0.995)

    elif phase_num == 5:
        # DCA 전략: 중간 엔트로피
        env = create_env(candles_data, crash_injection=True, crash_prob=0.03, dca_reward=True)
        model = create_model(env, ent_coef=0.08, learning_rate=2e-4)

    elif phase_num == 6:
        # 균형 전략: 모든 기능 활성
        env = create_env(candles_data, crash_injection=True, crash_prob=0.02, regime_aware=True)
        model = create_model(env, ent_coef=0.10, learning_rate=2e-4, gamma=0.995)

    # 훈련 실행
    model.learn(total_timesteps=steps, progress_bar=False)

    # 저장
    save_path = str(MODEL_DIR / f"phase{phase_num}")
    model.save(save_path)

    # 평가
    metrics = evaluate_model(model, candles_data)
    elapsed = time.time() - t0

    log(f"완료: {name} — {elapsed:.0f}초, "
        f"수익률 {metrics['mean_profit']:+.2f}%, "
        f"거래 {metrics['mean_trades']:.0f}회, "
        f"최대수익 {metrics['max_profit']:+.2f}%", f"P{phase_num}")

    env.close()
    return {
        "phase": phase_num,
        "name": name,
        "elapsed": elapsed,
        "steps": steps,
        "metrics": metrics,
        "model_path": save_path,
    }


def main():
    log("╔══════════════════════════════════════════════════╗")
    log("║   병렬 RL 훈련 — 6개 Phase 동시 실행 (V2.2)      ║")
    log("╚══════════════════════════════════════════════════╝")

    start_time = time.time()

    # 데이터 로드 (메인 프로세스에서 1회)
    candles = load_data(days=180, interval="4h")
    log(f"데이터 로드 완료: {len(candles)}봉")

    if len(candles) < 100:
        log(f"[ERROR] 데이터 부족: {len(candles)}봉")
        return

    # Phase별 스텝 수
    phase_steps = {
        1: 500_000,
        2: 300_000,
        3: 300_000,
        4: 300_000,
        5: 300_000,
        6: 300_000,
    }

    log(f"총 스텝: {sum(phase_steps.values()):,} (6개 병렬)")
    log("─" * 50)

    # 6개 Phase 병렬 실행
    results = {}
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(train_phase, phase, candles, steps): phase
            for phase, steps in phase_steps.items()
        }

        for future in as_completed(futures):
            phase_num = futures[future]
            try:
                result = future.result()
                results[phase_num] = result
            except Exception as e:
                log(f"Phase {phase_num} 실패: {e}", f"P{phase_num}")
                results[phase_num] = {"phase": phase_num, "error": str(e)}

    # 결과 요약
    elapsed = time.time() - start_time
    log("")
    log("╔══════════════════════════════════════════════════╗")
    log("║              병렬 훈련 결과 요약                  ║")
    log("╚══════════════════════════════════════════════════╝")
    log(f"  총 소요: {elapsed / 60:.1f}분 (병렬)")

    best_phase = None
    best_profit = -999

    for phase_num in sorted(results.keys()):
        r = results[phase_num]
        if "error" in r:
            log(f"  Phase {phase_num}: 실패 — {r['error']}")
            continue

        m = r["metrics"]
        marker = ""
        if m["mean_profit"] > best_profit:
            best_profit = m["mean_profit"]
            best_phase = phase_num

        log(f"  Phase {phase_num} ({r['name']}): "
            f"수익률 {m['mean_profit']:+.2f}%, "
            f"거래 {m['mean_trades']:.0f}회, "
            f"최대 {m['max_profit']:+.2f}%, "
            f"보상 {m['mean_reward']:.1f}")

    # 최고 모델을 최종 모델로 복사
    if best_phase is not None:
        from stable_baselines3 import PPO
        best_path = str(MODEL_DIR / f"phase{best_phase}")
        best_model = PPO.load(best_path)

        final_path = str(MODEL_DIR / "final_v3")
        best_model.save(final_path)
        legacy_path = str(PROJECT_DIR / "data" / "rl_models" / "ppo_btc_v3.zip")
        best_model.save(legacy_path)

        log("")
        log(f"  ★ 최고 모델: Phase {best_phase} (수익률 {best_profit:+.2f}%)")
        log(f"  ★ 저장: {final_path}")

    # DB 기록
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env")

        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if url and key:
            headers = {
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            best_m = results.get(best_phase, {}).get("metrics", {})
            row = {
                "model_type": "ppo_v3_parallel",
                "training_steps": sum(phase_steps.values()),
                "reward_mean": best_m.get("mean_reward", 0),
                "reward_std": best_m.get("std_reward", 0),
                "notes": json.dumps({
                    "method": "parallel_6phase",
                    "best_phase": best_phase,
                    "elapsed_min": round(elapsed / 60, 1),
                    "results": {str(k): v.get("metrics", {}) for k, v in results.items()},
                }, ensure_ascii=False, default=str),
            }
            requests.post(f"{url}/rest/v1/rl_training_log", json=row, headers=headers, timeout=10)
            log("DB 기록 완료")
    except Exception as e:
        log(f"DB 기록 실패: {e}")

    log("═══ 병렬 훈련 완료 ═══")


if __name__ == "__main__":
    main()
