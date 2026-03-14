"""종합 RL 훈련 스크립트 — 6가지 훈련 갭 해결

6개 Phase를 순차 실행:
  Phase 1: 정책 붕괴 수정 — 높은 엔트로피 + 행동 다양성 보상으로 신규 모델 훈련
  Phase 2: 폭락/블랙스완 — 합성 급락 시나리오 주입 훈련
  Phase 3: 외부 데이터 — 현실적 FGI/뉴스/고래 시뮬레이션으로 재훈련
  Phase 4: 시장 국면별 — 상승/하락/횡보 데이터로 분류 후 개별 미세조정
  Phase 5: DCA/손절 — 단계적 진입/퇴출 보상 강화 훈련
  Phase 6: 주말/시간패턴 — 시간 기반 거래 패턴 학습

사용법:
  python scripts/train_comprehensive.py              # 전체 6 Phase (약 2~3시간)
  python scripts/train_comprehensive.py --phase 1    # Phase 1만
  python scripts/train_comprehensive.py --steps 100000  # 스텝 수 조정
"""

import io
import sys
import os
import json
import time
import argparse
import copy
from pathlib import Path
from datetime import datetime, timezone, timedelta

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "rl_models" / "v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = PROJECT_DIR / "logs" / "train_comprehensive.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    ts = datetime.now(KST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_data(days: int = 180, interval: str = "1h") -> list[dict]:
    """Upbit에서 캔들 데이터를 로드한다."""
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    loader = HistoricalDataLoader()
    raw = loader.load_candles(days=days, interval=interval)
    candles = loader.compute_indicators(raw)
    log(f"데이터 로드 완료: {len(candles)}봉 ({days}일 {interval})")
    return candles


def evaluate_model(model, env_class, candles, n_episodes=10, **env_kwargs):
    """모델 성능을 평가한다."""
    from stable_baselines3.common.evaluation import evaluate_policy

    eval_env = env_class(candles=candles, crash_injection=False, **env_kwargs)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_episodes)

    # 상세 메트릭
    profits = []
    trades = []
    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        profits.append(info.get("profit_pct", 0))
        trades.append(info.get("trade_count", 0))

    eval_env.close()

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_profit": float(np.mean(profits)),
        "mean_trades": float(np.mean(trades)),
        "min_profit": float(np.min(profits)),
        "max_profit": float(np.max(profits)),
    }


def classify_candles_by_regime(candles: list[dict]) -> dict[str, list[dict]]:
    """캔들 데이터를 시장 국면별로 분류한다."""
    from rl_hybrid.rl.environment_v2 import classify_regime

    regimes = {"bull": [], "bear": [], "sideways": [], "volatile": []}
    window = 100  # 100봉씩 구간 분류

    for i in range(0, len(candles) - window, window // 2):
        chunk = candles[i:i + window]
        regime = classify_regime(candles, i + window // 2)
        regimes[regime].extend(chunk)

    for k, v in regimes.items():
        # 중복 제거 (overlapping windows)
        seen = set()
        unique = []
        for c in v:
            ts = c.get("timestamp", id(c))
            if ts not in seen:
                seen.add(ts)
                unique.append(c)
        regimes[k] = unique

    return regimes


# ═══════════════════════════════════════════════════
# Phase 1: 정책 붕괴 수정
# ═══════════════════════════════════════════════════

def phase1_anti_collapse(candles: list[dict], steps: int = 300_000):
    """높은 엔트로피 + 행동 다양성 보상으로 새 모델을 처음부터 훈련."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2

    log("═══ Phase 1: 정책 붕괴 수정 ═══")
    log(f"  엔트로피 0.15, 행동 다양성 보상, {steps:,} 스텝")

    # 훈련/평가 환경
    train_env = BitcoinTradingEnvV2(
        candles=candles,
        crash_injection=False,  # Phase 1에서는 정상 데이터로
        anti_collapse=True,
        dca_reward=True,
    )

    eval_candles = candles[int(len(candles) * 0.8):]
    eval_env = BitcoinTradingEnvV2(
        candles=eval_candles,
        crash_injection=False,
        anti_collapse=False,
    )

    # 새 모델 (높은 엔트로피)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        ent_coef=0.15,          # 높은 엔트로피 → 탐험 장려
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},  # 더 큰 네트워크
        verbose=0,
    )

    # 평가 콜백
    best_dir = str(MODEL_DIR / "phase1_best")
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=max(steps // 10, 10000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=eval_callback, progress_bar=True)
    elapsed = time.time() - t0

    # 저장
    save_path = str(MODEL_DIR / "phase1_anti_collapse")
    model.save(save_path)

    # 평가
    metrics = evaluate_model(model, BitcoinTradingEnvV2, candles)
    log(f"  Phase 1 완료 ({elapsed:.0f}초)")
    log(f"  수익률: {metrics['mean_profit']:+.2f}%, 거래: {metrics['mean_trades']:.0f}회")
    log(f"  보상: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")

    train_env.close()
    eval_env.close()

    return model, metrics


# ═══════════════════════════════════════════════════
# Phase 2: 폭락/블랙스완 훈련
# ═══════════════════════════════════════════════════

def phase2_crash_training(model, candles: list[dict], steps: int = 200_000):
    """합성 폭락 시나리오를 주입하여 위기 대응을 학습."""
    from stable_baselines3 import PPO
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2

    log("═══ Phase 2: 폭락/블랙스완 훈련 ═══")

    # 폭락 주입 환경 (높은 확률)
    crash_env = BitcoinTradingEnvV2(
        candles=candles,
        crash_injection=True,
        crash_prob=0.05,  # 5%씩 폭락 주입 (강화)
        anti_collapse=True,
        dca_reward=True,
    )

    model.set_env(crash_env)
    # 엔트로피 약간 낮추고 학습률 낮춰서 미세조정
    model.ent_coef = 0.10
    model.learning_rate = 1e-4

    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=True)
    elapsed = time.time() - t0

    save_path = str(MODEL_DIR / "phase2_crash")
    model.save(save_path)

    metrics = evaluate_model(model, BitcoinTradingEnvV2, candles)
    log(f"  Phase 2 완료 ({elapsed:.0f}초)")
    log(f"  수익률: {metrics['mean_profit']:+.2f}%, 거래: {metrics['mean_trades']:.0f}회")

    # 폭락 데이터로 별도 평가
    crash_metrics = evaluate_model(
        model, BitcoinTradingEnvV2, candles,
        crash_injection=True, crash_prob=0.08,
    )
    log(f"  폭락 시나리오 수익률: {crash_metrics['mean_profit']:+.2f}%")

    crash_env.close()
    return model, metrics


# ═══════════════════════════════════════════════════
# Phase 3: 현실적 외부 데이터 훈련
# ═══════════════════════════════════════════════════

def phase3_external_data(model, candles: list[dict], steps: int = 200_000):
    """현실적 FGI/뉴스/고래 시뮬레이션으로 재훈련."""
    from stable_baselines3 import PPO
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2

    log("═══ Phase 3: 현실적 외부 데이터 훈련 ═══")
    log("  상관 모델 기반 FGI/뉴스/펀딩/김프 시뮬레이션")

    # V2 환경은 이미 현실적 외부 데이터를 사용
    env = BitcoinTradingEnvV2(
        candles=candles,
        crash_injection=False,
        anti_collapse=True,
        dca_reward=True,
    )

    model.set_env(env)
    model.ent_coef = 0.08
    model.learning_rate = 5e-5

    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=True)
    elapsed = time.time() - t0

    save_path = str(MODEL_DIR / "phase3_external")
    model.save(save_path)

    metrics = evaluate_model(model, BitcoinTradingEnvV2, candles)
    log(f"  Phase 3 완료 ({elapsed:.0f}초)")
    log(f"  수익률: {metrics['mean_profit']:+.2f}%, 거래: {metrics['mean_trades']:.0f}회")

    env.close()
    return model, metrics


# ═══════════════════════════════════════════════════
# Phase 4: 시장 국면별 미세조정
# ═══════════════════════════════════════════════════

def phase4_regime_specific(model, candles: list[dict], steps_per_regime: int = 100_000):
    """상승/하락/횡보 데이터로 분류 후 개별 미세조정."""
    from stable_baselines3 import PPO
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2

    log("═══ Phase 4: 시장 국면별 훈련 ═══")

    regimes = classify_candles_by_regime(candles)

    for regime_name, regime_candles in regimes.items():
        if len(regime_candles) < 100:
            log(f"  {regime_name}: 데이터 부족 ({len(regime_candles)}봉), 건너뜀")
            continue

        log(f"  {regime_name} 국면: {len(regime_candles)}봉, {steps_per_regime:,} 스텝")

        env = BitcoinTradingEnvV2(
            candles=regime_candles,
            crash_injection=(regime_name == "bear"),  # 하락장에만 폭락 추가
            anti_collapse=True,
            dca_reward=True,
        )

        # 국면별 모델 복사 + 미세조정
        regime_model = PPO.load(str(MODEL_DIR / "phase3_external"), env=env)
        regime_model.ent_coef = 0.05
        regime_model.learning_rate = 3e-5

        regime_model.learn(total_timesteps=steps_per_regime, progress_bar=True)

        save_path = str(MODEL_DIR / f"phase4_{regime_name}")
        regime_model.save(save_path)

        metrics = evaluate_model(regime_model, BitcoinTradingEnvV2, regime_candles)
        log(f"    {regime_name} 수익률: {metrics['mean_profit']:+.2f}%, "
            f"거래: {metrics['mean_trades']:.0f}회")

        env.close()

    # 메인 모델은 전체 데이터로 추가 학습
    full_env = BitcoinTradingEnvV2(
        candles=candles,
        crash_injection=True,
        crash_prob=0.02,
        anti_collapse=True,
        regime_aware=True,
    )
    model.set_env(full_env)
    model.ent_coef = 0.06
    model.learning_rate = 3e-5

    model.learn(total_timesteps=steps_per_regime, progress_bar=True)

    save_path = str(MODEL_DIR / "phase4_regime_aware")
    model.save(save_path)

    metrics = evaluate_model(model, BitcoinTradingEnvV2, candles)
    log(f"  Phase 4 전체 수익률: {metrics['mean_profit']:+.2f}%, 거래: {metrics['mean_trades']:.0f}회")

    full_env.close()
    return model, metrics


# ═══════════════════════════════════════════════════
# Phase 5: DCA/손절 타이밍 훈련
# ═══════════════════════════════════════════════════

def phase5_dca_stoploss(model, candles: list[dict], steps: int = 150_000):
    """단계적 진입/퇴출 보상을 강화하여 DCA 타이밍 학습."""
    from stable_baselines3 import PPO
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2

    log("═══ Phase 5: DCA/손절 타이밍 훈련 ═══")

    env = BitcoinTradingEnvV2(
        candles=candles,
        crash_injection=True,
        crash_prob=0.03,
        anti_collapse=True,
        dca_reward=True,  # DCA 보상 강화
    )

    model.set_env(env)
    model.ent_coef = 0.06
    model.learning_rate = 3e-5

    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=True)
    elapsed = time.time() - t0

    save_path = str(MODEL_DIR / "phase5_dca")
    model.save(save_path)

    metrics = evaluate_model(model, BitcoinTradingEnvV2, candles)
    log(f"  Phase 5 완료 ({elapsed:.0f}초)")
    log(f"  수익률: {metrics['mean_profit']:+.2f}%, 거래: {metrics['mean_trades']:.0f}회")

    env.close()
    return model, metrics


# ═══════════════════════════════════════════════════
# Phase 6: 주말/시간 패턴 훈련
# ═══════════════════════════════════════════════════

def phase6_time_patterns(model, candles: list[dict], steps: int = 150_000):
    """주말/야간 거래량 감소 패턴 학습."""
    from stable_baselines3 import PPO
    from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2

    log("═══ Phase 6: 주말/시간 패턴 훈련 ═══")

    # 전체 기능 활성화된 최종 환경
    env = BitcoinTradingEnvV2(
        candles=candles,
        crash_injection=True,
        crash_prob=0.02,
        anti_collapse=True,
        dca_reward=True,
        regime_aware=True,
    )

    model.set_env(env)
    model.ent_coef = 0.05
    model.learning_rate = 2e-5

    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=True)
    elapsed = time.time() - t0

    # 최종 모델 저장
    final_path = str(MODEL_DIR / "final_v2")
    model.save(final_path)

    # 기존 모델 디렉토리에도 복사 (실전 연동)
    legacy_path = str(PROJECT_DIR / "data" / "rl_models" / "ppo_btc_v2.zip")
    model.save(legacy_path)

    metrics = evaluate_model(model, BitcoinTradingEnvV2, candles)
    log(f"  Phase 6 완료 ({elapsed:.0f}초)")
    log(f"  최종 수익률: {metrics['mean_profit']:+.2f}%, 거래: {metrics['mean_trades']:.0f}회")
    log(f"  최종 모델 저장: {final_path}")

    env.close()
    return model, metrics


# ═══════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════

def record_to_db(phase_results: dict):
    """훈련 결과를 DB에 기록한다."""
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env")

        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        row = {
            "model_type": "ppo_v2_comprehensive",
            "training_steps": phase_results.get("total_steps", 0),
            "reward_mean": phase_results.get("final_reward", 0),
            "reward_std": phase_results.get("final_reward_std", 0),
            "notes": json.dumps(phase_results, ensure_ascii=False, default=str),
        }

        requests.post(
            f"{url}/rest/v1/rl_training_log",
            json=row,
            headers=headers,
            timeout=10,
        )
        log("DB 기록 완료")
    except Exception as e:
        log(f"DB 기록 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="종합 RL 훈련 (6 Phase)")
    parser.add_argument("--phase", type=int, default=0, help="특정 Phase만 실행 (1-6, 0=전체)")
    parser.add_argument("--steps", type=int, default=0, help="Phase당 스텝 수 오버라이드")
    parser.add_argument("--days", type=int, default=180, help="훈련 데이터 일수")
    parser.add_argument("--interval", type=str, default="4h", help="캔들 간격 (1h/4h)")
    args = parser.parse_args()

    log("╔═══════════════════════════════════════════╗")
    log("║   종합 RL 훈련 — 6가지 훈련 갭 해결       ║")
    log("╚═══════════════════════════════════════════╝")

    start_time = time.time()

    # 데이터 로드
    candles = load_data(days=args.days, interval=args.interval)
    if len(candles) < 100:
        log(f"[ERROR] 데이터 부족: {len(candles)}봉 (최소 100봉 필요)")
        return

    results = {}
    total_steps = 0
    model = None

    # Phase별 기본 스텝 수
    default_steps = {
        1: 300_000,
        2: 200_000,
        3: 200_000,
        4: 100_000,  # per regime
        5: 150_000,
        6: 150_000,
    }

    run_phase = args.phase  # 0이면 전체

    # Phase 1
    if run_phase in (0, 1):
        s = args.steps or default_steps[1]
        model, m = phase1_anti_collapse(candles, steps=s)
        results["phase1"] = m
        total_steps += s

    # Phase 2
    if run_phase in (0, 2):
        s = args.steps or default_steps[2]
        if model is None:
            from stable_baselines3 import PPO
            from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
            p1_path = str(MODEL_DIR / "phase1_anti_collapse.zip")
            if Path(p1_path).exists():
                env = BitcoinTradingEnvV2(candles=candles)
                model = PPO.load(p1_path, env=env)
            else:
                log("[WARN] Phase 1 모델 없음, Phase 1부터 실행 필요")
                return
        model, m = phase2_crash_training(model, candles, steps=s)
        results["phase2"] = m
        total_steps += s

    # Phase 3
    if run_phase in (0, 3):
        s = args.steps or default_steps[3]
        if model is None:
            from stable_baselines3 import PPO
            from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
            p2_path = str(MODEL_DIR / "phase2_crash.zip")
            if Path(p2_path).exists():
                env = BitcoinTradingEnvV2(candles=candles)
                model = PPO.load(p2_path, env=env)
            else:
                log("[WARN] 이전 Phase 모델 없음")
                return
        model, m = phase3_external_data(model, candles, steps=s)
        results["phase3"] = m
        total_steps += s

    # Phase 4
    if run_phase in (0, 4):
        s = args.steps or default_steps[4]
        if model is None:
            from stable_baselines3 import PPO
            from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
            p3_path = str(MODEL_DIR / "phase3_external.zip")
            if Path(p3_path).exists():
                env = BitcoinTradingEnvV2(candles=candles)
                model = PPO.load(p3_path, env=env)
            else:
                log("[WARN] 이전 Phase 모델 없음")
                return
        model, m = phase4_regime_specific(model, candles, steps_per_regime=s)
        results["phase4"] = m
        total_steps += s * 5  # 4개 국면 + 전체

    # Phase 5
    if run_phase in (0, 5):
        s = args.steps or default_steps[5]
        if model is None:
            from stable_baselines3 import PPO
            from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
            p4_path = str(MODEL_DIR / "phase4_regime_aware.zip")
            if Path(p4_path).exists():
                env = BitcoinTradingEnvV2(candles=candles)
                model = PPO.load(p4_path, env=env)
            else:
                log("[WARN] 이전 Phase 모델 없음")
                return
        model, m = phase5_dca_stoploss(model, candles, steps=s)
        results["phase5"] = m
        total_steps += s

    # Phase 6
    if run_phase in (0, 6):
        s = args.steps or default_steps[6]
        if model is None:
            from stable_baselines3 import PPO
            from rl_hybrid.rl.environment_v2 import BitcoinTradingEnvV2
            p5_path = str(MODEL_DIR / "phase5_dca.zip")
            if Path(p5_path).exists():
                env = BitcoinTradingEnvV2(candles=candles)
                model = PPO.load(p5_path, env=env)
            else:
                log("[WARN] 이전 Phase 모델 없음")
                return
        model, m = phase6_time_patterns(model, candles, steps=s)
        results["phase6"] = m
        total_steps += s

    # 결과 요약
    elapsed = time.time() - start_time
    log("")
    log("╔═══════════════════════════════════════════╗")
    log("║             훈련 결과 요약                 ║")
    log("╚═══════════════════════════════════════════╝")
    log(f"  총 소요: {elapsed / 60:.1f}분")
    log(f"  총 스텝: {total_steps:,}")

    for phase_name, m in results.items():
        log(f"  {phase_name}: 수익률 {m['mean_profit']:+.2f}%, "
            f"거래 {m['mean_trades']:.0f}회, "
            f"보상 {m['mean_reward']:.3f}")

    # DB 기록
    final_metrics = results.get("phase6") or results.get(list(results.keys())[-1]) if results else {}
    record_to_db({
        "total_steps": total_steps,
        "elapsed_min": round(elapsed / 60, 1),
        "phases": list(results.keys()),
        "final_reward": final_metrics.get("mean_reward", 0),
        "final_reward_std": final_metrics.get("std_reward", 0),
        "final_profit": final_metrics.get("mean_profit", 0),
        "final_trades": final_metrics.get("mean_trades", 0),
        "phase_results": results,
    })

    log("═══ 훈련 완료 ═══")


if __name__ == "__main__":
    main()
