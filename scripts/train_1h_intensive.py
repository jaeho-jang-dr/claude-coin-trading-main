#!/usr/bin/env python3
"""
RL 1시간 집중 훈련 스크립트

5-Phase 커리큘럼 학습 + 멀티 리워드 비교 + 스트레스 테스트
총 ~500K steps, 약 60분 소요 (CPU 기준)

Phase 1: 워밍업 — v6 Sharpe, 30일, 50K steps (~6분)
Phase 2: 트렌드 학습 — v7 Trend, 60일, 80K steps (~10분)
Phase 3: 하이브리드 메인 — v8 Hybrid, 90일, 150K steps (~18분)
Phase 4: 풀데이터 심화 — v8 Hybrid, 180일, 150K steps (~18분)
Phase 5: 스트레스 테스트 — v8 Hybrid, 4h 캔들, 70K steps (~8분)

사용법:
  python scripts/train_1h_intensive.py                    # 전체 5-Phase
  python scripts/train_1h_intensive.py --phase 3          # 특정 Phase만
  python scripts/train_1h_intensive.py --quick             # 축소판 (~20분)
  python scripts/train_1h_intensive.py --reward v8         # 리워드 고정
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_1h")

KST = timezone(timedelta(hours=9))

# ─── v8.2 최적 하이퍼파라미터 (정책 붕괴 방지) ───
BEST_HP = {
    "lr": 3e-4,
    "ent_coef": 0.08,       # 0.02→0.05→0.08: 상수행동 탈출 강화
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

BALANCE = 10_000_000

# ─── Phase 정의 ───
PHASES = [
    {
        "id": 1,
        "name": "워밍업 (v6 Sharpe)",
        "reward": "v6",
        "days": 30,
        "interval": "1h",
        "steps": 50_000,
        "lr_ratio": 1.0,
        "ent_coef_override": None,
        "description": "안정적 Sharpe 보상으로 기본 정책 학습",
    },
    {
        "id": 2,
        "name": "트렌드 학습 (v7 Trend)",
        "reward": "v7",
        "days": 60,
        "interval": "1h",
        "steps": 80_000,
        "lr_ratio": 0.7,
        "ent_coef_override": 0.015,
        "description": "트렌드 팔로잉 보상으로 방향성 학습",
    },
    {
        "id": 3,
        "name": "하이브리드 메인 (v8 Hybrid)",
        "reward": "v8",
        "days": 90,
        "interval": "1h",
        "steps": 150_000,
        "lr_ratio": 0.5,
        "ent_coef_override": None,
        "description": "Sharpe 30% + Trend 15% + PnL 30% 하이브리드",
    },
    {
        "id": 4,
        "name": "풀데이터 심화 (v8 180일)",
        "reward": "v8",
        "days": 180,
        "interval": "1h",
        "steps": 150_000,
        "lr_ratio": 0.3,
        "ent_coef_override": None,
        "description": "6개월 전체 데이터로 장기 패턴 학습",
    },
    {
        "id": 5,
        "name": "스트레스 테스트 (4h 캔들)",
        "reward": "v8",
        "days": 180,
        "interval": "4h",
        "steps": 70_000,
        "lr_ratio": 0.2,
        "ent_coef_override": 0.01,
        "description": "저빈도 캔들로 미세조정 + 과적합 방지",
    },
]

QUICK_PHASES = [
    {**PHASES[0], "steps": 20_000},
    {**PHASES[2], "steps": 50_000, "days": 60},
    {**PHASES[4], "steps": 30_000},
]


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def evaluate(model, env, episodes=10) -> dict:
    """모델 평가 — 여러 에피소드 평균"""
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
        "win_rate": float(np.mean([s.get("win_rate", 0) for s in all_stats])),
    }


def multi_reward_eval(model, candles, balance=BALANCE) -> dict:
    """3가지 리워드 함수로 동시 평가"""
    results = {}
    for rv in ("v6", "v7", "v8"):
        from rl_hybrid.rl.environment import BitcoinTradingEnv
        env = BitcoinTradingEnv(candles=candles, initial_balance=balance, reward_version=rv)
        stats = evaluate(model, env, episodes=5)
        results[rv] = stats
    return results


def stress_test(model, candles, balance=BALANCE) -> dict:
    """시장 레짐별 스트레스 테스트"""
    from rl_hybrid.rl.environment import BitcoinTradingEnv

    n = len(candles)
    quarter = n // 4
    regimes = {
        "recent": candles[-quarter:],
        "mid": candles[quarter:quarter*2],
        "early": candles[:quarter],
        "full": candles,
    }

    results = {}
    for name, data in regimes.items():
        if len(data) < 50:
            continue
        env = BitcoinTradingEnv(candles=data, initial_balance=balance, reward_version="v8")
        stats = evaluate(model, env, episodes=3)
        results[name] = stats

    # 변동성 구간 감지
    if n > 100:
        prices = [c["close"] for c in candles]
        returns = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = [np.std(returns[max(0,i-24):i+1]) for i in range(len(returns))]

        # 상위 25% 고변동 구간 추출
        vol_threshold = np.percentile(volatility, 75)
        high_vol_indices = [i for i, v in enumerate(volatility) if v >= vol_threshold]
        if len(high_vol_indices) > 50:
            start_idx = high_vol_indices[0]
            end_idx = min(high_vol_indices[-1] + 1, n)
            high_vol_candles = candles[start_idx:end_idx]
            if len(high_vol_candles) >= 50:
                env = BitcoinTradingEnv(candles=high_vol_candles, initial_balance=balance, reward_version="v8")
                results["high_volatility"] = evaluate(model, env, episodes=3)

    return results


def run_phase(phase: dict, model=None, force_reward: str = None) -> tuple:
    """단일 Phase 실행

    Returns:
        (model, result_dict)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    from rl_hybrid.rl.environment import BitcoinTradingEnv
    from rl_hybrid.rl.policy import MODEL_DIR

    phase_id = phase["id"]
    reward_ver = force_reward or phase["reward"]
    start = time.time()

    logger.info(f"\n{'═' * 60}")
    logger.info(f"  Phase {phase_id}: {phase['name']}")
    logger.info(f"  {phase['description']}")
    logger.info(f"  reward={reward_ver}, days={phase['days']}, interval={phase['interval']}")
    logger.info(f"  steps={phase['steps']:,}, lr_ratio={phase['lr_ratio']}")
    logger.info(f"{'═' * 60}")

    # 데이터 로드
    loader = HistoricalDataLoader()
    try:
        candles = loader.compute_indicators(
            loader.load_candles(days=phase["days"], interval=phase["interval"])
        )
    except Exception as e:
        logger.error(f"Phase {phase_id} 데이터 로드 실패: {e}")
        return model, {"phase": phase_id, "error": str(e)}

    if len(candles) < 100:
        logger.warning(f"Phase {phase_id} 데이터 부족: {len(candles)}개")
        return model, {"phase": phase_id, "error": "insufficient_data", "candles": len(candles)}

    split = int(len(candles) * 0.8)
    train_c = candles[:split]
    eval_c = candles[split:]

    logger.info(f"  데이터: {len(candles)}개 (학습 {len(train_c)}, 평가 {len(eval_c)})")

    train_env = BitcoinTradingEnv(
        candles=train_c, initial_balance=BALANCE, reward_version=reward_ver
    )
    eval_env = BitcoinTradingEnv(
        candles=eval_c, initial_balance=BALANCE, reward_version=reward_ver
    )

    # 모델 생성 또는 계속 학습
    actual_lr = BEST_HP["lr"] * phase["lr_ratio"]
    ent_coef = phase["ent_coef_override"] or BEST_HP["ent_coef"]

    def _create_fresh_model(env, lr, ec):
        """새 모델 생성 (정책 붕괴 탈출용)"""
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(lr),
            n_steps=BEST_HP["n_steps"],
            batch_size=BEST_HP["batch_size"],
            n_epochs=BEST_HP["n_epochs"],
            gamma=BEST_HP["gamma"],
            gae_lambda=BEST_HP["gae_lambda"],
            clip_range=BEST_HP["clip_range"],
            ent_coef=ec,
            vf_coef=BEST_HP["vf_coef"],
            max_grad_norm=BEST_HP["max_grad_norm"],
            verbose=1,
            policy_kwargs={
                "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
            },
        )

    if model is None:
        # Phase 1: 기존 모델 로드 시도
        model_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
        if not os.path.exists(model_path + ".zip"):
            model_path = os.path.join(MODEL_DIR, "v3_final")

        if os.path.exists(model_path + ".zip"):
            logger.info(f"  기존 모델 로드: {model_path}")
            model = PPO.load(model_path, env=train_env)

            # 정책 붕괴 감지: trades<=2면 새로 시작
            collapse_check = evaluate(model, eval_env, episodes=3)
            if collapse_check["trades"] <= 2:
                logger.warning(f"  정책 붕괴 감지 (trades={collapse_check['trades']:.0f}) → 새 모델로 재시작")
                model = _create_fresh_model(train_env, actual_lr, ent_coef)
        else:
            logger.info("  새 모델 생성")
            model = _create_fresh_model(train_env, actual_lr, ent_coef)
    else:
        # 이전 Phase에서 이어받은 모델도 붕괴 체크
        model.set_env(train_env)
        collapse_check = evaluate(model, eval_env, episodes=3)
        if collapse_check["trades"] <= 2 and phase_id <= 2:
            logger.warning(f"  정책 붕괴 지속 (trades={collapse_check['trades']:.0f}) → 새 모델로 재시작")
            model = _create_fresh_model(train_env, actual_lr, ent_coef)

    # LR / 엔트로피 조정
    model.learning_rate = linear_schedule(actual_lr)
    model.ent_coef = ent_coef

    # 베이스라인 평가
    logger.info("  베이스라인 평가...")
    baseline = evaluate(model, eval_env, episodes=8)
    logger.info(
        f"  Baseline: return={baseline['return_pct']:.2f}%, "
        f"sharpe={baseline['sharpe']:.3f}, mdd={baseline['mdd']:.2%}, "
        f"trades={baseline['trades']:.0f}"
    )

    # 학습
    save_dir = os.path.join(MODEL_DIR, f"1h_phase{phase_id}")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=max(phase["steps"] // 5, 5000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    logger.info(f"  학습: {phase['steps']:,} steps, LR={actual_lr:.1e}, ent={ent_coef}")
    model.learn(
        total_timesteps=phase["steps"],
        callback=eval_cb,
        reset_num_timesteps=True,
    )

    # 학습 후 평가
    logger.info("  학습 후 평가...")
    new_stats = evaluate(model, eval_env, episodes=8)
    logger.info(
        f"  After: return={new_stats['return_pct']:.2f}%, "
        f"sharpe={new_stats['sharpe']:.3f}, mdd={new_stats['mdd']:.2%}, "
        f"trades={new_stats['trades']:.0f}"
    )

    # 멀티 리워드 비교 (Phase 3, 5에서만)
    multi_reward = None
    if phase_id in (3, 5):
        logger.info("  멀티 리워드 비교 평가...")
        multi_reward = multi_reward_eval(model, eval_c)
        for rv, stats in multi_reward.items():
            logger.info(f"    {rv}: return={stats['return_pct']:.2f}%, sharpe={stats['sharpe']:.3f}")

    # 스트레스 테스트 (Phase 5에서만)
    stress = None
    if phase_id == 5:
        logger.info("  스트레스 테스트...")
        stress = stress_test(model, candles)
        for regime, stats in stress.items():
            logger.info(f"    {regime}: return={stats['return_pct']:.2f}%, sharpe={stats['sharpe']:.3f}")

    elapsed = time.time() - start
    improved = new_stats["sharpe"] > baseline["sharpe"] - 0.05

    result = {
        "phase": phase_id,
        "name": phase["name"],
        "reward_version": reward_ver,
        "baseline": baseline,
        "new_stats": new_stats,
        "improved": improved,
        "elapsed_sec": elapsed,
        "steps": phase["steps"],
        "days": phase["days"],
        "interval": phase["interval"],
        "lr": actual_lr,
        "ent_coef": ent_coef,
        "candles_count": len(candles),
        "multi_reward": multi_reward,
        "stress_test": stress,
    }

    sharpe_delta = new_stats["sharpe"] - baseline["sharpe"]
    status = "IMPROVED" if improved else "ROLLBACK"
    logger.info(
        f"  Phase {phase_id} 완료: {status} "
        f"(sharpe {baseline['sharpe']:.3f} → {new_stats['sharpe']:.3f}, "
        f"delta={sharpe_delta:+.4f}, {elapsed:.0f}초)"
    )

    return model, result


def run_full_training(
    phases: list = None,
    single_phase: int = None,
    force_reward: str = None,
) -> dict:
    """전체 1시간 집중 훈련 실행"""
    from rl_hybrid.rl.policy import MODEL_DIR

    total_start = time.time()
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")

    if phases is None:
        phases = PHASES

    if single_phase:
        phases = [p for p in phases if p["id"] == single_phase]
        if not phases:
            logger.error(f"Phase {single_phase} 없음")
            return {"error": f"Phase {single_phase} not found"}

    total_steps = sum(p["steps"] for p in phases)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  RL 1시간 집중 훈련 시작")
    logger.info(f"  {len(phases)} Phases, 총 {total_steps:,} steps")
    logger.info(f"  시작: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}")
    logger.info(f"{'=' * 60}")

    model = None
    phase_results = []
    best_sharpe = -float("inf")
    best_phase = None

    for phase in phases:
        model, result = run_phase(phase, model=model, force_reward=force_reward)
        phase_results.append(result)

        if "error" not in result:
            sharpe = result["new_stats"]["sharpe"]
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_phase = phase["id"]

    total_elapsed = time.time() - total_start

    # 최종 모델 저장
    if model is not None:
        version_id = f"sb3_1h_{ts}"
        model.save(os.path.join(MODEL_DIR, "ppo_btc_latest"))

        version_dir = os.path.join(MODEL_DIR, "versions", version_id)
        os.makedirs(version_dir, exist_ok=True)
        model.save(os.path.join(version_dir, "model"))

        # 레지스트리 업데이트
        registry_path = os.path.join(MODEL_DIR, "sb3_registry.json")
        registry = (
            json.loads(open(registry_path, encoding="utf-8").read())
            if os.path.exists(registry_path)
            else {"current": None, "versions": [], "rollbacks": 0}
        )

        final_phase = phase_results[-1] if phase_results else {}
        final_stats = final_phase.get("new_stats", {})

        registry["versions"].append({
            "version_id": version_id,
            "metrics": final_stats,
            "tier": "1h_intensive",
            "phases_completed": len([r for r in phase_results if "error" not in r]),
            "total_steps": total_steps,
            "created_at": datetime.now(KST).isoformat(),
        })
        registry["current"] = version_id

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        logger.info(f"\n  최종 모델: {version_id}")
    else:
        version_id = None

    # 종합 리포트
    summary = {
        "version_id": version_id,
        "phases": phase_results,
        "total_steps": total_steps,
        "total_elapsed_sec": total_elapsed,
        "total_elapsed_min": total_elapsed / 60,
        "best_sharpe": best_sharpe,
        "best_phase": best_phase,
        "timestamp": datetime.now(KST).isoformat(),
    }

    # 결과 저장
    results_dir = PROJECT_DIR / "data" / "training_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"1h_intensive_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, indent=2, ensure_ascii=False)

    # DB 저장
    _save_to_db(summary)

    # 텔레그램 알림
    _notify_result(summary)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  1시간 집중 훈련 완료!")
    logger.info(f"  총 시간: {total_elapsed/60:.1f}분")
    logger.info(f"  최고 Sharpe: {best_sharpe:.4f} (Phase {best_phase})")
    logger.info(f"  결과: {result_path}")
    logger.info(f"{'=' * 60}")

    return summary


def _save_to_db(summary: dict):
    """Supabase에 학습 결과 저장 (반드시 저장 — 실패 시 로컬 백업)"""
    import requests as req

    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        logger.error("DB 저장 불가: SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY 미설정")
        _save_db_fallback(summary)
        return

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    saved = 0
    failed = 0

    for phase_result in summary.get("phases", []):
        if "error" in phase_result:
            continue

        baseline = phase_result.get("baseline", {})
        new_stats = phase_result.get("new_stats", {})

        row = {
            "created_at": summary.get("timestamp"),
            "tier": "1h_intensive",
            "tier_name": f"Phase {phase_result['phase']}: {phase_result['name']}",
            "steps": phase_result["steps"],
            "data_days": phase_result["days"],
            "lr_ratio": phase_result["lr"] / BEST_HP["lr"],
            "candles_count": phase_result.get("candles_count"),
            "baseline_return_pct": baseline.get("return_pct"),
            "baseline_sharpe": baseline.get("sharpe"),
            "baseline_mdd": baseline.get("mdd"),
            "baseline_trades": baseline.get("trades"),
            "new_return_pct": new_stats.get("return_pct"),
            "new_sharpe": new_stats.get("sharpe"),
            "new_mdd": new_stats.get("mdd"),
            "new_trades": new_stats.get("trades"),
            "improved": phase_result.get("improved", False),
            "version_id": summary.get("version_id"),
            "elapsed_sec": phase_result.get("elapsed_sec"),
            "rollback": not phase_result.get("improved", True),
        }

        try:
            r = req.post(
                f"{url}/rest/v1/rl_training_log",
                headers=headers,
                json=row,
                timeout=10,
            )
            if r.status_code in (200, 201):
                saved += 1
            else:
                logger.error(f"DB 저장 실패 Phase {phase_result['phase']}: {r.status_code} {r.text[:200]}")
                failed += 1
        except Exception as e:
            logger.error(f"DB 저장 예외 Phase {phase_result['phase']}: {e}")
            failed += 1

    if saved > 0:
        logger.info(f"DB 저장 완료: {saved}건 성공, {failed}건 실패")
    if failed > 0:
        _save_db_fallback(summary)


def _save_db_fallback(summary: dict):
    """DB 저장 실패 시 로컬 JSON 백업"""
    fallback_dir = PROJECT_DIR / "data" / "db_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    path = fallback_dir / f"training_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, indent=2, ensure_ascii=False)
    logger.warning(f"DB 실패 → 로컬 백업 저장: {path}")


def _notify_result(summary: dict):
    """텔레그램으로 훈련 결과 알림"""
    try:
        import subprocess
        phases = summary.get("phases", [])
        good = sum(1 for p in phases if p.get("improved"))
        total = len([p for p in phases if "error" not in p])

        msg = f"[RL 1H Training] {good}/{total} Phases 개선"

        lines = [
            f"총 시간: {summary['total_elapsed_min']:.1f}분",
            f"총 Steps: {summary['total_steps']:,}",
            f"최고 Sharpe: {summary['best_sharpe']:.4f} (Phase {summary['best_phase']})",
            "",
        ]

        for p in phases:
            if "error" in p:
                lines.append(f"Phase {p['phase']}: ERROR - {p['error']}")
                continue
            status = "v" if p.get("improved") else "x"
            b = p["baseline"]
            n = p["new_stats"]
            lines.append(
                f"[{status}] P{p['phase']} {p['name'][:15]}: "
                f"sharpe {b['sharpe']:.3f}->{n['sharpe']:.3f}, "
                f"ret {b['return_pct']:.1f}%->{n['return_pct']:.1f}%"
            )

        if summary.get("version_id"):
            lines.append(f"\nModel: {summary['version_id']}")

        detail = "\n".join(lines)
        subprocess.run(
            [sys.executable, "scripts/notify_telegram.py", "trade", msg, detail],
            cwd=str(PROJECT_DIR),
            check=False,
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="RL 1시간 집중 훈련")
    parser.add_argument("--phase", type=int, help="특정 Phase만 실행 (1-5)")
    parser.add_argument("--quick", action="store_true", help="축소판 (~20분)")
    parser.add_argument("--reward", choices=["v6", "v7", "v8"], help="리워드 함수 고정")
    args = parser.parse_args()

    phases = QUICK_PHASES if args.quick else None
    result = run_full_training(
        phases=phases,
        single_phase=args.phase,
        force_reward=args.reward,
    )

    # stdout에 JSON 출력
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    print(json.dumps(convert(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
