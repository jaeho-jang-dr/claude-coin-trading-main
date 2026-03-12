#!/usr/bin/env python3
"""
RL 학습 통합 스케줄러

3-tier 학습 전략:
  Tier 1 (Quick)  : 6시간마다 — 50K steps, 30일 데이터, 증분 학습 (LR 30%)
  Tier 2 (Daily)  : 매일 03:00 — 150K steps, 60일 데이터, 증분 학습 (LR 50%)
  Tier 3 (Weekly) : 매주 일요일 04:00 — 300K steps, 180일 데이터, 풀 재학습 (LR 100%)

사용법:
  python scripts/training_scheduler.py tier1          # Quick 1회 실행
  python scripts/training_scheduler.py tier2          # Daily 1회 실행
  python scripts/training_scheduler.py tier3          # Weekly 1회 실행
  python scripts/training_scheduler.py status         # 학습 현황 확인
  python scripts/training_scheduler.py register       # Task Scheduler 등록
  python scripts/training_scheduler.py remove         # Task Scheduler 해제
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("training_scheduler")

# ─── 학습 티어 정의 ───

TIERS = {
    "tier1": {
        "name": "Quick Incremental",
        "steps": 50_000,
        "days": 30,
        "lr_ratio": 0.3,       # 베이스 LR의 30%
        "eval_episodes": 10,
        "description": "6시간 간격 빠른 증분 학습",
        "schedule": {"type": "hourly", "interval": 6},
    },
    "tier2": {
        "name": "Daily Training",
        "steps": 150_000,
        "days": 60,
        "lr_ratio": 0.5,       # 베이스 LR의 50%
        "eval_episodes": 15,
        "description": "매일 03:00 중간 강도 학습",
        "schedule": {"type": "daily", "time": "03:00"},
    },
    "tier3": {
        "name": "Weekly Full Retrain",
        "steps": 300_000,
        "days": 180,
        "lr_ratio": 1.0,       # 풀 LR
        "eval_episodes": 20,
        "description": "매주 일요일 04:00 전체 재학습",
        "schedule": {"type": "weekly", "day": "Sunday", "time": "04:00"},
    },
    "tier4": {
        "name": "Monthly 1H Intensive",
        "steps": 500_000,
        "days": 180,
        "lr_ratio": 1.0,       # 5-Phase 커리큘럼
        "eval_episodes": 20,
        "description": "매월 1일 02:00 1시간 집중 훈련 (5-Phase)",
        "schedule": {"type": "monthly", "day": 1, "time": "02:00"},
    },
}

# v8 최적 HP (보상 함수 v8에 맞춤)
BEST_HP = {
    "lr": 3e-4,          # 약간 낮춰서 안정적 학습
    "ent_coef": 0.02,    # 엔트로피 강화 → 정책 붕괴(always-buy) 방지
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
}

HISTORY_PATH = PROJECT_DIR / "data" / "training_history.json"

# Supabase 설정
from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _save_to_db(result: dict) -> bool:
    """학습 결과를 Supabase rl_training_log 테이블에 저장"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase 설정 없음 — DB 저장 건너뜀")
        return False

    baseline = result.get("baseline", {})
    new_stats = result.get("new_stats", {})

    row = {
        "created_at": result.get("timestamp", datetime.now().isoformat()),
        "tier": result["tier"],
        "tier_name": result.get("tier_name"),
        "steps": result["steps"],
        "data_days": result["data_days"],
        "lr_ratio": result["lr_ratio"],
        "candles_count": result.get("candles_count"),
        "baseline_return_pct": baseline.get("return_pct"),
        "baseline_sharpe": baseline.get("sharpe"),
        "baseline_mdd": baseline.get("mdd"),
        "baseline_trades": baseline.get("trades"),
        "new_return_pct": new_stats.get("return_pct"),
        "new_sharpe": new_stats.get("sharpe"),
        "new_mdd": new_stats.get("mdd"),
        "new_trades": new_stats.get("trades"),
        "improved": result.get("improved", False),
        "version_id": result.get("version_id"),
        "elapsed_sec": result.get("elapsed_sec"),
        "rollback": not result.get("improved", True),
    }

    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/rl_training_log",
            headers=_supabase_headers(),
            json=row,
            timeout=10,
        )
        if r.status_code in (200, 201):
            logger.info("DB 저장 완료 (rl_training_log)")
            return True
        else:
            logger.warning(f"DB 저장 실패: {r.status_code} {r.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"DB 저장 예외: {e}")
        return False


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def load_history() -> list:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def save_history(history: list):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(history[-100:], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def run_tier(tier_key: str) -> dict:
    """지정 티어의 학습을 1회 실행"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    from rl_hybrid.rl.environment import BitcoinTradingEnv
    from rl_hybrid.rl.policy import MODEL_DIR

    tier = TIERS[tier_key]
    start = time.time()
    balance = 10_000_000

    logger.info(f"═══ {tier['name']} ({tier_key}) 시작 ═══")
    logger.info(f"  steps={tier['steps']}, days={tier['days']}, lr_ratio={tier['lr_ratio']}")

    # 1. 데이터 로드
    loader = HistoricalDataLoader()
    logger.info(f"최근 {tier['days']}일 데이터 로드...")
    try:
        candles = loader.compute_indicators(
            loader.load_candles(days=tier["days"], interval="1h")
        )
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return {"error": str(e), "tier": tier_key}

    if len(candles) < 100:
        logger.warning(f"데이터 부족: {len(candles)}개")
        return {"error": "insufficient_data", "tier": tier_key}

    split = int(len(candles) * 0.8)
    train_c = candles[:split]
    eval_c = candles[split:]

    logger.info(f"데이터: 학습 {len(train_c)}개, 평가 {len(eval_c)}개")

    train_env = BitcoinTradingEnv(candles=train_c, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_c, initial_balance=balance)

    # 2. 모델 로드 (tier3는 새로 생성할 수도 있음)
    model_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
    if not os.path.exists(model_path + ".zip"):
        model_path = os.path.join(MODEL_DIR, "v3_final")

    if tier_key == "tier3" and not os.path.exists(model_path + ".zip"):
        # tier3: 모델이 없으면 새로 생성
        logger.info("Tier3: 새 모델 생성")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(BEST_HP["lr"] * tier["lr_ratio"]),
            n_steps=BEST_HP["n_steps"],
            batch_size=BEST_HP["batch_size"],
            n_epochs=BEST_HP["n_epochs"],
            ent_coef=BEST_HP["ent_coef"],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
        )
    else:
        logger.info(f"모델 로드: {model_path}")
        model = PPO.load(model_path, env=train_env)

    # 3. 베이스라인 평가
    logger.info("베이스라인 평가...")
    baseline = _evaluate(model, eval_env, tier["eval_episodes"])
    logger.info(
        f"베이스라인: return={baseline['return_pct']:.2f}%, "
        f"sharpe={baseline['sharpe']:.3f}, mdd={baseline['mdd']:.2%}, "
        f"trades={baseline['trades']:.0f}"
    )

    # 4. 학습
    actual_lr = BEST_HP["lr"] * tier["lr_ratio"]
    logger.info(f"학습 시작: {tier['steps']} steps, LR={actual_lr:.1e}")
    model.learning_rate = linear_schedule(actual_lr)
    model.ent_coef = BEST_HP["ent_coef"]

    save_dir = os.path.join(MODEL_DIR, f"continuous_{tier_key}")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=max(tier["steps"] // 5, 10000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model.learn(
        total_timesteps=tier["steps"],
        callback=eval_cb,
        reset_num_timesteps=True,
    )

    # 5. 학습 후 평가
    logger.info("학습 후 평가...")
    new_stats = _evaluate(model, eval_env, tier["eval_episodes"])
    logger.info(
        f"학습 후: return={new_stats['return_pct']:.2f}%, "
        f"sharpe={new_stats['sharpe']:.3f}, mdd={new_stats['mdd']:.2%}, "
        f"trades={new_stats['trades']:.0f}"
    )

    # 6. 모델 저장 / 롤백
    registry_path = os.path.join(MODEL_DIR, "sb3_registry.json")
    registry = json.loads(open(registry_path, encoding="utf-8").read()) if os.path.exists(registry_path) else {"current": None, "versions": [], "rollbacks": 0}

    # Sharpe 비교 (tier3는 더 엄격: 개선되어야만 저장)
    if tier_key == "tier3":
        improved = new_stats["sharpe"] > baseline["sharpe"]
    else:
        improved = new_stats["sharpe"] > baseline["sharpe"] - 0.05  # 약간의 여유

    result = {
        "tier": tier_key,
        "tier_name": tier["name"],
        "baseline": baseline,
        "new_stats": new_stats,
        "improved": improved,
        "elapsed_sec": time.time() - start,
        "timestamp": datetime.now().isoformat(),
        "data_days": tier["days"],
        "steps": tier["steps"],
        "lr_ratio": tier["lr_ratio"],
        "candles_count": len(candles),
    }

    if improved:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"sb3_{tier_key}_{ts}"

        model.save(os.path.join(MODEL_DIR, "ppo_btc_latest"))

        version_dir = os.path.join(MODEL_DIR, "versions", version_id)
        os.makedirs(version_dir, exist_ok=True)
        model.save(os.path.join(version_dir, "model"))

        registry["versions"].append({
            "version_id": version_id,
            "metrics": new_stats,
            "baseline": baseline,
            "tier": tier_key,
            "created_at": result["timestamp"],
        })
        registry["current"] = version_id

        result["version_id"] = version_id
        logger.info(f"모델 업데이트: {version_id}")
    else:
        registry["rollbacks"] += 1
        logger.warning(
            f"롤백: sharpe {baseline['sharpe']:.3f} → {new_stats['sharpe']:.3f} (저하)"
        )

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # 7. 히스토리 저장 (로컬 JSON + Supabase DB)
    history = load_history()
    history.append(result)
    save_history(history)
    _save_to_db(result)

    # 8. 텔레그램 알림 (tier2, tier3, tier4)
    if tier_key in ("tier2", "tier3", "tier4"):
        _notify_result(result)

    elapsed = result["elapsed_sec"]
    logger.info(f"═══ {tier['name']} 완료 ({elapsed:.0f}초) ═══")
    return result


def _evaluate(model, env, episodes) -> dict:
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


def _notify_result(result: dict):
    """텔레그램으로 학습 결과 알림"""
    try:
        import subprocess
        tier = result["tier"]
        name = result["tier_name"]
        improved = "개선" if result["improved"] else "롤백"
        b = result["baseline"]
        n = result["new_stats"]

        msg = f"[RL {name}] {improved}"
        detail = (
            f"Tier: {tier}\n"
            f"Steps: {result['steps']:,}\n"
            f"Data: {result['data_days']}일 ({result.get('candles_count', '?')}개)\n"
            f"Time: {result['elapsed_sec']:.0f}초\n\n"
            f"Before → After:\n"
            f"  Return: {b['return_pct']:.2f}% → {n['return_pct']:.2f}%\n"
            f"  Sharpe: {b['sharpe']:.3f} → {n['sharpe']:.3f}\n"
            f"  MDD: {b['mdd']:.2%} → {n['mdd']:.2%}\n"
            f"  Trades: {b['trades']:.0f} → {n['trades']:.0f}"
        )
        if result.get("version_id"):
            detail += f"\n\nModel: {result['version_id']}"

        subprocess.run(
            [sys.executable, "scripts/notify_telegram.py", "trade", msg, detail],
            cwd=str(PROJECT_DIR),
            check=False,
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass


def show_status():
    """학습 현황 표시"""
    from rl_hybrid.rl.policy import MODEL_DIR

    print("\n═══ RL 학습 스케줄 ═══\n")

    for key, tier in TIERS.items():
        sched = tier["schedule"]
        if sched["type"] == "hourly":
            when = f"매 {sched['interval']}시간"
        elif sched["type"] == "daily":
            when = f"매일 {sched['time']}"
        elif sched["type"] == "monthly":
            when = f"매월 {sched['day']}일 {sched['time']}"
        else:
            when = f"매주 {sched['day']} {sched['time']}"
        print(f"  [{key}] {tier['name']}")
        print(f"    스케줄: {when}")
        print(f"    설정: {tier['steps']:,} steps, {tier['days']}일 데이터, LR×{tier['lr_ratio']}")
        print()

    # 레지스트리 상태
    registry_path = os.path.join(MODEL_DIR, "sb3_registry.json")
    if os.path.exists(registry_path):
        reg = json.loads(open(registry_path, encoding="utf-8").read())
        print(f"  현재 모델: {reg.get('current', 'N/A')}")
        print(f"  전체 버전: {len(reg.get('versions', []))}개")
        print(f"  롤백 횟수: {reg.get('rollbacks', 0)}회")
    else:
        print("  레지스트리: 없음")

    # 최근 학습 히스토리
    history = load_history()
    if history:
        print(f"\n  최근 학습 (최근 5건):")
        for h in history[-5:]:
            status = "✅" if h.get("improved") else "❌"
            ts = h.get("timestamp", "")[:16]
            tier = h.get("tier", "?")
            sharpe_before = h.get("baseline", {}).get("sharpe", 0)
            sharpe_after = h.get("new_stats", {}).get("sharpe", 0)
            elapsed = h.get("elapsed_sec", 0)
            print(f"    {status} [{ts}] {tier}: sharpe {sharpe_before:.3f}→{sharpe_after:.3f} ({elapsed:.0f}s)")

    # Task Scheduler 상태
    print("\n  Task Scheduler:")
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-ScheduledTask -TaskName 'CoinTrading_RL_*' -ErrorAction SilentlyContinue | "
             "ForEach-Object { $info = Get-ScheduledTaskInfo -TaskName $_.TaskName -ErrorAction SilentlyContinue; "
             "\"    $($_.State) | $($_.TaskName) | Next: $($info.NextRunTime)\" }"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("    (등록된 태스크 없음)")
    except Exception:
        print("    (확인 불가)")
    print()


def register_tasks():
    """Windows Task Scheduler에 3-tier 학습 태스크 등록"""
    python = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
    script = str(PROJECT_DIR / "scripts" / "training_scheduler.py")

    ps_commands = []

    for tier_key, tier in TIERS.items():
        task_name = f"CoinTrading_RL_{tier_key}"
        sched = tier["schedule"]

        action_cmd = (
            f"$action = New-ScheduledTaskAction "
            f"-Execute '{python}' "
            f"-Argument '-u \"{script}\" {tier_key}' "
            f"-WorkingDirectory '{PROJECT_DIR}'"
        )

        if sched["type"] == "hourly":
            trigger_cmd = (
                f"$trigger = New-ScheduledTaskTrigger -Once -At '00:00' "
                f"-RepetitionInterval (New-TimeSpan -Hours {sched['interval']}) "
                f"-RepetitionDuration (New-TimeSpan -Days 365)"
            )
        elif sched["type"] == "daily":
            trigger_cmd = (
                f"$trigger = New-ScheduledTaskTrigger -Daily -At '{sched['time']}'"
            )
        elif sched["type"] == "monthly":
            # Monthly: 매월 특정 일에 실행
            trigger_cmd = (
                f"$trigger = New-ScheduledTaskTrigger -Once -At '{sched['time']}'\n"
                f"$trigger.Repetition.Interval = 'P31D'"
            )
        else:  # weekly
            trigger_cmd = (
                f"$trigger = New-ScheduledTaskTrigger -Weekly "
                f"-DaysOfWeek {sched['day']} -At '{sched['time']}'"
            )

        # tier4=90분, tier3=60분, 나머지=30분
        time_limit = 90 if tier_key == "tier4" else 60 if tier_key == "tier3" else 30

        settings_cmd = (
            f"$settings = New-ScheduledTaskSettingsSet "
            f"-AllowStartIfOnBatteries "
            f"-DontStopIfGoingOnBatteries "
            f"-StartWhenAvailable "
            f"-ExecutionTimeLimit (New-TimeSpan -Minutes {time_limit}) "
            f"-MultipleInstances IgnoreNew"
        )

        register_cmd = (
            f"Register-ScheduledTask "
            f"-TaskName '{task_name}' "
            f"-Description '{tier['description']}' "
            f"-Action $action "
            f"-Trigger $trigger "
            f"-Settings $settings "
            f"-RunLevel Highest "
            f"-Force | Out-Null"
        )

        ps_commands.append(
            f"{action_cmd}; {trigger_cmd}; {settings_cmd}; {register_cmd}; "
            f"Write-Host '  [OK] {task_name} - {tier['description']}'"
        )

    # PowerShell 스크립트를 임시 파일로 저장 후 실행
    ps_script = PROJECT_DIR / "scripts" / "_register_rl_tasks.ps1"
    ps_content = "Write-Host '=== RL 학습 태스크 등록 ==='\n"
    for cmd in ps_commands:
        ps_content += cmd + "\n"
    ps_content += "Write-Host '등록 완료!'\n"

    ps_script.write_text(ps_content, encoding="utf-8-sig")
    print(f"PowerShell 스크립트 생성: {ps_script}")
    print("관리자 권한으로 실행하세요:")
    print(f"  powershell -ExecutionPolicy Bypass -File {ps_script}")


def remove_tasks():
    """Task Scheduler에서 RL 학습 태스크 제거"""
    ps_script = PROJECT_DIR / "scripts" / "_remove_rl_tasks.ps1"
    lines = ["Write-Host '=== RL 학습 태스크 제거 ==='"]
    for tier_key in TIERS:
        name = f"CoinTrading_RL_{tier_key}"
        lines.append(
            f"$t = Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue; "
            f"if ($t) {{ Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false; "
            f"Write-Host '  [삭제] {name}' }} else {{ Write-Host '  [없음] {name}' }}"
        )
    ps_script.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"PowerShell 스크립트 생성: {ps_script}")
    print("관리자 권한으로 실행하세요:")
    print(f"  powershell -ExecutionPolicy Bypass -File {ps_script}")


def backfill_db():
    """기존 training_history.json 데이터를 Supabase DB로 일괄 업로드"""
    history = load_history()
    if not history:
        print("학습 히스토리 없음")
        return

    print(f"총 {len(history)}건 백필 시작...")
    success = 0
    for h in history:
        if _save_to_db(h):
            success += 1
    print(f"완료: {success}/{len(history)}건 저장")


def main():
    parser = argparse.ArgumentParser(description="RL 학습 통합 스케줄러")
    parser.add_argument(
        "action",
        choices=["tier1", "tier2", "tier3", "tier4", "status", "register", "remove", "backfill"],
        help="실행할 작업",
    )
    args = parser.parse_args()

    if args.action == "status":
        show_status()
    elif args.action == "register":
        register_tasks()
    elif args.action == "remove":
        remove_tasks()
    elif args.action == "backfill":
        backfill_db()
    elif args.action == "tier4":
        # tier4는 1시간 집중 훈련 스크립트 호출
        from scripts.train_1h_intensive import run_full_training
        result = run_full_training()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        result = run_tier(args.action)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
