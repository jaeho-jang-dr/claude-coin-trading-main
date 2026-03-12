"""2시간 자율 강화학습 스케줄러

비트코인 매매봇(PPO v6)과 김치랑봇(PPO + DQN)을 동시에 훈련한다.

Phase 구성 (2시간):
  Phase 1 (0~30분): BTC PPO v6 30일 + 김치랑 PPO 90일 (병렬)
  Phase 2 (30~60분): BTC PPO v6 60일 + 김치랑 DQN 730일 (병렬)
  Phase 3 (60~90분): BTC PPO v6 90일 + 김치랑 PPO 앙상블 검증
  Phase 4 (90~120분): BTC PPO v6 180일 + 최종 평가 + 모델 배포

사용법:
  python scripts/train_2h_autonomous.py           # 2시간 전체
  python scripts/train_2h_autonomous.py --quick    # 30분 빠른 버전
  python scripts/train_2h_autonomous.py --phase 1  # 특정 Phase만
"""

import hide_console

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "train_2h_autonomous.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("train_2h")

# ============================================================
# Supabase DB 저장 유틸
# ============================================================

def _get_supabase():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    return url, key


def save_to_db(table: str, row: dict) -> bool:
    """Supabase REST API로 저장"""
    import requests
    url, key = _get_supabase()
    if not url or not key:
        return False
    try:
        resp = requests.post(
            f"{url}/rest/v1/{table}",
            json=row,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return True
        logger.warning(f"DB 저장 실패 ({table}): {resp.status_code} {resp.text[:100]}")
        return False
    except Exception as e:
        logger.warning(f"DB 저장 오류 ({table}): {e}")
        return False


def save_local(filename: str, row: dict):
    """로컬 JSONL 백업"""
    path = os.path.join(PROJECT_DIR, "data", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row["_saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def record_training(phase: str, bot: str, result: dict):
    """훈련 결과를 DB + 로컬에 이중 기록"""
    # Supabase rl_training_log 스키마에 맞춤
    tier_name = f"{bot}_{result.get('model_type', 'ppo')}_{phase[:20]}"
    db_row = {
        "tier": f"2h_{bot}",
        "tier_name": tier_name,
        "steps": result.get("total_steps", 0),
        "data_days": result.get("data_days", 0),
        "lr_ratio": 0.3,
        "candles_count": 0,
        "baseline_return_pct": 0,
        "baseline_sharpe": result.get("baseline_sharpe", 0) or 0,
        "baseline_mdd": 0,
        "baseline_trades": 0,
        "new_return_pct": result.get("eval_return_pct", 0) or 0,
        "new_sharpe": result.get("eval_sharpe", 0) or 0,
        "new_mdd": 0,
        "new_trades": result.get("eval_trades", 0) or 0,
        "improved": result.get("improved", False),
        "version_id": f"2h_{bot}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "elapsed_sec": result.get("elapsed_sec", 0),
        "rollback": not result.get("improved", True),
        "notes": json.dumps({
            "phase": phase,
            "bot": bot,
            "model_type": result.get("model_type"),
            "eval_pnl": result.get("eval_pnl"),
            "error": result.get("error"),
        }, ensure_ascii=False),
    }
    save_to_db("rl_training_log", db_row)

    # 로컬에는 풀 정보 저장
    local_row = {
        "phase": phase,
        "bot": bot,
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_local("training_2h_log.jsonl", local_row)


# ============================================================
# BTC 매매봇 훈련
# ============================================================

def train_btc_ppo(data_days: int, steps: int, phase_name: str) -> dict:
    """BTC PPO v6 증분 학습"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from rl_hybrid.rl.data_loader import HistoricalDataLoader
    from rl_hybrid.rl.environment import BitcoinTradingEnv
    from rl_hybrid.rl.policy import MODEL_DIR

    start = time.time()
    logger.info(f"[BTC] {phase_name}: {data_days}일 데이터, {steps:,} 스텝")

    try:
        # 데이터 로드
        loader = HistoricalDataLoader()
        candles = loader.compute_indicators(
            loader.load_candles(days=data_days, interval="1h")
        )
        if len(candles) < 100:
            return {"error": f"데이터 부족: {len(candles)}개", "data_days": data_days}

        split = int(len(candles) * 0.8)
        train_env = BitcoinTradingEnv(candles=candles[:split], initial_balance=10_000_000)
        eval_env = BitcoinTradingEnv(candles=candles[split:], initial_balance=10_000_000)

        # 모델 로드 또는 신규 생성
        model_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
        use_existing = False
        if os.path.exists(model_path + ".zip"):
            logger.info(f"[BTC] 기존 모델 로드: {model_path}")
            model = PPO.load(model_path, env=train_env)
            # 정책 붕괴 검사: trades<=2면 신규 생성
            test_stats = _evaluate_btc(model, eval_env, episodes=3)
            if test_stats["trades"] <= 2:
                logger.warning(f"[BTC] 정책 붕괴 감지 (trades={test_stats['trades']:.0f}) → 신규 모델 생성")
                use_existing = False
            else:
                use_existing = True

        if not use_existing:
            logger.info("[BTC] 신규 모델 생성 (높은 탐색률)")
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                ent_coef=0.15,  # 높은 엔트로피로 다양한 행동 유도
                clip_range=0.3,
                verbose=0,
            )

        # 증분 학습 파라미터
        if use_existing:
            model.learning_rate = 3e-4 * 0.3
            model.ent_coef = 0.12
        else:
            model.ent_coef = 0.15

        save_dir = os.path.join(MODEL_DIR, "2h_best")
        os.makedirs(save_dir, exist_ok=True)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=steps // 5,
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )

        model.learn(total_timesteps=steps, callback=eval_cb, reset_num_timesteps=True)

        # 평가
        eval_stats = _evaluate_btc(model, eval_env, episodes=5)
        baseline_stats = None

        # 기존 모델과 비교
        if os.path.exists(model_path + ".zip"):
            try:
                old_model = PPO.load(model_path, env=eval_env)
                baseline_stats = _evaluate_btc(old_model, eval_env, episodes=5)
            except Exception:
                pass

        improved = True
        if baseline_stats:
            improved = eval_stats["sharpe"] > baseline_stats["sharpe"] - 0.05
            logger.info(
                f"[BTC] 비교: sharpe {baseline_stats['sharpe']:.3f} → {eval_stats['sharpe']:.3f} "
                f"({'개선' if improved else '롤백'})"
            )

        if improved:
            model.save(model_path)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ver_dir = os.path.join(MODEL_DIR, "versions", f"2h_{ts}")
            os.makedirs(ver_dir, exist_ok=True)
            model.save(os.path.join(ver_dir, "model"))
            logger.info(f"[BTC] 모델 저장 완료: 2h_{ts}")

        elapsed = time.time() - start
        return {
            "model_type": "ppo",
            "total_steps": steps,
            "data_days": data_days,
            "eval_return_pct": eval_stats["return_pct"],
            "eval_sharpe": eval_stats["sharpe"],
            "eval_trades": eval_stats["trades"],
            "improved": improved,
            "elapsed_sec": elapsed,
            "baseline_sharpe": baseline_stats["sharpe"] if baseline_stats else None,
        }

    except Exception as e:
        logger.error(f"[BTC] 훈련 에러: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "data_days": data_days, "elapsed_sec": time.time() - start}


def _evaluate_btc(model, env, episodes=5) -> dict:
    """BTC 모델 평가"""
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


# ============================================================
# 김치랑봇 훈련
# ============================================================

def train_kimchirang_ppo(days: int, steps: int, phase_name: str) -> dict:
    """김치랑 PPO 학습"""
    start = time.time()
    logger.info(f"[김치랑 PPO] {phase_name}: {days}일 데이터, {steps:,} 스텝")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from kimchirang.rl_env import KPHistoricalData, KimchirangEnv

        # 데이터 수집
        kp_data = KPHistoricalData(days=days)
        if not kp_data.collect():
            return {"error": "KP 데이터 수집 실패", "data_days": days}

        logger.info(f"[김치랑 PPO] KP 데이터: {len(kp_data.kp_series)}개, "
                     f"범위 {min(kp_data.kp_series):.2f}% ~ {max(kp_data.kp_series):.2f}%")

        env = KimchirangEnv(kp_data)
        eval_env = KimchirangEnv(kp_data)

        model_dir = os.path.join(PROJECT_DIR, "data", "rl_models", "kimchirang")
        os.makedirs(model_dir, exist_ok=True)

        # 기존 모델 로드 또는 신규 생성
        latest_path = os.path.join(model_dir, "ppo_kimchirang_latest")
        best_path = os.path.join(model_dir, "best_model")

        if os.path.exists(latest_path + ".zip"):
            logger.info(f"[김치랑 PPO] 기존 모델 로드")
            model = PPO.load(latest_path, env=env)
            model.learning_rate = 3e-4 * 0.3
        else:
            logger.info("[김치랑 PPO] 신규 모델 생성")
            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.08,
                verbose=0,
            )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=os.path.join(model_dir, "logs"),
            eval_freq=max(steps // 5, 2048),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )

        model.learn(total_timesteps=steps, callback=eval_cb)

        # 저장
        model.save(latest_path)
        logger.info(f"[김치랑 PPO] 모델 저장: {latest_path}")

        # 평가
        eval_result = _evaluate_kimchirang(model, eval_env, episodes=5)

        elapsed = time.time() - start
        return {
            "model_type": "ppo",
            "total_steps": steps,
            "data_days": days,
            "eval_pnl": eval_result["avg_pnl"],
            "eval_trades": eval_result["avg_trades"],
            "improved": True,
            "elapsed_sec": elapsed,
        }

    except Exception as e:
        logger.error(f"[김치랑 PPO] 에러: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "data_days": days, "elapsed_sec": time.time() - start}


def train_kimchirang_dqn(days: int, steps: int, phase_name: str) -> dict:
    """김치랑 DQN 학습"""
    start = time.time()
    logger.info(f"[김치랑 DQN] {phase_name}: {days}일 데이터, {steps:,} 스텝")

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
        from kimchirang.rl_env import KPHistoricalData, KimchirangEnv

        kp_data = KPHistoricalData(days=days)
        if not kp_data.collect():
            return {"error": "KP 데이터 수집 실패", "data_days": days}

        env = KimchirangEnv(kp_data)
        eval_env = KimchirangEnv(kp_data)

        model_dir = os.path.join(PROJECT_DIR, "data", "rl_models", "kimchirang", "dqn")
        os.makedirs(model_dir, exist_ok=True)

        # 기존 모델 로드 또는 신규 생성
        latest_path = os.path.join(model_dir, "dqn_kimchirang_latest")
        if os.path.exists(latest_path + ".zip"):
            logger.info("[김치랑 DQN] 기존 모델 로드")
            model = DQN.load(latest_path, env=env)
        else:
            logger.info("[김치랑 DQN] 신규 모델 생성")
            model = DQN(
                "MlpPolicy", env,
                learning_rate=1e-4,
                buffer_size=100_000,
                learning_starts=5000,
                batch_size=128,
                gamma=0.99,
                tau=0.005,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.15,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0,
            )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=os.path.join(model_dir, "logs"),
            eval_freq=max(steps // 5, 5000),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )

        model.learn(total_timesteps=steps, callback=eval_cb)

        model.save(latest_path)
        logger.info(f"[김치랑 DQN] 모델 저장: {latest_path}")

        eval_result = _evaluate_kimchirang(model, eval_env, episodes=5)

        elapsed = time.time() - start
        return {
            "model_type": "dqn",
            "total_steps": steps,
            "data_days": days,
            "eval_pnl": eval_result["avg_pnl"],
            "eval_trades": eval_result["avg_trades"],
            "improved": True,
            "elapsed_sec": elapsed,
        }

    except Exception as e:
        logger.error(f"[김치랑 DQN] 에러: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "data_days": days, "elapsed_sec": time.time() - start}


def _evaluate_kimchirang(model, env, episodes=5) -> dict:
    """김치랑 모델 평가"""
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        results.append(info)
    return {
        "avg_pnl": float(np.mean([r.get("total_pnl", 0) for r in results])),
        "avg_trades": float(np.mean([r.get("trade_count", 0) for r in results])),
    }


# ============================================================
# 앙상블 검증
# ============================================================

def verify_ensemble(phase_name: str = "", **kwargs) -> dict:
    """김치랑 PPO + DQN 앙상블 합의 검증"""
    logger.info("[앙상블] PPO + DQN 앙상블 검증 시작")
    start = time.time()

    try:
        from stable_baselines3 import PPO, DQN
        from kimchirang.rl_env import KPHistoricalData, KimchirangEnv

        model_dir = os.path.join(PROJECT_DIR, "data", "rl_models", "kimchirang")
        ppo_path = os.path.join(model_dir, "best_model")
        dqn_path = os.path.join(model_dir, "dqn", "best_model")

        has_ppo = os.path.exists(ppo_path + ".zip")
        has_dqn = os.path.exists(dqn_path + ".zip")

        if not has_ppo and not has_dqn:
            return {"error": "모델 없음", "mode": "none"}

        kp_data = KPHistoricalData(days=90)
        if not kp_data.collect():
            return {"error": "데이터 수집 실패"}

        env = KimchirangEnv(kp_data)

        # 개별 평가
        results = {}
        if has_ppo:
            ppo_model = PPO.load(ppo_path)
            results["ppo"] = _evaluate_kimchirang(ppo_model, env, episodes=5)
            logger.info(f"[앙상블] PPO: PnL={results['ppo']['avg_pnl']:.2f}%, "
                        f"거래={results['ppo']['avg_trades']:.0f}회")

        if has_dqn:
            dqn_model = DQN.load(dqn_path)
            results["dqn"] = _evaluate_kimchirang(dqn_model, env, episodes=5)
            logger.info(f"[앙상블] DQN: PnL={results['dqn']['avg_pnl']:.2f}%, "
                        f"거래={results['dqn']['avg_trades']:.0f}회")

        # 앙상블 시뮬레이션 (합의 기반)
        if has_ppo and has_dqn:
            ensemble_results = []
            for ep in range(5):
                obs, _ = env.reset()
                done = False
                while not done:
                    ppo_action, _ = ppo_model.predict(obs, deterministic=True)
                    dqn_action, _ = dqn_model.predict(obs, deterministic=True)

                    # 앙상블: 합의 시만 행동, 불일치 시 Hold
                    if int(ppo_action) == int(dqn_action):
                        action = int(ppo_action)
                    else:
                        action = 0  # Hold

                    obs, _, t, tr, info = env.step(action)
                    done = t or tr
                ensemble_results.append(info)

            results["ensemble"] = {
                "avg_pnl": float(np.mean([r.get("total_pnl", 0) for r in ensemble_results])),
                "avg_trades": float(np.mean([r.get("trade_count", 0) for r in ensemble_results])),
            }
            logger.info(f"[앙상블] 합의: PnL={results['ensemble']['avg_pnl']:.2f}%, "
                        f"거래={results['ensemble']['avg_trades']:.0f}회")

        mode = "ensemble" if has_ppo and has_dqn else ("ppo" if has_ppo else "dqn")
        return {
            "mode": mode,
            "results": results,
            "elapsed_sec": time.time() - start,
        }

    except Exception as e:
        logger.error(f"[앙상블] 에러: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "elapsed_sec": time.time() - start}


# ============================================================
# Phase 실행기
# ============================================================

PHASES = {
    1: {
        "name": "Phase 1: 기초 학습",
        "duration_min": 30,
        "tasks": [
            {"bot": "btc", "fn": "train_btc_ppo", "args": {"data_days": 30, "steps": 500_000}},
            {"bot": "kimchirang", "fn": "train_kimchirang_ppo", "args": {"days": 90, "steps": 300_000}},
        ],
    },
    2: {
        "name": "Phase 2: 심화 학습",
        "duration_min": 30,
        "tasks": [
            {"bot": "btc", "fn": "train_btc_ppo", "args": {"data_days": 60, "steps": 500_000}},
            {"bot": "kimchirang", "fn": "train_kimchirang_dqn", "args": {"days": 365, "steps": 500_000}},
        ],
    },
    3: {
        "name": "Phase 3: 장기 패턴",
        "duration_min": 30,
        "tasks": [
            {"bot": "btc", "fn": "train_btc_ppo", "args": {"data_days": 90, "steps": 500_000}},
            {"bot": "kimchirang", "fn": "train_kimchirang_ppo", "args": {"days": 365, "steps": 300_000}},
        ],
    },
    4: {
        "name": "Phase 4: 최종 강화 + 앙상블 검증",
        "duration_min": 30,
        "tasks": [
            {"bot": "btc", "fn": "train_btc_ppo", "args": {"data_days": 180, "steps": 500_000}},
            {"bot": "kimchirang", "fn": "verify_ensemble", "args": {}},
        ],
    },
}

QUICK_PHASES = {
    1: {
        "name": "Quick Phase 1: 기초",
        "duration_min": 15,
        "tasks": [
            {"bot": "btc", "fn": "train_btc_ppo", "args": {"data_days": 30, "steps": 200_000}},
            {"bot": "kimchirang", "fn": "train_kimchirang_ppo", "args": {"days": 90, "steps": 100_000}},
        ],
    },
    2: {
        "name": "Quick Phase 2: DQN + 앙상블",
        "duration_min": 15,
        "tasks": [
            {"bot": "btc", "fn": "train_btc_ppo", "args": {"data_days": 60, "steps": 200_000}},
            {"bot": "kimchirang", "fn": "train_kimchirang_dqn", "args": {"days": 365, "steps": 200_000}},
        ],
    },
}

FN_MAP = {
    "train_btc_ppo": train_btc_ppo,
    "train_kimchirang_ppo": train_kimchirang_ppo,
    "train_kimchirang_dqn": train_kimchirang_dqn,
    "verify_ensemble": verify_ensemble,
}


def run_phase(phase_num: int, phase_config: dict):
    """Phase 실행 (병렬)"""
    phase_name = phase_config["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  {phase_name}")
    logger.info(f"{'='*60}")

    results = {}
    tasks = phase_config["tasks"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for task in tasks:
            fn = FN_MAP[task["fn"]]
            args = task["args"].copy()
            args["phase_name"] = phase_name
            futures[executor.submit(fn, **args)] = task["bot"]

        for future in as_completed(futures):
            bot = futures[future]
            try:
                result = future.result()
                results[bot] = result

                if "error" in result:
                    logger.warning(f"  [{bot}] 에러: {result['error']}")
                else:
                    elapsed = result.get("elapsed_sec", 0)
                    logger.info(f"  [{bot}] 완료 ({elapsed:.0f}초)")

                # DB 기록
                record_training(phase_name, bot, result)

            except Exception as e:
                logger.error(f"  [{bot}] 예외: {e}")
                results[bot] = {"error": str(e)}

    return results


def send_telegram_report(all_results: dict, total_elapsed: float):
    """훈련 완료 텔레그램 알림"""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_DIR, ".env"))

        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_USER_ID")
        if not token or not chat_id:
            return

        import requests

        lines = ["🤖 *2시간 자율 RL 훈련 완료*\n"]
        lines.append(f"⏱️ 총 소요: {total_elapsed/60:.0f}분\n")

        for phase_name, phase_results in all_results.items():
            lines.append(f"📌 *{phase_name}*")
            for bot, result in phase_results.items():
                if "error" in result:
                    lines.append(f"  ❌ {bot}: {result['error'][:50]}")
                else:
                    if bot == "btc":
                        sharpe = result.get("eval_sharpe", 0)
                        trades = result.get("eval_trades", 0)
                        improved = "✅" if result.get("improved") else "🔄"
                        lines.append(f"  {improved} BTC: sharpe={sharpe:.3f}, trades={trades:.0f}")
                    elif "results" in result:  # 앙상블 검증
                        mode = result.get("mode", "?")
                        lines.append(f"  🔗 앙상블: mode={mode}")
                        for k, v in result.get("results", {}).items():
                            lines.append(f"    {k}: PnL={v['avg_pnl']:.2f}%, trades={v['avg_trades']:.0f}")
                    else:
                        pnl = result.get("eval_pnl", 0)
                        trades = result.get("eval_trades", 0)
                        mtype = result.get("model_type", "?")
                        lines.append(f"  ✅ 김치랑 {mtype}: PnL={pnl:.2f}%, trades={trades:.0f}")

        text = "\n".join(lines)
        # Escape MarkdownV2 special chars
        for ch in ['_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
            text = text.replace(ch, f'\\{ch}')

        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2"},
            timeout=10,
        )
        logger.info("텔레그램 알림 전송 완료")

    except Exception as e:
        logger.warning(f"텔레그램 알림 실패: {e}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="2시간 자율 RL 훈련")
    parser.add_argument("--quick", action="store_true", help="30분 빠른 버전")
    parser.add_argument("--phase", type=int, help="특정 Phase만 실행 (1-4)")
    args = parser.parse_args()

    phases = QUICK_PHASES if args.quick else PHASES
    total_start = time.time()

    logger.info("\n" + "=" * 60)
    logger.info("  🚀 2시간 자율 RL 훈련 시작")
    logger.info(f"  모드: {'빠른 (30분)' if args.quick else '전체 (2시간)'}")
    logger.info(f"  Phase: {list(phases.keys())}")
    logger.info("=" * 60)

    all_results = {}

    for phase_num, phase_config in phases.items():
        if args.phase and phase_num != args.phase:
            continue

        phase_start = time.time()
        results = run_phase(phase_num, phase_config)
        phase_elapsed = time.time() - phase_start

        all_results[phase_config["name"]] = results
        logger.info(f"  Phase {phase_num} 완료: {phase_elapsed/60:.1f}분")

    total_elapsed = time.time() - total_start

    # 최종 리포트
    logger.info("\n" + "=" * 60)
    logger.info("  📊 2시간 자율 RL 훈련 결과")
    logger.info("=" * 60)

    for phase_name, phase_results in all_results.items():
        logger.info(f"\n  {phase_name}:")
        for bot, result in phase_results.items():
            if "error" in result:
                logger.info(f"    [{bot}] ❌ {result['error'][:80]}")
            elif "results" in result:  # 앙상블
                logger.info(f"    [{bot}] 🔗 앙상블 mode={result.get('mode')}")
                for k, v in result.get("results", {}).items():
                    logger.info(f"      {k}: PnL={v['avg_pnl']:.2f}%, trades={v['avg_trades']:.0f}")
            else:
                mtype = result.get("model_type", "?")
                sharpe = result.get("eval_sharpe")
                pnl = result.get("eval_pnl")
                trades = result.get("eval_trades", 0)
                improved = result.get("improved")
                metric = f"sharpe={sharpe:.3f}" if sharpe else f"PnL={pnl:.2f}%"
                status = "✅" if improved else "🔄"
                logger.info(f"    [{bot}] {status} {mtype}: {metric}, trades={trades:.0f}")

    logger.info(f"\n  총 소요: {total_elapsed/60:.1f}분")

    # 텔레그램 알림
    send_telegram_report(all_results, total_elapsed)

    # 결과 JSON 저장
    summary_path = os.path.join(PROJECT_DIR, "data", "train_2h_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "quick" if args.quick else "full",
            "total_elapsed_sec": total_elapsed,
            "phases": {k: v for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"  결과 저장: {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
