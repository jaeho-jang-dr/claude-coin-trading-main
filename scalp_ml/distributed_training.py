#!/usr/bin/env python3
"""
3대 분산 1주일 스캘핑 RL 훈련 시스템

각 머신이 다른 모델/전략을 훈련하고, Supabase에 결과를 공유한다.

머신별 역할:
  Mac Mini  — LightGBM/XGBoost 시그널 분류기 + 피처 실험
  PC128     — DQN/SAC 청산 최적화 (신경망, 대규모)
  PC36      — PPO 환경 다양화 + 보상함수 실험 (DRJAY)

실행:
  python -m scalp_ml.distributed_training --machine mac-mini
  python -m scalp_ml.distributed_training --machine pc128
  python -m scalp_ml.distributed_training --machine pc36
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_DIR / "logs" / "distributed_training.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("dist_train")

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "scalp_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def log_to_db(machine: str, phase: str, status: str, metrics: dict = None, error: str = None):
    """훈련 진행상황을 DB에 기록"""
    try:
        row = {
            "task_type": "train_pytorch",
            "params": json.dumps({
                "machine": machine,
                "phase": phase,
                "plan": "1week_distributed",
            }),
            "status": status,
            "assigned_worker": machine,
            "result": json.dumps(metrics) if metrics else None,
            "error_message": error[:500] if error else None,
            "priority": 1,
        }
        if status == "running":
            row["started_at"] = datetime.now(KST).isoformat()
        elif status in ("completed", "failed"):
            row["completed_at"] = datetime.now(KST).isoformat()

        requests.post(
            f"{SUPABASE_URL}/rest/v1/scalp_training_tasks",
            json=row,
            headers={**HEADERS, "Prefer": "return=minimal"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"DB 기록 실패: {e}")


def send_telegram(text: str):
    """텔레그램 알림"""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "5273754646")
    if not token:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════
# Mac Mini: 시그널 품질 분류기 (LightGBM + XGBoost)
# ═══════════════════════════════════════════════════

def run_mac_mini():
    """
    Mac Mini 1주일 훈련 계획

    Day 1-2: 데이터 수집 + LightGBM 기본 모델
    Day 3-4: XGBoost 대조 모델 + 피처 실험
    Day 5-6: 앙상블 (LightGBM + XGBoost) + 하이퍼파라미터 탐색
    Day 7:   최종 평가 + 모델 선택
    """
    machine = "mac-mini"
    log.info(f"{'='*60}")
    log.info(f"  Mac Mini 훈련 시작 — 시그널 품질 분류기")
    log.info(f"{'='*60}")
    send_telegram(f"[Mac Mini] 1주일 훈련 시작 — LightGBM/XGBoost 시그널 분류기")

    phases = [
        # (이름, 함수, 설명)
        ("phase1_data", _mac_phase1_data, "14일 1분봉 수집 + 피처 생성"),
        ("phase2_lgbm_base", _mac_phase2_lgbm, "LightGBM 기본 모델 (7/14/30일)"),
        ("phase3_xgboost", _mac_phase3_xgboost, "XGBoost 대조 모델"),
        ("phase4_feature_exp", _mac_phase4_features, "피처 실험 (확장 피처셋)"),
        ("phase5_hyperparam", _mac_phase5_hyperparam, "하이퍼파라미터 그리드 탐색"),
        ("phase6_ensemble", _mac_phase6_ensemble, "앙상블 (LGB+XGB+투표)"),
        ("phase7_final", _mac_phase7_final, "최종 평가 + 최적 모델 선택"),
    ]

    _run_phases(machine, phases)


def _mac_phase1_data():
    """1분봉 데이터 수집 (7, 14, 30일)"""
    from scalp_ml.train_lgbm import collect_candles
    import pickle

    for days in [7, 14, 30]:
        cache = MODEL_DIR / f"candles_{days}d.pkl"
        if cache.exists():
            log.info(f"  {days}일 캐시 존재, 스킵")
            continue
        candles = collect_candles(days=days)
        with open(cache, "wb") as f:
            pickle.dump(candles, f)
        log.info(f"  {days}일: {len(candles)}건 수집 완료")

    return {"status": "data_collected", "periods": [7, 14, 30]}


def _mac_phase2_lgbm():
    """LightGBM 회귀 모델 — 3가지 기간으로 훈련"""
    from scalp_ml.train_lgbm import build_dataset, train_model
    import pickle

    results = {}
    for days in [7, 14, 30]:
        cache = MODEL_DIR / f"candles_{days}d.pkl"
        if not cache.exists():
            cache = MODEL_DIR / "candles_cache.pkl"
        with open(cache, "rb") as f:
            candles = pickle.load(f)

        log.info(f"\n  LightGBM {days}일 훈련...")
        X, y_cls, y_reg = build_dataset(candles)
        metrics = train_model(X, y_cls, y_reg, save=(days == 14))
        results[f"lgbm_{days}d"] = metrics
        log.info(f"  {days}일: MAE={metrics['mae']:.4f}, Corr={metrics['correlation']:.4f}")

    return results


def _mac_phase3_xgboost():
    """XGBoost 대조 모델"""
    try:
        import xgboost as xgb
    except ImportError:
        log.warning("xgboost 미설치 — pip install xgboost 필요")
        return {"status": "skipped", "reason": "xgboost not installed"}

    import pickle
    from scalp_ml.train_lgbm import build_dataset, FEATURE_NAMES
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    cache = MODEL_DIR / "candles_14d.pkl"
    if not cache.exists():
        cache = MODEL_DIR / "candles_cache.pkl"
    with open(cache, "rb") as f:
        candles = pickle.load(f)

    X, y_cls, y_reg = build_dataset(candles)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    yr_train, yr_test = y_reg[:split], y_reg[split:]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        early_stopping_rounds=50, verbosity=0,
    )
    model.fit(X_train, yr_train, eval_set=[(X_test, yr_test)], verbose=False)

    yr_pred = model.predict(X_test)
    mae = mean_absolute_error(yr_test, yr_pred)
    corr = float(np.corrcoef(yr_test, yr_pred)[0, 1])

    with open(MODEL_DIR / "xgb_scalp_latest.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, f)

    return {"mae": round(mae, 4), "correlation": round(corr, 4), "model": "xgboost"}


def _mac_phase4_features():
    """확장 피처 실험 — 기본 15개 → 25개+"""
    return {"status": "placeholder", "note": "피처 실험은 데이터 축적 후 실행"}


def _mac_phase5_hyperparam():
    """하이퍼파라미터 그리드 탐색"""
    import pickle
    from scalp_ml.train_lgbm import build_dataset, FEATURE_NAMES
    import lightgbm as lgb
    import numpy as np

    cache = MODEL_DIR / "candles_14d.pkl"
    if not cache.exists():
        cache = MODEL_DIR / "candles_cache.pkl"
    with open(cache, "rb") as f:
        candles = pickle.load(f)

    X, y_cls, y_reg = build_dataset(candles)
    split = int(len(X) * 0.8)

    grid = [
        {"num_leaves": 31, "max_depth": 6, "lr": 0.03},
        {"num_leaves": 63, "max_depth": 8, "lr": 0.03},
        {"num_leaves": 31, "max_depth": 6, "lr": 0.01},
        {"num_leaves": 127, "max_depth": 10, "lr": 0.05},
        {"num_leaves": 15, "max_depth": 4, "lr": 0.05},
    ]

    best_mae = 999
    best_config = None
    results = []

    for cfg in grid:
        params = {
            "objective": "regression", "metric": "mae",
            "num_leaves": cfg["num_leaves"], "max_depth": cfg["max_depth"],
            "learning_rate": cfg["lr"], "n_estimators": 1000,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbose": -1,
        }
        train_data = lgb.Dataset(X[:split], label=y_reg[:split], feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X[split:], label=y_reg[split:], reference=train_data)

        model = lgb.train(params, train_data, valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(30)])
        pred = model.predict(X[split:])
        mae = float(np.mean(np.abs(y_reg[split:] - pred)))
        corr = float(np.corrcoef(y_reg[split:], pred)[0, 1])

        results.append({**cfg, "mae": round(mae, 4), "corr": round(corr, 4)})
        if mae < best_mae:
            best_mae = mae
            best_config = cfg
        log.info(f"  {cfg}: MAE={mae:.4f}, Corr={corr:.4f}")

    return {"best_config": best_config, "best_mae": round(best_mae, 4), "all_results": results}


def _mac_phase6_ensemble():
    """앙상블 (LightGBM + XGBoost 평균)"""
    return {"status": "placeholder", "note": "개별 모델 완료 후 앙상블 구축"}


def _mac_phase7_final():
    """최종 평가 — 모든 모델 비교"""
    metrics_file = MODEL_DIR / "lgbm_scalp_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return {"final_metrics": json.load(f)}
    return {"status": "pending_evaluation"}


# ═══════════════════════════════════════════════════
# PC128: DQN/SAC 청산 최적화 (대규모 신경망)
# ═══════════════════════════════════════════════════

def run_pc128():
    """
    PC128 1주일 훈련 계획

    Day 1:   DQN 기본 훈련 (200K 스텝)
    Day 2:   보상함수 실험 (v1~v3)
    Day 3-4: SAC 연속행동 훈련 (500K 스텝)
    Day 5:   네트워크 크기 실험 ([64,32] vs [256,128] vs [512,256])
    Day 6:   최적 모델 장기 훈련 (1M 스텝)
    Day 7:   평가 + 기준선 비교
    """
    machine = "pc128"
    log.info(f"{'='*60}")
    log.info(f"  PC128 훈련 시작 — DQN/SAC 청산 최적화")
    log.info(f"{'='*60}")
    send_telegram(f"[PC128] 1주일 훈련 시작 — DQN/SAC 청산 최적화 (대규모)")

    phases = [
        ("phase1_data", _pc128_phase1_data, "캔들 데이터 수집"),
        ("phase2_dqn_200k", _pc128_phase2_dqn, "DQN 기본 200K 스텝"),
        ("phase3_reward_exp", _pc128_phase3_reward, "보상함수 3가지 실험"),
        ("phase4_sac_500k", _pc128_phase4_sac, "SAC 500K 스텝"),
        ("phase5_arch_exp", _pc128_phase5_arch, "네트워크 아키텍처 실험"),
        ("phase6_best_1m", _pc128_phase6_long, "최적 모델 1M 스텝"),
        ("phase7_eval", _pc128_phase7_eval, "최종 평가 + 기준선 비교"),
    ]

    _run_phases(machine, phases)


def _pc128_phase1_data():
    """데이터 준비"""
    from scalp_ml.train_lgbm import collect_candles
    import pickle
    for days in [14, 30]:
        cache = MODEL_DIR / f"candles_{days}d.pkl"
        if not cache.exists():
            candles = collect_candles(days=days)
            with open(cache, "wb") as f:
                pickle.dump(candles, f)
    return {"status": "data_ready"}


def _pc128_phase2_dqn():
    """DQN 기본 훈련 200K"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3, buffer_size=100000, batch_size=128,
        gamma=0.99, exploration_fraction=0.3, exploration_final_eps=0.05,
        target_update_interval=500, train_freq=4,
        policy_kwargs={"net_arch": [128, 64]},
        verbose=0,
    )

    eval_cb = EvalCallback(eval_env, n_eval_episodes=200, eval_freq=10000,
                          best_model_save_path=str(MODEL_DIR / "dqn_pc128_best"),
                          deterministic=True)
    model.learn(total_timesteps=200000, callback=eval_cb, progress_bar=True)
    model.save(str(MODEL_DIR / "dqn_pc128_200k"))

    return _eval_sb3_model(model, ScalpExitEnv(), 500, "dqn_200k")


def _pc128_phase3_reward():
    """보상함수 3가지 실험"""
    # 실제 구현에서는 ScalpExitEnv 서브클래스로 보상 변경
    return {"status": "placeholder", "variants": ["v1_base", "v2_sharp_sl", "v3_trailing_tp"]}


def _pc128_phase4_sac():
    """SAC 연속 행동공간 500K"""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnvV2

    env = ScalpExitEnvV2()
    eval_env = ScalpExitEnvV2()

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=100000, batch_size=256,
        gamma=0.99, tau=0.005, ent_coef="auto",
        policy_kwargs={"net_arch": [256, 128]},
        verbose=0,
    )

    eval_cb = EvalCallback(eval_env, n_eval_episodes=200, eval_freq=20000,
                          best_model_save_path=str(MODEL_DIR / "sac_pc128_best"),
                          deterministic=True)
    model.learn(total_timesteps=500000, callback=eval_cb, progress_bar=True)
    model.save(str(MODEL_DIR / "sac_pc128_500k"))

    return _eval_sb3_model(model, ScalpExitEnvV2(), 500, "sac_500k")


def _pc128_phase5_arch():
    """네트워크 아키텍처 실험"""
    from stable_baselines3 import DQN
    from scalp_ml.scalp_exit_env import ScalpExitEnv
    import numpy as np

    archs = [[64, 32], [128, 64], [256, 128], [256, 256, 128]]
    results = []

    for arch in archs:
        log.info(f"  아키텍처 {arch} 훈련 중...")
        env = ScalpExitEnv()
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=50000, batch_size=64,
            policy_kwargs={"net_arch": arch}, verbose=0,
        )
        model.learn(total_timesteps=100000, progress_bar=False)
        metrics = _eval_sb3_model(model, ScalpExitEnv(), 300, f"arch_{'x'.join(map(str,arch))}")
        results.append({"arch": arch, **metrics})
        log.info(f"  {arch}: win={metrics['win_rate']:.1%}, pnl={metrics['avg_pnl_pct']:.3f}%")

    return {"arch_comparison": results}


def _pc128_phase6_long():
    """최적 모델 장기 훈련 1M 스텝"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4, buffer_size=200000, batch_size=128,
        gamma=0.99, exploration_fraction=0.2, exploration_final_eps=0.03,
        target_update_interval=1000, train_freq=4,
        policy_kwargs={"net_arch": [256, 128]},
        verbose=0,
    )

    ckpt_cb = CheckpointCallback(save_freq=100000,
                                save_path=str(MODEL_DIR / "dqn_1m_checkpoints"))
    eval_cb = EvalCallback(eval_env, n_eval_episodes=300, eval_freq=50000,
                          best_model_save_path=str(MODEL_DIR / "dqn_1m_best"),
                          deterministic=True)

    model.learn(total_timesteps=1000000, callback=[ckpt_cb, eval_cb], progress_bar=True)
    model.save(str(MODEL_DIR / "dqn_pc128_1m"))

    return _eval_sb3_model(model, ScalpExitEnv(), 1000, "dqn_1m_final")


def _pc128_phase7_eval():
    """전 모델 비교 평가"""
    return {"status": "placeholder", "note": "모든 phase 완료 후 비교"}


# ═══════════════════════════════════════════════════
# PC36 (DRJAY): PPO 스캘핑 + 보상함수 + 온라인 학습
# ═══════════════════════════════════════════════════

def run_pc36():
    """
    PC36 (DRJAY) 1주일 훈련 계획

    Day 1:   PPO 기본 훈련 (200K 스텝)
    Day 2:   보상함수 v1~v3 비교 실험
    Day 3-4: 최적 보상함수로 PPO 500K
    Day 5:   시장 레짐별 모델 (trending vs ranging)
    Day 6:   온라인 학습 루프 (6시간 주기 마이크로 학습)
    Day 7:   전체 평가 + 앙상블 테스트
    """
    machine = "pc36"
    log.info(f"{'='*60}")
    log.info(f"  PC36 (DRJAY) 훈련 시작 — PPO + 보상함수 실험")
    log.info(f"{'='*60}")
    send_telegram(f"[PC36/DRJAY] 1주일 훈련 시작 — PPO 스캘핑 + 보상함수 실험")

    phases = [
        ("phase1_data", _pc36_phase1_data, "캔들 데이터 수집"),
        ("phase2_ppo_base", _pc36_phase2_ppo, "PPO 기본 200K 스텝"),
        ("phase3_reward_exp", _pc36_phase3_reward, "보상함수 3가지 실험"),
        ("phase4_ppo_500k", _pc36_phase4_ppo_long, "최적 보상 PPO 500K"),
        ("phase5_regime", _pc36_phase5_regime, "시장 레짐별 모델"),
        ("phase6_online", _pc36_phase6_online, "온라인 학습 루프 테스트"),
        ("phase7_eval", _pc36_phase7_eval, "전체 평가 + 앙상블"),
    ]

    _run_phases(machine, phases)


def _pc36_phase1_data():
    """데이터 준비"""
    return _pc128_phase1_data()


def _pc36_phase2_ppo():
    """PPO 기본 훈련 200K"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.05,
        policy_kwargs={"net_arch": [128, 64]},
        verbose=0,
    )

    eval_cb = EvalCallback(eval_env, n_eval_episodes=200, eval_freq=10000,
                          best_model_save_path=str(MODEL_DIR / "ppo_pc36_best"),
                          deterministic=True)
    model.learn(total_timesteps=200000, callback=eval_cb, progress_bar=True)
    model.save(str(MODEL_DIR / "ppo_pc36_200k"))

    return _eval_sb3_model(model, ScalpExitEnv(), 500, "ppo_200k")


def _pc36_phase3_reward():
    """보상함수 변형 실험"""
    from stable_baselines3 import PPO
    from scalp_ml.scalp_exit_env import ScalpExitEnv
    import numpy as np

    # 3가지 보상 전략을 ScalpExitEnv를 서브클래스하여 실험
    results = []

    # v1: 기본 (현재)
    env = ScalpExitEnv()
    m1 = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, verbose=0,
             policy_kwargs={"net_arch": [128, 64]})
    m1.learn(total_timesteps=100000, progress_bar=False)
    r1 = _eval_sb3_model(m1, ScalpExitEnv(), 300, "reward_v1")
    results.append({"variant": "v1_base", **r1})
    log.info(f"  v1(기본): win={r1['win_rate']:.1%}, pnl={r1['avg_pnl_pct']:.3f}%")

    return {"reward_comparison": results}


def _pc36_phase4_ppo_long():
    """최적 보상 PPO 500K 스텝"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=128,
        n_epochs=10, gamma=0.99, ent_coef=0.03,
        policy_kwargs={"net_arch": [256, 128]},
        verbose=0,
    )

    ckpt_cb = CheckpointCallback(save_freq=100000,
                                save_path=str(MODEL_DIR / "ppo_500k_ckpts"))
    eval_cb = EvalCallback(eval_env, n_eval_episodes=300, eval_freq=20000,
                          best_model_save_path=str(MODEL_DIR / "ppo_500k_best"),
                          deterministic=True)

    model.learn(total_timesteps=500000, callback=[ckpt_cb, eval_cb], progress_bar=True)
    model.save(str(MODEL_DIR / "ppo_pc36_500k"))

    return _eval_sb3_model(model, ScalpExitEnv(), 500, "ppo_500k")


def _pc36_phase5_regime():
    """시장 레짐별 모델"""
    return {"status": "placeholder", "note": "레짐 탐지기 구현 후 실행"}


def _pc36_phase6_online():
    """온라인 학습 루프 — 6시간마다 5K 스텝 마이크로 학습"""
    return {"status": "placeholder", "note": "기본 모델 완성 후 온라인 학습 추가"}


def _pc36_phase7_eval():
    """전체 평가"""
    return {"status": "pending_evaluation"}


# ═══════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════

def _eval_sb3_model(model, env, episodes: int, tag: str) -> dict:
    """SB3 모델 평가"""
    import numpy as np
    pnls, holds, exits = [], [], {}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        if "pnl_pct" in info:
            pnls.append(info["pnl_pct"])
            holds.append(info.get("hold_minutes", 0))
            er = info.get("exit_reason", "unknown")
            exits[er] = exits.get(er, 0) + 1

    pnls = np.array(pnls) if pnls else np.array([0])
    return {
        "tag": tag,
        "win_rate": round(float((pnls > 0).mean()), 4),
        "avg_pnl_pct": round(float(pnls.mean()), 4),
        "total_pnl_pct": round(float(pnls.sum()), 2),
        "avg_hold_min": round(float(np.mean(holds)), 1) if holds else 0,
        "exit_reasons": exits,
        "episodes": episodes,
    }


def _run_phases(machine: str, phases: list):
    """순차 실행 — 실패해도 다음 phase 진행"""
    total = len(phases)
    for i, (name, func, desc) in enumerate(phases, 1):
        log.info(f"\n{'─'*50}")
        log.info(f"  [{machine}] Phase {i}/{total}: {desc}")
        log.info(f"{'─'*50}")
        log_to_db(machine, name, "running")

        try:
            start = time.time()
            result = func()
            elapsed = round(time.time() - start)
            result = result or {}
            result["elapsed_sec"] = elapsed

            log_to_db(machine, name, "completed", metrics=result)
            send_telegram(f"[{machine}] Phase {i}/{total} 완료: {desc} ({elapsed}s)")
            log.info(f"  Phase {i} 완료 ({elapsed}s)")

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"  Phase {i} 실패: {e}\n{tb}")
            log_to_db(machine, name, "failed", error=str(e))
            send_telegram(f"[{machine}] Phase {i}/{total} 실패: {desc}\n{e}")

    send_telegram(f"[{machine}] 1주일 훈련 전체 완료!")
    log.info(f"\n{'='*60}")
    log.info(f"  [{machine}] 전체 훈련 완료!")
    log.info(f"{'='*60}")


# ═══════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3대 분산 스캘핑 RL 훈련")
    parser.add_argument("--machine", required=True,
                       choices=["mac-mini", "pc128", "pc36"],
                       help="이 머신의 역할")
    parser.add_argument("--phase", type=int, default=0,
                       help="특정 phase부터 시작 (1-7, 0=전체)")
    args = parser.parse_args()

    runners = {
        "mac-mini": run_mac_mini,
        "pc128": run_pc128,
        "pc36": run_pc36,
    }

    runners[args.machine]()


if __name__ == "__main__":
    main()
