"""SB3 기반 지속 학습 스크립트

v3 최적 모델(SB3 PPO)을 주기적으로 최신 데이터로 재학습한다.
기존 continuous_learner.py는 DistributedTrainer(PyTorch)용이므로,
SB3 PPO 모델에 맞게 새로 작성.

사용법:
  python scripts/continuous_learn_sb3.py              # 1회 학습 사이클
  python scripts/continuous_learn_sb3.py --daemon      # 백그라운드 루프
  python scripts/continuous_learn_sb3.py --interval 4  # 4시간 간격
"""

import hide_console
import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

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
logger = logging.getLogger("continuous_sb3")

# v8.2 최적 하이퍼파라미터 (정책 붕괴 방지)
BEST_HP = {
    "lr": 3e-4,
    "ent_coef": 0.08,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
}

REGISTRY_PATH = os.path.join(MODEL_DIR, "sb3_registry.json")


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def evaluate(model, env, episodes=10) -> dict:
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


def load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current": None, "versions": [], "rollbacks": 0}


def save_registry(reg: dict):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def retrain_cycle(
    incremental_steps: int = 50_000,
    data_days: int = 30,
    eval_episodes: int = 10,
    balance: int = 10_000_000,
) -> dict:
    """1회 재학습 사이클

    Returns:
        결과 dict (improved, baseline, new_stats, version_id)
    """
    start = time.time()
    loader = HistoricalDataLoader()

    # 1. 최신 데이터 로드
    logger.info(f"최근 {data_days}일 데이터 로드...")
    try:
        candles = loader.compute_indicators(
            loader.load_candles(days=data_days, interval="1h")
        )
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return {"error": str(e)}

    if len(candles) < 100:
        logger.warning(f"데이터 부족: {len(candles)}개")
        return {"error": "insufficient_data"}

    split = int(len(candles) * 0.8)
    train_c = candles[:split]
    eval_c = candles[split:]

    train_env = BitcoinTradingEnv(candles=train_c, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_c, initial_balance=balance)

    # 2. 현재 모델 로드
    model_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
    if not os.path.exists(model_path + ".zip"):
        model_path = os.path.join(MODEL_DIR, "v3_final")
    if not os.path.exists(model_path + ".zip"):
        logger.error("모델 파일 없음")
        return {"error": "no_model"}

    logger.info(f"모델 로드: {model_path}")
    model = PPO.load(model_path, env=train_env)

    # 3. 베이스라인 평가
    logger.info("베이스라인 평가...")
    baseline = evaluate(model, eval_env, episodes=eval_episodes)
    logger.info(
        f"베이스라인: return={baseline['return_pct']:.2f}%, "
        f"sharpe={baseline['sharpe']:.3f}, mdd={baseline['mdd']:.2%}"
    )

    # 4. 증분 학습 (낮은 LR로)
    logger.info(f"증분 학습: {incremental_steps} 스텝...")
    model.learning_rate = linear_schedule(BEST_HP["lr"] * 0.3)  # 기본의 30%
    model.ent_coef = BEST_HP["ent_coef"]

    save_dir = os.path.join(MODEL_DIR, "continuous_best")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=incremental_steps // 5,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model.learn(
        total_timesteps=incremental_steps,
        callback=eval_cb,
        reset_num_timesteps=True,
    )

    # 5. 학습 후 평가
    logger.info("학습 후 평가...")
    new_stats = evaluate(model, eval_env, episodes=eval_episodes)
    logger.info(
        f"학습 후: return={new_stats['return_pct']:.2f}%, "
        f"sharpe={new_stats['sharpe']:.3f}, mdd={new_stats['mdd']:.2%}"
    )

    # 6. 성능 비교 + 모델 저장/롤백
    registry = load_registry()
    improved = new_stats["sharpe"] > baseline["sharpe"] - 0.05

    result = {
        "baseline": baseline,
        "new_stats": new_stats,
        "improved": improved,
        "elapsed_sec": time.time() - start,
        "timestamp": datetime.now().isoformat(),
        "data_days": data_days,
        "incremental_steps": incremental_steps,
    }

    if improved:
        # 모델 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"sb3_{ts}"

        model.save(os.path.join(MODEL_DIR, "ppo_btc_latest"))

        version_dir = os.path.join(MODEL_DIR, "versions", version_id)
        os.makedirs(version_dir, exist_ok=True)
        model.save(os.path.join(version_dir, "model"))

        registry["versions"].append({
            "version_id": version_id,
            "metrics": new_stats,
            "baseline": baseline,
            "created_at": result["timestamp"],
        })
        registry["current"] = version_id
        save_registry(registry)

        result["version_id"] = version_id
        logger.info(f"모델 업데이트 완료: {version_id}")
    else:
        # 성능 저하 → 롤백 (기존 모델 유지)
        registry["rollbacks"] += 1
        save_registry(registry)

        logger.warning(
            f"성능 저하 → 롤백 (baseline sharpe={baseline['sharpe']:.3f}, "
            f"new sharpe={new_stats['sharpe']:.3f})"
        )

    logger.info(f"사이클 완료: {result['elapsed_sec']:.1f}초")
    return result


def daemon_loop(interval_hours: float, **kwargs):
    """백그라운드 재학습 루프"""
    interval_sec = interval_hours * 3600
    logger.info(f"지속 학습 데몬 시작 (간격: {interval_hours}h)")

    while True:
        try:
            result = retrain_cycle(**kwargs)
            if "error" not in result:
                status = "개선" if result["improved"] else "롤백"
                logger.info(
                    f"사이클 결과: {status}, "
                    f"sharpe: {result['baseline']['sharpe']:.3f} → "
                    f"{result['new_stats']['sharpe']:.3f}"
                )
        except Exception as e:
            logger.error(f"사이클 에러: {e}", exc_info=True)

        logger.info(f"다음 학습까지 {interval_hours}시간 대기...")
        time.sleep(interval_sec)


def main():
    parser = argparse.ArgumentParser(description="SB3 PPO 지속 학습")
    parser.add_argument("--daemon", action="store_true", help="백그라운드 루프 모드")
    parser.add_argument("--interval", type=float, default=6, help="학습 간격 (시간)")
    parser.add_argument("--steps", type=int, default=50_000, help="증분 학습 스텝")
    parser.add_argument("--days", type=int, default=30, help="학습 데이터 기간 (일)")
    parser.add_argument("--episodes", type=int, default=10, help="평가 에피소드 수")
    args = parser.parse_args()

    kwargs = {
        "incremental_steps": args.steps,
        "data_days": args.days,
        "eval_episodes": args.episodes,
    }

    if args.daemon:
        daemon_loop(interval_hours=args.interval, **kwargs)
    else:
        result = retrain_cycle(**kwargs)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
