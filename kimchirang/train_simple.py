"""Kimchirang Simple Training -- 김치프리미엄 RL 간단 학습

사용법:
  python -m kimchirang.train_simple                # 기본 50K 스텝, 90일
  python -m kimchirang.train_simple --steps 100000 # 스텝 변경
  python -m kimchirang.train_simple --days 180     # 데이터 기간 변경
"""

import argparse
import logging
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from kimchirang.rl_env import KPHistoricalData, KimchirangEnv

logger = logging.getLogger("kimchirang.train")


def evaluate(model, env, n_episodes: int = 5) -> dict:
    """학습된 모델 평가"""
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "total_pnl": info.get("total_pnl", 0),
            "trade_count": info.get("trade_count", 0),
        })

        logger.info(
            f"  평가 에피소드 {ep+1}: "
            f"PnL={info.get('total_pnl', 0):.2f}% "
            f"거래={info.get('trade_count', 0)}회 "
            f"보상={total_reward:.2f}"
        )

    avg_pnl = np.mean([r["total_pnl"] for r in results])
    avg_trades = np.mean([r["trade_count"] for r in results])

    return {
        "avg_pnl": avg_pnl,
        "avg_trades": avg_trades,
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="김치랑 RL 간단 학습")
    parser.add_argument("--steps", type=int, default=50_000, help="학습 스텝 수")
    parser.add_argument("--days", type=int, default=90, help="히스토리컬 데이터 일수")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(PROJECT_DIR, "logs", "kimchirang_train.log"),
                encoding="utf-8",
            ),
        ],
    )

    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)

    # Step 1: 데이터 수집
    logger.info(f"=== 김치랑 RL 학습 시작 ({args.steps:,} 스텝, {args.days}일) ===")

    kp_data = KPHistoricalData(days=args.days)
    if not kp_data.collect():
        logger.error("데이터 수집 실패 -- 종료")
        sys.exit(1)

    logger.info(f"KP 데이터: {len(kp_data.kp_series)}개 시간봉")
    logger.info(f"KP 범위: {min(kp_data.kp_series):.2f}% ~ {max(kp_data.kp_series):.2f}%")
    logger.info(f"KP 평균: {np.mean(kp_data.kp_series):.2f}%")

    # Step 2: 환경 생성
    env = KimchirangEnv(kp_data)

    # Step 3: PPO 학습
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        logger.error("stable-baselines3 미설치: pip install stable-baselines3")
        sys.exit(1)

    model_dir = os.path.join(PROJECT_DIR, "data", "rl_models", "kimchirang")
    os.makedirs(model_dir, exist_ok=True)

    eval_env = KimchirangEnv(kp_data)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=os.path.join(model_dir, "logs"),
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.05,  # 높은 탐색률로 다양한 행동 시도
        verbose=1,
        # tensorboard_log=os.path.join(model_dir, "tb_logs"),
    )

    logger.info("PPO 학습 시작...")
    model.learn(total_timesteps=args.steps, callback=eval_callback)

    # Step 4: 최종 모델 저장
    latest_path = os.path.join(model_dir, "ppo_kimchirang_latest")
    model.save(latest_path)
    logger.info(f"최종 모델 저장: {latest_path}")

    # Step 5: 평가
    logger.info("=== 학습 완료 — 평가 시작 ===")
    eval_result = evaluate(model, eval_env)

    logger.info(
        f"\n=== 평가 결과 ===\n"
        f"  평균 PnL: {eval_result['avg_pnl']:.2f}%\n"
        f"  평균 거래: {eval_result['avg_trades']:.0f}회\n"
    )

    # Best 모델도 평가
    best_path = os.path.join(model_dir, "best_model")
    if os.path.exists(best_path + ".zip"):
        logger.info("=== Best 모델 평가 ===")
        best_model = PPO.load(best_path)
        best_result = evaluate(best_model, eval_env)
        logger.info(
            f"  Best 평균 PnL: {best_result['avg_pnl']:.2f}%\n"
            f"  Best 평균 거래: {best_result['avg_trades']:.0f}회\n"
        )

    logger.info("=== 김치랑 RL 학습 완료 ===")


if __name__ == "__main__":
    main()
