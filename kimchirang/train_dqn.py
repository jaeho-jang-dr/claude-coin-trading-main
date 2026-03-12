"""Kimchirang DQN Training -- Experience Replay로 희귀 이벤트 반복 학습

DQN 장점 (김치랑):
  - Discrete(3) 액션에 최적화된 Q-value 직접 계산
  - Experience Replay: 10% 프리미엄, 대청산 같은 극단 이벤트를 반복 학습
  - 높은 샘플 효율: 같은 데이터로 더 많이 배움

사용법:
  python -m kimchirang.train_dqn                    # 기본 500K 스텝, 730일
  python -m kimchirang.train_dqn --steps 1000000    # 스텝 변경
  python -m kimchirang.train_dqn --days 365         # 데이터 기간 변경
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

logger = logging.getLogger("kimchirang.train_dqn")


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
    parser = argparse.ArgumentParser(description="김치랑 DQN 학습")
    parser.add_argument("--steps", type=int, default=500_000, help="학습 스텝 수")
    parser.add_argument("--days", type=int, default=730, help="히스토리컬 데이터 일수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="리플레이 버퍼 크기")
    parser.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(PROJECT_DIR, "logs", "kimchirang_train_dqn.log"),
                encoding="utf-8",
            ),
        ],
    )

    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)

    # Step 1: 데이터 수집
    logger.info(f"=== 김치랑 DQN 학습 시작 ({args.steps:,} 스텝, {args.days}일) ===")

    kp_data = KPHistoricalData(days=args.days)
    if not kp_data.collect():
        logger.error("데이터 수집 실패 -- 종료")
        sys.exit(1)

    logger.info(f"KP 데이터: {len(kp_data.kp_series)}개 시간봉")
    logger.info(f"KP 범위: {min(kp_data.kp_series):.2f}% ~ {max(kp_data.kp_series):.2f}%")
    logger.info(f"KP 평균: {np.mean(kp_data.kp_series):.2f}%")

    # Step 2: 환경 생성
    env = KimchirangEnv(kp_data)

    # Step 3: DQN 학습
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        logger.error("stable-baselines3 미설치: pip install stable-baselines3")
        sys.exit(1)

    model_dir = os.path.join(PROJECT_DIR, "data", "rl_models", "kimchirang")
    os.makedirs(model_dir, exist_ok=True)

    eval_env = KimchirangEnv(kp_data)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "dqn"),
        log_path=os.path.join(model_dir, "dqn", "logs"),
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,      # 리플레이 버퍼 (극단 이벤트 반복 학습)
        learning_starts=5000,              # 5K 스텝 탐색 후 학습 시작
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.005,                         # 소프트 타겟 업데이트
        train_freq=4,                      # 4스텝마다 학습
        target_update_interval=1000,       # 타겟 네트워크 업데이트 주기
        exploration_fraction=0.15,         # 15% 구간 탐색
        exploration_initial_eps=1.0,       # 초기 100% 랜덤
        exploration_final_eps=0.05,        # 최종 5% 랜덤
        policy_kwargs=dict(
            net_arch=[256, 256],           # 더 넓은 네트워크
        ),
        verbose=1,
    )

    logger.info("DQN 학습 시작...")
    logger.info(f"  리플레이 버퍼: {args.buffer_size:,}")
    logger.info(f"  배치 크기: {args.batch_size}")
    logger.info(f"  학습률: {args.lr}")
    model.learn(total_timesteps=args.steps, callback=eval_callback)

    # Step 4: 최종 모델 저장
    latest_path = os.path.join(model_dir, "dqn", "dqn_kimchirang_latest")
    model.save(latest_path)
    logger.info(f"최종 모델 저장: {latest_path}")

    # Step 5: 평가
    logger.info("=== DQN 학습 완료 — 평가 시작 ===")
    eval_result = evaluate(model, eval_env)

    logger.info(
        f"\n=== DQN 평가 결과 ===\n"
        f"  평균 PnL: {eval_result['avg_pnl']:.2f}%\n"
        f"  평균 거래: {eval_result['avg_trades']:.0f}회\n"
    )

    # Best 모델 평가
    best_path = os.path.join(model_dir, "dqn", "best_model")
    if os.path.exists(best_path + ".zip"):
        logger.info("=== DQN Best 모델 평가 ===")
        best_model = DQN.load(best_path)
        best_result = evaluate(best_model, eval_env)
        logger.info(
            f"  Best 평균 PnL: {best_result['avg_pnl']:.2f}%\n"
            f"  Best 평균 거래: {best_result['avg_trades']:.0f}회\n"
        )

        # PPO best와 비교
        ppo_best_path = os.path.join(model_dir, "best_model")
        if os.path.exists(ppo_best_path + ".zip"):
            from stable_baselines3 import PPO
            ppo_model = PPO.load(ppo_best_path)
            ppo_result = evaluate(ppo_model, eval_env)
            logger.info(
                f"\n=== PPO vs DQN 비교 ===\n"
                f"  PPO  — PnL: {ppo_result['avg_pnl']:.2f}%, 거래: {ppo_result['avg_trades']:.0f}회\n"
                f"  DQN  — PnL: {best_result['avg_pnl']:.2f}%, 거래: {best_result['avg_trades']:.0f}회\n"
            )

    logger.info("=== 김치랑 DQN 학습 완료 ===")


if __name__ == "__main__":
    main()
