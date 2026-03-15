#!/usr/bin/env python3
"""
스캘핑 청산 DQN 훈련

ScalpExitEnv에서 HOLD/TP/SL 결정을 학습한다.
진입은 규칙 기반, 청산 타이밍만 RL이 최적화.

실행:
  python -m scalp_ml.train_exit_dqn              # 기본 50K 스텝
  python -m scalp_ml.train_exit_dqn --steps 100000
  python -m scalp_ml.train_exit_dqn --eval        # 평가만
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_exit_dqn")

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "scalp_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train(total_steps: int = 50000, eval_episodes: int = 500):
    """DQN 훈련"""
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        log.error("stable-baselines3 필요: pip install stable-baselines3")
        # fallback: 순수 numpy DQN
        return train_simple_dqn(total_steps, eval_episodes)

    from scalp_ml.scalp_exit_env import ScalpExitEnv

    log.info(f"=== DQN 훈련 시작 ({total_steps:,} 스텝) ===")

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        policy_kwargs={"net_arch": [128, 64]},
        verbose=0,
    )

    # 콜백: 매 5000스텝 평가
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=100,
        eval_freq=5000,
        best_model_save_path=str(MODEL_DIR / "dqn_exit_best"),
        log_path=str(MODEL_DIR / "dqn_exit_logs"),
        deterministic=True,
    )

    start = time.time()
    model.learn(total_timesteps=total_steps, callback=eval_cb, progress_bar=True)
    elapsed = time.time() - start

    # 최종 평가
    metrics = evaluate_model(model, eval_env, eval_episodes)
    metrics["training_steps"] = total_steps
    metrics["training_time_sec"] = round(elapsed)

    # 저장
    model.save(str(MODEL_DIR / "dqn_exit_latest"))
    log.info(f"모델 저장: {MODEL_DIR / 'dqn_exit_latest.zip'}")

    with open(MODEL_DIR / "dqn_exit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_simple_dqn(total_steps: int = 50000, eval_episodes: int = 500):
    """stable-baselines3 없을 때 간단한 Q-learning"""
    from scalp_ml.scalp_exit_env import ScalpExitEnv
    import random
    from collections import deque

    log.info(f"=== Simple Q-Learning 훈련 ({total_steps:,} 스텝) ===")
    log.info("  (stable-baselines3 없음 — 테이블 Q-learning 사용)")

    env = ScalpExitEnv()

    # 상태를 이산화하여 Q-테이블 사용
    # [pnl_bucket, hold_bucket, mom_bucket]
    def discretize(obs):
        pnl = int(np.clip(obs[0] * 2, -5, 5)) + 5      # 11 bins
        hold = int(min(obs[1], 10))                       # 11 bins
        mom = int(np.clip(obs[2] * 2, -3, 3)) + 3        # 7 bins
        rsi_b = int(obs[5] // 20)                         # 5 bins
        return (pnl, hold, mom, rsi_b)

    q_table = {}
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05

    replay_buffer = deque(maxlen=10000)
    batch_size = 32

    episode_rewards = []
    episode_pnls = []
    start = time.time()

    obs, _ = env.reset()
    ep_reward = 0

    for step in range(total_steps):
        state = discretize(obs)

        # epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = [q_table.get((state, a), 0.0) for a in range(3)]
            action = int(np.argmax(q_vals))

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        next_state = discretize(next_obs)
        replay_buffer.append((state, action, reward, next_state, terminated))

        # 배치 학습
        if len(replay_buffer) >= batch_size:
            batch = random.sample(list(replay_buffer), batch_size)
            for s, a, r, ns, done in batch:
                q_old = q_table.get((s, a), 0.0)
                if done:
                    q_target = r
                else:
                    q_next = max(q_table.get((ns, na), 0.0) for na in range(3))
                    q_target = r + gamma * q_next
                q_table[(s, a)] = q_old + alpha * (q_target - q_old)

        if terminated:
            episode_rewards.append(ep_reward)
            if "pnl_pct" in info:
                episode_pnls.append(info["pnl_pct"])
            obs, _ = env.reset()
            ep_reward = 0
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        else:
            obs = next_obs

        if (step + 1) % 10000 == 0:
            recent_rew = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            recent_pnl = np.mean(episode_pnls[-100:]) if episode_pnls else 0
            win_rate = env.win_rate
            log.info(f"  Step {step+1:,}: eps={epsilon:.3f}, "
                     f"reward={recent_rew:.3f}, pnl={recent_pnl:.3f}%, "
                     f"win={win_rate:.1%}, Q-states={len(q_table)}")

    elapsed = time.time() - start

    # 평가
    log.info(f"\n=== 평가 ({eval_episodes} 에피소드) ===")
    eval_env = ScalpExitEnv()
    eval_pnls = []
    eval_exits = {"take_profit": 0, "stop_loss": 0, "timeout": 0, "forced_sl": 0}
    eval_holds = []

    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            state = discretize(obs)
            q_vals = [q_table.get((state, a), 0.0) for a in range(3)]
            action = int(np.argmax(q_vals))
            obs, _, done, _, info = eval_env.step(action)

        if "pnl_pct" in info:
            eval_pnls.append(info["pnl_pct"])
            exit_r = info.get("exit_reason", "unknown")
            if exit_r in eval_exits:
                eval_exits[exit_r] += 1
            eval_holds.append(info.get("hold_minutes", 0))

    eval_pnls = np.array(eval_pnls)
    win_rate = (eval_pnls > 0).mean() if len(eval_pnls) > 0 else 0
    avg_pnl = eval_pnls.mean() if len(eval_pnls) > 0 else 0
    total_pnl = eval_pnls.sum() if len(eval_pnls) > 0 else 0

    metrics = {
        "model_type": "q_learning",
        "win_rate": round(float(win_rate), 4),
        "avg_pnl_pct": round(float(avg_pnl), 4),
        "total_pnl_pct": round(float(total_pnl), 2),
        "avg_hold_min": round(float(np.mean(eval_holds)), 1) if eval_holds else 0,
        "exit_reasons": eval_exits,
        "episodes": eval_episodes,
        "training_steps": total_steps,
        "training_time_sec": round(elapsed),
        "q_states": len(q_table),
    }

    log.info(f"  승률: {win_rate:.1%}")
    log.info(f"  평균 PnL: {avg_pnl:.3f}%")
    log.info(f"  합계 PnL: {total_pnl:.2f}%")
    log.info(f"  평균 보유: {metrics['avg_hold_min']}분")
    log.info(f"  청산 이유: {eval_exits}")

    # 기준선 비교 (항상 HOLD)
    log.info(f"\n=== 기준선 비교 ===")
    baseline_env = ScalpExitEnv()
    baseline_pnls = []
    for _ in range(eval_episodes):
        obs, _ = baseline_env.reset()
        done = False
        while not done:
            obs, _, done, _, info = baseline_env.step(0)  # 항상 HOLD
        if "pnl_pct" in info:
            baseline_pnls.append(info["pnl_pct"])

    bl_pnls = np.array(baseline_pnls)
    bl_win = (bl_pnls > 0).mean() if len(bl_pnls) > 0 else 0
    bl_avg = bl_pnls.mean() if len(bl_pnls) > 0 else 0
    log.info(f"  기준선(항상HOLD): 승률 {bl_win:.1%}, 평균 {bl_avg:.3f}%")
    log.info(f"  RL 모델:          승률 {win_rate:.1%}, 평균 {avg_pnl:.3f}%")
    log.info(f"  개선:             승률 {(win_rate-bl_win)*100:+.1f}pp, "
             f"PnL {(avg_pnl-bl_avg)*100:+.1f}bp")

    metrics["baseline_win_rate"] = round(float(bl_win), 4)
    metrics["baseline_avg_pnl"] = round(float(bl_avg), 4)

    # 저장
    import pickle
    with open(MODEL_DIR / "q_table_exit.pkl", "wb") as f:
        pickle.dump({"q_table": q_table, "metrics": metrics}, f)
    log.info(f"\nQ-테이블 저장: {MODEL_DIR / 'q_table_exit.pkl'}")

    with open(MODEL_DIR / "dqn_exit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_model(model, env, episodes: int = 500) -> dict:
    """SB3 모델 평가"""
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

    pnls = np.array(pnls)
    return {
        "win_rate": round(float((pnls > 0).mean()), 4),
        "avg_pnl_pct": round(float(pnls.mean()), 4),
        "total_pnl_pct": round(float(pnls.sum()), 2),
        "avg_hold_min": round(float(np.mean(holds)), 1),
        "exit_reasons": exits,
        "episodes": episodes,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval", action="store_true", help="평가만")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    if args.eval:
        # 저장된 모델 평가
        from scalp_ml.scalp_exit_env import ScalpExitEnv
        try:
            from stable_baselines3 import DQN
            model = DQN.load(str(MODEL_DIR / "dqn_exit_latest"))
            env = ScalpExitEnv()
            metrics = evaluate_model(model, env, args.episodes)
        except Exception:
            import pickle
            with open(MODEL_DIR / "q_table_exit.pkl", "rb") as f:
                data = pickle.load(f)
            metrics = data["metrics"]
        log.info(json.dumps(metrics, indent=2))
        return

    metrics = train(total_steps=args.steps, eval_episodes=args.episodes)
    log.info(f"\n{'='*50}")
    log.info(f"훈련 완료!")
    log.info(f"  승률: {metrics['win_rate']:.1%}")
    log.info(f"  평균 PnL: {metrics['avg_pnl_pct']:.3f}%")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    main()
