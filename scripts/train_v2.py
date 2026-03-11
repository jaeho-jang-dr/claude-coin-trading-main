"""RL 강화학습 v2 — 하이퍼파라미터 튜닝 + 보상함수 v7 + 커리큘럼 학습

개선사항:
  1. 보상함수 v7 (트렌드 팔로잉 + 적응형 MDD)
  2. 학습률 스케줄링 (linear decay)
  3. 엔트로피 계수 증가 (탐색 강화)
  4. 커리큘럼 학습: 짧은 에피소드 → 긴 에피소드
  5. Early stopping (성능 저하 시 학습 중단)
  6. 기존 모델과 비교 평가
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.reward_v7 import RewardCalculatorV7
from rl_hybrid.rl.policy import MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_v2")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "training_results",
)


class BitcoinTradingEnvV7(BitcoinTradingEnv):
    """보상함수 v7을 사용하는 환경"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_calc = RewardCalculatorV7()

    def step(self, action):
        action_val = float(np.clip(action[0], -1, 1))
        candle = self.candles[self.current_step]
        price = candle["close"]
        prev_value = self._portfolio_value(price)

        self._execute_action(action_val, price)

        self.current_step += 1
        next_candle = self.candles[self.current_step]
        next_price = next_candle["close"]
        curr_value = self._portfolio_value(next_price)

        self.total_value_history.append(curr_value)
        self.action_history.append(action_val)

        reward_info = self.reward_calc.calculate(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=curr_value,
            action=action_val,
            prev_action=self.prev_action,
            step=self.current_step,
            price=next_price,
        )
        self.prev_action = action_val

        terminated = False
        truncated = self.current_step >= self.end_idx

        if curr_value < self.initial_balance * 0.1:
            terminated = True
            reward_info["reward"] -= 1.0

        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_info["components"]

        return obs, reward_info["reward"], terminated, truncated, info


class EarlyStoppingCallback(BaseCallback):
    """성능 저하 시 학습 중단"""

    def __init__(self, check_freq=10000, patience=3, min_improvement=0.01, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_reward = -np.inf
        self.no_improve_count = 0

    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            # rollout 평균 보상 확인
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                if mean_reward > self.best_reward + self.min_improvement:
                    self.best_reward = mean_reward
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                if self.verbose:
                    logger.info(
                        f"[EarlyStopping] step={self.num_timesteps} "
                        f"mean_reward={mean_reward:.4f} best={self.best_reward:.4f} "
                        f"patience={self.no_improve_count}/{self.patience}"
                    )

                if self.no_improve_count >= self.patience:
                    logger.info(f"Early stopping at step {self.num_timesteps}")
                    return False
        return True


def linear_schedule(initial_value: float):
    """학습률 선형 감소 스케줄"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def evaluate_model(model, env, episodes=10, label=""):
    """모델 평가"""
    all_stats = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        stats = env.get_episode_stats()
        all_stats.append(stats)

    avg = {
        "return_pct": np.mean([s["total_return_pct"] for s in all_stats]),
        "sharpe": np.mean([s["sharpe_ratio"] for s in all_stats]),
        "mdd": np.mean([s["max_drawdown"] for s in all_stats]),
        "trades": np.mean([s["trade_count"] for s in all_stats]),
    }
    logger.info(
        f"[{label}] 수익률: {avg['return_pct']:.2f}% | "
        f"샤프: {avg['sharpe']:.3f} | MDD: {avg['mdd']:.2%} | "
        f"거래: {avg['trades']:.0f}회"
    )
    return avg


def buy_and_hold(candles):
    initial = candles[0]["close"]
    final = candles[-1]["close"]
    ret = (final - initial) / initial * 100
    prices = [c["close"] for c in candles]
    peak = prices[0]
    max_dd = 0
    for p in prices:
        peak = max(peak, p)
        max_dd = max(max_dd, (peak - p) / peak)
    return {"return_pct": ret, "mdd": max_dd}


def run_curriculum_training(balance=10_000_000):
    """커리큘럼 학습: 3단계 점진적 학습"""

    loader = HistoricalDataLoader()
    all_results = {"started_at": datetime.now().isoformat()}
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  v2 강화학습 시작 (보상v7 + 커리큘럼 + 스케줄링)")
    logger.info("=" * 60)

    # === Stage 1: 최근 데이터로 빠른 적응 (90일, 100K) ===
    logger.info("\n" + "=" * 60)
    logger.info("Stage 1: 최근 시장 적응 (90일, 100K 스텝)")
    logger.info("=" * 60)

    candles_90 = loader.compute_indicators(loader.load_candles(days=90, interval="1h"))
    split = int(len(candles_90) * 0.8)

    train_env = BitcoinTradingEnvV7(candles=candles_90[:split], initial_balance=balance)
    eval_env = BitcoinTradingEnvV7(candles=candles_90[split:], initial_balance=balance)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(5e-4),  # 높은 LR → 빠른 적응
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # v6 대비 2배 (탐색 강화)
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs={
            "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
        },
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v7"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    early_cb = EarlyStoppingCallback(check_freq=20_000, patience=5)

    start = time.time()
    model.learn(total_timesteps=100_000, callback=CallbackList([eval_cb, early_cb]))
    s1_time = time.time() - start

    s1_result = evaluate_model(model, eval_env, episodes=10, label="Stage 1")
    s1_result["elapsed_sec"] = s1_time
    all_results["stage1"] = s1_result
    model.save(os.path.join(MODEL_DIR, "v7_stage1"))

    # === Stage 2: 중기 패턴 학습 (180일, 200K) ===
    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: 중기 패턴 학습 (180일, 200K 스텝)")
    logger.info("=" * 60)

    candles_180 = loader.compute_indicators(loader.load_candles(days=180, interval="1h"))
    split = int(len(candles_180) * 0.8)

    train_env2 = BitcoinTradingEnvV7(candles=candles_180[:split], initial_balance=balance)
    eval_env2 = BitcoinTradingEnvV7(candles=candles_180[split:], initial_balance=balance)

    model.set_env(train_env2)

    eval_cb2 = EvalCallback(
        eval_env2,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v7"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    early_cb2 = EarlyStoppingCallback(check_freq=40_000, patience=4)

    start = time.time()
    model.learn(total_timesteps=200_000, callback=CallbackList([eval_cb2, early_cb2]), reset_num_timesteps=False)
    s2_time = time.time() - start

    s2_result = evaluate_model(model, eval_env2, episodes=10, label="Stage 2")
    s2_result["elapsed_sec"] = s2_time
    all_results["stage2"] = s2_result
    model.save(os.path.join(MODEL_DIR, "v7_stage2"))

    # === Stage 3: 장기 일반화 (365일, 200K, 낮은 LR) ===
    logger.info("\n" + "=" * 60)
    logger.info("Stage 3: 장기 일반화 (365일, 200K 스텝, LR 감소)")
    logger.info("=" * 60)

    candles_365 = loader.compute_indicators(loader.load_candles(days=365, interval="1h"))
    split = int(len(candles_365) * 0.8)

    train_env3 = BitcoinTradingEnvV7(candles=candles_365[:split], initial_balance=balance)
    eval_env3 = BitcoinTradingEnvV7(candles=candles_365[split:], initial_balance=balance)

    model.set_env(train_env3)
    # LR을 낮춰서 미세 조정 (과적합 방지)
    model.learning_rate = linear_schedule(1e-4)

    eval_cb3 = EvalCallback(
        eval_env3,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v7"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    early_cb3 = EarlyStoppingCallback(check_freq=40_000, patience=3)

    start = time.time()
    model.learn(total_timesteps=200_000, callback=CallbackList([eval_cb3, early_cb3]), reset_num_timesteps=False)
    s3_time = time.time() - start

    s3_result = evaluate_model(model, eval_env3, episodes=10, label="Stage 3")
    s3_result["elapsed_sec"] = s3_time
    all_results["stage3"] = s3_result
    model.save(os.path.join(MODEL_DIR, "v7_final"))
    model.save(os.path.join(MODEL_DIR, "ppo_btc_latest"))

    # === Stage 4: 멀티 타임프레임 미세조정 (4h, 100K) ===
    logger.info("\n" + "=" * 60)
    logger.info("Stage 4: 멀티 타임프레임 4h 미세조정 (180일, 100K)")
    logger.info("=" * 60)

    candles_4h = loader.compute_indicators(loader.load_candles(days=180, interval="4h"))
    split = int(len(candles_4h) * 0.8)

    train_env4 = BitcoinTradingEnvV7(candles=candles_4h[:split], initial_balance=balance)
    eval_env4 = BitcoinTradingEnvV7(candles=candles_4h[split:], initial_balance=balance)

    model.set_env(train_env4)
    model.learning_rate = linear_schedule(5e-5)  # 매우 낮은 LR

    eval_cb4 = EvalCallback(
        eval_env4,
        best_model_save_path=os.path.join(MODEL_DIR, "best_v7"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    start = time.time()
    model.learn(total_timesteps=100_000, callback=eval_cb4, reset_num_timesteps=False)
    s4_time = time.time() - start

    s4_result = evaluate_model(model, eval_env4, episodes=10, label="Stage 4 (4h)")
    s4_result["elapsed_sec"] = s4_time
    all_results["stage4_4h"] = s4_result
    model.save(os.path.join(MODEL_DIR, "v7_final"))
    model.save(os.path.join(MODEL_DIR, "ppo_btc_latest"))

    # === 최종 비교 평가 ===
    logger.info("\n" + "=" * 60)
    logger.info("최종 비교 평가")
    logger.info("=" * 60)

    final_eval = {}
    for days, label in [(90, "90일"), (180, "180일")]:
        candles = loader.compute_indicators(loader.load_candles(days=days, interval="1h"))
        split = int(len(candles) * 0.8)
        eval_c = candles[split:]
        e_env = BitcoinTradingEnvV7(candles=eval_c, initial_balance=balance)

        # v7 모델
        v7_result = evaluate_model(model, e_env, episodes=10, label=f"v7 {label}")

        # 기존 v6 모델 (있으면)
        v6_path = os.path.join(MODEL_DIR, "best", "best_model")
        v6_result = {"return_pct": 0, "sharpe": 0, "mdd": 0, "trades": 0}
        if os.path.exists(v6_path + ".zip"):
            e_env_v6 = BitcoinTradingEnv(candles=eval_c, initial_balance=balance)
            v6_model = PPO.load(v6_path, env=e_env_v6)
            v6_result_raw = []
            for ep in range(10):
                obs, _ = e_env_v6.reset()
                done = False
                while not done:
                    action, _ = v6_model.predict(obs, deterministic=True)
                    obs, r, t, tr, i = e_env_v6.step(action)
                    done = t or tr
                v6_result_raw.append(e_env_v6.get_episode_stats())
            v6_result = {
                "return_pct": np.mean([s["total_return_pct"] for s in v6_result_raw]),
                "sharpe": np.mean([s["sharpe_ratio"] for s in v6_result_raw]),
                "mdd": np.mean([s["max_drawdown"] for s in v6_result_raw]),
                "trades": np.mean([s["trade_count"] for s in v6_result_raw]),
            }
            logger.info(
                f"[v6 {label}] 수익률: {v6_result['return_pct']:.2f}% | "
                f"샤프: {v6_result['sharpe']:.3f} | MDD: {v6_result['mdd']:.2%}"
            )

        bnh = buy_and_hold(eval_c)
        logger.info(f"[B&H {label}] 수익률: {bnh['return_pct']:.2f}% | MDD: {bnh['mdd']:.2%}")

        improvement = v7_result["return_pct"] - v6_result["return_pct"]
        alpha = v7_result["return_pct"] - bnh["return_pct"]
        logger.info(f"[{label}] v7 vs v6: {improvement:+.2f}%p | v7 vs B&H: {alpha:+.2f}%p")

        final_eval[label] = {
            "v7": v7_result,
            "v6": v6_result,
            "buy_and_hold": bnh,
            "v7_vs_v6": improvement,
            "alpha_vs_bnh": alpha,
        }

    all_results["final_eval"] = final_eval
    total_elapsed = time.time() - total_start
    all_results["total_elapsed_sec"] = total_elapsed
    all_results["finished_at"] = datetime.now().isoformat()

    # 결과 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"v2_training_{timestamp}.json")

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info(f"  v2 학습 완료! 총 {total_elapsed/60:.1f}분")
    logger.info(f"  결과: {path}")
    logger.info("=" * 60)

    return all_results


if __name__ == "__main__":
    run_curriculum_training()
