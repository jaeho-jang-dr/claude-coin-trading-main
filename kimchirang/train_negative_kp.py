"""Kimchirang Negative KP Training -- 마이너스 KP 구간 전략 학습

현재 시장 상황 (2026-03): KP가 -1.5% ~ 0% 왔다갔다.
기존 모델은 KP +3% 진입만 학습했으므로, 마이너스 구간에서의 최적 행동을 학습한다.

학습 목표:
  1. Hold 최적화: KP가 마이너스일 때 진입하지 않고 기다리는 것이 최선임을 학습
  2. 반등 감지: KP가 -1.5%→0%→+로 올라갈 때 진입 타이밍 포착
  3. 역 프리미엄 활용: KP가 충분히 마이너스(-2% 이하)일 때 역방향 차익 가능성
  4. 과매매 방지: 좁은 레인지에서 쓸데없이 진입/청산 반복 억제

사용법:
  python -m kimchirang.train_negative_kp                    # 기본 2시간 학습
  python -m kimchirang.train_negative_kp --steps 500000     # 스텝 지정
  python -m kimchirang.train_negative_kp --days 180         # 데이터 기간
"""

import argparse
import logging
import os
import sys
import time as time_mod
from collections import deque

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import gymnasium as gym
from kimchirang.rl_env import KPHistoricalData, UPBIT_FEE, BINANCE_FEE, TOTAL_ROUND_TRIP

logger = logging.getLogger("kimchirang.train_neg_kp")


class NegativeKPEnv(gym.Env):
    """마이너스 KP 구간 특화 RL 환경

    기존 KimchirangEnv와 다른 점:
      1. Hold 보상: KP가 마이너스일 때 Hold하면 소폭 양의 보상 (올바른 판단)
      2. 잘못된 진입 패널티: KP < 0.5%에서 진입하면 강한 패널티
      3. 반등 보너스: KP가 상승 추세에서 적절히 진입하면 보너스
      4. 레인지 판단: 좁은 KP 변동폭에서 거래하면 패널티 (수수료 > 수익)
      5. 역 프리미엄 진입: KP가 -2% 이하에서 역방향 진입 보상
    """

    metadata = {"render_modes": []}

    def __init__(self, kp_data: KPHistoricalData, window: int = 20):
        super().__init__()

        self.kp_data = kp_data
        self.window = window

        # 12차원 관측 (기존과 동일 구조)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        # 0=Hold, 1=Enter(정방향: KP+ 차익), 2=Exit, 3=ReverseEnter(역방향: KP- 차익)
        self.action_space = gym.spaces.Discrete(4)

        self._reset_state()

    def _reset_state(self):
        self._idx = 0
        self._position_open = False
        self._position_direction = None  # 'forward' or 'reverse'
        self._entry_kp = 0.0
        self._entry_idx = 0
        self._total_pnl = 0.0
        self._trade_count = 0
        self._last_trade_idx = 0
        self._consecutive_holds = 0
        self._kp_history = deque(maxlen=max(self.window * 5, 100))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self._idx = self.window

        for i in range(self.window):
            self._kp_history.append(self.kp_data.kp_series[i])

        return self._get_obs(), {}

    def step(self, action):
        action_val = int(action)
        reward = 0.0

        mid_kp = self.kp_data.kp_series[self._idx]
        entry_kp = self.kp_data.entry_kp_series[self._idx]
        exit_kp = self.kp_data.exit_kp_series[self._idx]

        self._kp_history.append(mid_kp)
        history = list(self._kp_history)
        arr = np.array(history)

        steps_since_last = self._idx - self._last_trade_idx

        # KP 통계
        kp_ma = np.mean(arr[-20:]) if len(arr) >= 20 else np.mean(arr)
        kp_std = np.std(arr[-20:]) if len(arr) >= 20 else max(np.std(arr), 0.01)
        kp_velocity = arr[-1] - arr[-2] if len(arr) >= 2 else 0.0
        kp_range = max(arr[-20:]) - min(arr[-20:]) if len(arr) >= 20 else 0.0

        # === 액션 처리 ===

        if action_val == 1 and not self._position_open:
            # ENTER 정방향 (KP+ 차익: Upbit 매수, Binance 숏)
            self._position_open = True
            self._position_direction = 'forward'
            self._entry_kp = entry_kp
            self._entry_idx = self._idx
            self._last_trade_idx = self._idx
            self._consecutive_holds = 0

            if mid_kp < 0.5:
                # KP가 낮은데 정방향 진입 = 나쁜 판단
                penalty = min(abs(mid_kp) * 0.15, 0.5)
                reward = -0.1 - penalty
            elif mid_kp >= 2.0 and kp_velocity > 0:
                # KP 2%+ & 상승 추세에서 진입 = 아주 좋음
                reward = 0.2
            elif mid_kp >= 1.0:
                # KP 1%+ 진입 = 괜찮음
                reward = 0.05
            else:
                # KP 0.5~1% = 수수료 고려하면 애매
                reward = -0.02

            if steps_since_last < 4:
                reward -= 0.2

        elif action_val == 3 and not self._position_open:
            # REVERSE ENTER (역 프리미엄 차익: Upbit 매도, Binance 롱)
            # KP가 충분히 마이너스일 때 역방향 진입
            self._position_open = True
            self._position_direction = 'reverse'
            self._entry_kp = mid_kp  # 마이너스 값
            self._entry_idx = self._idx
            self._last_trade_idx = self._idx
            self._consecutive_holds = 0

            if mid_kp > -1.0:
                # KP가 별로 안 내려갔는데 역방향 진입 = 위험
                reward = -0.15
            elif mid_kp <= -2.0 and kp_velocity < 0:
                # KP -2% 이하 & 계속 하락 = 역방향 차익 기회
                reward = 0.15
            elif mid_kp <= -1.5:
                # KP -1.5% = 괜찮은 역방향 진입
                reward = 0.05
            else:
                reward = -0.05

            if steps_since_last < 4:
                reward -= 0.2

        elif action_val == 2 and self._position_open:
            # EXIT (청산)
            if self._position_direction == 'forward':
                # 정방향: entry_kp - exit_kp = 수익
                kp_profit = self._entry_kp - exit_kp
            else:
                # 역방향: KP가 0으로 회귀하면 수익
                # entry_kp (음수) → 현재 mid_kp (더 0에 가까우면 수익)
                kp_profit = mid_kp - self._entry_kp  # 예: 0% - (-2%) = +2%

            net_profit = kp_profit - TOTAL_ROUND_TRIP * 100
            reward = net_profit

            # 수수료보다 작은 수익이면 추가 패널티 (쓸데없는 거래)
            if 0 < kp_profit < TOTAL_ROUND_TRIP * 100:
                reward -= 0.1

            self._total_pnl += net_profit
            self._trade_count += 1
            self._position_open = False
            self._position_direction = None
            self._last_trade_idx = self._idx
            self._consecutive_holds = 0

        elif (action_val == 1 or action_val == 3) and self._position_open:
            # 이미 포지션인데 진입 시도
            reward = -0.02

        elif action_val == 2 and not self._position_open:
            # 포지션 없는데 청산 시도
            reward = -0.02

        elif action_val == 0:
            # HOLD
            self._consecutive_holds += 1

            if self._position_open:
                # 포지션 보유 중 Hold
                if self._position_direction == 'forward':
                    unrealized = self._entry_kp - mid_kp
                else:
                    unrealized = mid_kp - self._entry_kp
                reward = unrealized * 0.005

                hold_hours = self._idx - self._entry_idx
                if hold_hours > 48:
                    reward -= 0.003
            else:
                # 포지션 없이 Hold
                if mid_kp < 0.5 and mid_kp > -1.5:
                    # KP가 -1.5% ~ 0.5% 레인지 = Hold가 정답
                    reward = 0.01  # 올바른 판단 보상
                elif mid_kp >= 2.0:
                    # KP 2%+인데 Hold = 기회 놓침
                    reward = -0.03
                elif mid_kp <= -2.0:
                    # KP -2% 이하인데 Hold = 역방향 기회 놓침 (약한 패널티)
                    reward = -0.01
                else:
                    reward = 0.0

                # 좁은 레인지 (KP 변동폭 0.5% 미만) = Hold 보너스
                if kp_range < 0.5:
                    reward += 0.005

        # 다음 스텝
        self._idx += 1
        terminated = self._idx >= len(self.kp_data.kp_series) - 1
        truncated = False

        # 강제 청산
        if terminated and self._position_open:
            if self._position_direction == 'forward':
                kp_profit = self._entry_kp - exit_kp
            else:
                kp_profit = mid_kp - self._entry_kp
            net_profit = kp_profit - TOTAL_ROUND_TRIP * 100
            reward += net_profit
            self._total_pnl += net_profit
            self._position_open = False

        obs = self._get_obs() if not terminated else np.zeros(12, dtype=np.float32)

        return obs, reward, terminated, truncated, {
            "total_pnl": self._total_pnl,
            "trade_count": self._trade_count,
        }

    def _get_obs(self) -> np.ndarray:
        """12차원 관측 벡터 (기존과 동일 구조)"""
        kp = self.kp_data.kp_series[self._idx]
        entry_kp = self.kp_data.entry_kp_series[self._idx]
        exit_kp = self.kp_data.exit_kp_series[self._idx]

        history = list(self._kp_history)
        arr = np.array(history) if len(history) > 0 else np.array([kp])

        short_window = min(5, len(arr))
        long_window = min(20, len(arr))
        kp_ma_short = np.mean(arr[-short_window:])
        kp_ma_long = np.mean(arr[-long_window:])

        std = np.std(arr[-long_window:]) if len(arr) >= 2 else 1.0
        z_score = (kp - kp_ma_long) / max(std, 0.01)

        velocity = arr[-1] - arr[-2] if len(arr) >= 2 else 0.0

        state = np.array([
            kp / 10,
            entry_kp / 10,
            exit_kp / 10,
            kp_ma_short / 10,
            kp_ma_long / 10,
            np.clip(z_score / 3, -1, 1),
            np.clip(velocity, -1, 1),
            0.1,     # spread cost
            0.0,     # funding rate
            0.0,     # placeholder
            0.0,     # placeholder
            1.0 if self._position_open else 0.0,
        ], dtype=np.float32)

        return state


def evaluate(model, env, n_episodes: int = 5) -> dict:
    """학습된 모델 평가"""
    results = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        results.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "total_pnl": info.get("total_pnl", 0),
            "trade_count": info.get("trade_count", 0),
            "steps": steps,
        })

        logger.info(
            f"  평가 #{ep+1}: PnL={info.get('total_pnl', 0):.2f}% "
            f"거래={info.get('trade_count', 0)}회 보상={total_reward:.2f}"
        )

    total_actions = sum(action_counts.values())
    logger.info(
        f"  액션 분포: Hold={action_counts[0]/total_actions*100:.1f}% "
        f"Enter={action_counts[1]/total_actions*100:.1f}% "
        f"Exit={action_counts[2]/total_actions*100:.1f}% "
        f"ReverseEnter={action_counts[3]/total_actions*100:.1f}%"
    )

    return {
        "avg_pnl": np.mean([r["total_pnl"] for r in results]),
        "avg_trades": np.mean([r["trade_count"] for r in results]),
        "action_distribution": action_counts,
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="김치랑 마이너스 KP 구간 학습")
    parser.add_argument("--hours", type=float, default=2.0, help="학습 시간 (시간)")
    parser.add_argument("--steps", type=int, default=0, help="스텝 수 (0=시간 기반)")
    parser.add_argument("--days", type=int, default=180, help="히스토리컬 데이터 일수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(PROJECT_DIR, "logs", "kimchirang_train_neg_kp.log"),
                encoding="utf-8",
            ),
        ],
    )

    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)

    # Step 1: 데이터 수집
    logger.info(f"=== 김치랑 마이너스 KP 학습 시작 ({args.days}일 데이터) ===")
    logger.info(f"학습 목표: KP -1.5% ~ 0% 구간에서의 최적 행동 학습")

    kp_data = KPHistoricalData(days=args.days)
    if not kp_data.collect():
        logger.error("데이터 수집 실패 -- 종료")
        sys.exit(1)

    # KP 분포 분석
    kp_arr = np.array(kp_data.kp_series)
    neg_ratio = np.mean(kp_arr < 0) * 100
    target_ratio = np.mean((kp_arr >= -1.5) & (kp_arr <= 0)) * 100

    logger.info(f"KP 데이터: {len(kp_data.kp_series)}개 시간봉")
    logger.info(f"KP 범위: {kp_arr.min():.2f}% ~ {kp_arr.max():.2f}%")
    logger.info(f"KP 평균: {kp_arr.mean():.2f}%")
    logger.info(f"마이너스 KP 비율: {neg_ratio:.1f}%")
    logger.info(f"타겟 구간 (-1.5%~0%) 비율: {target_ratio:.1f}%")

    # Step 2: 환경 생성
    env = NegativeKPEnv(kp_data)
    eval_env = NegativeKPEnv(kp_data)

    # Step 3: PPO + DQN 병렬 학습
    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        logger.error("stable-baselines3 미설치: pip install stable-baselines3")
        sys.exit(1)

    model_dir = os.path.join(PROJECT_DIR, "data", "rl_models", "kimchirang", "neg_kp")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "dqn"), exist_ok=True)

    # 시간 기반 학습 스텝 계산
    if args.steps > 0:
        total_steps = args.steps
    else:
        # 2시간 = ~500K PPO + ~500K DQN (각 1시간)
        total_steps = 500_000

    logger.info(f"총 학습 스텝: PPO {total_steps:,} + DQN {total_steps:,}")
    start_time = time_mod.time()

    # === Phase 1: PPO 학습 ===
    logger.info("\n=== Phase 1: PPO 학습 시작 ===")

    ppo_eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=os.path.join(model_dir, "logs"),
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.08,  # 높은 탐색률 (새로운 행동 패턴 학습)
        verbose=1,
    )

    ppo_model.learn(total_timesteps=total_steps, callback=ppo_eval_callback)
    ppo_model.save(os.path.join(model_dir, "ppo_neg_kp_latest"))

    ppo_elapsed = time_mod.time() - start_time
    logger.info(f"PPO 학습 완료 ({ppo_elapsed/60:.1f}분)")

    # PPO 평가
    logger.info("=== PPO 평가 ===")
    ppo_result = evaluate(ppo_model, eval_env)

    # === Phase 2: DQN 학습 ===
    logger.info("\n=== Phase 2: DQN 학습 시작 ===")

    dqn_eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "dqn"),
        log_path=os.path.join(model_dir, "dqn", "logs"),
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    dqn_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        buffer_size=100_000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,  # 20% 탐색 (새로운 영역이므로 더 넓게)
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )

    dqn_model.learn(total_timesteps=total_steps, callback=dqn_eval_callback)
    dqn_model.save(os.path.join(model_dir, "dqn", "dqn_neg_kp_latest"))

    total_elapsed = time_mod.time() - start_time
    logger.info(f"DQN 학습 완료 (총 {total_elapsed/60:.1f}분)")

    # DQN 평가
    logger.info("=== DQN 평가 ===")
    dqn_result = evaluate(dqn_model, eval_env)

    # === 최종 비교 ===
    logger.info(
        f"\n{'='*60}\n"
        f"=== 최종 비교 (마이너스 KP 학습) ===\n"
        f"{'='*60}\n"
        f"  PPO — 평균 PnL: {ppo_result['avg_pnl']:.2f}%, "
        f"거래: {ppo_result['avg_trades']:.0f}회\n"
        f"  DQN — 평균 PnL: {dqn_result['avg_pnl']:.2f}%, "
        f"거래: {dqn_result['avg_trades']:.0f}회\n"
        f"\n  총 학습 시간: {total_elapsed/60:.1f}분 ({total_elapsed/3600:.1f}시간)\n"
        f"  데이터: {args.days}일, {len(kp_data.kp_series)}개 시간봉\n"
        f"  타겟 구간 비율: {target_ratio:.1f}%\n"
        f"{'='*60}\n"
    )

    logger.info("=== 김치랑 마이너스 KP 학습 완료 ===")


if __name__ == "__main__":
    main()
