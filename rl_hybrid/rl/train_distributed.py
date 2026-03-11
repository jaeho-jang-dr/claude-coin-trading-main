"""RL 초기 훈련 — DistributedTrainer(순수 PyTorch) 기반

SB3 없이 자체 PPO로 히스토리컬 데이터 학습.

사용법:
    python -m rl_hybrid.rl.train_distributed                     # 기본 (180일, 100K 스텝)
    python -m rl_hybrid.rl.train_distributed --days 365 --steps 200000
    python -m rl_hybrid.rl.train_distributed --eval              # 평가만
    python -m rl_hybrid.rl.train_distributed --baseline          # Buy&Hold 벤치마크
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.distributed_trainer import DistributedTrainer
from rl_hybrid.rl.trajectory import TrajectoryBuffer, Transition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.train_distributed")


def prepare_data(days: int = 180, interval: str = "1h"):
    """훈련/평가 데이터 분리"""
    loader = HistoricalDataLoader()
    logger.info(f"Upbit에서 {days}일 {interval} 캔들 로드 중...")
    raw_candles = loader.load_candles(days=days, interval=interval)
    candles = loader.compute_indicators(raw_candles)

    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    logger.info(
        f"데이터 준비 완료: 훈련={len(train_candles)}봉, 평가={len(eval_candles)}봉 "
        f"(총 {len(candles)}봉, {days}일 {interval})"
    )
    return train_candles, eval_candles


def train(days: int, total_steps: int, balance: float, rollout_steps: int = 2048):
    """DistributedTrainer 기반 PPO 훈련"""
    train_candles, eval_candles = prepare_data(days)

    if len(train_candles) < 100:
        logger.error(f"훈련 데이터 부족: {len(train_candles)}봉")
        return

    trainer = DistributedTrainer(
        obs_dim=42,
        action_dim=1,
        lr=3e-4,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=10,
        batch_size=64,
        min_trajectories_for_update=1,  # 단일 노드이므로 1
    )

    env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    buffer = TrajectoryBuffer(max_size=rollout_steps * 2)

    obs, info = env.reset()
    steps_done = 0
    episode_count = 0
    best_eval_sharpe = -999
    start_time = time.time()

    logger.info(
        f"=== PPO 훈련 시작 ===\n"
        f"  총 스텝: {total_steps:,}\n"
        f"  롤아웃: {rollout_steps} 스텝/배치\n"
        f"  초기 잔고: {balance:,.0f} KRW\n"
        f"  모델 파라미터: {sum(p.numel() for p in trainer.model.parameters()):,}"
    )

    while steps_done < total_steps:
        # === 롤아웃 수집 ===
        buffer.clear()
        trainer.model.eval()

        for _ in range(rollout_steps):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                action, log_prob, value = trainer.model.get_action(obs_t, deterministic=False)
                action_np = float(action.squeeze().item())
                action_clipped = np.clip(action_np, -1.0, 1.0)

            next_obs, reward, terminated, truncated, step_info = env.step(
                np.array([action_clipped])
            )
            done = terminated or truncated

            buffer.add(Transition(
                obs=obs.copy(),
                action=np.array([action_clipped]),
                reward=reward,
                next_obs=next_obs.copy(),
                done=done,
                value=float(value.squeeze().item()),
                log_prob=float(log_prob.squeeze().item()),
            ))

            obs = next_obs
            steps_done += 1

            if done:
                episode_count += 1
                obs, info = env.reset()

        # === PPO 업데이트 ===
        if buffer.is_ready(min_steps=64):
            traj_data = buffer.serialize()
            trainer.receive_trajectory(traj_data, worker_id="local_trainer")
            stats = trainer.update()

            if stats:
                elapsed = time.time() - start_time
                fps = steps_done / elapsed
                logger.info(
                    f"[{steps_done:>7,}/{total_steps:,}] "
                    f"policy={stats['policy_loss']:.4f} "
                    f"value={stats['value_loss']:.4f} "
                    f"entropy={stats['entropy']:.4f} "
                    f"episodes={episode_count} "
                    f"fps={fps:.0f}"
                )

        # === 주기적 평가 (20K 스텝마다) ===
        if steps_done % 20000 < rollout_steps and steps_done > rollout_steps:
            eval_stats = evaluate(trainer, eval_candles, balance, episodes=5, quiet=True)
            avg_sharpe = eval_stats["avg_sharpe"]
            avg_return = eval_stats["avg_return"]

            logger.info(
                f"  [평가] return={avg_return:+.2f}% sharpe={avg_sharpe:.3f} "
                f"mdd={eval_stats['avg_mdd']:.2%} trades={eval_stats['avg_trades']:.0f}"
            )

            if avg_sharpe > best_eval_sharpe:
                best_eval_sharpe = avg_sharpe
                trainer.save_model()
                logger.info(f"  [Best] 새 최고 sharpe={avg_sharpe:.3f} → 모델 저장")

    # === 최종 평가 ===
    trainer.save_model()
    elapsed = time.time() - start_time

    logger.info(f"\n=== 훈련 완료: {elapsed:.0f}초, {steps_done:,} 스텝, {episode_count} 에피소드 ===")

    final_stats = evaluate(trainer, eval_candles, balance, episodes=10)
    backtest_baseline(eval_candles)

    return final_stats


def evaluate(
    trainer: DistributedTrainer = None,
    eval_candles: list = None,
    balance: float = 10_000_000,
    episodes: int = 10,
    quiet: bool = False,
) -> dict:
    """모델 평가"""
    if trainer is None:
        trainer = DistributedTrainer(obs_dim=42)
        trainer.load_model()

    if eval_candles is None:
        _, eval_candles = prepare_data(180)

    if len(eval_candles) < 50:
        logger.warning(f"평가 데이터 부족: {len(eval_candles)}봉")
        return {"avg_return": 0, "avg_sharpe": 0, "avg_mdd": 0, "avg_trades": 0}

    env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)
    all_stats = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False

        while not done:
            action, _, _ = trainer.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated

        stats = env.get_episode_stats()
        all_stats.append(stats)

        if not quiet:
            logger.info(
                f"  [에피소드 {ep+1}/{episodes}] "
                f"수익률: {stats['total_return_pct']:+.2f}% | "
                f"샤프: {stats['sharpe_ratio']:.3f} | "
                f"MDD: {stats['max_drawdown']:.2%} | "
                f"거래: {stats['trade_count']}회"
            )

    result = {
        "avg_return": float(np.mean([s["total_return_pct"] for s in all_stats])),
        "avg_sharpe": float(np.mean([s["sharpe_ratio"] for s in all_stats])),
        "avg_mdd": float(np.mean([s["max_drawdown"] for s in all_stats])),
        "avg_trades": float(np.mean([s["trade_count"] for s in all_stats])),
    }

    if not quiet:
        logger.info(
            f"\n=== 평가 요약 ({episodes} 에피소드) ===\n"
            f"  평균 수익률: {result['avg_return']:+.2f}%\n"
            f"  평균 샤프:   {result['avg_sharpe']:.3f}\n"
            f"  평균 MDD:    {result['avg_mdd']:.2%}\n"
            f"  평균 거래수: {result['avg_trades']:.1f}\n"
        )

    return result


def backtest_baseline(eval_candles: list = None):
    """Buy & Hold 벤치마크"""
    if eval_candles is None:
        _, eval_candles = prepare_data(180)

    if not eval_candles:
        logger.error("평가 데이터 없음")
        return

    initial_price = eval_candles[0]["close"]
    final_price = eval_candles[-1]["close"]
    bnh_return = (final_price - initial_price) / initial_price * 100

    prices = [c["close"] for c in eval_candles]
    peak = prices[0]
    max_dd = 0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak
        max_dd = max(max_dd, dd)

    logger.info(
        f"\n=== Buy & Hold 벤치마크 ===\n"
        f"  기간: {len(eval_candles)}봉\n"
        f"  시작가: {initial_price:,.0f}원\n"
        f"  종가:   {final_price:,.0f}원\n"
        f"  수익률: {bnh_return:+.2f}%\n"
        f"  MDD:    {max_dd:.2%}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC RL 트레이딩 모델 훈련 (PyTorch PPO)")
    parser.add_argument("--days", type=int, default=180, help="훈련 데이터 기간 (일)")
    parser.add_argument("--steps", type=int, default=100_000, help="총 훈련 스텝")
    parser.add_argument("--balance", type=float, default=10_000_000, help="초기 잔고 (KRW)")
    parser.add_argument("--rollout", type=int, default=2048, help="롤아웃 스텝 수")
    parser.add_argument("--eval", action="store_true", help="평가만 실행")
    parser.add_argument("--baseline", action="store_true", help="Buy & Hold 벤치마크")

    args = parser.parse_args()

    if args.baseline:
        backtest_baseline()
    elif args.eval:
        evaluate(episodes=10)
    else:
        train(args.days, args.steps, args.balance, args.rollout)
