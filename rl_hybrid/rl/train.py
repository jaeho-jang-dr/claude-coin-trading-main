"""RL 훈련 스크립트 — Upbit 히스토리컬 데이터로 PPO 모델 훈련

사용법:
    python -m rl_hybrid.rl.train                    # 기본 훈련 (180일, 100K 스텝)
    python -m rl_hybrid.rl.train --days 365 --steps 500000
    python -m rl_hybrid.rl.train --eval             # 기존 모델 평가만
"""

import argparse
import logging
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.train")


def prepare_data(days: int = 180, interval: str = "1h"):
    """훈련/평가 데이터 분리"""
    loader = HistoricalDataLoader()
    logger.info(f"Upbit에서 {days}일 {interval} 캔들 로드 중...")
    raw_candles = loader.load_candles(days=days, interval=interval)
    candles = loader.compute_indicators(raw_candles)

    # 80/20 분리
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    logger.info(
        f"데이터 준비 완료: 훈련={len(train_candles)}봉, 평가={len(eval_candles)}봉 "
        f"(총 {len(candles)}봉, {days}일 {interval})"
    )
    return train_candles, eval_candles


def train(days: int, total_steps: int, balance: float, interval: str = "1h"):
    """PPO 모델 훈련"""
    from rl_hybrid.rl.policy import PPOTrader, SB3_AVAILABLE

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3를 설치하세요: pip install stable-baselines3")
        return

    train_candles, eval_candles = prepare_data(days, interval)

    train_env = BitcoinTradingEnv(
        candles=train_candles,
        initial_balance=balance,
    )
    eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=balance,
    )

    trader = PPOTrader(env=train_env)
    trader.train(
        total_timesteps=total_steps,
        eval_env=eval_env,
        save_freq=total_steps // 10,
    )

    # 최종 평가
    evaluate(trader, eval_env, episodes=10)
    backtest_baseline(days, interval)


def evaluate(trader=None, env=None, episodes: int = 10, model_path: str = None):
    """모델 평가"""
    if trader is None:
        from rl_hybrid.rl.policy import PPOTrader
        _, eval_candles = prepare_data(180)
        env = BitcoinTradingEnv(candles=eval_candles, initial_balance=10_000_000)
        trader = PPOTrader(env=env, model_path=model_path)

    all_stats = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = trader.predict(obs)
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated

        stats = env.get_episode_stats()
        all_stats.append(stats)
        logger.info(
            f"[에피소드 {ep+1}/{episodes}] "
            f"수익률: {stats['total_return_pct']:.2f}% | "
            f"샤프: {stats['sharpe_ratio']:.3f} | "
            f"MDD: {stats['max_drawdown']:.2%} | "
            f"거래: {stats['trade_count']}회"
        )

    # 종합 통계
    avg_return = np.mean([s["total_return_pct"] for s in all_stats])
    avg_sharpe = np.mean([s["sharpe_ratio"] for s in all_stats])
    avg_mdd = np.mean([s["max_drawdown"] for s in all_stats])
    avg_trades = np.mean([s["trade_count"] for s in all_stats])

    logger.info(
        f"\n=== 평가 요약 ({episodes} 에피소드) ===\n"
        f"  평균 수익률: {avg_return:.2f}%\n"
        f"  평균 샤프:   {avg_sharpe:.3f}\n"
        f"  평균 MDD:    {avg_mdd:.2%}\n"
        f"  평균 거래수: {avg_trades:.1f}\n"
    )

    return all_stats


def backtest_baseline(days: int = 180, interval: str = "1h"):
    """Buy & Hold 벤치마크"""
    train_candles, eval_candles = prepare_data(days, interval)
    candles = eval_candles

    if not candles:
        logger.error("평가 데이터 없음")
        return

    initial_price = candles[0]["close"]
    final_price = candles[-1]["close"]
    bnh_return = (final_price - initial_price) / initial_price * 100

    # 가격 히스토리에서 MDD 계산
    prices = [c["close"] for c in candles]
    peak = prices[0]
    max_dd = 0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak
        max_dd = max(max_dd, dd)

    logger.info(
        f"\n=== Buy & Hold 벤치마크 ===\n"
        f"  기간: {candles[0]['timestamp']} ~ {candles[-1]['timestamp']}\n"
        f"  시작가: {initial_price:,.0f}원\n"
        f"  종가:   {final_price:,.0f}원\n"
        f"  수익률: {bnh_return:.2f}%\n"
        f"  MDD:    {max_dd:.2%}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC RL 트레이딩 모델 훈련")
    parser.add_argument("--days", type=int, default=180, help="훈련 데이터 기간 (일)")
    parser.add_argument("--steps", type=int, default=100_000, help="총 훈련 스텝")
    parser.add_argument("--balance", type=float, default=10_000_000, help="초기 잔고 (KRW)")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "4h", "1d"],
                        help="캔들 타임프레임 (1h, 4h, 1d)")
    parser.add_argument("--eval", action="store_true", help="평가만 실행")
    parser.add_argument("--baseline", action="store_true", help="Buy & Hold 벤치마크")
    parser.add_argument("--model", type=str, default=None, help="로드할 모델 경로")
    parser.add_argument("--multi", action="store_true",
                        help="멀티 타임프레임 훈련 (1h + 4h)")

    args = parser.parse_args()

    if args.baseline:
        backtest_baseline(args.days, args.interval)
    elif args.eval:
        evaluate(model_path=args.model, episodes=10)
    elif args.multi:
        # === 멀티 타임프레임 훈련 ===
        logger.info("=== 멀티 타임프레임 훈련 시작 ===")

        # Track 1: 1시간봉 (단기 타이밍)
        logger.info("\n[Track 1] 1시간봉 훈련 (단기 타이밍)")
        train(args.days, args.steps, args.balance, interval="1h")

        # Track 2: 4시간봉 (트렌드 방향)
        logger.info("\n[Track 2] 4시간봉 훈련 (트렌드 방향)")
        train(min(args.days, 365), args.steps, args.balance, interval="4h")

        logger.info("\n=== 멀티 타임프레임 훈련 완료 ===")
    else:
        train(args.days, args.steps, args.balance, args.interval)
