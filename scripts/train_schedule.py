"""RL 강화학습 스케줄 — 단계별 체계적 훈련

Phase 1: 기존 모델 평가 (베이스라인 측정)
Phase 2: 단기 데이터 워밍업 (90일, 50K 스텝)
Phase 3: 중기 본학습 (180일, 200K 스텝)
Phase 4: 장기 심화학습 (365일, 300K 스텝)
Phase 5: 멀티 타임프레임 (1h + 4h 교차 학습)
Phase 6: 최종 평가 + Buy & Hold 비교

CPU 환경 기준 예상 소요: Phase별 10~30분, 전체 약 2시간
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

from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.policy import PPOTrader, MODEL_DIR, SB3_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_schedule")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training_results")


def evaluate_model(trader, env, episodes=10, label=""):
    """모델 평가 → 통계 반환"""
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


def buy_and_hold_baseline(candles):
    """Buy & Hold 벤치마크"""
    initial = candles[0]["close"]
    final = candles[-1]["close"]
    bnh_return = (final - initial) / initial * 100

    prices = [c["close"] for c in candles]
    peak = prices[0]
    max_dd = 0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak
        max_dd = max(max_dd, dd)

    return {"return_pct": bnh_return, "mdd": max_dd}


def phase1_baseline(balance):
    """Phase 1: 기존 모델 베이스라인 평가"""
    logger.info("=" * 60)
    logger.info("Phase 1: 기존 모델 베이스라인 평가")
    logger.info("=" * 60)

    loader = HistoricalDataLoader()
    candles = loader.compute_indicators(loader.load_candles(days=180, interval="1h"))
    split = int(len(candles) * 0.8)
    eval_candles = candles[split:]
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

    # 기존 best 모델 로드
    best_path = os.path.join(MODEL_DIR, "best")
    if os.path.exists(best_path + ".zip"):
        trader = PPOTrader(env=eval_env, model_path=best_path)
        result = evaluate_model(trader, eval_env, episodes=10, label="기존 best 모델")
    else:
        logger.info("기존 모델 없음 — 스킵")
        result = {"return_pct": 0, "sharpe": 0, "mdd": 0, "trades": 0}

    # Buy & Hold
    bnh = buy_and_hold_baseline(eval_candles)
    logger.info(f"[Buy & Hold] 수익률: {bnh['return_pct']:.2f}% | MDD: {bnh['mdd']:.2%}")

    return {"model": result, "buy_and_hold": bnh}


def phase2_warmup(balance):
    """Phase 2: 단기 워밍업 (90일, 50K 스텝)"""
    logger.info("=" * 60)
    logger.info("Phase 2: 단기 워밍업 학습 (90일, 50K 스텝)")
    logger.info("=" * 60)

    loader = HistoricalDataLoader()
    candles = loader.compute_indicators(loader.load_candles(days=90, interval="1h"))
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

    trader = PPOTrader(env=train_env)

    start = time.time()
    trader.train(total_timesteps=50_000, eval_env=eval_env, save_freq=10_000)
    elapsed = time.time() - start

    result = evaluate_model(trader, eval_env, episodes=10, label="Phase 2 워밍업")
    result["elapsed_sec"] = elapsed
    logger.info(f"Phase 2 소요: {elapsed:.0f}초")

    return result


def phase3_main_training(balance):
    """Phase 3: 중기 본학습 (180일, 200K 스텝) — 기존 모델 이어서 학습"""
    logger.info("=" * 60)
    logger.info("Phase 3: 중기 본학습 (180일, 200K 스텝)")
    logger.info("=" * 60)

    loader = HistoricalDataLoader()
    candles = loader.compute_indicators(loader.load_candles(days=180, interval="1h"))
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

    # Phase 2에서 저장된 모델 로드하여 이어서 학습
    latest_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
    if os.path.exists(latest_path + ".zip"):
        from stable_baselines3 import PPO
        model = PPO.load(latest_path, env=train_env)
        trader = PPOTrader(env=train_env)
        trader.model = model
        logger.info("Phase 2 모델 로드 → 이어서 학습")
    else:
        trader = PPOTrader(env=train_env)

    start = time.time()
    trader.train(total_timesteps=200_000, eval_env=eval_env, save_freq=20_000)
    elapsed = time.time() - start

    result = evaluate_model(trader, eval_env, episodes=10, label="Phase 3 본학습")
    result["elapsed_sec"] = elapsed
    logger.info(f"Phase 3 소요: {elapsed:.0f}초")

    return result


def phase4_deep_training(balance):
    """Phase 4: 장기 심화학습 (365일, 300K 스텝)"""
    logger.info("=" * 60)
    logger.info("Phase 4: 장기 심화학습 (365일, 300K 스텝)")
    logger.info("=" * 60)

    loader = HistoricalDataLoader()
    candles = loader.compute_indicators(loader.load_candles(days=365, interval="1h"))
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

    # Phase 3 모델 이어서 학습
    latest_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
    if os.path.exists(latest_path + ".zip"):
        from stable_baselines3 import PPO
        model = PPO.load(latest_path, env=train_env)
        trader = PPOTrader(env=train_env)
        trader.model = model
        logger.info("Phase 3 모델 로드 → 이어서 학습")
    else:
        trader = PPOTrader(env=train_env)

    start = time.time()
    trader.train(total_timesteps=300_000, eval_env=eval_env, save_freq=30_000)
    elapsed = time.time() - start

    result = evaluate_model(trader, eval_env, episodes=10, label="Phase 4 심화학습")
    result["elapsed_sec"] = elapsed
    logger.info(f"Phase 4 소요: {elapsed:.0f}초")

    return result


def phase5_multi_timeframe(balance):
    """Phase 5: 멀티 타임프레임 (4시간봉 추가 학습)"""
    logger.info("=" * 60)
    logger.info("Phase 5: 멀티 타임프레임 4h 추가학습 (180일, 100K 스텝)")
    logger.info("=" * 60)

    loader = HistoricalDataLoader()
    candles = loader.compute_indicators(loader.load_candles(days=180, interval="4h"))
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

    # Phase 4 모델 이어서 4h 데이터로 학습
    latest_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
    if os.path.exists(latest_path + ".zip"):
        from stable_baselines3 import PPO
        model = PPO.load(latest_path, env=train_env)
        trader = PPOTrader(env=train_env)
        trader.model = model
        logger.info("Phase 4 모델 로드 → 4h 타임프레임 학습")
    else:
        trader = PPOTrader(env=train_env)

    start = time.time()
    trader.train(total_timesteps=100_000, eval_env=eval_env, save_freq=20_000)
    elapsed = time.time() - start

    result = evaluate_model(trader, eval_env, episodes=10, label="Phase 5 멀티TF")
    result["elapsed_sec"] = elapsed
    logger.info(f"Phase 5 소요: {elapsed:.0f}초")

    return result


def phase6_final_evaluation(balance):
    """Phase 6: 최종 종합 평가"""
    logger.info("=" * 60)
    logger.info("Phase 6: 최종 종합 평가")
    logger.info("=" * 60)

    loader = HistoricalDataLoader()
    results = {}

    # 여러 기간으로 평가
    for days, label in [(90, "최근 90일"), (180, "최근 180일")]:
        candles = loader.compute_indicators(loader.load_candles(days=days, interval="1h"))
        split = int(len(candles) * 0.8)
        eval_candles = candles[split:]
        eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)

        latest_path = os.path.join(MODEL_DIR, "ppo_btc_latest")
        trader = PPOTrader(env=eval_env, model_path=latest_path)
        model_result = evaluate_model(trader, eval_env, episodes=10, label=f"최종모델 {label}")

        bnh = buy_and_hold_baseline(eval_candles)
        logger.info(f"[Buy & Hold {label}] 수익률: {bnh['return_pct']:.2f}% | MDD: {bnh['mdd']:.2%}")

        alpha = model_result["return_pct"] - bnh["return_pct"]
        logger.info(f"[Alpha {label}] {alpha:+.2f}%p (모델 - B&H)")

        results[label] = {"model": model_result, "buy_and_hold": bnh, "alpha": alpha}

    return results


def save_results(all_results):
    """결과 저장"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"schedule_{timestamp}.json")

    # numpy float → python float 변환
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

    logger.info(f"결과 저장: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="RL 강화학습 스케줄")
    parser.add_argument("--phase", type=int, default=0,
                        help="특정 Phase만 실행 (1-6, 0=전체)")
    parser.add_argument("--balance", type=float, default=10_000_000,
                        help="초기 잔고 (KRW)")
    parser.add_argument("--skip-to", type=int, default=1,
                        help="이 Phase부터 시작 (이전 Phase 스킵)")
    args = parser.parse_args()

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3가 필요합니다")
        return

    balance = args.balance
    all_results = {"started_at": datetime.now().isoformat(), "balance": balance}
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  RL 강화학습 스케줄 시작")
    logger.info(f"  초기 잔고: {balance:,.0f} KRW")
    logger.info(f"  디바이스: CPU (PyTorch {__import__('torch').__version__})")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Phase 1: 기존 모델 베이스라인 평가")
    logger.info("  Phase 2: 단기 워밍업 (90일, 50K)")
    logger.info("  Phase 3: 중기 본학습 (180일, 200K)")
    logger.info("  Phase 4: 장기 심화 (365일, 300K)")
    logger.info("  Phase 5: 멀티 타임프레임 (4h, 100K)")
    logger.info("  Phase 6: 최종 종합 평가")
    logger.info("")

    phases = {
        1: ("Phase 1: 베이스라인", phase1_baseline),
        2: ("Phase 2: 워밍업", phase2_warmup),
        3: ("Phase 3: 본학습", phase3_main_training),
        4: ("Phase 4: 심화학습", phase4_deep_training),
        5: ("Phase 5: 멀티TF", phase5_multi_timeframe),
        6: ("Phase 6: 최종평가", phase6_final_evaluation),
    }

    run_phases = [args.phase] if args.phase > 0 else range(args.skip_to, 7)

    for p in run_phases:
        if p not in phases:
            continue
        name, func = phases[p]
        try:
            result = func(balance)
            all_results[f"phase{p}"] = result
            logger.info(f"{name} 완료\n")
        except Exception as e:
            logger.error(f"{name} 실패: {e}", exc_info=True)
            all_results[f"phase{p}"] = {"error": str(e)}

    total_elapsed = time.time() - total_start
    all_results["total_elapsed_sec"] = total_elapsed
    all_results["finished_at"] = datetime.now().isoformat()

    # 결과 저장
    result_path = save_results(all_results)

    # 최종 요약
    logger.info("=" * 60)
    logger.info("  학습 스케줄 완료")
    logger.info(f"  총 소요: {total_elapsed/60:.1f}분")
    logger.info(f"  결과: {result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
