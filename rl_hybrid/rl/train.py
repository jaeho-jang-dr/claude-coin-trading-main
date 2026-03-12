"""RL 훈련 스크립트 — Upbit 히스토리컬 데이터로 PPO/SAC/TD3 모델 훈련

사용법:
    python -m rl_hybrid.rl.train                    # 기본 훈련 (PPO, 180일, 100K 스텝)
    python -m rl_hybrid.rl.train --algo sac         # SAC 알고리즘으로 훈련
    python -m rl_hybrid.rl.train --algo td3         # TD3 알고리즘으로 훈련
    python -m rl_hybrid.rl.train --algo all         # 3개 알고리즘 비교 훈련
    python -m rl_hybrid.rl.train --days 365 --steps 500000
    python -m rl_hybrid.rl.train --eval             # 기존 모델 평가만
    python -m rl_hybrid.rl.train --edge-cases       # 엣지 케이스 합성 데이터 혼합 훈련
    python -m rl_hybrid.rl.train --edge-cases --synthetic-ratio 0.4  # 합성 비율 40%
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
from rl_hybrid.rl.scenario_generator import ScenarioGenerator
from rl_hybrid.config import config as system_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.train")


def get_trader_class(algo: str):
    """Return the appropriate trader class for the algorithm name.

    Args:
        algo: 'ppo', 'sac', or 'td3'

    Returns:
        해당 알고리즘의 Trader 클래스
    """
    from rl_hybrid.rl.policy import PPOTrader, SACTrader, TD3Trader
    mapping = {"ppo": PPOTrader, "sac": SACTrader, "td3": TD3Trader}
    if algo not in mapping:
        raise ValueError(f"지원하지 않는 알고리즘: {algo}. 선택: {list(mapping.keys())}")
    return mapping[algo]


def prepare_data(days: int = 180, interval: str = "1h"):
    """훈련/평가 데이터 분리 (캔들 + 외부 시그널)

    Returns:
        (train_candles, eval_candles, train_signals, eval_signals)
        *_signals는 None일 수 있음 (Supabase 미연결 시)
    """
    loader = HistoricalDataLoader()
    logger.info(f"Upbit에서 {days}일 {interval} 캔들 로드 중...")
    raw_candles = loader.load_candles(days=days, interval=interval)
    candles = loader.compute_indicators(raw_candles)

    # 외부 시그널 로드 및 정렬
    aligned_signals = None
    try:
        raw_signals = loader.load_external_signals(days=days)
        if raw_signals:
            aligned_signals = loader.align_external_to_candles(candles, raw_signals)
    except Exception as e:
        logger.warning(f"외부 시그널 로드/정렬 실패 — 기본값 사용: {e}")

    # 80/20 분리
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    train_signals = None
    eval_signals = None
    if aligned_signals is not None:
        train_signals = aligned_signals[:split]
        eval_signals = aligned_signals[split:]
        logger.info(
            f"외부 시그널 분리 완료: 훈련={len(train_signals)}건, "
            f"평가={len(eval_signals)}건"
        )
    else:
        logger.info("외부 시그널 없음 — 환경에서 기본 상수값 사용")

    logger.info(
        f"데이터 준비 완료: 훈련={len(train_candles)}봉, 평가={len(eval_candles)}봉 "
        f"(총 {len(candles)}봉, {days}일 {interval})"
    )
    return train_candles, eval_candles, train_signals, eval_signals


def prepare_edge_case_data(
    days: int = 180,
    interval: str = "1h",
    synthetic_ratio: float = 0.3,
    variations: int = 3,
):
    """실제 데이터 + 엣지 케이스 합성 데이터 혼합 준비

    Args:
        days: 실제 데이터 기간 (일)
        interval: 캔들 타임프레임
        synthetic_ratio: 합성 데이터 비율 (0.3 = 30%)
        variations: 시나리오당 변형 수

    Returns:
        (train_candles, eval_candles, train_signals, eval_signals)
    """
    loader = HistoricalDataLoader()
    logger.info(f"Upbit에서 {days}일 {interval} 캔들 로드 중...")
    raw_candles = loader.load_candles(days=days, interval=interval)

    # 합성 데이터 생성 (실제 가격 기반)
    base_price = raw_candles[-1]["close"] if raw_candles else 100_000_000
    gen = ScenarioGenerator(base_price=base_price)
    mixed_raw = gen.mix_with_real(raw_candles, synthetic_ratio=synthetic_ratio,
                                  variations=variations)

    real_count = len(raw_candles)
    synth_count = len(mixed_raw) - real_count
    logger.info(
        f"엣지 케이스 혼합: 실제={real_count}봉, 합성={synth_count}봉, "
        f"합성비율={synth_count / len(mixed_raw):.1%}"
    )

    # 기술 지표 계산 (합성 데이터에도 적용)
    candles = loader.compute_indicators(mixed_raw)

    # 외부 시그널 (실제 데이터만 — 합성 구간은 기본값)
    aligned_signals = None
    try:
        raw_signals = loader.load_external_signals(days=days)
        if raw_signals:
            # 실제 캔들 구간만 정렬하고, 합성 구간은 기본값 채우기
            real_aligned = loader.align_external_to_candles(
                candles[:real_count], raw_signals
            )
            synth_defaults = [
                loader._default_external_signal() for _ in range(synth_count)
            ]
            aligned_signals = real_aligned + synth_defaults
    except Exception as e:
        logger.warning(f"외부 시그널 로드/정렬 실패 — 기본값 사용: {e}")

    # 80/20 분리
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    train_signals = None
    eval_signals = None
    if aligned_signals is not None:
        train_signals = aligned_signals[:split]
        eval_signals = aligned_signals[split:]

    logger.info(
        f"엣지 케이스 데이터 준비 완료: 훈련={len(train_candles)}봉, "
        f"평가={len(eval_candles)}봉"
    )
    return train_candles, eval_candles, train_signals, eval_signals


def train(days: int, total_steps: int, balance: float, interval: str = "1h",
          model_path: str = None, algo: str = "ppo",
          edge_cases: bool = False, synthetic_ratio: float = 0.3):
    """RL 모델 훈련

    Args:
        days: 훈련 데이터 기간 (일)
        total_steps: 총 훈련 스텝
        balance: 초기 잔고 (KRW)
        interval: 캔들 타임프레임
        model_path: 기존 모델 경로. 지정 시 해당 모델에서 이어 학습 (fine-tuning)
        algo: 알고리즘 ('ppo', 'sac', 'td3')
    """
    from rl_hybrid.rl.policy import SB3_AVAILABLE

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3를 설치하세요: pip install stable-baselines3")
        return

    TraderClass = get_trader_class(algo)
    if edge_cases:
        train_candles, eval_candles, train_signals, eval_signals = \
            prepare_edge_case_data(days, interval, synthetic_ratio)
    else:
        train_candles, eval_candles, train_signals, eval_signals = prepare_data(days, interval)

    train_env = BitcoinTradingEnv(
        candles=train_candles,
        initial_balance=balance,
        external_signals=train_signals,
    )
    eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=balance,
        external_signals=eval_signals,
    )

    if model_path:
        # 기존 모델에서 이어 학습
        logger.info(f"기존 모델 로드하여 fine-tuning: {model_path}")
        trader = TraderClass(env=train_env)
        trader.load(model_path)
        trader.model.set_env(train_env)
    else:
        trader = TraderClass(env=train_env)

    trader.train(
        total_timesteps=total_steps,
        eval_env=eval_env,
        save_freq=total_steps // 10,
    )

    # 최종 평가
    evaluate(trader, eval_env, episodes=10)
    backtest_baseline(days, interval)

    return trader


def train_all(days: int, total_steps: int, balance: float, interval: str = "1h",
              edge_cases: bool = False, synthetic_ratio: float = 0.3):
    """PPO, SAC, TD3 세 알고리즘을 동일 데이터로 훈련 후 비교

    최적 모델을 data/rl_models/best/best_model.zip에 저장하고,
    model_info.json에 사용된 알고리즘을 기록한다.
    """
    from rl_hybrid.rl.policy import SB3_AVAILABLE, MODEL_DIR

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3를 설치하세요: pip install stable-baselines3")
        return

    if edge_cases:
        train_candles, eval_candles, train_signals, eval_signals = \
            prepare_edge_case_data(days, interval, synthetic_ratio)
    else:
        train_candles, eval_candles, train_signals, eval_signals = prepare_data(days, interval)
    algos = ["ppo", "sac", "td3"]
    results = {}

    for algo in algos:
        logger.info(f"\n{'='*60}")
        logger.info(f"  {algo.upper()} 훈련 시작")
        logger.info(f"{'='*60}")

        TraderClass = get_trader_class(algo)

        train_env = BitcoinTradingEnv(
            candles=train_candles,
            initial_balance=balance,
            external_signals=train_signals,
        )
        eval_env = BitcoinTradingEnv(
            candles=eval_candles,
            initial_balance=balance,
            external_signals=eval_signals,
        )

        trader = TraderClass(env=train_env)
        trader.train(
            total_timesteps=total_steps,
            eval_env=eval_env,
            save_freq=total_steps // 10,
        )

        # 평가 (10 에피소드)
        logger.info(f"\n--- {algo.upper()} 평가 ---")
        eval_env_fresh = BitcoinTradingEnv(
            candles=eval_candles,
            initial_balance=balance,
            external_signals=eval_signals,
        )
        stats = evaluate(trader, eval_env_fresh, episodes=10)

        avg_return = np.mean([s["total_return_pct"] for s in stats])
        avg_sharpe = np.mean([s["sharpe_ratio"] for s in stats])
        avg_mdd = np.mean([s["max_drawdown"] for s in stats])
        avg_trades = np.mean([s["trade_count"] for s in stats])

        results[algo] = {
            "trader": trader,
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_mdd": avg_mdd,
            "avg_trades": avg_trades,
        }

    # 비교 테이블 출력
    logger.info(f"\n{'='*60}")
    logger.info("  알고리즘 비교 결과")
    logger.info(f"{'='*60}")
    logger.info(f"{'알고리즘':>8} | {'수익률':>10} | {'샤프':>8} | {'MDD':>10} | {'거래수':>8}")
    logger.info("-" * 60)
    for algo in algos:
        r = results[algo]
        logger.info(
            f"{algo.upper():>8} | "
            f"{r['avg_return']:>9.2f}% | "
            f"{r['avg_sharpe']:>8.3f} | "
            f"{r['avg_mdd']:>9.2%} | "
            f"{r['avg_trades']:>8.1f}"
        )

    # 최적 알고리즘 선정 (샤프 비율 기준, 동점 시 수익률)
    best_algo = max(
        algos,
        key=lambda a: (results[a]["avg_sharpe"], results[a]["avg_return"]),
    )
    logger.info(f"\n최적 알고리즘: {best_algo.upper()} "
                f"(샤프={results[best_algo]['avg_sharpe']:.3f})")

    # 최적 모델을 best/ 디렉토리에 저장
    best_dir = os.path.join(MODEL_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)
    best_model_path = os.path.join(best_dir, "best_model")
    results[best_algo]["trader"].save(best_model_path)

    # model_info.json 저장
    info_path = os.path.join(best_dir, "model_info.json")
    model_info = {
        "algorithm": best_algo,
        "avg_return_pct": round(float(results[best_algo]["avg_return"]), 4),
        "avg_sharpe": round(float(results[best_algo]["avg_sharpe"]), 4),
        "avg_mdd": round(float(results[best_algo]["avg_mdd"]), 6),
        "training_steps": total_steps,
        "training_days": days,
        "interval": interval,
        "comparison": {
            a: {
                "avg_return_pct": round(float(results[a]["avg_return"]), 4),
                "avg_sharpe": round(float(results[a]["avg_sharpe"]), 4),
                "avg_mdd": round(float(results[a]["avg_mdd"]), 6),
            }
            for a in algos
        },
    }
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"모델 정보 저장: {info_path}")

    backtest_baseline(days, interval)


def evaluate(trader=None, env=None, episodes: int = 10, model_path: str = None):
    """모델 평가"""
    if trader is None:
        from rl_hybrid.rl.policy import PPOTrader
        _, eval_candles, _, eval_signals = prepare_data(180)
        env = BitcoinTradingEnv(
            candles=eval_candles, initial_balance=10_000_000,
            external_signals=eval_signals,
        )
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


def train_multi_objective(
    days: int,
    total_steps: int,
    balance: float,
    interval: str = "4h",
    algo: str = "ppo",
    envelope_morl: bool = False,
    adaptive_weights: bool = True,
    edge_cases: bool = False,
    synthetic_ratio: float = 0.3,
):
    """다중 목표 RL 훈련

    5가지 목표(profit, risk, efficiency, sharpe, tail_risk)를
    동시에 최적화하며, Pareto 최적 모델을 발견한다.

    Args:
        days: 훈련 데이터 기간
        total_steps: 총 훈련 스텝
        balance: 초기 잔고
        interval: 캔들 타임프레임
        algo: RL 알고리즘
        envelope_morl: Envelope MORL 모드 (가중치를 관측에 concat)
        adaptive_weights: 적응적 가중치 조절 사용 여부
        edge_cases: 엣지 케이스 합성 데이터 혼합
        synthetic_ratio: 합성 데이터 비율
    """
    from rl_hybrid.rl.policy import SB3_AVAILABLE

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3를 설치하세요: pip install stable-baselines3")
        return

    from rl_hybrid.rl.multi_objective_reward import (
        create_training_pipeline,
        OBJECTIVE_NAMES,
    )

    mo_config = system_config.multi_objective_rl

    logger.info(f"\n{'='*60}")
    logger.info("  Multi-Objective RL 훈련 시작")
    logger.info(f"{'='*60}")
    logger.info(f"  알고리즘: {algo.upper()}")
    logger.info(f"  Envelope MORL: {envelope_morl}")
    logger.info(f"  적응적 가중치: {adaptive_weights}")
    logger.info(f"  가중치: {mo_config.weights}")
    logger.info(f"  제약: MDD<{mo_config.max_mdd:.0%}, "
                f"trades/day<{mo_config.max_trades_per_day:.0f}")
    logger.info(f"{'='*60}\n")

    TraderClass = get_trader_class(algo)

    if edge_cases:
        train_candles, eval_candles, train_signals, eval_signals = \
            prepare_edge_case_data(days, interval, synthetic_ratio)
    else:
        train_candles, eval_candles, train_signals, eval_signals = \
            prepare_data(days, interval)

    # 기본 환경 생성
    base_train_env = BitcoinTradingEnv(
        candles=train_candles,
        initial_balance=balance,
        external_signals=train_signals,
    )
    base_eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=balance,
        external_signals=eval_signals,
    )

    # 다중 목표 래핑
    mo_train_env, mo_eval_env, sb3_callback = create_training_pipeline(
        train_env=base_train_env,
        eval_env=base_eval_env,
        weights=mo_config.weights,
        envelope_morl=envelope_morl,
        adaptive_weights=adaptive_weights,
        pareto_max_k=mo_config.pareto_max_k,
        eval_freq=mo_config.eval_freq,
    )

    # 훈련
    trader = TraderClass(env=mo_train_env)

    callbacks = []
    if sb3_callback is not None:
        callbacks.append(sb3_callback)

    trader.train(
        total_timesteps=total_steps,
        eval_env=mo_eval_env,
        save_freq=total_steps // 10,
        callbacks=callbacks if callbacks else None,
    )

    # 최종 평가
    logger.info("\n--- Multi-Objective 평가 ---")
    all_stats = evaluate(trader, mo_eval_env, episodes=10)

    # 목표별 통계 출력
    logger.info(f"\n{'='*60}")
    logger.info("  목표별 평균 점수")
    logger.info(f"{'='*60}")
    for name in OBJECTIVE_NAMES:
        scores = [
            s.get("objective_means", {}).get(name, 0.0) for s in all_stats
        ]
        avg = np.mean(scores) if scores else 0.0
        logger.info(f"  {name:>12}: {avg:+.4f}")

    # 가중치 최종 상태
    final_stats = mo_eval_env.get_episode_stats()
    logger.info(f"\n최종 가중치: {final_stats.get('weights', {})}")

    backtest_baseline(days, interval)

    return trader


def backtest_baseline(days: int = 180, interval: str = "1h"):
    """Buy & Hold 벤치마크"""
    _, eval_candles, _, _ = prepare_data(days, interval)
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
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3", "all"],
                        help="RL 알고리즘 (ppo, sac, td3, all)")
    parser.add_argument("--eval", action="store_true", help="평가만 실행")
    parser.add_argument("--baseline", action="store_true", help="Buy & Hold 벤치마크")
    parser.add_argument("--model", type=str, default=None, help="로드할 모델 경로")
    parser.add_argument("--multi", action="store_true",
                        help="멀티 타임프레임 훈련 (1h + 4h)")
    parser.add_argument("--edge-cases", action="store_true",
                        help="엣지 케이스 합성 데이터 혼합 훈련")
    parser.add_argument("--synthetic-ratio", type=float, default=0.3,
                        help="합성 데이터 비율 (0.0~1.0, default: 0.3)")
    parser.add_argument("--multi-objective", action="store_true",
                        help="다중 목표 RL 훈련 (profit/risk/efficiency/sharpe/tail_risk)")
    parser.add_argument("--envelope-morl", action="store_true",
                        help="Envelope MORL 모드 (가중치를 관측에 concat)")
    parser.add_argument("--no-adaptive-weights", action="store_true",
                        help="적응적 가중치 조절 비활성화")

    args = parser.parse_args()

    if args.baseline:
        backtest_baseline(args.days, args.interval)
    elif args.multi_objective:
        train_multi_objective(
            args.days, args.steps, args.balance, args.interval,
            algo=args.algo if args.algo != "all" else "ppo",
            envelope_morl=args.envelope_morl,
            adaptive_weights=not args.no_adaptive_weights,
            edge_cases=args.edge_cases,
            synthetic_ratio=args.synthetic_ratio,
        )
    elif args.eval:
        evaluate(model_path=args.model, episodes=10)
    elif args.algo == "all":
        train_all(args.days, args.steps, args.balance, args.interval,
                  edge_cases=args.edge_cases, synthetic_ratio=args.synthetic_ratio)
    elif args.multi:
        # === 멀티 타임프레임 훈련 ===
        logger.info("=== 멀티 타임프레임 훈련 시작 ===")

        # Track 1: 1시간봉 (단기 타이밍)
        logger.info("\n[Track 1] 1시간봉 훈련 (단기 타이밍)")
        train(args.days, args.steps, args.balance, interval="1h", algo=args.algo)

        # Track 2: 4시간봉 (트렌드 방향)
        logger.info("\n[Track 2] 4시간봉 훈련 (트렌드 방향)")
        train(min(args.days, 365), args.steps, args.balance, interval="4h", algo=args.algo)

        logger.info("\n=== 멀티 타임프레임 훈련 완료 ===")
    else:
        train(args.days, args.steps, args.balance, args.interval,
              model_path=args.model, algo=args.algo,
              edge_cases=args.edge_cases,
              synthetic_ratio=args.synthetic_ratio)
