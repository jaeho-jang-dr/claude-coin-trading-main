"""Trainer 전용 스크립트 — 훈련 실행 + 결과 DB 업로드

Tier 1 (Trainer) 사용자가 이 스크립트만 실행합니다.
코드 수정 없이 훈련 파라미터만 조절 가능합니다.

사용법:
    # 기본 훈련 (PPO, 180일, 100K 스텝)
    python -m rl_hybrid.rl.trainer_submit

    # 파라미터 변경
    python -m rl_hybrid.rl.trainer_submit --algo sac --steps 200000 --days 365

    # 엣지 케이스 혼합 훈련
    python -m rl_hybrid.rl.trainer_submit --edge-cases --synthetic-ratio 0.4

    # 특정 트레이너 이름 지정
    python -m rl_hybrid.rl.trainer_submit --trainer-id "trainer-mac-B"
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.rl.train import (
    evaluate,
    get_trader_class,
    prepare_data,
    prepare_edge_case_data,
)
from rl_hybrid.rl.environment import BitcoinTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.trainer_submit")


def get_default_trainer_id() -> str:
    """머신 기반 기본 trainer_id 생성"""
    hostname = platform.node().split(".")[0]
    return f"trainer-{hostname}"


def compute_model_hash(model_path: str) -> str:
    """모델 파일 SHA256 해시"""
    if not os.path.exists(model_path):
        return ""
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def get_current_best() -> dict:
    """현재 best 모델 정보 로드"""
    info_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "rl_models", "best", "model_info.json",
    )
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f)
    return {}


def evaluate_on_real_data(model_path: str, algo: str, days: int = 180,
                          episodes: int = 10) -> dict:
    """순수 실제 데이터로만 모델 평가 (공정 비교)"""
    logger.info("순수 실제 데이터로 공정 평가 중...")
    _, eval_candles, _, eval_signals = prepare_data(days, "1h")

    env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=10_000_000,
        external_signals=eval_signals,
    )

    TraderClass = get_trader_class(algo)
    load_path = model_path.removesuffix(".zip")  # SB3 adds .zip internally
    trader = TraderClass(env=env, model_path=load_path)
    stats = evaluate(trader, env, episodes=episodes)

    return {
        "avg_return_pct": round(float(np.mean([s["total_return_pct"] for s in stats])), 4),
        "avg_sharpe": round(float(np.mean([s["sharpe_ratio"] for s in stats])), 4),
        "avg_mdd": round(float(np.mean([s["max_drawdown"] for s in stats])), 6),
        "avg_trades": round(float(np.mean([s["trade_count"] for s in stats])), 1),
    }


def submit_to_db(result: dict) -> bool:
    """훈련 결과를 Supabase DB에 업로드"""
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_key:
        logger.warning("SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY 미설정 -- 로컬 JSON으로 저장")
        return save_local(result)

    try:
        import requests
        resp = requests.post(
            f"{supabase_url}/rest/v1/rl_training_results",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json=result,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            record_id = data[0]["id"] if data else "?"
            logger.info(f"DB 업로드 성공: id={record_id}")
            return True
        else:
            logger.error(f"DB 업로드 실패: {resp.status_code} {resp.text}")
            return save_local(result)
    except Exception as e:
        logger.error(f"DB 연결 실패: {e}")
        return save_local(result)


def save_local(result: dict) -> bool:
    """DB 연결 불가 시 로컬 JSON 저장 (나중에 수동 업로드)"""
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "training_submissions",
    )
    os.makedirs(base_dir, exist_ok=True)

    filename = f"{result['trainer_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(base_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"로컬 저장: {filepath}")
    logger.info("나중에 Admin에게 전달하거나 DB 연결 후 재업로드하세요.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RL 훈련 실행 + 결과 업로드 (Trainer 전용)"
    )
    parser.add_argument("--trainer-id", type=str, default=None,
                        help="훈련자 식별자 (기본: trainer-{hostname})")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3"],
                        help="RL 알고리즘")
    parser.add_argument("--days", type=int, default=180,
                        help="훈련 데이터 기간 (일)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="총 훈련 스텝")
    parser.add_argument("--balance", type=float, default=10_000_000,
                        help="초기 잔고 (KRW)")
    parser.add_argument("--edge-cases", action="store_true",
                        help="엣지 케이스 합성 데이터 혼합")
    parser.add_argument("--synthetic-ratio", type=float, default=0.3,
                        help="합성 데이터 비율")
    parser.add_argument("--model", type=str, default=None,
                        help="기존 모델 경로 (fine-tuning)")

    args = parser.parse_args()
    trainer_id = args.trainer_id or get_default_trainer_id()

    logger.info(f"{'='*60}")
    logger.info(f"  Trainer: {trainer_id}")
    logger.info(f"  알고리즘: {args.algo.upper()}")
    logger.info(f"  설정: {args.days}일, {args.steps:,} 스텝, 엣지케이스={args.edge_cases}")
    logger.info(f"{'='*60}")

    # ── Step 1: 훈련 ──
    start_time = time.time()

    from rl_hybrid.rl.policy import SB3_AVAILABLE
    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 미설치. pip install stable-baselines3")
        return

    TraderClass = get_trader_class(args.algo)

    if args.edge_cases:
        train_candles, eval_candles, train_signals, eval_signals = \
            prepare_edge_case_data(args.days, "1h", args.synthetic_ratio)
    else:
        train_candles, eval_candles, train_signals, eval_signals = \
            prepare_data(args.days, "1h")

    train_env = BitcoinTradingEnv(
        candles=train_candles,
        initial_balance=args.balance,
        external_signals=train_signals,
    )
    eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=args.balance,
        external_signals=eval_signals,
    )

    if args.model:
        logger.info(f"기존 모델에서 fine-tuning: {args.model}")
        trader = TraderClass(env=train_env)
        trader.load(args.model)
        trader.model.set_env(train_env)
    else:
        trader = TraderClass(env=train_env)

    trader.train(
        total_timesteps=args.steps,
        eval_env=eval_env,
        save_freq=args.steps // 10,
    )

    elapsed = time.time() - start_time
    logger.info(f"훈련 완료: {elapsed:.0f}초")

    # ── Step 2: 순수 실제 데이터로 공정 평가 ──
    from rl_hybrid.rl.policy import MODEL_DIR
    model_path = os.path.join(MODEL_DIR, f"{args.algo}_btc_latest.zip")
    eval_result = evaluate_on_real_data(model_path, args.algo, args.days)

    # ── Step 3: 현재 best와 비교 ──
    current_best = get_current_best()
    best_return = current_best.get("avg_return_pct", 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"  결과 비교")
    logger.info(f"{'='*60}")
    logger.info(f"  현재 best:  {best_return:.2f}%")
    logger.info(f"  이번 훈련:  {eval_result['avg_return_pct']:.2f}%")
    logger.info(f"  차이:       {eval_result['avg_return_pct'] - best_return:+.2f}%p")

    # ── Step 4: 결과 업로드 ──
    submission = {
        "trainer_id": trainer_id,
        "algorithm": args.algo,
        "training_days": args.days,
        "training_steps": args.steps,
        "interval": "1h",
        "edge_cases": args.edge_cases,
        "synthetic_ratio": args.synthetic_ratio if args.edge_cases else 0.0,
        "avg_return_pct": eval_result["avg_return_pct"],
        "avg_sharpe": eval_result["avg_sharpe"],
        "avg_mdd": eval_result["avg_mdd"],
        "avg_trades": eval_result["avg_trades"],
        "observation_dim": 42,
        "model_hash": compute_model_hash(model_path),
        "best_return_pct": best_return,
        "improvement": round(eval_result["avg_return_pct"] - best_return, 4),
    }

    submit_to_db(submission)

    # ── Step 5: 로컬 모델 보관 ──
    candidate_dir = os.path.join(
        MODEL_DIR, "submissions", f"{trainer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(candidate_dir, exist_ok=True)

    import shutil
    shutil.copy2(model_path, os.path.join(candidate_dir, "model.zip"))
    with open(os.path.join(candidate_dir, "result.json"), "w") as f:
        json.dump(submission, f, indent=2, default=str)

    logger.info(f"모델 보관: {candidate_dir}")
    logger.info("Admin이 리뷰 후 best 승격 여부를 결정합니다.")


if __name__ == "__main__":
    main()
