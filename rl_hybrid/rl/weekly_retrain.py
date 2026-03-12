"""주간 자동 재학습 — 매주 최신 데이터로 RL 모델 재훈련

매주 1회 cron으로 실행하여:
1. 현재 best 모델을 최신 데이터로 fine-tuning (MORL 래퍼 적용 가능)
2. 새 모델을 scratch에서 훈련
3. 3개 알고리즘(PPO/SAC/TD3) 비교
4. 기존 best 대비 개선된 경우에만 교체
5. 추가 RL 모듈 훈련 (Offline RL, Decision Transformer, Multi-Agent, LLM Projection, Self-Tuning)
6. 결과를 텔레그램으로 알림

사용법:
    python -m rl_hybrid.rl.weekly_retrain
    python -m rl_hybrid.rl.weekly_retrain --days 90 --steps 200000
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.rl.train import prepare_data, evaluate, get_trader_class
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.config import SystemConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.weekly_retrain")

KST = timezone(timedelta(hours=9))
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_DIR / "data" / "rl_models"
BEST_DIR = MODEL_DIR / "best"
BEST_MODEL = BEST_DIR / "best_model"
INFO_PATH = BEST_DIR / "model_info.json"
HISTORY_DIR = MODEL_DIR / "retrain_history"


def load_current_best_info() -> dict:
    """현재 best 모델 정보 로드"""
    if INFO_PATH.exists():
        try:
            with open(INFO_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"algorithm": "ppo", "avg_return_pct": 0, "avg_sharpe": 0}


def evaluate_model(trader, eval_candles, eval_signals, balance, episodes=10, sys_config=None) -> dict:
    """모델 평가 후 통계 반환"""
    eval_env = BitcoinTradingEnv(
        candles=eval_candles,
        initial_balance=balance,
        external_signals=eval_signals,
    )
    if sys_config is not None:
        eval_env = _maybe_wrap_morl(eval_env, sys_config)
    stats = evaluate(trader, eval_env, episodes=episodes)
    return {
        "avg_return": float(np.mean([s["total_return_pct"] for s in stats])),
        "avg_sharpe": float(np.mean([s["sharpe_ratio"] for s in stats])),
        "avg_mdd": float(np.mean([s["max_drawdown"] for s in stats])),
        "avg_trades": float(np.mean([s["trade_count"] for s in stats])),
    }


def notify_telegram(message: str):
    """텔레그램 알림 전송"""
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/notify_telegram.py", "info", "RL 주간 재학습", message],
            cwd=str(PROJECT_DIR), check=False, timeout=15,
        )
    except Exception as e:
        logger.warning(f"텔레그램 알림 실패: {e}")


def _maybe_wrap_morl(env, sys_config: SystemConfig):
    """MORL_ENVELOPE=true인 경우 환경을 MultiObjectiveEnv로 래핑"""
    morl_cfg = sys_config.multi_objective_rl
    if not morl_cfg.envelope_morl:
        return env
    try:
        from rl_hybrid.rl.multi_objective_reward import create_multi_objective_env
        wrapped = create_multi_objective_env(
            base_env=env,
            weights=morl_cfg.weights,
            envelope_morl=True,
            adaptive_weights=morl_cfg.adaptive_weights,
        )
        logger.info("MORL Envelope 래퍼 적용 완료")
        return wrapped
    except Exception as e:
        logger.warning(f"MORL 래퍼 적용 실패 (원본 환경 사용): {e}")
        return env


def _train_extra_stages(days: int, sys_config: SystemConfig) -> dict:
    """Stage 2~6: 추가 RL 모듈 훈련 (각 단계 독립 실행, 실패해도 다음 진행)

    Returns:
        각 스테이지별 결과 딕셔너리
    """
    results = {}

    # Stage 2: Offline RL (CQL) — 축적된 DB 데이터로 오프라인 학습
    logger.info("\n--- Stage 2: Offline RL (CQL) ---")
    offline_cycle_id = None
    try:
        from rl_hybrid.rl.offline_rl import train_offline
        t0 = time.time()
        try:
            from rl_hybrid.rl.rl_db_logger import log_training_start, log_training_complete
            offline_cycle_id = log_training_start(
                cycle_type="weekly", algorithm="cql", module="offline_rl",
                training_epochs=50, data_days=days,
            )
        except Exception:
            pass
        offline_result = train_offline(
            algorithm="cql",
            epochs=50,
            min_data_points=20,
            output_dir=str(MODEL_DIR / "offline"),
        )
        elapsed = time.time() - t0
        results["offline_rl"] = {"status": "ok", "elapsed": f"{elapsed:.0f}s", **offline_result}
        logger.info(f"Offline RL 완료: {offline_result} ({elapsed:.0f}s)")
        if offline_cycle_id:
            try:
                eval_m = offline_result.get("eval_metrics", {})
                log_training_complete(
                    cycle_id=offline_cycle_id,
                    direction_accuracy=eval_m.get("direction_accuracy"),
                    q_loss=offline_result.get("train_metrics", {}).get("final_q_loss"),
                    cql_penalty=offline_result.get("train_metrics", {}).get("final_cql_penalty"),
                    model_path=offline_result.get("model_path"),
                    model_version=offline_result.get("version_id"),
                    elapsed_seconds=elapsed, status="completed",
                )
            except Exception:
                pass
    except Exception as e:
        results["offline_rl"] = {"status": "skipped", "reason": str(e)}
        logger.warning(f"Offline RL 훈련 건너뜀: {e}")
        if offline_cycle_id:
            try:
                log_training_complete(cycle_id=offline_cycle_id, status="failed",
                                     error_message=str(e)[:500], elapsed_seconds=time.time() - t0)
            except Exception:
                pass

    # Stage 3: Decision Transformer
    logger.info("\n--- Stage 3: Decision Transformer ---")
    dt_cycle_id = None
    try:
        from rl_hybrid.rl.decision_transformer import train_dt
        t0 = time.time()
        try:
            from rl_hybrid.rl.rl_db_logger import log_training_start, log_training_complete
            dt_cycle_id = log_training_start(
                cycle_type="weekly", algorithm="dt", module="decision_transformer",
                training_epochs=50, data_days=days, interval="4h",
            )
        except Exception:
            pass
        dt_result = train_dt(days=days, interval="4h", n_epochs=50)
        elapsed = time.time() - t0
        results["decision_transformer"] = {"status": "ok", "elapsed": f"{elapsed:.0f}s", **dt_result}
        logger.info(f"Decision Transformer 완료: {dt_result} ({elapsed:.0f}s)")
        if dt_cycle_id:
            try:
                log_training_complete(
                    cycle_id=dt_cycle_id,
                    best_eval_loss=dt_result.get("best_eval_loss"),
                    n_sequences=dt_result.get("n_sequences"),
                    model_version=dt_result.get("registry_version"),
                    elapsed_seconds=elapsed, status="completed",
                )
            except Exception:
                pass
    except Exception as e:
        results["decision_transformer"] = {"status": "skipped", "reason": str(e)}
        logger.warning(f"DT 훈련 건너뜀: {e}")
        if dt_cycle_id:
            try:
                log_training_complete(cycle_id=dt_cycle_id, status="failed",
                                     error_message=str(e)[:500], elapsed_seconds=time.time() - t0)
            except Exception:
                pass

    # Stage 4: Multi-Agent Consensus (설정에서 enabled인 경우)
    if sys_config.multi_agent.enabled:
        logger.info("\n--- Stage 4: Multi-Agent Consensus ---")
        ma_cycle_id = None
        try:
            from rl_hybrid.rl.multi_agent_consensus import MultiAgentTrainer
            t0 = time.time()
            try:
                from rl_hybrid.rl.rl_db_logger import log_training_start, log_training_complete
                ma_cycle_id = log_training_start(
                    cycle_type="weekly", algorithm="multi_agent",
                    module="multi_agent_consensus",
                    training_steps=sys_config.multi_agent.scalping_steps + sys_config.multi_agent.swing_steps,
                    data_days=days,
                )
            except Exception:
                pass
            trainer = MultiAgentTrainer(
                scalping_steps=sys_config.multi_agent.scalping_steps,
                swing_steps=sys_config.multi_agent.swing_steps,
                weight_learner_steps=sys_config.multi_agent.weight_learner_steps,
            )
            trainer.train(joint_finetune=False)
            elapsed = time.time() - t0
            results["multi_agent"] = {"status": "ok", "elapsed": f"{elapsed:.0f}s"}
            logger.info(f"Multi-Agent Consensus 훈련 완료 ({elapsed:.0f}s)")
            if ma_cycle_id:
                try:
                    log_training_complete(
                        cycle_id=ma_cycle_id, elapsed_seconds=elapsed, status="completed",
                    )
                except Exception:
                    pass
        except Exception as e:
            results["multi_agent"] = {"status": "skipped", "reason": str(e)}
            logger.warning(f"Multi-Agent 훈련 건너뜀: {e}")
            if ma_cycle_id:
                try:
                    log_training_complete(cycle_id=ma_cycle_id, status="failed",
                                         error_message=str(e)[:500], elapsed_seconds=time.time() - t0)
                except Exception:
                    pass
    else:
        results["multi_agent"] = {"status": "disabled"}
        logger.info("Multi-Agent Consensus: 비활성화 (MULTI_AGENT_ENABLED=false)")

    # Stage 5: LLM Projection (설정에서 enabled이고 충분한 데이터가 있는 경우)
    if sys_config.llm_state.enabled:
        logger.info("\n--- Stage 5: LLM Projection Layer ---")
        try:
            from rl_hybrid.rl.llm_state_encoder import train_projection
            t0 = time.time()
            proj_result = train_projection(epochs=30)
            elapsed = time.time() - t0
            results["llm_projection"] = {"status": "ok", "elapsed": f"{elapsed:.0f}s", **proj_result}
            logger.info(f"LLM Projection 훈련 완료: {proj_result} ({elapsed:.0f}s)")
        except Exception as e:
            results["llm_projection"] = {"status": "skipped", "reason": str(e)}
            logger.warning(f"LLM Projection 훈련 건너뜀: {e}")
    else:
        results["llm_projection"] = {"status": "disabled"}
        logger.info("LLM Projection: 비활성화 (LLM_STATE_ENCODER_ENABLED=false)")

    # Stage 6: Self-Tuning Agent (경량, 항상 실행)
    logger.info("\n--- Stage 6: Self-Tuning Agent ---")
    try:
        from rl_hybrid.rl.self_tuning_rl import TuningAgent, TuningEnvironment
        t0 = time.time()
        tuning_env = TuningEnvironment()
        tuner = TuningAgent(env=tuning_env)
        tuner.train(total_timesteps=50_000)
        tuner.save()
        elapsed = time.time() - t0
        results["self_tuning"] = {"status": "ok", "elapsed": f"{elapsed:.0f}s"}
        logger.info(f"Self-Tuning Agent 훈련 완료 ({elapsed:.0f}s)")
    except Exception as e:
        results["self_tuning"] = {"status": "skipped", "reason": str(e)}
        logger.warning(f"Self-Tuning 훈련 건너뜀: {e}")

    return results


def _build_summary_message(
    stage1_msg: str,
    extra_results: dict,
) -> str:
    """전체 재학습 결과를 텔레그램 알림 메시지로 조합"""
    lines = [stage1_msg, ""]

    ok_count = sum(1 for v in extra_results.values() if v.get("status") == "ok")
    skip_count = sum(1 for v in extra_results.values() if v.get("status") == "skipped")
    disabled_count = sum(1 for v in extra_results.values() if v.get("status") == "disabled")

    if extra_results:
        lines.append(f"[추가 모듈] OK={ok_count} / 건너뜀={skip_count} / 비활성={disabled_count}")
        for name, result in extra_results.items():
            status = result.get("status", "?")
            elapsed = result.get("elapsed", "")
            if status == "ok":
                detail = f" ({elapsed})" if elapsed else ""
                lines.append(f"  {name}: OK{detail}")
            elif status == "skipped":
                reason = result.get("reason", "")[:60]
                lines.append(f"  {name}: 건너뜀 - {reason}")
            else:
                lines.append(f"  {name}: {status}")

    return "\n".join(lines)


def weekly_retrain(days: int = 90, total_steps: int = 200_000, balance: float = 10_000_000):
    """주간 재학습 메인 로직"""
    from rl_hybrid.rl.policy import SB3_AVAILABLE
    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 미설치")
        return

    sys_config = SystemConfig()
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== 주간 재학습 시작 ({timestamp}) ===")
    use_morl = sys_config.multi_objective_rl.envelope_morl
    if use_morl:
        logger.info("MORL Envelope 모드 활성화 -- 환경에 MultiObjectiveEnv 래퍼 적용")

    # 데이터 준비
    train_candles, eval_candles, train_signals, eval_signals = prepare_data(days)

    # 현재 best 모델 정보
    current_info = load_current_best_info()
    current_algo = current_info.get("algorithm", "ppo")
    logger.info(f"현재 best: {current_algo.upper()} (수익률={current_info.get('avg_return_pct', '?')}%)")

    candidates = {}

    # ================================================================
    # Stage 1: PPO/SAC/TD3 훈련 (기존 로직)
    # ================================================================
    logger.info("\n=== Stage 1: PPO/SAC/TD3 훈련 ===")

    # 후보 1: 현재 best fine-tuning (같은 알고리즘)
    if (BEST_MODEL.parent / "best_model.zip").exists():
        try:
            logger.info(f"\n--- 후보 1: {current_algo.upper()} fine-tuning ---")
            TraderClass = get_trader_class(current_algo)
            train_env = BitcoinTradingEnv(
                candles=train_candles, initial_balance=balance,
                external_signals=train_signals,
            )
            train_env = _maybe_wrap_morl(train_env, sys_config)
            trader_ft = TraderClass(env=train_env)
            trader_ft.load(str(BEST_MODEL))
            trader_ft.model.set_env(train_env)

            eval_env_ft = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=balance,
                external_signals=eval_signals,
            )
            eval_env_ft = _maybe_wrap_morl(eval_env_ft, sys_config)
            trader_ft.train(
                total_timesteps=total_steps,
                eval_env=eval_env_ft,
                save_freq=total_steps // 5,
            )

            stats = evaluate_model(trader_ft, eval_candles, eval_signals, balance, sys_config=sys_config)
            candidates[f"{current_algo}_finetune"] = {"trader": trader_ft, **stats}
            logger.info(f"  fine-tune 결과: 수익률={stats['avg_return']:.2f}%, 샤프={stats['avg_sharpe']:.3f}")
        except Exception as e:
            logger.warning(f"fine-tuning 실패: {e}")

    # 후보 2-4: 알고리즘 scratch 훈련
    # SAC/TD3는 월 1회(1일)만, PPO는 매주
    day_of_month = datetime.now(KST).day
    if day_of_month <= 7:
        algos_to_train = ["ppo", "sac", "td3"]
        logger.info("월초 -- PPO + SAC + TD3 전체 비교")
    else:
        algos_to_train = ["ppo"]
        logger.info("주간 -- PPO만 훈련 (SAC/TD3는 월초에만)")

    for algo in algos_to_train:
        algo_cycle_id = None
        algo_start = time.time()
        try:
            logger.info(f"\n--- 후보: {algo.upper()} scratch ---")

            # DB 로깅: 훈련 시작
            try:
                from rl_hybrid.rl.rl_db_logger import log_training_start, log_training_complete
                algo_cycle_id = log_training_start(
                    cycle_type="weekly",
                    algorithm=algo,
                    module="weekly_retrain",
                    training_steps=total_steps,
                    data_days=days,
                    morl_enabled=use_morl,
                )
            except Exception:
                pass

            TraderClass = get_trader_class(algo)
            train_env = BitcoinTradingEnv(
                candles=train_candles, initial_balance=balance,
                external_signals=train_signals,
            )
            train_env = _maybe_wrap_morl(train_env, sys_config)
            eval_env = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=balance,
                external_signals=eval_signals,
            )
            eval_env = _maybe_wrap_morl(eval_env, sys_config)
            trader = TraderClass(env=train_env)
            trader.train(
                total_timesteps=total_steps,
                eval_env=eval_env,
                save_freq=total_steps // 5,
            )

            stats = evaluate_model(trader, eval_candles, eval_signals, balance, sys_config=sys_config)
            candidates[f"{algo}_scratch"] = {"trader": trader, **stats}
            logger.info(f"  {algo} scratch 결과: 수익률={stats['avg_return']:.2f}%, 샤프={stats['avg_sharpe']:.3f}")

            # DB 로깅: 훈련 완료
            if algo_cycle_id:
                try:
                    log_training_complete(
                        cycle_id=algo_cycle_id,
                        avg_return_pct=stats['avg_return'],
                        avg_sharpe=stats['avg_sharpe'],
                        avg_mdd=stats['avg_mdd'],
                        avg_trades=stats['avg_trades'],
                        elapsed_seconds=time.time() - algo_start,
                        status="completed",
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"{algo} scratch 훈련 실패: {e}")
            if algo_cycle_id:
                try:
                    log_training_complete(
                        cycle_id=algo_cycle_id,
                        elapsed_seconds=time.time() - algo_start,
                        status="failed",
                        error_message=str(e)[:500],
                    )
                except Exception:
                    pass

    if not candidates:
        logger.error("모든 후보 훈련 실패")
        notify_telegram("주간 재학습 실패: 모든 후보 훈련 실패")
        return

    # 현재 best 평가 (동일 eval 데이터로)
    current_stats = None
    if (BEST_MODEL.parent / "best_model.zip").exists():
        try:
            TraderClass = get_trader_class(current_algo)
            eval_env = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=balance,
                external_signals=eval_signals,
            )
            eval_env = _maybe_wrap_morl(eval_env, sys_config)
            trader_current = TraderClass(env=eval_env, model_path=str(BEST_MODEL))
            current_stats = evaluate_model(trader_current, eval_candles, eval_signals, balance, sys_config=sys_config)
            candidates["current_best"] = {"trader": None, **current_stats}
            logger.info(f"\n현재 best 평가: 수익률={current_stats['avg_return']:.2f}%, 샤프={current_stats['avg_sharpe']:.3f}")
        except Exception as e:
            logger.warning(f"현재 best 평가 실패: {e}")

    # 비교 테이블
    logger.info(f"\n{'='*60}")
    logger.info("  주간 재학습 비교 결과")
    logger.info(f"{'='*60}")
    logger.info(f"{'후보':>20} | {'수익률':>10} | {'샤프':>8} | {'MDD':>10} | {'거래수':>8}")
    logger.info("-" * 65)
    for name, r in sorted(candidates.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True):
        logger.info(
            f"{name:>20} | {r['avg_return']:>9.2f}% | {r['avg_sharpe']:>8.3f} | "
            f"{r['avg_mdd']:>9.2%} | {r['avg_trades']:>8.1f}"
        )

    # 최적 후보 선정 (current_best 제외한 새 후보 중)
    new_candidates = {k: v for k, v in candidates.items() if k != "current_best"}
    if not new_candidates:
        logger.info("새 후보 없음, 현재 best 유지")
        return

    best_name = max(new_candidates, key=lambda k: (new_candidates[k]["avg_sharpe"], new_candidates[k]["avg_return"]))
    best_result = new_candidates[best_name]

    # 교체 판단: 현재 best 대비 샤프 or 수익률이 개선되어야 함
    should_replace = True
    if current_stats:
        sharpe_better = best_result["avg_sharpe"] > current_stats["avg_sharpe"] - 0.05
        return_better = best_result["avg_return"] > current_stats["avg_return"]
        should_replace = sharpe_better and return_better
        if not should_replace:
            logger.info(f"\n새 최적({best_name})이 현재 best보다 나쁨 → 교체 안 함")

    if should_replace and best_result.get("trader"):
        # 기존 best 백업
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = HISTORY_DIR / f"best_{timestamp}.zip"
        if (BEST_MODEL.parent / "best_model.zip").exists():
            shutil.copy2(BEST_MODEL.parent / "best_model.zip", backup_path)
            logger.info(f"기존 best 백업: {backup_path}")

        # 새 best 저장
        # 알고리즘 정보에서 algo 추출 (저장 전에 먼저 추출)
        algo_name = best_name.split("_")[0]
        best_result["trader"].save(str(BEST_MODEL))
        latest_name = f"{algo_name}_btc_latest.zip"
        shutil.copy2(BEST_MODEL.parent / "best_model.zip",
                      MODEL_DIR / latest_name)

        model_info = {
            "algorithm": algo_name,
            "avg_return_pct": round(best_result["avg_return"], 4),
            "avg_sharpe": round(best_result["avg_sharpe"], 4),
            "avg_mdd": round(best_result["avg_mdd"], 6),
            "training_steps": total_steps,
            "training_days": days,
            "retrained_at": datetime.now(KST).isoformat(),
            "candidate_name": best_name,
            "morl_enabled": use_morl,
        }
        with open(INFO_PATH, "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"\n새 best 모델 교체 완료: {best_name} ({algo_name.upper()})")
        stage1_msg = (
            f"주간 재학습 완료 -- 모델 교체\n"
            f"후보: {best_name}\n"
            f"수익률: {best_result['avg_return']:.2f}%\n"
            f"샤프: {best_result['avg_sharpe']:.3f}\n"
            f"MDD: {best_result['avg_mdd']:.2%}"
        )
    else:
        logger.info("\n현재 best 모델 유지")
        stage1_msg = (
            f"주간 재학습 완료 -- 현재 모델 유지\n"
            f"최적 후보({best_name}): 수익률={best_result['avg_return']:.2f}%\n"
            f"현재 best: 수익률={current_stats['avg_return']:.2f}%" if current_stats
            else f"주간 재학습 완료 -- 모델 유지"
        )

    # ================================================================
    # Stage 2~6: 추가 RL 모듈 훈련
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("  추가 RL 모듈 훈련 시작 (Stage 2~6)")
    logger.info(f"{'='*60}")

    extra_results = _train_extra_stages(days, sys_config)

    # 최종 요약 메시지 (Stage 1 + Stage 2~6 통합)
    final_msg = _build_summary_message(stage1_msg, extra_results)
    notify_telegram(final_msg)
    logger.info(f"\n=== 주간 재학습 전체 완료 ===")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(description="RL 주간 자동 재학습")
    parser.add_argument("--days", type=int, default=90, help="훈련 데이터 기간 (일)")
    parser.add_argument("--steps", type=int, default=200_000, help="총 훈련 스텝")
    parser.add_argument("--balance", type=float, default=10_000_000, help="초기 잔고")
    args = parser.parse_args()

    weekly_retrain(args.days, args.steps, args.balance)
