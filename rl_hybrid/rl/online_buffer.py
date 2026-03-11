"""온라인 학습 버퍼 — 실매매 결과를 축적하여 주기적 미세 학습

매 파이프라인 실행(4h)마다:
1. RL advisory의 (obs, action, market_state) 저장
2. retrospective.py가 1h/4h 후 실제 수익률 기록
3. 버퍼가 TRIGGER_SIZE(50건)에 도달하면 미세 학습 실행
4. 성능 저하 없으면 best 모델 업데이트

사용법:
    # run_agents.py Phase 6.5에서 자동 호출
    from rl_hybrid.rl.online_buffer import OnlineExperienceBuffer
    buf = OnlineExperienceBuffer()
    buf.add_experience(market_data, external_data, portfolio, agent_state, action, decision)
    if buf.should_train():
        result = buf.micro_train()
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger("rl.online_buffer")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_DIR / "data" / "rl_models"
BEST_DIR = MODEL_DIR / "best"
BUFFER_PATH = MODEL_DIR / "online_buffer.json"

KST = timezone(timedelta(hours=9))

TRIGGER_SIZE = 50       # 이 건수 이상이면 micro-train 트리거
MICRO_TRAIN_STEPS = 5000  # 미세 학습 스텝 수
EVAL_EPISODES = 5         # 평가 에피소드 수


class OnlineExperienceBuffer:
    """실매매 경험 축적 + 주기적 미세 학습"""

    def __init__(self):
        self.buffer = self._load()

    def _load(self) -> list:
        if BUFFER_PATH.exists():
            try:
                with open(BUFFER_PATH) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(BUFFER_PATH, "w") as f:
            json.dump(self.buffer, f, ensure_ascii=False, indent=1)

    def add_experience(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict,
        rl_action: float,
        agent_decision: str,
    ):
        """실매매 경험 1건 추가"""
        # 핵심 시장 상태만 저장 (용량 절약)
        ticker = market_data.get("ticker", {})
        indicators = market_data.get("indicators", {})
        fgi = market_data.get("fear_greed", {})

        entry = {
            "timestamp": datetime.now(KST).isoformat(),
            "market_state": {
                "price": ticker.get("trade_price"),
                "change_24h": ticker.get("signed_change_rate"),
                "rsi_14": indicators.get("rsi_14"),
                "sma_20": indicators.get("sma_20"),
                "fgi_value": fgi.get("value"),
                "volume_24h": ticker.get("acc_trade_volume_24h"),
            },
            "rl_action": float(rl_action),
            "agent_decision": agent_decision,
            "outcome_pct": None,  # retrospective.py가 나중에 채움
            "outcome_filled": False,
        }
        self.buffer.append(entry)
        self._save()
        logger.info(f"온라인 버퍼: {len(self.buffer)}/{TRIGGER_SIZE}건 축적")

    def update_outcomes(self, recent_outcomes: list[dict]):
        """retrospective 결과로 outcome 업데이트

        Args:
            recent_outcomes: [{"timestamp": ..., "outcome_4h_pct": ...}, ...]
        """
        updated = 0
        for outcome in recent_outcomes:
            ts = outcome.get("timestamp", "")
            pct = outcome.get("outcome_4h_pct")
            if pct is None:
                continue
            for entry in self.buffer:
                if not entry["outcome_filled"] and entry["timestamp"][:16] == ts[:16]:
                    entry["outcome_pct"] = float(pct)
                    entry["outcome_filled"] = True
                    updated += 1
                    break
        if updated:
            self._save()
            logger.info(f"온라인 버퍼: {updated}건 outcome 업데이트")

    def should_train(self) -> bool:
        """미세 학습 실행 조건 확인"""
        filled = sum(1 for e in self.buffer if e["outcome_filled"])
        return len(self.buffer) >= TRIGGER_SIZE and filled >= TRIGGER_SIZE * 0.5

    def micro_train(self) -> dict:
        """미세 학습 실행

        Returns:
            {"success": bool, "message": str, "new_return": float, "old_return": float}
        """
        try:
            from rl_hybrid.rl.policy import SB3_AVAILABLE
            if not SB3_AVAILABLE:
                return {"success": False, "message": "SB3 미설치"}

            from rl_hybrid.rl.train import prepare_data, evaluate, get_trader_class
            from rl_hybrid.rl.environment import BitcoinTradingEnv

            # 최근 14일 데이터로 환경 구성
            train_candles, eval_candles, train_signals, eval_signals = prepare_data(14)

            # 현재 best 모델 로드
            info_path = BEST_DIR / "model_info.json"
            algo = "ppo"
            if info_path.exists():
                try:
                    with open(info_path) as f:
                        algo = json.load(f).get("algorithm", "ppo")
                except Exception:
                    pass

            best_model_path = str(BEST_DIR / "best_model")
            if not (BEST_DIR / "best_model.zip").exists():
                return {"success": False, "message": "best 모델 없음"}

            TraderClass = get_trader_class(algo)

            # 현재 best 성능 측정
            eval_env_old = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=10_000_000,
                external_signals=eval_signals,
            )
            trader_old = TraderClass(env=eval_env_old, model_path=best_model_path)
            old_stats = evaluate(trader_old, eval_env_old, episodes=EVAL_EPISODES)
            old_return = float(np.mean([s["total_return_pct"] for s in old_stats]))
            old_sharpe = float(np.mean([s["sharpe_ratio"] for s in old_stats]))

            # 미세 학습
            train_env = BitcoinTradingEnv(
                candles=train_candles, initial_balance=10_000_000,
                external_signals=train_signals,
            )
            eval_env_new = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=10_000_000,
                external_signals=eval_signals,
            )

            trader_new = TraderClass(env=train_env)
            trader_new.load(best_model_path)
            trader_new.model.set_env(train_env)

            trader_new.train(
                total_timesteps=MICRO_TRAIN_STEPS,
                eval_env=eval_env_new,
                save_freq=MICRO_TRAIN_STEPS,
            )

            # 새 모델 성능 측정
            eval_env_check = BitcoinTradingEnv(
                candles=eval_candles, initial_balance=10_000_000,
                external_signals=eval_signals,
            )
            new_stats = evaluate(trader_new, eval_env_check, episodes=EVAL_EPISODES)
            new_return = float(np.mean([s["total_return_pct"] for s in new_stats]))
            new_sharpe = float(np.mean([s["sharpe_ratio"] for s in new_stats]))

            # 교체 판단: 성능 저하 없으면(샤프 -0.1 이내) 교체
            improved = new_sharpe >= old_sharpe - 0.1

            result = {
                "success": True,
                "old_return": round(old_return, 2),
                "old_sharpe": round(old_sharpe, 3),
                "new_return": round(new_return, 2),
                "new_sharpe": round(new_sharpe, 3),
                "replaced": improved,
            }

            if improved:
                # 백업 후 교체
                backup_path = MODEL_DIR / "retrain_history" / f"micro_{datetime.now(KST).strftime('%Y%m%d_%H%M')}.zip"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                if (BEST_DIR / "best_model.zip").exists():
                    shutil.copy2(BEST_DIR / "best_model.zip", backup_path)
                trader_new.save(best_model_path)
                result["message"] = f"미세 학습 성공: {old_return:.2f}% → {new_return:.2f}%"
                logger.info(f"미세 학습 모델 교체: {old_return:.2f}% → {new_return:.2f}%")
            else:
                result["message"] = f"미세 학습 성능 저하: {old_sharpe:.3f} → {new_sharpe:.3f}, 교체 안 함"
                logger.info(result["message"])

            # 버퍼 클리어
            self.buffer.clear()
            self._save()

            return result

        except Exception as e:
            logger.error(f"미세 학습 실패: {e}")
            return {"success": False, "message": str(e)}

    def get_stats(self) -> dict:
        """버퍼 상태 통계"""
        total = len(self.buffer)
        filled = sum(1 for e in self.buffer if e["outcome_filled"])
        return {
            "total": total,
            "outcome_filled": filled,
            "trigger_size": TRIGGER_SIZE,
            "ready_to_train": self.should_train(),
        }
