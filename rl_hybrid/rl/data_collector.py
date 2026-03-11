"""Supabase 라이브 데이터 수집기 — 과거 매매 결과를 RL 훈련 데이터로 변환

Supabase의 decisions, market_context_log, agent_switches 테이블에서
실제 매매 결과를 가져와 RL 환경의 리플레이 버퍼로 구성한다.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rl_hybrid.config import config

logger = logging.getLogger("rl.data_collector")


class LiveDataCollector:
    """Supabase에서 라이브 매매 데이터를 수집하여 RL 훈련 데이터로 변환"""

    def __init__(self):
        self.db_url = config.supabase.db_url
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL이 설정되지 않았습니다")

    def _get_conn(self):
        return psycopg2.connect(self.db_url)

    def collect_decisions(self, days: int = 30, limit: int = 500) -> list[dict]:
        """과거 매매 결정 + 시장 맥락 + 결과 수집

        Returns:
            [{"decision", "confidence", "current_price", "rsi", "fgi",
              "outcome_4h_pct", "outcome_24h_pct", "was_correct_24h",
              "market_context", ...}, ...]
        """
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT
                    d.id::text,
                    d.decision,
                    d.confidence,
                    d.current_price,
                    d.rsi_value,
                    d.fear_greed_value,
                    d.sma20_price,
                    d.trade_amount,
                    d.trade_volume,
                    d.profit_loss,
                    d.outcome_4h_pct,
                    d.outcome_24h_pct,
                    d.was_correct_4h,
                    d.was_correct_24h,
                    d.created_at::text,
                    d.market_data_snapshot,
                    esl.fusion_score,
                    esl.fusion_signal,
                    esl.binance_score,
                    esl.long_short_ratio,
                    esl.funding_rate,
                    esl.whale_score,
                    esl.macro_score,
                    esl.fgi_value AS esl_fgi,
                    esl.news_sentiment,
                    esl.kimchi_premium_pct,
                    esl.eth_btc_score
                FROM decisions d
                LEFT JOIN external_signal_log esl ON d.external_signal_id = esl.id
                WHERE d.created_at > NOW() - INTERVAL '%s days'
                  AND d.outcome_24h_pct IS NOT NULL
                ORDER BY d.created_at DESC
                LIMIT %s
            """, (days, limit))

            results = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()

            logger.info(f"매매 결정 수집: {len(results)}건 ({days}일)")
            return results

        except Exception as e:
            logger.error(f"매매 결정 수집 실패: {e}")
            return []

    def collect_switch_outcomes(self, days: int = 30) -> list[dict]:
        """전략 전환 성과 데이터 수집 (메타 학습용)"""
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT
                    from_agent,
                    to_agent,
                    switch_reason,
                    danger_score,
                    opportunity_score,
                    market_state,
                    profit_after_4h,
                    profit_after_24h,
                    switch_time::text,
                    CASE
                        WHEN profit_after_24h > 1.0 THEN 'good'
                        WHEN profit_after_24h < -1.0 THEN 'bad'
                        ELSE 'neutral'
                    END AS outcome
                FROM agent_switches
                WHERE switch_time > NOW() - INTERVAL '%s days'
                  AND profit_after_24h IS NOT NULL
                ORDER BY switch_time DESC
            """, (days,))

            results = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()

            logger.info(f"전환 성과 수집: {len(results)}건")
            return results

        except Exception as e:
            logger.error(f"전환 성과 수집 실패: {e}")
            return []

    def collect_portfolio_history(self, days: int = 30) -> list[dict]:
        """포트폴리오 스냅샷 히스토리 (보상 계산용)"""
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT
                    total_value_krw,
                    btc_balance,
                    krw_balance,
                    btc_avg_price,
                    btc_current_price,
                    profit_loss_pct,
                    created_at::text
                FROM portfolio_snapshots
                WHERE created_at > NOW() - INTERVAL '%s days'
                ORDER BY created_at ASC
            """, (days,))

            results = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()

            logger.info(f"포트폴리오 히스토리: {len(results)}건")
            return results

        except Exception as e:
            logger.error(f"포트폴리오 히스토리 수집 실패: {e}")
            return []

    def build_training_episodes(self, days: int = 30) -> list[dict]:
        """수집된 데이터를 RL 훈련 에피소드로 변환

        각 매매 결정을 (state, action, reward) 튜플로 변환한다.

        Returns:
            [{"state": dict, "action": float, "reward": float,
              "outcome": str, "timestamp": str}, ...]
        """
        decisions = self.collect_decisions(days)
        if not decisions:
            return []

        episodes = []
        for d in decisions:
            # 행동 매핑: decision → continuous action [-1, 1]
            action_map = {"buy": 0.5, "sell": -0.5, "hold": 0.0}
            action = action_map.get(d["decision"], 0.0)

            # 신뢰도로 행동 강도 조절
            confidence = float(d.get("confidence") or 0.5)
            action *= (0.5 + confidence)  # 0.25 ~ 1.0 범위로 스케일

            # 보상: 24시간 후 결과
            outcome_pct = float(d.get("outcome_24h_pct") or 0)

            # 행동 방향과 결과 일치도 기반 보상
            if d["decision"] == "buy":
                reward = outcome_pct / 10  # 매수 후 상승 → 양의 보상
            elif d["decision"] == "sell":
                reward = -outcome_pct / 10  # 매도 후 하락 → 양의 보상
            else:
                reward = -abs(outcome_pct) / 20  # 관망 시 큰 변동 → 음의 보상

            # 상태 벡터 구성요소
            state = {
                "current_price": float(d.get("current_price") or 0),
                "rsi": float(d.get("rsi_value") or 50),
                "fgi": float(d.get("fear_greed_value") or 50),
                "sma20": float(d.get("sma20_price") or 0),
                "fusion_score": float(d.get("fusion_score") or 0),
                "binance_score": float(d.get("binance_score") or 0),
                "long_short_ratio": float(d.get("long_short_ratio") or 1.0),
                "funding_rate": float(d.get("funding_rate") or 0),
                "whale_score": float(d.get("whale_score") or 0),
                "macro_score": float(d.get("macro_score") or 0),
                "news_sentiment": float(d.get("news_sentiment") or 0),
                "kimchi_premium": float(d.get("kimchi_premium_pct") or 0),
            }

            episodes.append({
                "state": state,
                "action": action,
                "reward": reward,
                "outcome_pct": outcome_pct,
                "was_correct": d.get("was_correct_24h"),
                "decision": d["decision"],
                "confidence": confidence,
                "timestamp": d.get("created_at", ""),
            })

        logger.info(
            f"훈련 에피소드 변환: {len(episodes)}건, "
            f"avg_reward={np.mean([e['reward'] for e in episodes]):.4f}"
        )
        return episodes

    def get_training_stats(self, days: int = 30) -> dict:
        """훈련 데이터 통계 요약"""
        decisions = self.collect_decisions(days)
        if not decisions:
            return {"available": False, "count": 0}

        outcomes = [float(d.get("outcome_24h_pct") or 0) for d in decisions]
        correct = [d for d in decisions if d.get("was_correct_24h")]

        return {
            "available": True,
            "count": len(decisions),
            "days": days,
            "win_rate": len(correct) / len(decisions) if decisions else 0,
            "avg_outcome_pct": float(np.mean(outcomes)),
            "std_outcome_pct": float(np.std(outcomes)),
            "by_decision": {
                "buy": len([d for d in decisions if d["decision"] == "buy"]),
                "sell": len([d for d in decisions if d["decision"] == "sell"]),
                "hold": len([d for d in decisions if d["decision"] == "hold"]),
            },
        }
