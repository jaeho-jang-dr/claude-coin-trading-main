#!/usr/bin/env python3
"""
시그널 분석기

시그널 품질, 전략별 성과, 필터 효과를 분석한다.
워커의 'analyze' 작업에서 호출된다.
"""

from __future__ import annotations

import logging
from collections import defaultdict

log = logging.getLogger("analyzer")


class SignalAnalyzer:
    """시그널 품질 분석"""

    def __init__(self, db_client):
        self.db = db_client

    def run(self, params: dict) -> dict:
        """분석 실행 — params에 따라 다른 분석 수행"""
        analysis_type = params.get("type", "signal_quality")

        if analysis_type == "signal_quality":
            return self.analyze_signal_quality(params)
        elif analysis_type == "filter_effectiveness":
            return self.analyze_filter_effectiveness(params)
        elif analysis_type == "hourly_pattern":
            return self.analyze_hourly_pattern(params)
        elif analysis_type == "strategy_comparison":
            return self.analyze_strategy_comparison(params)
        else:
            return {"error": f"알 수 없는 분석 타입: {analysis_type}"}

    def analyze_signal_quality(self, params: dict) -> dict:
        """전략별 시그널 품질 분석"""
        days = params.get("days", 7)

        signals = self.db.get("signal_attempt_log", {
            "select": "strategy,signal_type,action,outcome_5m_pct,would_have_won_5m,outcome_15m_pct,best_exit_pct,worst_drawdown_pct",
            "signal_type": "neq.no_signal",
            "outcome_5m_pct": "not.is.null",
            "order": "recorded_at.desc",
            "limit": "1000",
        })

        if not signals:
            return {"error": "분석 데이터 없음", "count": 0}

        # 전략별 집계
        by_strategy = defaultdict(lambda: {
            "total": 0, "generated": 0, "blocked": 0,
            "win_5m": 0, "loss_5m": 0,
            "avg_outcome_5m": [], "avg_best_exit": [], "avg_worst_dd": [],
        })

        for sig in signals:
            strategy = sig.get("strategy", "unknown")
            stats = by_strategy[strategy]
            stats["total"] += 1

            if sig.get("signal_type") == "generated":
                stats["generated"] += 1
            elif sig.get("signal_type") == "blocked":
                stats["blocked"] += 1

            if sig.get("would_have_won_5m") is True:
                stats["win_5m"] += 1
            elif sig.get("would_have_won_5m") is False:
                stats["loss_5m"] += 1

            if sig.get("outcome_5m_pct") is not None:
                stats["avg_outcome_5m"].append(sig["outcome_5m_pct"])
            if sig.get("best_exit_pct") is not None:
                stats["avg_best_exit"].append(sig["best_exit_pct"])
            if sig.get("worst_drawdown_pct") is not None:
                stats["avg_worst_dd"].append(sig["worst_drawdown_pct"])

        # 결과 정리
        result = {"strategies": {}, "total_signals": len(signals)}
        for strategy, stats in by_strategy.items():
            evaluated = stats["win_5m"] + stats["loss_5m"]
            result["strategies"][strategy] = {
                "total": stats["total"],
                "generated": stats["generated"],
                "blocked": stats["blocked"],
                "win_rate_5m": round(stats["win_5m"] / max(evaluated, 1) * 100, 1),
                "avg_outcome_5m": round(sum(stats["avg_outcome_5m"]) / max(len(stats["avg_outcome_5m"]), 1), 3),
                "avg_best_exit": round(sum(stats["avg_best_exit"]) / max(len(stats["avg_best_exit"]), 1), 3),
                "avg_worst_dd": round(sum(stats["avg_worst_dd"]) / max(len(stats["avg_worst_dd"]), 1), 3),
            }

        return result

    def analyze_filter_effectiveness(self, params: dict) -> dict:
        """필터별 차단 효과 분석"""
        blocked = self.db.get("signal_attempt_log", {
            "select": "block_filter,would_have_won_5m,outcome_5m_pct,best_exit_pct",
            "signal_type": "eq.blocked",
            "outcome_5m_pct": "not.is.null",
            "order": "recorded_at.desc",
            "limit": "500",
        })

        if not blocked:
            return {"error": "차단 데이터 없음"}

        by_filter = defaultdict(lambda: {"total": 0, "saved": 0, "missed": 0, "outcomes": []})

        for sig in blocked:
            filt = sig.get("block_filter", "unknown")
            stats = by_filter[filt]
            stats["total"] += 1

            if sig.get("would_have_won_5m") is True:
                stats["missed"] += 1  # 필터가 수익 기회를 놓침
            elif sig.get("would_have_won_5m") is False:
                stats["saved"] += 1  # 필터가 손실 방어

            if sig.get("outcome_5m_pct") is not None:
                stats["outcomes"].append(sig["outcome_5m_pct"])

        result = {"filters": {}}
        for filt, stats in by_filter.items():
            evaluated = stats["saved"] + stats["missed"]
            result["filters"][filt] = {
                "total_blocked": stats["total"],
                "save_rate": round(stats["saved"] / max(evaluated, 1) * 100, 1),
                "missed_opportunities": stats["missed"],
                "avg_outcome_if_traded": round(sum(stats["outcomes"]) / max(len(stats["outcomes"]), 1), 3),
            }

        return result

    def analyze_hourly_pattern(self, params: dict) -> dict:
        """시간대별 패턴 분석"""
        # Phase 2에서 구현 (scalp_trade_log에서 시간대별 집계)
        return {"status": "not_implemented", "message": "Phase 2에서 구현 예정"}

    def analyze_strategy_comparison(self, params: dict) -> dict:
        """전략 비교 분석"""
        # Phase 2에서 구현
        return {"status": "not_implemented", "message": "Phase 2에서 구현 예정"}
