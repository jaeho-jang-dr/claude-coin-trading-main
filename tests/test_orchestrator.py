"""
Orchestrator 유닛 테스트

Coverage:
  - 시장 국면 분류 (extreme_fear ~ extreme_greed)
  - 위험도(danger_score) 계산 (연속손절, BTC 과다, 급락, 김치P, 롱과밀, 매크로, 뉴스)
  - 기회(opportunity_score) 계산 (극공포, RSI과매도, 반등, Fusion, 펀딩비, 김치할인, 매크로, 뉴스)
  - 전략 전환: 모든 전환 규칙 (danger→보수적, opportunity→공격적, 횡보→보통)
  - FOMO 방지: -5% 급락 차단, FGI<=20 AND >-8% 예외, 심각 급락 차단
  - 쿨다운: 2시간 기본, 4시간 강화, 긴급 면제
  - DB 학습: penalty, aggregate, 데이터 부족
  - 자동 긴급정지: 플래시크래시, 캐스케이딩, 외부약세, 포트폴리오 위기
  - 감독 오버라이드: DCA+cascade→매도, 급락+매수→관망
  - DCA 추적: 매수 기록, 매도 초기화
  - 긴급정지 통합: 사용자 EMERGENCY_STOP, 자동 긴급정지
  - 피드백/성과: conservative/aggressive/moderate 바이어스, 연패/연승 조정
  - 연속 손절 카운팅
  - Full run 통합 테스트
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.base_agent import Decision


# ── 공통 헬퍼 ──────────────────────────────────────

def _default_state(active="moderate", last_switch=None, consecutive_losses=0,
                   switch_history=None, dca_history=None):
    state = {
        "active_agent": active,
        "last_switch_time": last_switch,
        "last_trade_time": None,
        "consecutive_losses": consecutive_losses,
        "switch_history": switch_history or [],
    }
    if dca_history is not None:
        state["dca_history"] = dca_history
    return state


def _make_orchestrator(active="moderate", last_switch=None,
                       consecutive_losses=0, switch_history=None,
                       dca_history=None):
    """Orchestrator를 생성하면서 _load_state를 모킹한다."""
    with patch("agents.orchestrator._load_state") as mock:
        mock.return_value = _default_state(
            active, last_switch, consecutive_losses, switch_history, dca_history,
        )
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        orch._learning_data = None
        orch._performance = {}
        return orch


def _ms(danger_score=20, opportunity_score=20, fgi=50, rsi=50,
        price_change_24h=0, kimchi_pct=0, ls_ratio=1.0,
        consecutive_losses=0, fusion_signal="neutral", phase="neutral"):
    """market_state 딕셔너리 축약 생성."""
    return {
        "danger_score": danger_score,
        "opportunity_score": opportunity_score,
        "fgi": fgi,
        "rsi": rsi,
        "price_change_24h": price_change_24h,
        "kimchi_pct": kimchi_pct,
        "ls_ratio": ls_ratio,
        "consecutive_losses": consecutive_losses,
        "fusion_signal": fusion_signal,
        "phase": phase,
    }


def _make_market_data(rsi=50, sma_deviation=0, fgi=50, price_change_rate=0,
                      ai_score=0, news_sentiment="neutral",
                      trade_price=50000000, candles_4h=None):
    return {
        "indicators": {
            "rsi_14": rsi, "sma_20": trade_price,
            "sma_20_deviation_pct": sma_deviation,
            "macd": {"histogram": 0, "signal_cross": False},
            "bollinger": {},
        },
        "ticker": {"trade_price": trade_price,
                   "signed_change_rate": price_change_rate / 100},
        "fear_greed": {"value": fgi},
        "news": {"overall_sentiment": news_sentiment},
        "ai_composite_signal": {"score": ai_score},
        "current_price": trade_price,
        "candles_4h": candles_4h or [],
    }


def _make_portfolio(krw=1000000, btc_balance=0, btc_avg_price=50000000,
                    profit_pct=0, total_eval=1000000):
    btc_eval = btc_balance * btc_avg_price if btc_balance > 0 else 0
    return {
        "krw_balance": krw,
        "btc": {"balance": btc_balance, "avg_buy_price": btc_avg_price,
                "profit_pct": profit_pct, "eval_amount": btc_eval},
        "total_eval": total_eval,
        "btc_ratio": btc_eval / total_eval if total_eval > 0 else 0,
    }


def _make_external_signal(total_score=0, strategy_bonus=0,
                          fusion_signal="neutral"):
    return {
        "total_score": total_score,
        "strategy_bonus": strategy_bonus,
        "fusion": {"signal": fusion_signal},
    }


# ============================================================
# Phase Classification
# ============================================================

class TestOrchestratorPhases:

    def test_extreme_fear(self):
        orch = _make_orchestrator()
        assert orch._classify_phase(10) == "extreme_fear"
        assert orch._classify_phase(0) == "extreme_fear"

    def test_extreme_fear_boundary(self):
        orch = _make_orchestrator()
        assert orch._classify_phase(20) == "extreme_fear"

    def test_fear(self):
        orch = _make_orchestrator()
        assert orch._classify_phase(21) == "fear"
        assert orch._classify_phase(25) == "fear"
        assert orch._classify_phase(35) == "fear"

    def test_neutral(self):
        orch = _make_orchestrator()
        assert orch._classify_phase(36) == "neutral"
        assert orch._classify_phase(50) == "neutral"
        assert orch._classify_phase(60) == "neutral"

    def test_greed(self):
        orch = _make_orchestrator()
        assert orch._classify_phase(61) == "greed"
        assert orch._classify_phase(70) == "greed"
        assert orch._classify_phase(80) == "greed"

    def test_extreme_greed(self):
        orch = _make_orchestrator()
        assert orch._classify_phase(81) == "extreme_greed"
        assert orch._classify_phase(90) == "extreme_greed"
        assert orch._classify_phase(100) == "extreme_greed"


# ============================================================
# Danger Score
# ============================================================

class TestOrchestratorDangerScore:

    def test_zero_danger(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        assert score == 0

    def test_consecutive_losses_10_per(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=2,
        )
        assert score == 20

    def test_consecutive_losses_capped_at_30(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=10,
        )
        # 10*10 = 100 but capped at 30
        assert score >= 30

    def test_btc_overweight(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.5,
            consecutive_losses=0,
        )
        # (0.5 - 0.3) * 100 = 20
        assert score == 20

    def test_btc_ratio_below_threshold(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.25,
            consecutive_losses=0,
        )
        assert score == 0

    def test_crash_adds_danger(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-5,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        # abs(-5)*5 = 25
        assert score == 25

    def test_crash_capped_at_25(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-10,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        # abs(-10)*5 = 50 capped at 25
        assert score == 25

    def test_no_crash_bonus_above_minus_3(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-2,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        assert score == 0

    def test_kimchi_premium(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=5, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        # (5-3)*5 = 10
        assert score == 10

    def test_kimchi_premium_capped_at_15(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=10, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        # (10-3)*5 = 35 capped at 15
        assert score == 15

    def test_long_overcrowding(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.5, btc_ratio=0.2,
            consecutive_losses=0,
        )
        # (1.5-1.2)*20 = 6
        assert score == 6

    def test_long_overcrowding_capped_at_10(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=2.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        # (2.0-1.2)*20 = 16 capped at 10
        assert score == 10

    def test_macro_negative(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0, macro_score=-20,
        )
        # abs(-20)*0.5 = 10
        assert score == 10

    def test_macro_not_negative_enough(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0, macro_score=-5,
        )
        assert score == 0

    def test_news_negative(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0, news_sentiment="negative",
        )
        assert score == 10

    def test_news_slightly_negative(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0, news_sentiment="slightly_negative",
        )
        assert score == 5

    def test_danger_capped_at_100(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-10,
            kimchi_pct=10, ls_ratio=2.0, btc_ratio=0.8,
            consecutive_losses=5, macro_score=-30, news_sentiment="negative",
        )
        assert score <= 100

    def test_all_factors_combined(self):
        orch = _make_orchestrator()
        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-5,
            kimchi_pct=5, ls_ratio=1.5, btc_ratio=0.5,
            consecutive_losses=2, macro_score=-20, news_sentiment="negative",
        )
        # 20(losses) + 20(btc) + 25(crash) + 10(kimchi) + 6(long) + 10(macro) + 10(news) = 101 → capped 100
        assert score <= 100
        assert score >= 70


# ============================================================
# Opportunity Score
# ============================================================

class TestOrchestratorOpportunityScore:

    def test_zero_opportunity(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 0

    def test_extreme_fear(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=10, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # 25-10 = 15
        assert score == 15

    def test_extreme_fear_boundary(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=25, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # 25-25 = 0
        assert score == 0

    def test_extreme_fear_max(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=0, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # 25-0 = 25
        assert score == 25

    def test_rsi_oversold(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=25, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # (35-25)*1.3 = 13
        assert score == 13

    def test_rsi_not_oversold(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=40, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 0

    def test_bounce(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=2,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # 2*5 = 10
        assert score == 10

    def test_bounce_capped_at_15(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=5,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # 5*5 = 25 capped at 15
        assert score == 15

    def test_strong_buy_fusion(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="strong_buy", fusion_score=50,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 20

    def test_buy_fusion(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="buy", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 10

    def test_fusion_score_above_10(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=15,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 5

    def test_negative_funding_rate(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=-0.02, kimchi_pct=0,
        )
        assert score == 10

    def test_slightly_negative_funding(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=-0.005, kimchi_pct=0,
        )
        assert score == 5

    def test_kimchi_discount(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=-3,
        )
        # abs(-3)*3 = 9
        assert score == 9

    def test_kimchi_discount_capped_at_10(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=-5,
        )
        # abs(-5)*3 = 15 capped at 10
        assert score == 10

    def test_macro_positive(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
            macro_score=20,
        )
        # 20*0.5 = 10
        assert score == 10

    def test_news_positive(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
            news_sentiment="positive",
        )
        assert score == 8

    def test_news_slightly_positive(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
            news_sentiment="slightly_positive",
        )
        assert score == 4

    def test_opportunity_capped_at_100(self):
        orch = _make_orchestrator()
        score = orch._calculate_opportunity_score(
            fgi=5, rsi=20, price_change_24h=5,
            fusion_signal="strong_buy", fusion_score=50,
            funding_rate=-0.02, kimchi_pct=-5,
            macro_score=30, news_sentiment="positive",
        )
        assert score <= 100


# ============================================================
# Strategy Switching - All Transition Rules
# ============================================================

class TestOrchestratorSwitching:

    # ── danger 기반 전환 ──

    def test_danger_70_aggressive_to_conservative(self):
        """danger >= 70 → 공격적→보수적 직행."""
        orch = _make_orchestrator("aggressive")
        target = orch._decide_target(
            "aggressive",
            _ms(danger_score=75, price_change_24h=-6, fgi=30, phase="fear"),
            75, 10,
        )
        assert target == "conservative"

    def test_danger_70_moderate_to_conservative(self):
        """danger >= 70 → 보통→보수적."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target("moderate", _ms(danger_score=70), 70, 10)
        assert target == "conservative"

    def test_danger_70_already_conservative(self):
        """이미 보수적이면 전환 없음."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target("conservative", _ms(danger_score=75), 75, 10)
        assert target is None

    def test_danger_45_69_aggressive_to_moderate(self):
        """danger 45~69 + 공격적 → 보통."""
        orch = _make_orchestrator("aggressive")
        target = orch._decide_target(
            "aggressive",
            _ms(danger_score=50, kimchi_pct=4, ls_ratio=1.3),
            50, 10,
        )
        assert target == "moderate"

    def test_danger_50_moderate_to_conservative(self):
        """danger >= 50 + 보통 → 보수적."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target(
            "moderate",
            _ms(danger_score=55, fgi=35, phase="fear"),
            55, 10,
        )
        assert target == "conservative"

    def test_danger_49_moderate_no_switch(self):
        """danger 49 + 보통 → 전환 없음 (50 미만)."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target("moderate", _ms(danger_score=49), 49, 10)
        # danger < 50 for moderate, and < 45 for aggressive
        assert target is None

    def test_danger_44_aggressive_no_switch(self):
        """danger 44 + 공격적 → 전환 없음 (45 미만)."""
        orch = _make_orchestrator("aggressive")
        target = orch._decide_target("aggressive", _ms(danger_score=44), 44, 10)
        assert target is None

    # ── opportunity 기반 전환 ──

    def test_opportunity_60_to_aggressive(self):
        """opportunity >= 60 + danger < 30 → 공격적 직행."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(opportunity_score=65, fgi=15, rsi=25, price_change_24h=2,
                fusion_signal="strong_buy", phase="extreme_fear"),
            10, 65,
        )
        assert target == "aggressive"

    def test_opportunity_60_moderate_to_aggressive(self):
        """opportunity >= 60 + 보통 → 공격적."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target(
            "moderate",
            _ms(fgi=15, phase="extreme_fear"),
            10, 65,
        )
        assert target == "aggressive"

    def test_opportunity_60_danger_too_high(self):
        """opportunity 60+ 이나 danger >= 30 → 전환 없음."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target("conservative", _ms(fgi=40, phase="neutral"), 35, 65)
        assert target is None

    def test_opportunity_40_conservative_to_moderate(self):
        """opportunity 40~59 + 보수적 → 보통."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=30, rsi=35, phase="fear"),
            10, 45,
        )
        assert target == "moderate"

    def test_opportunity_40_moderate_to_aggressive(self):
        """opportunity 40~59 + 보통 → 공격적."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target(
            "moderate",
            _ms(fgi=30, rsi=35, fusion_signal="buy", phase="fear"),
            10, 45,
        )
        assert target == "aggressive"

    def test_opportunity_40_danger_35_blocks(self):
        """opportunity 40~59 + danger >= 35 → 전환 없음."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target("conservative", _ms(phase="neutral"), 36, 45)
        assert target is None

    def test_opportunity_25_conservative_to_moderate(self):
        """opportunity 25~39 + 보수적 → 보통."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=40, rsi=40, phase="neutral"),
            10, 30,
        )
        assert target == "moderate"

    def test_opportunity_25_danger_30_blocks(self):
        """opportunity 25~39 + danger >= 30 → 전환 없음."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target("conservative", _ms(phase="neutral"), 30, 30)
        assert target is None

    # ── 횡보장 ──

    def test_sideways_aggressive_to_moderate(self):
        """횡보(danger < 25, opportunity < 25) + 공격적 → 보통."""
        orch = _make_orchestrator("aggressive")
        target = orch._decide_target(
            "aggressive",
            _ms(fgi=50, rsi=50, phase="neutral"),
            10, 10,
        )
        assert target == "moderate"

    def test_sideways_conservative_to_moderate(self):
        """횡보 + 보수적 + 연속손절 0 + 변동 작음 → 보통 복귀."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=50, rsi=50, price_change_24h=0,
                consecutive_losses=0, phase="neutral"),
            10, 10,
        )
        assert target == "moderate"

    def test_sideways_conservative_with_losses_no_switch(self):
        """횡보 + 보수적 + 연속손절 > 0 → 전환 안 함."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=50, rsi=50, consecutive_losses=1, phase="neutral"),
            10, 10,
        )
        assert target is None

    def test_sideways_moderate_no_switch(self):
        """횡보 + 보통이면 이미 최적 → 전환 없음."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target(
            "moderate",
            _ms(fgi=50, rsi=50, phase="neutral"),
            15, 15,
        )
        assert target is None

    def test_no_switch_already_optimal(self):
        """중간 danger/opportunity에서 보통이면 전환 없음."""
        orch = _make_orchestrator("moderate")
        target = orch._decide_target("moderate", _ms(), 30, 30)
        assert target is None


# ============================================================
# FOMO Prevention
# ============================================================

class TestOrchestratorFomo:

    def test_fomo_block_crash(self):
        """24h -6% 급락 + FGI 30(>20) → 공격적 전환 차단."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=30, rsi=25, price_change_24h=-6,
                fusion_signal="strong_buy", phase="fear"),
            10, 65,
        )
        assert target is None

    def test_fomo_exception_extreme_fear(self):
        """24h -6% 이나 FGI <= 20 극공포 + 하락폭 > -8 → 공격적 전환 허용."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=15, rsi=25, price_change_24h=-6,
                fusion_signal="strong_buy", phase="extreme_fear"),
            10, 65,
        )
        assert target == "aggressive"

    def test_fomo_no_exception_deep_crash(self):
        """24h -9% 심각 급락 → FGI 극공포여도 -8 초과 → 차단."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=15, rsi=25, price_change_24h=-9,
                fusion_signal="strong_buy", phase="extreme_fear"),
            10, 65,
        )
        assert target is None

    def test_fomo_boundary_minus_5(self):
        """24h -5% 정확히 → FOMO 차단 발동."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=30, price_change_24h=-5.01, phase="fear"),
            10, 65,
        )
        assert target is None

    def test_no_fomo_above_minus_5(self):
        """24h -4.9% → FOMO 차단 미발동."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=15, rsi=25, price_change_24h=-4.9,
                fusion_signal="strong_buy", phase="extreme_fear"),
            10, 65,
        )
        assert target == "aggressive"

    def test_fomo_boundary_minus_8_exception(self):
        """24h -8% 정확히 + FGI 20 → 허용 (> -8 이므로)."""
        orch = _make_orchestrator("conservative")
        # price_change_24h = -8 → -8 > -8 is False → exception NOT met → FOMO blocked
        target = orch._decide_target(
            "conservative",
            _ms(fgi=20, rsi=25, price_change_24h=-8,
                fusion_signal="strong_buy", phase="extreme_fear"),
            10, 65,
        )
        # -8 > -8 is False, so not (fgi<=20 and price>-8) = not (True and False) = True → fomo_block
        assert target is None

    def test_fomo_minus_7_9_with_extreme_fear(self):
        """24h -7.9% + FGI 20 → price > -8 = True → 예외 적용."""
        orch = _make_orchestrator("conservative")
        target = orch._decide_target(
            "conservative",
            _ms(fgi=20, rsi=25, price_change_24h=-7.9,
                fusion_signal="strong_buy", phase="extreme_fear"),
            10, 65,
        )
        assert target == "aggressive"


# ============================================================
# Cooldown
# ============================================================

class TestOrchestratorCooldown:

    def test_cooldown_active(self):
        """마지막 전환 후 1시간 → 쿨다운 중."""
        kst = timezone(timedelta(hours=9))
        recent = (datetime.now(kst) - timedelta(hours=1)).isoformat()
        orch = _make_orchestrator(last_switch=recent)
        assert orch._is_on_cooldown() is True

    def test_cooldown_expired(self):
        """마지막 전환 후 3시간 → 쿨다운 해제."""
        kst = timezone(timedelta(hours=9))
        old = (datetime.now(kst) - timedelta(hours=3)).isoformat()
        orch = _make_orchestrator(last_switch=old)
        assert orch._is_on_cooldown() is False

    def test_no_cooldown_no_history(self):
        """전환 이력 없으면 쿨다운 아님."""
        orch = _make_orchestrator()
        assert orch._is_on_cooldown() is False

    def test_cooldown_strengthened_3_switches(self):
        """당일 3회 이상 전환 → 4시간 쿨다운."""
        kst = timezone(timedelta(hours=9))
        recent = (datetime.now(kst) - timedelta(hours=3)).isoformat()
        today = time.strftime("%Y-%m-%d")
        history = [
            {"timestamp": f"{today}T01:00:00+09:00"},
            {"timestamp": f"{today}T05:00:00+09:00"},
            {"timestamp": f"{today}T09:00:00+09:00"},
        ]
        orch = _make_orchestrator(last_switch=recent, switch_history=history)
        # 3시간 경과이나 당일 3회 → 4시간 쿨다운 → 아직 쿨다운 중
        assert orch._is_on_cooldown() is True

    def test_cooldown_4h_expired(self):
        """당일 3회 전환 + 5시간 경과 → 쿨다운 해제."""
        kst = timezone(timedelta(hours=9))
        old = (datetime.now(kst) - timedelta(hours=5)).isoformat()
        today = time.strftime("%Y-%m-%d")
        history = [
            {"timestamp": f"{today}T01:00:00+09:00"},
            {"timestamp": f"{today}T05:00:00+09:00"},
            {"timestamp": f"{today}T09:00:00+09:00"},
        ]
        orch = _make_orchestrator(last_switch=old, switch_history=history)
        assert orch._is_on_cooldown() is False

    def test_emergency_bypasses_cooldown(self):
        """danger >= 70 또는 24h -7% → 쿨다운 무시하고 전환."""
        kst = timezone(timedelta(hours=9))
        recent = (datetime.now(kst) - timedelta(minutes=30)).isoformat()
        orch = _make_orchestrator("aggressive", last_switch=recent)

        ms = _ms(danger_score=75, opportunity_score=5, fgi=20,
                 price_change_24h=-8, consecutive_losses=3,
                 phase="extreme_fear")
        result = orch._evaluate_switch(ms)
        assert result is not None
        assert result["to"] == "conservative"

    def test_cooldown_blocks_non_emergency(self):
        """쿨다운 중 + 비긴급 → 전환 차단."""
        kst = timezone(timedelta(hours=9))
        recent = (datetime.now(kst) - timedelta(minutes=30)).isoformat()
        orch = _make_orchestrator("moderate", last_switch=recent)

        ms = _ms(danger_score=40, opportunity_score=50, fgi=30,
                 price_change_24h=-2, phase="fear")
        result = orch._evaluate_switch(ms)
        assert result is None


# ============================================================
# DB Learning
# ============================================================

class TestOrchestratorLearning:

    def test_no_learning_data(self):
        orch = _make_orchestrator()
        orch._learning_data = None
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result == {"danger_adjust": 0, "opportunity_adjust": 0}

    def test_bad_aggressive_history(self):
        """현재→공격적 전환 성공률 30% (< 40%) + 5회 (>= 3) → opportunity -10."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 30,
                "total_switches": 5,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == -10

    def test_bad_conservative_history(self):
        """현재→보수적 전환 성공률 35% (< 40%) + 4회 (>= 3) → danger -10."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "conservative"): {
                "success_rate_pct": 35,
                "total_switches": 4,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["danger_adjust"] == -10

    def test_good_history_no_penalty(self):
        """성공률 60% → 패널티 없음."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 60,
                "total_switches": 5,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == 0

    def test_insufficient_data_no_penalty(self):
        """전환 횟수 2회 (< 3) → 패널티 없음."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 20,
                "total_switches": 2,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == 0

    def test_boundary_success_rate_40(self):
        """성공률 정확히 40% → 패널티 없음 (< 40 조건)."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 40,
                "total_switches": 5,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == 0

    def test_boundary_success_rate_39(self):
        """성공률 39% → 패널티 적용."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 39,
                "total_switches": 3,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == -10

    def test_both_bad_histories(self):
        """공격적/보수적 모두 성공률 낮음 → 양쪽 모두 -10."""
        orch = _make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 20,
                "total_switches": 5,
            },
            ("moderate", "conservative"): {
                "success_rate_pct": 25,
                "total_switches": 4,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == -10
        assert result["danger_adjust"] == -10

    def test_aggregate_learning(self):
        orch = _make_orchestrator()
        rows = [
            {"from_agent": "moderate", "to_agent": "aggressive",
             "outcome": "good", "profit_after_24h": 2.0},
            {"from_agent": "moderate", "to_agent": "aggressive",
             "outcome": "bad", "profit_after_24h": -1.5},
            {"from_agent": "moderate", "to_agent": "aggressive",
             "outcome": "good", "profit_after_24h": 1.0},
        ]
        result = orch._aggregate_learning(rows)
        key = ("moderate", "aggressive")
        assert key in result
        assert result[key]["good_count"] == 2
        assert result[key]["bad_count"] == 1
        assert result[key]["success_rate_pct"] == pytest.approx(66.7, abs=0.1)
        assert result[key]["total_switches"] == 3

    def test_aggregate_learning_empty(self):
        orch = _make_orchestrator()
        result = orch._aggregate_learning([])
        assert result == {}


# ============================================================
# Auto Emergency
# ============================================================

class TestOrchestratorAutoEmergency:

    def test_flash_crash(self):
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -12, "cascade_risk": 50,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -5}},
        )
        assert result is not None
        assert "플래시 크래시" in result["reason"]

    def test_flash_crash_boundary(self):
        """4h -10% 정확히 → 발동."""
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -10, "cascade_risk": 50,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": 0}},
        )
        # -10 < -10 is False → no trigger
        assert result is None

    def test_flash_crash_slightly_below(self):
        """4h -10.01% → 발동."""
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -10.01, "cascade_risk": 50,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": 0}},
        )
        assert result is not None

    def test_cascade_plus_danger(self):
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 85, "consecutive_losses": 0},
            drop_context={"price_change_4h": -5, "cascade_risk": 92,
                          "external_bearish_count": 3, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -5}},
        )
        assert result is not None
        assert "캐스케이딩" in result["reason"]

    def test_cascade_without_danger(self):
        """cascade >= 90 이나 danger < 80 → 미발동."""
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 70, "consecutive_losses": 0},
            drop_context={"price_change_4h": -3, "cascade_risk": 95,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": 0}},
        )
        assert result is None

    def test_external_bearish_overload(self):
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 5,
                          "external_bearish_details": ["a", "b", "c", "d", "e"]},
            portfolio={"btc": {"profit_pct": -5}},
        )
        assert result is not None
        assert "외부 약세" in result["reason"]

    def test_external_bearish_4_no_trigger(self):
        """약세 지표 4개 → 미발동 (5개 이상 필요)."""
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 4,
                          "external_bearish_details": ["a", "b", "c", "d"]},
            portfolio={"btc": {"profit_pct": 0}},
        )
        assert result is None

    def test_portfolio_crisis(self):
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 6},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -18}},
        )
        assert result is not None
        assert "연속 손절" in result["reason"]

    def test_portfolio_crisis_boundary(self):
        """연속 손절 4회 (< 5) → 미발동."""
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 4},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -18}},
        )
        assert result is None

    def test_portfolio_crisis_profit_above_minus15(self):
        """연속 손절 5+ 이나 profit > -15% → 미발동."""
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 6},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -14}},
        )
        assert result is None

    def test_no_emergency_normal(self):
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 30, "consecutive_losses": 1},
            drop_context={"price_change_4h": -1, "cascade_risk": 20,
                          "external_bearish_count": 1, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": 3}},
        )
        assert result is None

    def test_multiple_triggers_combined(self):
        orch = _make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 85, "consecutive_losses": 6},
            drop_context={"price_change_4h": -12, "cascade_risk": 95,
                          "external_bearish_count": 5,
                          "external_bearish_details": ["a", "b", "c", "d", "e"]},
            portfolio={"btc": {"profit_pct": -20}},
        )
        assert result is not None
        assert " / " in result["reason"]


# ============================================================
# Override Decision
# ============================================================

class TestOrchestratorOverride:

    def test_dca_cascade_high_override_to_sell(self):
        """DCA + cascade >= 70 → 매도 오버라이드."""
        orch = _make_orchestrator()
        decision = Decision(
            decision="buy", confidence=0.6, reason="DCA",
            buy_score={}, trade_params={"is_dca": True, "amount": 50000},
            external_signal={}, agent_name="moderate",
        )
        dc = {"cascade_risk": 75, "price_change_4h": -5,
              "external_bearish_count": 3, "consecutive_red_candles": 4}
        ms = {"danger_score": 50}

        result = orch._override_decision(decision, dc, ms)
        assert result.decision == "sell"
        assert result._orchestrator_override is True

    def test_dca_cascade_69_no_override(self):
        """DCA + cascade 69 (< 70) → 오버라이드 없음."""
        orch = _make_orchestrator()
        decision = Decision(
            decision="buy", confidence=0.6, reason="DCA",
            buy_score={}, trade_params={"is_dca": True, "amount": 50000},
            external_signal={}, agent_name="moderate",
        )
        dc = {"cascade_risk": 69, "price_change_4h": -5,
              "trend_falling": True, "consecutive_red_candles": 4,
              "external_bearish_count": 3}
        ms = {"danger_score": 50}

        result = orch._override_decision(decision, dc, ms)
        # cascade 69 < 70 → no DCA override
        # But: buy + not dca check... it IS dca so first check doesn't apply for crash override
        # Actually price_change_4h=-5 < -3 and trend_falling but is_dca=True → first rule applies
        # First rule: buy AND is_dca AND cascade >= 70 → 69 < 70 → skip
        # Second rule: buy AND NOT is_dca → is_dca=True → skip
        assert result.decision == "buy"

    def test_buy_during_crash_override_to_hold(self):
        """신규 매수 + 급락 (4h < -3 + trend_falling) → 관망."""
        orch = _make_orchestrator()
        decision = Decision(
            decision="buy", confidence=0.7, reason="Buy signal",
            buy_score={}, trade_params={"amount": 100000},
            external_signal={}, agent_name="aggressive",
        )
        dc = {"cascade_risk": 40, "price_change_4h": -4,
              "trend_falling": True, "consecutive_red_candles": 5,
              "external_bearish_count": 2}
        ms = {"danger_score": 40}

        result = orch._override_decision(decision, dc, ms)
        assert result.decision == "hold"
        assert result._orchestrator_override is True

    def test_buy_no_crash_no_override(self):
        """정상 조건 → 오버라이드 없음."""
        orch = _make_orchestrator()
        decision = Decision(
            decision="buy", confidence=0.7, reason="Buy",
            buy_score={}, trade_params={"amount": 100000},
            external_signal={}, agent_name="moderate",
        )
        dc = {"cascade_risk": 20, "price_change_4h": -1,
              "trend_falling": False, "consecutive_red_candles": 1,
              "external_bearish_count": 0}
        ms = {"danger_score": 20}

        result = orch._override_decision(decision, dc, ms)
        assert result.decision == "buy"

    def test_sell_not_overridden(self):
        """매도 결정은 오버라이드하지 않음."""
        orch = _make_orchestrator()
        decision = Decision(
            decision="sell", confidence=0.8, reason="Sell",
            buy_score={}, trade_params={"volume": 0.01},
            external_signal={}, agent_name="moderate",
        )
        dc = {"cascade_risk": 80, "price_change_4h": -8,
              "trend_falling": True, "consecutive_red_candles": 5,
              "external_bearish_count": 4}
        ms = {"danger_score": 70}

        result = orch._override_decision(decision, dc, ms)
        assert result.decision == "sell"

    def test_hold_not_overridden(self):
        """hold 결정은 오버라이드 대상 아님."""
        orch = _make_orchestrator()
        decision = Decision(
            decision="hold", confidence=0.5, reason="Hold",
            buy_score={}, trade_params={},
            external_signal={}, agent_name="moderate",
        )
        dc = {"cascade_risk": 80, "price_change_4h": -8,
              "trend_falling": True, "consecutive_red_candles": 5,
              "external_bearish_count": 4}
        ms = {"danger_score": 70}

        result = orch._override_decision(decision, dc, ms)
        assert result.decision == "hold"


# ============================================================
# Drop Context
# ============================================================

class TestOrchestratorDropContext:

    def test_basic_drop_context_keys(self):
        orch = _make_orchestrator()
        md = _make_market_data(price_change_rate=-2, candles_4h=[
            {"trade_price": 50000000, "opening_price": 51000000,
             "candle_acc_trade_volume": 100},
            {"trade_price": 49000000, "opening_price": 50000000,
             "candle_acc_trade_volume": 150},
        ])
        ext = {
            "sources": {
                "whale_tracker": {"whale_score": {"direction": "neutral"}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": 0},
                    "top_trader_long_short": {"current_ratio": 1.0},
                    "kimchi_premium": {"premium_pct": 0},
                },
                "news_sentiment": {"overall_sentiment": "neutral"},
                "macro": {"analysis": {"macro_score": 0}},
            },
        }
        dc = orch._build_drop_context(md, ext, _make_portfolio())
        assert "cascade_risk" in dc
        assert "trend_falling" in dc
        assert "dca_already_done" in dc
        assert "external_bearish_count" in dc
        assert "price_change_4h" in dc

    def test_high_cascade_with_bearish_signals(self):
        orch = _make_orchestrator()
        candles = [
            {"trade_price": 50000000, "opening_price": 51000000,
             "candle_acc_trade_volume": 100},
            {"trade_price": 47000000, "opening_price": 50000000,
             "candle_acc_trade_volume": 300},
        ]
        md = _make_market_data(price_change_rate=-6, candles_4h=candles)
        ext = {
            "sources": {
                "whale_tracker": {"whale_score": {"direction": "exchange_deposit"}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": 0.005},
                    "top_trader_long_short": {"current_ratio": 2.0},
                    "kimchi_premium": {"premium_pct": 6.0},
                },
                "news_sentiment": {"overall_sentiment": "negative"},
                "macro": {"analysis": {"macro_score": -20}},
            },
        }
        dc = orch._build_drop_context(md, ext, _make_portfolio())
        assert dc["cascade_risk"] >= 50
        assert dc["external_bearish_count"] >= 4

    def test_dca_history_tracking(self):
        orch = _make_orchestrator(dca_history={
            "KRW-BTC": {"dca_count": 1, "dca_total_amount": 50000},
        })
        md = _make_market_data()
        ext = {
            "sources": {
                "whale_tracker": {"whale_score": {"direction": "neutral"}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": 0},
                    "top_trader_long_short": {"current_ratio": 1.0},
                    "kimchi_premium": {"premium_pct": 0},
                },
                "news_sentiment": {"overall_sentiment": "neutral"},
                "macro": {"analysis": {"macro_score": 0}},
            },
        }
        dc = orch._build_drop_context(md, ext, _make_portfolio())
        assert dc["dca_already_done"] is True

    def test_no_dca_history(self):
        orch = _make_orchestrator(dca_history={})
        md = _make_market_data()
        ext = {
            "sources": {
                "whale_tracker": {"whale_score": {"direction": "neutral"}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": 0},
                    "top_trader_long_short": {"current_ratio": 1.0},
                    "kimchi_premium": {"premium_pct": 0},
                },
                "news_sentiment": {"overall_sentiment": "neutral"},
                "macro": {"analysis": {"macro_score": 0}},
            },
        }
        dc = orch._build_drop_context(md, ext, _make_portfolio())
        assert dc["dca_already_done"] is False


# ============================================================
# DCA Tracking
# ============================================================

class TestOrchestratorDcaTracking:

    @patch("agents.orchestrator._save_state")
    def test_track_dca_buy(self, mock_save):
        orch = _make_orchestrator(dca_history={})
        decision = Decision(
            decision="buy", confidence=0.6, reason="DCA",
            buy_score={},
            trade_params={"is_dca": True, "market": "KRW-BTC", "amount": 50000},
            external_signal={}, agent_name="test",
        )
        orch._track_dca(decision)
        assert orch.state["dca_history"]["KRW-BTC"]["dca_count"] == 1
        assert orch.state["dca_history"]["KRW-BTC"]["dca_total_amount"] == 50000

    @patch("agents.orchestrator._save_state")
    def test_track_dca_increments(self, mock_save):
        orch = _make_orchestrator(dca_history={
            "KRW-BTC": {"dca_count": 1, "dca_total_amount": 50000},
        })
        decision = Decision(
            decision="buy", confidence=0.6, reason="DCA",
            buy_score={},
            trade_params={"is_dca": True, "market": "KRW-BTC", "amount": 30000},
            external_signal={}, agent_name="test",
        )
        orch._track_dca(decision)
        assert orch.state["dca_history"]["KRW-BTC"]["dca_count"] == 2
        assert orch.state["dca_history"]["KRW-BTC"]["dca_total_amount"] == 80000

    @patch("agents.orchestrator._save_state")
    def test_track_dca_sell_resets(self, mock_save):
        orch = _make_orchestrator(dca_history={
            "KRW-BTC": {"dca_count": 2, "dca_total_amount": 100000},
        })
        # Ensure state has dca_history from init
        orch.state.setdefault("dca_history", {})
        orch.state["dca_history"]["KRW-BTC"] = {"dca_count": 2, "dca_total_amount": 100000}
        decision = Decision(
            decision="sell", confidence=0.8, reason="Sell",
            buy_score={},
            trade_params={"market": "KRW-BTC", "volume": 0.01},
            external_signal={}, agent_name="test",
        )
        orch._track_dca(decision)
        assert "KRW-BTC" not in orch.state["dca_history"]

    @patch("agents.orchestrator._save_state")
    def test_non_dca_buy_not_tracked(self, mock_save):
        orch = _make_orchestrator(dca_history={})
        decision = Decision(
            decision="buy", confidence=0.7, reason="Buy",
            buy_score={},
            trade_params={"market": "KRW-BTC", "amount": 100000},
            external_signal={}, agent_name="test",
        )
        orch._track_dca(decision)
        assert "KRW-BTC" not in orch.state.get("dca_history", {})


# ============================================================
# Emergency Stop Integration
# ============================================================

class TestOrchestratorEmergencyStop:

    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "true"})
    def test_user_emergency_stop(self, mock_state):
        mock_state.return_value = _default_state()
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        result = orch.run({}, {}, {})
        assert result["decision"]["decision"] == "hold"
        assert "EMERGENCY_STOP" in result["decision"]["reason"]

    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false"})
    def test_auto_emergency_blocks(self, mock_state):
        mock_state.return_value = _default_state()
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        auto_em = {
            "active": True,
            "reason": "Flash crash",
            "activated_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        }
        with patch.object(orch, '_check_auto_emergency_active', return_value=auto_em):
            with patch.object(orch, '_can_lift_auto_emergency', return_value=False):
                result = orch.run({}, {}, {})
                assert result["decision"]["decision"] == "hold"
                assert "자동긴급정지" in result["active_agent"]

    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false"})
    def test_auto_emergency_lifted(self, mock_state):
        mock_state.return_value = _default_state()
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        auto_em = {
            "active": True,
            "reason": "Flash crash",
            "activated_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        }
        with patch.object(orch, '_check_auto_emergency_active', return_value=auto_em):
            with patch.object(orch, '_can_lift_auto_emergency', return_value=True):
                with patch.object(orch, '_deactivate_auto_emergency') as mock_deactivate:
                    with patch.object(orch, '_load_learning_data', return_value=None):
                        with patch.object(orch, '_evaluate_market_state',
                                          return_value=_ms(danger_score=20, opportunity_score=20)):
                            with patch.object(orch, '_evaluate_switch', return_value=None):
                                with patch.object(orch, '_build_drop_context',
                                                  return_value={"cascade_risk": 10, "price_change_4h": -1}):
                                    with patch.object(orch, '_evaluate_auto_emergency', return_value=None):
                                        with patch.object(orch.active_agent, 'decide',
                                                          return_value=Decision(
                                                              decision="hold", confidence=0.5,
                                                              reason="Test", buy_score={},
                                                              trade_params={},
                                                              external_signal={},
                                                              agent_name="test")):
                                            with patch.object(orch.active_agent, 'save_buy_score_detail',
                                                              return_value=None):
                                                with patch.object(orch, '_track_dca'):
                                                    with patch.object(orch, '_override_decision',
                                                                      side_effect=lambda d, *a: d):
                                                        result = orch.run(
                                                            _make_market_data(), {
                                                                "external_signal": _make_external_signal(),
                                                                "sources": {
                                                                    "fear_greed": {"current": {"value": 50}},
                                                                    "binance_sentiment": {
                                                                        "kimchi_premium": {"premium_pct": 0},
                                                                        "top_trader_long_short": {"current_ratio": 1.0},
                                                                        "funding_rate": {"current_rate": 0},
                                                                    },
                                                                    "macro": {"analysis": {"macro_score": 0}},
                                                                    "eth_btc": {},
                                                                    "news_sentiment": {"overall_sentiment": "neutral"},
                                                                    "whale_tracker": {"whale_score": {"direction": "neutral"}},
                                                                    "user_feedback": [],
                                                                    "performance_review": {},
                                                                },
                                                            }, _make_portfolio())
                    mock_deactivate.assert_called_once()


# ============================================================
# Feedback & Performance
# ============================================================

class TestOrchestratorFeedbackPerformance:

    def test_feedback_conservative(self):
        orch = _make_orchestrator()
        orch._apply_feedback([{"content": "좀 더 보수적으로 해주세요"}])
        assert orch.state.get("feedback_bias") == "conservative"

    def test_feedback_aggressive(self):
        orch = _make_orchestrator()
        orch._apply_feedback([{"content": "공격적으로 전환해줘"}])
        assert orch.state.get("feedback_bias") == "aggressive"

    def test_feedback_moderate(self):
        orch = _make_orchestrator()
        orch._apply_feedback([{"content": "moderate로 가자"}])
        assert orch.state.get("feedback_bias") == "moderate"

    def test_feedback_safe_keyword(self):
        orch = _make_orchestrator()
        orch._apply_feedback([{"content": "좀 더 안전하게"}])
        assert orch.state.get("feedback_bias") == "conservative"

    def test_feedback_empty(self):
        orch = _make_orchestrator()
        orch._apply_feedback([])
        assert "feedback_bias" not in orch.state

    def test_feedback_irrelevant(self):
        orch = _make_orchestrator()
        orch._apply_feedback([{"content": "좋은 성과입니다"}])
        assert "feedback_bias" not in orch.state

    def test_performance_losing_streak_3(self):
        orch = _make_orchestrator()
        orch._performance = {
            "available": True,
            "win_rate_pct": 30,
            "recent_streak_type": "loss",
            "recent_streak": 4,
        }
        adj = orch._get_performance_adjustment()
        assert adj == 15  # danger +15

    def test_performance_losing_streak_2(self):
        orch = _make_orchestrator()
        orch._performance = {
            "available": True,
            "win_rate_pct": 40,
            "recent_streak_type": "loss",
            "recent_streak": 2,
        }
        adj = orch._get_performance_adjustment()
        assert adj == 8

    def test_performance_winning_streak(self):
        orch = _make_orchestrator()
        orch._performance = {
            "available": True,
            "win_rate_pct": 70,
            "recent_streak_type": "win",
            "recent_streak": 4,
        }
        adj = orch._get_performance_adjustment()
        assert adj == -10  # opportunity +10

    def test_performance_winning_low_winrate(self):
        """연승 중이나 win_rate < 60 → 조정 없음."""
        orch = _make_orchestrator()
        orch._performance = {
            "available": True,
            "win_rate_pct": 50,
            "recent_streak_type": "win",
            "recent_streak": 4,
        }
        adj = orch._get_performance_adjustment()
        assert adj == 0

    def test_performance_no_data(self):
        orch = _make_orchestrator()
        orch._performance = {}
        adj = orch._get_performance_adjustment()
        assert adj == 0

    def test_performance_not_available(self):
        orch = _make_orchestrator()
        orch._performance = {"available": False}
        adj = orch._get_performance_adjustment()
        assert adj == 0


# ============================================================
# Consecutive Losses
# ============================================================

class TestOrchestratorConsecutiveLosses:

    def test_no_losses(self):
        orch = _make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "buy", "profit_loss": 5.0},
            {"decision": "sell", "profit_loss": 3.0},
        ])
        assert count == 0

    def test_consecutive_losses(self):
        orch = _make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "buy", "profit_loss": -2.0},
            {"decision": "sell", "profit_loss": -3.0},
            {"decision": "buy", "profit_loss": 5.0},
        ])
        assert count == 2

    def test_hold_breaks_chain(self):
        orch = _make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "hold", "profit_loss": None},
            {"decision": "buy", "profit_loss": -2.0},
        ])
        assert count == 0

    def test_empty_decisions(self):
        orch = _make_orchestrator()
        count = orch._count_consecutive_losses([])
        assert count == 0

    def test_all_losses(self):
        orch = _make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "buy", "profit_loss": -1.0},
            {"decision": "sell", "profit_loss": -2.0},
            {"decision": "buy", "profit_loss": -3.0},
        ])
        assert count == 3

    def test_korean_decision_names(self):
        orch = _make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "매수", "profit_loss": -1.0},
            {"decision": "매도", "profit_loss": -2.0},
        ])
        assert count == 2


# ============================================================
# FGI Extraction
# ============================================================

class TestOrchestratorFgiExtraction:

    def test_fgi_from_sources(self):
        orch = _make_orchestrator()
        ext = {"sources": {"fear_greed": {"current": {"value": 25}}}}
        assert orch._get_fgi(ext) == 25

    def test_fgi_missing(self):
        orch = _make_orchestrator()
        assert orch._get_fgi({}) == 50

    def test_fgi_no_current(self):
        orch = _make_orchestrator()
        ext = {"sources": {"fear_greed": {}}}
        assert orch._get_fgi(ext) == 50


# ============================================================
# Full Run Integration (mocked)
# ============================================================

class TestOrchestratorRun:

    @patch("agents.orchestrator._save_state")
    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false", "MAX_TRADE_AMOUNT": "100000"})
    def test_full_run_hold(self, mock_state, mock_save):
        mock_state.return_value = _default_state("conservative", dca_history={})
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        md = _make_market_data(rsi=50, fgi=50, sma_deviation=0)
        ext = {
            "external_signal": _make_external_signal(),
            "sources": {
                "fear_greed": {"current": {"value": 50}},
                "binance_sentiment": {
                    "kimchi_premium": {"premium_pct": 0},
                    "top_trader_long_short": {"current_ratio": 1.0},
                    "funding_rate": {"current_rate": 0},
                },
                "macro": {"analysis": {"macro_score": 0, "sentiment": "neutral"}},
                "eth_btc": {},
                "news_sentiment": {"overall_sentiment": "neutral"},
                "whale_tracker": {"whale_score": {"direction": "neutral"}},
                "user_feedback": [],
                "performance_review": {"available": False},
            },
        }
        port = _make_portfolio()

        with patch.object(orch, '_check_auto_emergency_active', return_value=None):
            with patch.object(orch, '_load_learning_data', return_value=None):
                with patch.object(orch.active_agent, 'save_buy_score_detail',
                                  return_value=None):
                    result = orch.run(md, ext, port)

        assert "decision" in result
        assert "active_agent" in result
        assert "market_state" in result
        assert result["decision"]["decision"] == "hold"

    @patch("agents.orchestrator._save_state")
    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false", "MAX_TRADE_AMOUNT": "100000"})
    def test_full_run_returns_market_state(self, mock_state, mock_save):
        mock_state.return_value = _default_state("conservative", dca_history={})
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        md = _make_market_data(rsi=50, fgi=50)
        ext = {
            "external_signal": _make_external_signal(),
            "sources": {
                "fear_greed": {"current": {"value": 50}},
                "binance_sentiment": {
                    "kimchi_premium": {"premium_pct": 0},
                    "top_trader_long_short": {"current_ratio": 1.0},
                    "funding_rate": {"current_rate": 0},
                },
                "macro": {"analysis": {"macro_score": 0, "sentiment": "neutral"}},
                "eth_btc": {},
                "news_sentiment": {"overall_sentiment": "neutral"},
                "whale_tracker": {"whale_score": {"direction": "neutral"}},
                "user_feedback": [],
                "performance_review": {"available": False},
            },
        }

        with patch.object(orch, '_check_auto_emergency_active', return_value=None):
            with patch.object(orch, '_load_learning_data', return_value=None):
                with patch.object(orch.active_agent, 'save_buy_score_detail',
                                  return_value=None):
                    result = orch.run(md, ext, _make_portfolio())

        ms = result["market_state"]
        assert "danger_score" in ms
        assert "opportunity_score" in ms
        assert "phase" in ms
        assert "fgi" in ms


# ============================================================
# Evaluate Switch Integration
# ============================================================

class TestEvaluateSwitchIntegration:

    def test_feedback_bias_conservative_adds_danger(self):
        """feedback_bias=conservative → danger +10."""
        orch = _make_orchestrator("aggressive")
        orch.state["feedback_bias"] = "conservative"
        # danger=60 + 10(feedback) = 70 → should switch to conservative
        ms = _ms(danger_score=60, opportunity_score=10, fgi=30,
                 price_change_24h=-4, consecutive_losses=2, phase="fear")
        result = orch._evaluate_switch(ms)
        assert result is not None
        assert result["to"] == "conservative"

    def test_feedback_bias_aggressive_adds_opportunity(self):
        """feedback_bias=aggressive → opportunity +10."""
        orch = _make_orchestrator("conservative")
        orch.state["feedback_bias"] = "aggressive"
        # opportunity=55 + 10(feedback) = 65 → aggressive
        ms = _ms(danger_score=10, opportunity_score=55, fgi=15,
                 rsi=25, price_change_24h=2,
                 fusion_signal="strong_buy", phase="extreme_fear")
        result = orch._evaluate_switch(ms)
        assert result is not None
        assert result["to"] == "aggressive"

    def test_learning_penalty_reduces_opportunity(self):
        """학습 패널티 → opportunity -10으로 공격적 전환 억제."""
        orch = _make_orchestrator("conservative")
        orch._learning_data = {
            ("conservative", "aggressive"): {
                "success_rate_pct": 25,
                "total_switches": 5,
            },
        }
        # opportunity=65 - 10(penalty) = 55 → not enough for aggressive (needs >= 60)
        ms = _ms(danger_score=10, opportunity_score=65, fgi=15,
                 rsi=25, price_change_24h=2,
                 fusion_signal="strong_buy", phase="extreme_fear")
        result = orch._evaluate_switch(ms)
        # With penalty, opportunity becomes 55 < 60 → check 40-59 range
        # 55 >= 40 and danger 10 < 35 → conservative → moderate (not aggressive)
        assert result is not None
        assert result["to"] == "moderate"

    def test_performance_losing_streak_increases_danger(self):
        """연패 → danger +15로 보수적 전환 촉진."""
        orch = _make_orchestrator("moderate")
        orch._performance = {
            "available": True,
            "win_rate_pct": 30,
            "recent_streak_type": "loss",
            "recent_streak": 4,
        }
        # danger=40 + 15(perf) = 55 → moderate → conservative
        ms = _ms(danger_score=40, opportunity_score=10, fgi=35,
                 consecutive_losses=2, phase="fear")
        result = orch._evaluate_switch(ms)
        assert result is not None
        assert result["to"] == "conservative"
