"""
BaseStrategyAgent 유닛 테스트

Coverage:
  - calculate_buy_score: 점수 계산, 부분 점수, 극단 보너스, 경계값
  - evaluate_sell: 하이브리드 손절, 캐스케이딩, DCA, 오버웨이트, 추세
  - _calculate_trade_amount: 안전장치, 주말 축소, 외부 보너스 면제
  - _extract_indicators: 지표 추출
  - Decision: to_dict 직렬화
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.base_agent import BaseStrategyAgent, Decision
from agents.conservative import ConservativeAgent
from agents.moderate import ModerateAgent
from agents.aggressive import AggressiveAgent


# ── Fixtures ──

@pytest.fixture
def conservative():
    return ConservativeAgent()


@pytest.fixture
def moderate():
    return ModerateAgent()


@pytest.fixture
def aggressive():
    return AggressiveAgent()


def _make_market_data(
    rsi=50, sma_deviation=0, fgi=50, price_change_rate=0,
    ai_score=0, news_sentiment="neutral", trade_price=50000000,
    macd_histogram=0, signal_cross=False, candles_4h=None,
):
    return {
        "indicators": {
            "rsi_14": rsi,
            "sma_20": trade_price,
            "sma_20_deviation_pct": sma_deviation,
            "macd": {"histogram": macd_histogram, "signal_cross": signal_cross},
            "bollinger": {},
        },
        "ticker": {
            "trade_price": trade_price,
            "signed_change_rate": price_change_rate / 100,
        },
        "fear_greed": {"value": fgi},
        "news": {"overall_sentiment": news_sentiment},
        "ai_composite_signal": {"score": ai_score},
        "current_price": trade_price,
        "candles_4h": candles_4h or [],
    }


# ============================================================
# calculate_buy_score
# ============================================================

class TestCalculateBuyScore:
    """점수 계산 로직 검증."""

    def test_all_conditions_met_conservative(self, conservative):
        """FGI, RSI, SMA, 뉴스 모두 충족 시 최대 점수."""
        score = conservative.calculate_buy_score(
            fgi=15, rsi=25, sma_deviation=-6.0,
            news_negative=False, external_bonus=0,
        )
        # FGI 30+5(극단), RSI 25, SMA 25, News 20 = 105
        assert score["total"] >= 95
        assert score["result"] == "buy"

    def test_no_conditions_met(self, conservative):
        """아무 조건도 충족하지 않으면 점수 0."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["total"] == 0
        assert score["result"] == "hold"

    # ── FGI 점수 ──

    def test_fgi_extreme_bonus(self, conservative):
        """FGI가 임계의 50% 이하면 +5 보너스."""
        score = conservative.calculate_buy_score(
            fgi=10, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        # 10 <= 35*0.5=17.5 → 30+5=35
        assert score["fgi"]["score"] == 35

    def test_fgi_at_exactly_half_threshold(self, conservative):
        """FGI가 정확히 임계의 50%일 때 극단 보너스."""
        half = int(conservative.fgi_threshold * 0.5)  # 17
        score = conservative.calculate_buy_score(
            fgi=half, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["fgi"]["score"] == 35  # 30 + 5

    def test_fgi_just_above_half_threshold(self, conservative):
        """FGI가 임계의 50% 바로 위면 극단 보너스 없이 만점만."""
        half = int(conservative.fgi_threshold * 0.5)  # 17
        score = conservative.calculate_buy_score(
            fgi=half + 1, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        # 18 <= 35 → full points 30 (no extreme bonus since 18 > 17.5)
        assert score["fgi"]["score"] == 30

    def test_fgi_partial(self, conservative):
        """FGI가 임계값 초과 10 이내면 부분 점수 50%."""
        score = conservative.calculate_buy_score(
            fgi=41, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["fgi"]["score"] == int(conservative.fgi_points * 0.5)
        assert score["fgi"].get("partial") is True

    def test_fgi_at_exact_threshold(self, conservative):
        """FGI가 정확히 임계값일 때 만점."""
        score = conservative.calculate_buy_score(
            fgi=conservative.fgi_threshold, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["fgi"]["score"] == conservative.fgi_points

    def test_fgi_one_above_threshold(self, conservative):
        """FGI가 임계값 +1일 때 부분 점수."""
        score = conservative.calculate_buy_score(
            fgi=conservative.fgi_threshold + 1, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["fgi"].get("partial") is True

    def test_fgi_at_partial_boundary_plus_10(self, conservative):
        """FGI가 임계값 +10일 때 부분 점수 (경계값)."""
        score = conservative.calculate_buy_score(
            fgi=conservative.fgi_threshold + 10, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["fgi"]["score"] == int(conservative.fgi_points * 0.5)

    def test_fgi_beyond_partial_range(self, conservative):
        """FGI가 임계값 +11이면 점수 0."""
        score = conservative.calculate_buy_score(
            fgi=conservative.fgi_threshold + 11, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["fgi"]["score"] == 0

    # ── RSI 점수 ──

    def test_rsi_full_score(self, conservative):
        """RSI가 임계값 이하면 만점."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=conservative.rsi_threshold, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["rsi"]["score"] == conservative.rsi_points

    def test_rsi_partial(self, conservative):
        """RSI가 임계값 +5 이내면 부분 점수 15점."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=38, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["rsi"]["score"] == 15
        assert score["rsi"].get("partial") is True

    def test_rsi_at_partial_boundary(self, conservative):
        """RSI가 정확히 임계값 +5일 때 부분 점수."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=conservative.rsi_threshold + 5, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["rsi"]["score"] == 15

    def test_rsi_beyond_partial(self, conservative):
        """RSI가 임계값 +6이면 점수 0."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=conservative.rsi_threshold + 6, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["rsi"]["score"] == 0

    # ── SMA 점수 ──

    def test_sma_full_score(self, conservative):
        """SMA 이탈이 임계값 이하면 만점."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=-6.0,
            news_negative=True, external_bonus=0,
        )
        assert score["sma"]["score"] == 25

    def test_sma_at_exact_threshold(self, conservative):
        """SMA 이탈이 정확히 임계값일 때 만점."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=conservative.sma_deviation_pct,
            news_negative=True, external_bonus=0,
        )
        assert score["sma"]["score"] == conservative.sma_points

    def test_sma_above_threshold(self, conservative):
        """SMA 이탈이 임계값보다 크면 0점 (부분 점수 경로 주의)."""
        # sma_deviation_pct = -3.0, half = -1.5
        # deviation 0 > -1.5 → 0 points
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=0,
            news_negative=True, external_bonus=0,
        )
        assert score["sma"]["score"] == 0

    # ── 뉴스 점수 ──

    def test_news_positive_gets_points(self, conservative):
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=False, external_bonus=0,
        )
        assert score["news"]["score"] == 20

    def test_news_negative_zero(self, conservative):
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["news"]["score"] == 0

    # ── MACD 보너스 ──

    def test_macd_bonus_moderate_only(self, moderate):
        """Moderate만 MACD 보너스 10점."""
        score = moderate.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
            macd_golden_cross=True,
        )
        assert score["macd"]["score"] == 10

    def test_macd_bonus_not_conservative(self, conservative):
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
            macd_golden_cross=True,
        )
        assert "macd" not in score

    def test_macd_bonus_not_aggressive(self, aggressive):
        score = aggressive.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
            macd_golden_cross=True,
        )
        assert "macd" not in score

    # ── 외부 보너스 ──

    def test_external_bonus_positive(self, conservative):
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=15,
        )
        assert score["external"]["score"] == 15
        assert score["total"] == 15

    def test_external_bonus_negative(self, conservative):
        score = conservative.calculate_buy_score(
            fgi=25, rsi=25, sma_deviation=-6.0,
            news_negative=False, external_bonus=-15,
        )
        assert score["external"]["score"] == -15
        total_without = 30 + 25 + 25 + 20  # 100
        assert score["total"] == total_without - 15

    def test_external_bonus_zero(self, conservative):
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        assert score["external"]["score"] == 0

    # ── 임계 점수 경계 ──

    def test_threshold_boundary_buy(self, conservative):
        """정확히 임계 점수일 때 매수."""
        score = conservative.calculate_buy_score(
            fgi=15, rsi=25, sma_deviation=-6.0,
            news_negative=True, external_bonus=0,
        )
        assert score["total"] >= conservative.buy_score_threshold
        assert score["result"] == "buy"

    def test_threshold_boundary_hold(self, conservative):
        """임계 점수 미만이면 관망."""
        score = conservative.calculate_buy_score(
            fgi=41, rsi=38, sma_deviation=-1.0,
            news_negative=True, external_bonus=0,
        )
        assert score["total"] < conservative.buy_score_threshold
        assert score["result"] == "hold"


# ============================================================
# evaluate_sell
# ============================================================

class TestEvaluateSell:
    """하이브리드 손절 로직 테스트."""

    # ── 기본 매도 조건 ──

    def test_target_profit_sell(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=16.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "target_profit"

    def test_target_profit_deferred_by_ai(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=16.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=25,
        )
        assert result["action"] == "hold_defer"

    def test_target_profit_not_deferred_low_ai(self, conservative):
        """AI 시그널이 20 이하면 유예하지 않음."""
        result = conservative.evaluate_sell(
            profit_pct=16.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=20,
        )
        assert result["action"] == "sell"

    def test_fgi_overbought_sell(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=80, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["type"] == "fgi_overbought"

    def test_fgi_at_exact_sell_threshold(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=conservative.sell_fgi_threshold,
            current_rsi=50, buy_score={}, ai_signal_score=0,
        )
        assert result["type"] == "fgi_overbought"

    def test_fgi_below_sell_threshold(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=conservative.sell_fgi_threshold - 1,
            current_rsi=50, buy_score={}, ai_signal_score=0,
        )
        # RSI is below sell threshold too → no sell
        assert result is None

    def test_rsi_overbought_sell(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=50, current_rsi=75,
            buy_score={}, ai_signal_score=0,
        )
        assert result["type"] == "rsi_overbought"

    def test_forced_stop_loss(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=-11.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["type"] == "forced_stop"

    def test_forced_stop_at_exact_boundary(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=-10.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["type"] == "forced_stop"

    def test_no_sell_in_profit_no_triggers(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result is None

    def test_slightly_above_stop_no_sell(self, conservative):
        """손절선 바로 위(-4.9%)면 매도하지 않음."""
        result = conservative.evaluate_sell(
            profit_pct=-4.9, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result is None

    def test_at_exact_stop_loss(self, conservative):
        """정확히 손절선(-5.0%)."""
        buy_score = {"fgi": {"score": 0}, "rsi": {"score": 0},
                     "sma": {"score": 0}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-5.0, current_fgi=50, current_rsi=50,
            buy_score=buy_score, ai_signal_score=0,
        )
        assert result is not None
        assert result["action"] == "sell"

    # ── 하이브리드 손절 ──

    def test_hybrid_dca_with_bottom_signals(self, conservative):
        """손절선 + 바닥 시그널 3개 + AI 중립 → DCA."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
        )
        assert result["action"] == "dca"
        assert result["type"] == "hybrid_dca"

    def test_hybrid_sell_insufficient_signals(self, conservative):
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 0},
                     "sma": {"score": 0}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=50, current_rsi=50,
            buy_score=buy_score, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "stop_loss"

    def test_ai_extreme_negative_forces_sell(self, conservative):
        """바닥 시그널 3개이나 AI 극단 매도 압력(-20 이하) → 매도."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=-25,
        )
        assert result["type"] == "hybrid_forced"

    def test_ai_minus_20_boundary(self, conservative):
        """AI -20은 극단 매도가 아님 (< -20 이어야 함)."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=-20,
        )
        # -20 is NOT < -20, so DCA should proceed
        assert result["action"] == "dca"

    # ── 캐스케이딩 위험 ──

    def test_cascade_risk_high_forces_sell(self, conservative):
        """캐스케이딩 위험 70+ → 무조건 손절."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 20}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=10,
            drop_context={"cascade_risk": 75, "price_change_4h": -5.0,
                          "volume_ratio": 2.5, "external_bearish_count": 3},
        )
        assert result["type"] == "cascade_stop"

    def test_cascade_risk_at_boundary_70(self, conservative):
        """캐스케이딩 위험 정확히 70 → 손절."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 20}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=10,
            drop_context={"cascade_risk": 70, "price_change_4h": -5.0,
                          "volume_ratio": 2.5, "external_bearish_count": 3},
        )
        assert result["type"] == "cascade_stop"

    def test_dca_already_done_forces_sell(self, conservative):
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 20}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=10,
            drop_context={"dca_already_done": True, "cascade_risk": 0},
        )
        assert result["type"] == "dca_exhausted"

    def test_cascade_moderate_dca_with_strong_signals(self, conservative):
        """캐스케이딩 40-69 + 4개 바닥 시그널 → 신중 DCA."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 20}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 50, "external_bearish_count": 0},
        )
        assert result["type"] == "hybrid_dca_cautious"

    def test_cascade_moderate_sell_without_signals(self, conservative):
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 0},
                     "sma": {"score": 0}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=50, current_rsi=50,
            buy_score=buy_score, ai_signal_score=0,
            drop_context={"cascade_risk": 50, "external_bearish_count": 0},
        )
        assert result["type"] == "cascade_moderate_stop"

    def test_external_bearish_override(self, conservative):
        """외부 약세 3개+ 겹침 → 바닥 시그널 있어도 매도."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 20, "external_bearish_count": 3},
        )
        assert result["type"] == "external_bearish_override"

    def test_external_bearish_2_no_override(self, conservative):
        """외부 약세 2개 → 오버라이드 없이 DCA 가능."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 20, "external_bearish_count": 2},
        )
        assert result["action"] == "dca"

    # ── 하락 추세 ──

    def test_trend_falling_dca_with_strong_signals(self, conservative):
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 20}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 20, "external_bearish_count": 1,
                          "trend_falling": True},
        )
        assert result["type"] == "hybrid_dca_trend"

    def test_trend_falling_sell_with_weak_signals(self, conservative):
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=3,
            drop_context={"cascade_risk": 20, "external_bearish_count": 1,
                          "trend_falling": True},
        )
        assert result["type"] == "trend_stop"

    def test_trend_falling_dca_with_high_ai(self, conservative):
        """하락 추세 + 3개 시그널 + AI >= 15 → DCA 허용."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 0}}
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=15,
            drop_context={"cascade_risk": 20, "external_bearish_count": 1,
                          "trend_falling": True},
        )
        assert result["action"] == "dca"

    # ── 오버웨이트 분할 매도 ──

    def test_overweight_partial_sell(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=6.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
            btc_position_ratio=0.7,
        )
        assert result["action"] == "sell_partial"
        assert result["type"] == "overweight_rebalance"
        assert "sell_ratio" in result

    def test_overweight_no_trigger_low_profit(self, conservative):
        result = conservative.evaluate_sell(
            profit_pct=3.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
            btc_position_ratio=0.7,
        )
        assert result is None

    def test_overweight_not_exceeded_high_profit(self, conservative):
        """비중이 MAX_POSITION_RATIO 이하면 분할 매도 안 함."""
        result = conservative.evaluate_sell(
            profit_pct=6.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
            btc_position_ratio=0.3,
        )
        assert result is None

    # ── 매도 조건 우선순위 ──

    def test_overweight_takes_priority_over_target(self, conservative):
        """포지션 과다 분할 매도가 목표 수익 매도보다 우선."""
        result = conservative.evaluate_sell(
            profit_pct=16.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
            btc_position_ratio=0.7,
        )
        assert result["type"] == "overweight_rebalance"

    def test_forced_stop_priority_over_cascade(self, conservative):
        """강제 손절이 캐스케이딩보다 먼저 (코드 순서상)."""
        buy_score = {"fgi": {"score": 30}, "rsi": {"score": 25},
                     "sma": {"score": 25}, "news": {"score": 20}}
        result = conservative.evaluate_sell(
            profit_pct=-11.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=10,
            drop_context={"cascade_risk": 80, "dca_already_done": True},
        )
        assert result["type"] == "forced_stop"


# ============================================================
# _calculate_trade_amount
# ============================================================

class TestCalculateTradeAmount:

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "200000"})
    def test_capped_by_max(self, conservative):
        amount = conservative._calculate_trade_amount(5000000, external_bonus=0)
        assert amount <= 200000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_weekend_reduction(self, conservative):
        with patch.object(conservative, '_is_weekend', return_value=True):
            amount = conservative._calculate_trade_amount(1000000, external_bonus=0)
            assert amount == 50000  # 100000 * 0.5

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_weekend_no_reduction_high_bonus(self, conservative):
        with patch.object(conservative, '_is_weekend', return_value=True):
            amount = conservative._calculate_trade_amount(1000000, external_bonus=15)
            assert amount == 100000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_weekend_no_reduction_at_threshold(self, conservative):
        """외부 보너스 10이면 주말 축소 면제."""
        with patch.object(conservative, '_is_weekend', return_value=True):
            amount = conservative._calculate_trade_amount(1000000, external_bonus=10)
            assert amount == 100000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_weekend_reduction_at_9(self, conservative):
        """외부 보너스 9이면 주말 축소 적용."""
        with patch.object(conservative, '_is_weekend', return_value=True):
            amount = conservative._calculate_trade_amount(1000000, external_bonus=9)
            assert amount == 50000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_zero_balance(self, conservative):
        amount = conservative._calculate_trade_amount(0)
        assert amount == 0

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_aggressive_higher_ratio(self, aggressive):
        with patch.object(aggressive, '_is_weekend', return_value=False):
            amount = aggressive._calculate_trade_amount(500000)
            assert amount == 100000  # 500000 * 0.2 = 100000


# ============================================================
# _extract_indicators & Decision
# ============================================================

class TestHelpers:

    def test_extract_indicators(self, conservative):
        md = _make_market_data(rsi=35, sma_deviation=-3.0, trade_price=60000000)
        ind = conservative._extract_indicators(md)
        assert ind["rsi"] == 35
        assert ind["sma_deviation"] == -3.0
        assert ind["current_price"] == 60000000

    def test_extract_indicators_defaults(self, conservative):
        """키가 없으면 기본값."""
        ind = conservative._extract_indicators({})
        assert ind["rsi"] == 50
        assert ind["current_price"] == 0

    def test_decision_to_dict(self):
        d = Decision(
            decision="buy", confidence=0.8, reason="test",
            buy_score={"total": 80}, trade_params={"amount": 100000},
            external_signal={"total_score": 10, "strategy_bonus": 5,
                             "fusion": {"signal": "buy"}},
            agent_name="test_agent",
        )
        result = d.to_dict()
        assert result["decision"] == "buy"
        assert result["confidence"] == 0.8
        assert result["external_signal_summary"]["fusion_signal"] == "buy"
        assert result["external_signal_summary"]["total_score"] == 10

    def test_decision_to_dict_empty_signal(self):
        d = Decision(
            decision="hold", confidence=0.5, reason="test",
            buy_score={}, trade_params={},
            external_signal={}, agent_name="test",
        )
        result = d.to_dict()
        assert result["external_signal_summary"]["total_score"] == 0
        assert result["external_signal_summary"]["fusion_signal"] == "unknown"

    def test_decision_has_timestamp(self):
        d = Decision(
            decision="hold", confidence=0.5, reason="test",
            buy_score={}, trade_params={},
            external_signal={}, agent_name="test",
        )
        assert d.timestamp is not None
        assert "T" in d.timestamp
