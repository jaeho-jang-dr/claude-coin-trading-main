"""
agents/ 디렉토리 종합 유닛 테스트

Coverage:
  - base_agent.py: 점수 계산, 매수/매도 판단, 하이브리드 손절
  - conservative.py: 임계값 검증, 리스크 평가
  - moderate.py: 균형 매매 로직
  - aggressive.py: 고위험 시나리오
  - external_data.py: 데이터 수집, fusion 계산, 에러 처리
  - orchestrator.py: 전략 전환, danger/opportunity, 쿨다운, FOMO, DB 학습, 자동 긴급정지
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.base_agent import BaseStrategyAgent, Decision
from agents.conservative import ConservativeAgent
from agents.moderate import ModerateAgent
from agents.aggressive import AggressiveAgent
from agents.external_data import (
    ExternalDataAgent,
    analyze_news_sentiment,
    _compress_news,
    _fetch_nvt_signal,
    _run_script,
    load_performance_review,
    load_user_feedback,
)


# ============================================================
# Fixtures
# ============================================================

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
    macd_histogram=0, signal_cross=False,
    candles_4h=None,
):
    """테스트용 시장 데이터 팩토리."""
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


def _make_portfolio(krw=1000000, btc_balance=0, btc_avg_price=50000000,
                    profit_pct=0, total_eval=1000000):
    btc_eval = btc_balance * btc_avg_price if btc_balance > 0 else 0
    return {
        "krw_balance": krw,
        "btc": {
            "balance": btc_balance,
            "avg_buy_price": btc_avg_price,
            "profit_pct": profit_pct,
            "eval_amount": btc_eval,
        },
        "total_eval": total_eval,
        "btc_ratio": btc_eval / total_eval if total_eval > 0 else 0,
    }


def _make_external_signal(total_score=0, strategy_bonus=0, fusion_signal="neutral"):
    return {
        "total_score": total_score,
        "strategy_bonus": strategy_bonus,
        "fusion": {"signal": fusion_signal},
    }


# ============================================================
# BaseStrategyAgent - calculate_buy_score
# ============================================================

class TestBaseAgentBuyScore:
    """BaseStrategyAgent.calculate_buy_score 테스트."""

    def test_all_conditions_met_conservative(self, conservative):
        """FGI, RSI, SMA, 뉴스 모두 충족 시 100점 만점."""
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

    def test_fgi_extreme_bonus(self, conservative):
        """FGI가 임계의 50% 이하면 +5 보너스."""
        score = conservative.calculate_buy_score(
            fgi=10, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        # FGI 10 <= 30*0.5=15 → 30+5=35
        assert score["fgi"]["score"] == 35

    def test_fgi_partial(self, conservative):
        """FGI가 임계값 초과 10 이내면 부분 점수 50%.
        v2: conservative.fgi_threshold=35, so FGI=41 falls in partial zone (35 < 41 <= 45).
        """
        score = conservative.calculate_buy_score(
            fgi=41, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        # 41 > 35 but <= 45 → partial (35*0.5=17, int→17)
        assert score["fgi"]["score"] == int(conservative.fgi_points * 0.5)
        assert score["fgi"].get("partial") is True

    def test_rsi_partial(self, conservative):
        """RSI가 임계값 +5 이내면 부분 점수 15점.
        v2: conservative.rsi_threshold=35, so RSI=38 falls in partial zone (35 < 38 <= 40).
        """
        score = conservative.calculate_buy_score(
            fgi=80, rsi=38, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
        )
        # RSI 38 > 35 but <= 40 → 15점 (partial fixed score)
        assert score["rsi"]["score"] == 15
        assert score["rsi"].get("partial") is True

    def test_sma_full_score(self, conservative):
        """SMA 이탈이 임계값 이하면 만점."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=-6.0,
            news_negative=True, external_bonus=0,
        )
        assert score["sma"]["score"] == 25

    def test_news_positive_gets_points(self, conservative):
        """뉴스가 부정적이지 않으면 뉴스 점수 부여."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=False, external_bonus=0,
        )
        assert score["news"]["score"] == 20

    def test_macd_bonus_moderate_only(self, moderate):
        """Moderate만 MACD 보너스 10점."""
        score = moderate.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
            macd_golden_cross=True,
        )
        assert score["macd"]["score"] == 10

    def test_macd_bonus_not_conservative(self, conservative):
        """Conservative는 MACD 보너스 없음."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
            macd_golden_cross=True,
        )
        assert "macd" not in score

    def test_external_bonus_added(self, conservative):
        """외부 보너스가 총점에 반영된다."""
        score = conservative.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=15,
        )
        assert score["external"]["score"] == 15
        assert score["total"] == 15

    def test_threshold_boundary_buy(self, conservative):
        """정확히 임계 점수일 때 매수."""
        score = conservative.calculate_buy_score(
            fgi=15, rsi=25, sma_deviation=-6.0,
            news_negative=True, external_bonus=0,
        )
        # FGI 35, RSI 25, SMA 25 = 85 >= 70
        assert score["result"] == "buy"

    def test_threshold_boundary_hold(self, conservative):
        """임계 점수 미만이면 관망.
        v2: threshold=60. FGI partial(41>35<=45→15), RSI partial(38>35<=40→15),
        SMA 0, News 0 = 30 < 60 → hold.
        """
        score = conservative.calculate_buy_score(
            fgi=41, rsi=38, sma_deviation=-1.0,
            news_negative=True, external_bonus=0,
        )
        # FGI partial 15 (int(30*0.5)), RSI partial 15, SMA 0, news 0 = 30 < 60
        assert score["total"] < conservative.buy_score_threshold
        assert score["result"] == "hold"


# ============================================================
# BaseStrategyAgent - evaluate_sell
# ============================================================

class TestBaseAgentEvaluateSell:
    """BaseStrategyAgent.evaluate_sell 하이브리드 손절 테스트."""

    def test_target_profit_sell(self, conservative):
        """목표 수익 달성 시 매도."""
        result = conservative.evaluate_sell(
            profit_pct=16.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result is not None
        assert result["action"] == "sell"
        assert result["type"] == "target_profit"

    def test_target_profit_deferred_by_ai(self, conservative):
        """목표 수익이지만 AI 강세면 유예."""
        result = conservative.evaluate_sell(
            profit_pct=16.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=25,
        )
        assert result["action"] == "hold_defer"

    def test_fgi_overbought_sell(self, conservative):
        """FGI 과열 시 매도."""
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=80, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "fgi_overbought"

    def test_rsi_overbought_sell(self, conservative):
        """RSI 과매수 시 매도."""
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=50, current_rsi=75,
            buy_score={}, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "rsi_overbought"

    def test_forced_stop_loss(self, conservative):
        """강제 손절선 도달 시 무조건 매도."""
        result = conservative.evaluate_sell(
            profit_pct=-11.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "forced_stop"

    def test_hybrid_dca_with_bottom_signals(self, conservative):
        """손절선 + 바닥 시그널 3개 + AI 중립 → DCA."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
        )
        assert result["action"] == "dca"
        assert result["type"] == "hybrid_dca"

    def test_hybrid_sell_insufficient_signals(self, conservative):
        """손절선 + 바닥 시그널 부족 → 매도."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 0},
            "sma": {"score": 0}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=50, current_rsi=50,
            buy_score=buy_score, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "stop_loss"

    def test_cascade_risk_high_forces_sell(self, conservative):
        """캐스케이딩 위험 70+ → 무조건 손절."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 20},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=10,
            drop_context={"cascade_risk": 75, "price_change_4h": -5.0,
                          "volume_ratio": 2.5, "external_bearish_count": 3},
        )
        assert result["action"] == "sell"
        assert result["type"] == "cascade_stop"

    def test_dca_already_done_forces_sell(self, conservative):
        """DCA 이미 실행 후 추가 물타기 금지."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 20},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=10,
            drop_context={"dca_already_done": True, "cascade_risk": 0},
        )
        assert result["action"] == "sell"
        assert result["type"] == "dca_exhausted"

    def test_cascade_moderate_dca_with_strong_signals(self, conservative):
        """캐스케이딩 40-69 + 4개 바닥 시그널 → 신중 DCA."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 20},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 50, "external_bearish_count": 0},
        )
        assert result["action"] == "dca"
        assert result["type"] == "hybrid_dca_cautious"

    def test_cascade_moderate_sell_without_signals(self, conservative):
        """캐스케이딩 40-69 + 바닥 시그널 부족 → 매도."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 0},
            "sma": {"score": 0}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=50, current_rsi=50,
            buy_score=buy_score, ai_signal_score=0,
            drop_context={"cascade_risk": 50, "external_bearish_count": 0},
        )
        assert result["action"] == "sell"
        assert result["type"] == "cascade_moderate_stop"

    def test_external_bearish_override(self, conservative):
        """외부 약세 3개+ 겹침 → 바닥 시그널 있어도 매도."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 20, "external_bearish_count": 3},
        )
        assert result["action"] == "sell"
        assert result["type"] == "external_bearish_override"

    def test_trend_falling_dca_with_strong_signals(self, conservative):
        """하락 추세 + 강한 바닥 시그널 → 신중 DCA."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 20},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=5,
            drop_context={"cascade_risk": 20, "external_bearish_count": 1,
                          "trend_falling": True},
        )
        assert result["action"] == "dca"
        assert result["type"] == "hybrid_dca_trend"

    def test_trend_falling_sell_with_weak_signals(self, conservative):
        """하락 추세 + 바닥 시그널 3개 미만 → 매도."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=3,
            drop_context={"cascade_risk": 20, "external_bearish_count": 1,
                          "trend_falling": True},
        )
        assert result["action"] == "sell"
        assert result["type"] == "trend_stop"

    def test_ai_extreme_negative_forces_sell(self, conservative):
        """바닥 시그널 3개이나 AI 극단 매도 압력 → 매도."""
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-6.0, current_fgi=20, current_rsi=25,
            buy_score=buy_score, ai_signal_score=-25,
        )
        assert result["action"] == "sell"
        assert result["type"] == "hybrid_forced"

    def test_no_sell_in_profit_no_triggers(self, conservative):
        """수익 중 + 트리거 없음 → 매도하지 않음."""
        result = conservative.evaluate_sell(
            profit_pct=5.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result is None

    def test_overweight_partial_sell(self, conservative):
        """포지션 과다 + 수익 중 → 분할 매도."""
        result = conservative.evaluate_sell(
            profit_pct=6.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
            btc_position_ratio=0.7,
        )
        assert result["action"] == "sell_partial"
        assert result["type"] == "overweight_rebalance"
        assert "sell_ratio" in result

    def test_overweight_no_trigger_low_profit(self, conservative):
        """포지션 과다이나 수익률이 기준 미만이면 분할 매도 안 함."""
        result = conservative.evaluate_sell(
            profit_pct=3.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
            btc_position_ratio=0.7,
        )
        # 수익 3% < overweight_profit_threshold 5% → 분할 매도 안 함
        assert result is None


# ============================================================
# BaseStrategyAgent - Other helpers
# ============================================================

class TestBaseAgentHelpers:
    """BaseStrategyAgent 유틸리티 메서드 테스트."""

    def test_extract_indicators(self, conservative):
        md = _make_market_data(rsi=35, sma_deviation=-3.0, trade_price=60000000)
        ind = conservative._extract_indicators(md)
        assert ind["rsi"] == 35
        assert ind["sma_deviation"] == -3.0
        assert ind["current_price"] == 60000000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "200000"})
    def test_calculate_trade_amount_capped(self, conservative):
        """MAX_TRADE_AMOUNT 안전장치가 적용된다."""
        amount = conservative._calculate_trade_amount(5000000, external_bonus=0)
        # 5000000 * 0.1 = 500000 → min(500000, 200000) = 200000
        assert amount <= 200000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_calculate_trade_amount_weekend_reduction(self, conservative):
        """주말에는 거래 금액이 축소된다."""
        with patch.object(conservative, '_is_weekend', return_value=True):
            amount = conservative._calculate_trade_amount(1000000, external_bonus=0)
            # 1000000 * 0.1 = 100000 → weekend: 100000 * 0.5 = 50000
            assert amount == 50000

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_calculate_trade_amount_weekend_no_reduction_high_bonus(self, conservative):
        """주말이지만 외부 보너스 10+ → 축소 룰 무효화."""
        with patch.object(conservative, '_is_weekend', return_value=True):
            amount = conservative._calculate_trade_amount(1000000, external_bonus=15)
            assert amount == 100000  # 축소 안 됨

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


# ============================================================
# ConservativeAgent
# ============================================================

class TestConservativeAgent:
    """ConservativeAgent 전략 테스트."""

    def test_thresholds(self, conservative):
        # v2 조정(2026-03-11): threshold 70→60, FGI 30→35, RSI 30→35, SMA -5→-3
        assert conservative.buy_score_threshold == 60
        assert conservative.fgi_threshold == 35
        assert conservative.rsi_threshold == 35
        assert conservative.sma_deviation_pct == -3.0
        assert conservative.stop_loss_pct == -5.0
        assert conservative.forced_stop_loss_pct == -10.0
        assert conservative.target_profit_pct == 15.0
        assert conservative.max_trade_ratio == 0.10
        assert conservative.macd_bonus is False

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_buy_all_signals(self, conservative):
        """모든 조건 충족 시 매수."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=15, ai_score=5)
        port = _make_portfolio(krw=1000000)
        ext = _make_external_signal(strategy_bonus=10)
        decision = conservative.decide(md, ext, port)
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_hold_low_score(self, conservative):
        """점수 부족 시 관망."""
        md = _make_market_data(rsi=60, sma_deviation=2.0, fgi=70, ai_score=0)
        port = _make_portfolio()
        ext = _make_external_signal()
        decision = conservative.decide(md, ext, port)
        assert decision.decision == "hold"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_ai_veto(self, conservative):
        """매수 점수 충족이나 AI 음수 → 보류 (FGI > 15일 때)."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=20, ai_score=-5)
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=10)
        decision = conservative.decide(md, ext, port)
        assert decision.decision == "hold"
        assert hasattr(decision, '_was_ai_vetoed')
        assert decision._was_ai_vetoed is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_ai_veto_override_extreme_fear(self, conservative):
        """AI 음수이나 FGI <= 15 극공포면 매수 허용."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=10, ai_score=-5)
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=10)
        decision = conservative.decide(md, ext, port)
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_sell_when_holding(self, conservative):
        """보유 중 FGI 과열 → 매도."""
        md = _make_market_data(fgi=80, rsi=50)
        port = _make_portfolio(btc_balance=0.01, profit_pct=5.0, total_eval=1500000)
        ext = _make_external_signal()
        decision = conservative.decide(md, ext, port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_dca_when_stop_loss_with_signals(self, conservative):
        """손절선 + 바닥 시그널 충분 → DCA."""
        md = _make_market_data(fgi=15, rsi=25, sma_deviation=-6.0, ai_score=5)
        port = _make_portfolio(btc_balance=0.01, btc_avg_price=55000000,
                               profit_pct=-6.0, total_eval=1000000)
        ext = _make_external_signal(strategy_bonus=5)
        dc = {"cascade_risk": 10, "external_bearish_count": 0,
              "dca_already_done": False, "trend_falling": False}
        decision = conservative.decide(md, ext, port, drop_context=dc)
        assert decision.decision == "buy"
        assert decision.trade_params.get("is_dca") is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_partial_sell_overweight(self, conservative):
        """포지션 과다 + 수익 중 → 분할 매도."""
        md = _make_market_data(fgi=50, rsi=50)
        btc_eval = 800000
        total_eval = 1000000
        port = {
            "krw_balance": 200000,
            "btc": {
                "balance": 0.01,
                "avg_buy_price": 50000000,
                "profit_pct": 7.0,
                "eval_amount": btc_eval,
            },
            "total_eval": total_eval,
            "btc_ratio": btc_eval / total_eval,
        }
        ext = _make_external_signal()
        decision = conservative.decide(md, ext, port)
        assert decision.decision == "sell"
        assert decision.trade_params.get("is_partial") is True


# ============================================================
# ModerateAgent
# ============================================================

class TestModerateAgent:

    def test_thresholds(self, moderate):
        # v2 조정(2026-03-11): threshold 55→50, SMA -3→-2
        assert moderate.buy_score_threshold == 50
        assert moderate.fgi_threshold == 45
        assert moderate.rsi_threshold == 40
        assert moderate.sma_deviation_pct == -2.0
        assert moderate.target_profit_pct == 10.0
        assert moderate.macd_bonus is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_buy_with_macd(self, moderate):
        """MACD 골든크로스 보너스로 매수 점수 도달."""
        md = _make_market_data(
            rsi=38, sma_deviation=-4.0, fgi=40, ai_score=5,
            macd_histogram=1.0, signal_cross=True,
        )
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=5)
        decision = moderate.decide(md, ext, port)
        # FGI 30, RSI 25, SMA 25, News 20, MACD 10, ext 5 = 115 >= 55
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_moderate_fgi_above_20(self, moderate):
        """Moderate: AI 음수 + FGI > 20 → 보류."""
        md = _make_market_data(rsi=35, sma_deviation=-4.0, fgi=30, ai_score=-3)
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=10)
        decision = moderate.decide(md, ext, port)
        assert decision.decision == "hold"
        assert decision._was_ai_vetoed is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_override_extreme_fear(self, moderate):
        """Moderate: AI 음수이나 FGI <= 20 → 매수 허용."""
        md = _make_market_data(rsi=35, sma_deviation=-4.0, fgi=18, ai_score=-3)
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=10)
        decision = moderate.decide(md, ext, port)
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_sell_lower_thresholds(self, moderate):
        """Moderate의 매도 FGI 임계(70)가 Conservative(75)보다 낮다."""
        md = _make_market_data(fgi=72, rsi=50)
        port = _make_portfolio(btc_balance=0.01, profit_pct=3.0, total_eval=1000000)
        ext = _make_external_signal()
        decision = moderate.decide(md, ext, port)
        assert decision.decision == "sell"


# ============================================================
# AggressiveAgent
# ============================================================

class TestAggressiveAgent:

    def test_thresholds(self, aggressive):
        # v2 조정(2026-03-11): threshold 45→40
        assert aggressive.buy_score_threshold == 40
        assert aggressive.fgi_threshold == 60
        assert aggressive.rsi_threshold == 50
        assert aggressive.target_profit_pct == 7.0
        assert aggressive.stop_loss_pct == -3.0
        assert aggressive.forced_stop_loss_pct == -7.0
        assert aggressive.weekend_reduction == 0.0

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_decide_buy_always_news_positive(self, aggressive):
        """Aggressive: 뉴스 무관 (항상 20점 자동 부여)."""
        md = _make_market_data(
            rsi=45, sma_deviation=-2.0, fgi=55,
            ai_score=0, news_sentiment="negative",
        )
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=5)
        decision = aggressive.decide(md, ext, port)
        # FGI partial(30*0.5=15), RSI partial 15, SMA 25, News 20, ext 5 = 80 >= 45
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_no_ai_veto(self, aggressive):
        """Aggressive: AI 필터 없음, 음수여도 매수."""
        md = _make_market_data(
            rsi=45, sma_deviation=-2.0, fgi=55, ai_score=-10,
        )
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=5)
        decision = aggressive.decide(md, ext, port)
        # news_negative=False(항상), 점수 충분하면 AI 상관없이 매수
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_sell_low_target_profit(self, aggressive):
        """Aggressive: 목표 수익 7%로 빠른 익절."""
        md = _make_market_data(fgi=50, rsi=50)
        port = _make_portfolio(btc_balance=0.01, profit_pct=8.0, total_eval=1000000)
        ext = _make_external_signal()
        decision = aggressive.decide(md, ext, port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_forced_stop_lower(self, aggressive):
        """Aggressive: 강제 손절 -7%."""
        md = _make_market_data(fgi=50, rsi=50)
        port = _make_portfolio(btc_balance=0.01, profit_pct=-8.0, total_eval=1000000)
        ext = _make_external_signal()
        decision = aggressive.decide(md, ext, port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "500000"})
    def test_higher_trade_ratio(self, aggressive):
        """Aggressive: 1회 매매 20%."""
        with patch.object(aggressive, '_is_weekend', return_value=False):
            amount = aggressive._calculate_trade_amount(1000000)
            assert amount == 200000


# ============================================================
# ExternalDataAgent - News Sentiment
# ============================================================

class TestNewsSentiment:

    def test_no_articles(self):
        result = analyze_news_sentiment({"articles": []})
        assert result["sentiment_score"] == 0
        assert result["overall_sentiment"] == "neutral"

    def test_positive_articles(self):
        articles = [
            {"title": "ETF approved! Major rally expected", "content": "bullish adoption"},
            {"title": "Institutional inflow breaks record", "content": ""},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] == 2
        assert result["sentiment_score"] > 0
        assert result["overall_sentiment"] in ("positive", "slightly_positive")

    def test_negative_articles(self):
        articles = [
            {"title": "Major hack exploit discovered", "content": "crash panic"},
            {"title": "SEC lawsuit against crypto exchange", "content": "bearish sell-off"},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["negative_count"] == 2
        assert result["sentiment_score"] < 0

    def test_mixed_articles(self):
        articles = [
            {"title": "ETF approved bullish rally", "content": ""},
            {"title": "Major hack crash panic", "content": ""},
            {"title": "No relevant crypto news", "content": ""},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] == 1
        assert result["negative_count"] == 1
        assert result["neutral_count"] == 1

    def test_key_signals_limited_to_5(self):
        # Create many strongly positive articles
        articles = [
            {"title": f"ETF approved rally bullish #{i}", "content": "adoption institutional"}
            for i in range(10)
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert len(result["key_signals"]) <= 5

    def test_korean_keywords(self):
        articles = [
            {"title": "비트코인 상승 반등 돌파", "content": "강세 매수세"},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] == 1

    def test_overall_sentiment_thresholds(self):
        # positive threshold: >= 30
        articles = [
            {"title": "ETF approved", "content": ""},
            {"title": "rally expected", "content": ""},
            {"title": "neutral story", "content": ""},
        ]
        result = analyze_news_sentiment({"articles": articles})
        # 2 pos, 0 neg, 1 neutral → 66% → score 66 >= 30 → "positive"
        assert result["overall_sentiment"] == "positive"


# ============================================================
# ExternalDataAgent - Compress News
# ============================================================

class TestCompressNews:

    def test_empty_articles(self):
        result = _compress_news({"articles": []})
        assert result == {"articles": []}

    def test_compression(self):
        articles = [
            {"title": "Title1", "content": "X" * 200, "category": "crypto", "score": 0.9},
            {"title": "Title2", "content": "Short", "category": "market"},
        ]
        result = _compress_news({"articles": articles, "timestamp": "2025-01-01"})
        assert result["articles_count"] == 2
        assert "crypto" in result["by_category"]
        # Content should be truncated to 100 chars
        assert len(result["categories"]["crypto"][0]["snippet"]) <= 103  # 100 + "..."


# ============================================================
# ExternalDataAgent - _run_script
# ============================================================

class TestRunScript:

    @patch("agents.external_data.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"key": "value"}', stderr="",
        )
        result = _run_script("test_script.py")
        assert result == {"key": "value"}

    @patch("agents.external_data.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error",
        )
        result = _run_script("test_script.py")
        assert "error" in result

    @patch("agents.external_data.subprocess.run")
    def test_timeout(self, mock_run):
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("cmd", 60)
        result = _run_script("test_script.py", timeout=60)
        assert "타임아웃" in result["error"]

    @patch("agents.external_data.subprocess.run")
    def test_json_decode_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not json", stderr="",
        )
        result = _run_script("test_script.py")
        assert "JSON" in result["error"]


# ============================================================
# ExternalDataAgent - NVT Signal
# ============================================================

class TestFetchNvt:

    @patch("agents.external_data.requests.get")
    def test_success(self, mock_get):
        mc_resp = MagicMock(ok=True)
        mc_resp.json.return_value = {"values": [{"y": 1000000000000}]}
        tv_resp = MagicMock(ok=True)
        tv_resp.json.return_value = {"values": [{"y": 10000000000}]}
        mock_get.side_effect = [mc_resp, tv_resp]

        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 100.0
        assert result["interpretation"] == "normal"

    @patch("agents.external_data.requests.get")
    def test_failure_returns_default(self, mock_get):
        mock_get.side_effect = Exception("network error")
        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 100.0
        assert "error" in result

    @patch("agents.external_data.requests.get")
    def test_overvalued(self, mock_get):
        mc_resp = MagicMock(ok=True)
        mc_resp.json.return_value = {"values": [{"y": 1500000000000}]}
        tv_resp = MagicMock(ok=True)
        tv_resp.json.return_value = {"values": [{"y": 5000000000}]}
        mock_get.side_effect = [mc_resp, tv_resp]

        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 300.0
        assert result["interpretation"] == "overvalued"


# ============================================================
# ExternalDataAgent - Fusion Calculation
# ============================================================

class TestExternalDataFusion:

    def test_inline_fusion_neutral(self):
        agent = ExternalDataAgent()
        result = agent._inline_fusion({})
        assert result["total_score"] == 0
        assert result["strategy_bonus"] == 0

    def test_inline_fusion_with_binance(self):
        agent = ExternalDataAgent()
        results = {
            "binance_sentiment": {"sentiment_score": {"score": -10}},
        }
        result = agent._inline_fusion(results)
        # binance_sentiment score is now added (bug fix: was inverted before)
        assert result["total_score"] == -10

    def test_inline_fusion_bonus_mapping(self):
        agent = ExternalDataAgent()
        # score >= 40 → bonus 20
        results = {
            "binance_sentiment": {"sentiment_score": {"score": 50}},
        }
        result = agent._inline_fusion(results)
        assert result["total_score"] == 50
        assert result["strategy_bonus"] == 20

    def test_inline_fusion_negative_bonus(self):
        agent = ExternalDataAgent()
        results = {
            "binance_sentiment": {"sentiment_score": {"score": -50}},
        }
        result = agent._inline_fusion(results)
        assert result["total_score"] == -50
        assert result["strategy_bonus"] == -20

    def test_enhance_fusion_macro(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 10, "strategy_bonus": 5}
        results = {
            "macro": {"analysis": {"macro_score": 20, "sentiment": "bullish"}},
            "eth_btc": {},
            "news_sentiment": {},
            "crypto_signals": {},
            "coinmarketcap": {},
        }
        result = agent._enhance_fusion(fusion, results)
        # macro_adj = min(15, max(-15, 20*0.5)) = 10
        assert result["total_score"] == 20  # 10 + 10
        assert result["extra_components"]["macro"]["score"] == 10

    def test_enhance_fusion_eth_btc_extreme(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = {
            "macro": {"analysis": {}},
            "eth_btc": {"eth_btc_z_score": -2.5},
            "news_sentiment": {},
            "crypto_signals": {},
            "coinmarketcap": {},
        }
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["eth_btc"]["score"] == 5

    def test_enhance_fusion_news_positive(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = {
            "macro": {"analysis": {}},
            "eth_btc": {},
            "news_sentiment": {"sentiment_score": 50, "overall_sentiment": "positive"},
            "crypto_signals": {},
            "coinmarketcap": {},
        }
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["news_sentiment"]["score"] == 10

    def test_enhance_fusion_crypto_signals_high(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = {
            "macro": {"analysis": {}},
            "eth_btc": {},
            "news_sentiment": {},
            "crypto_signals": {
                "btc": {"anomaly_level": "HIGH", "change_24h": 5.0},
                "anomaly_alerts": {"count": 10},
            },
            "coinmarketcap": {},
        }
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["crypto_signals"]["score"] == 10

    def test_enhance_fusion_cmc_dominance(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = {
            "macro": {"analysis": {}},
            "eth_btc": {},
            "news_sentiment": {},
            "crypto_signals": {},
            "coinmarketcap": {"status": "success", "btc_dominance": 58.0},
        }
        result = agent._enhance_fusion(fusion, results)
        assert result["total_score"] == 5


# ============================================================
# ExternalDataAgent - collect_all
# ============================================================

class TestExternalDataCollectAll:

    @patch("agents.external_data._fetch_nvt_signal")
    @patch("agents.external_data._run_script")
    @patch("agents.external_data.load_user_feedback", return_value=[])
    @patch("agents.external_data.load_performance_review", return_value={"available": False})
    def test_collect_all_returns_structure(self, mock_perf, mock_fb, mock_run, mock_nvt):
        mock_run.return_value = {"test": True}
        mock_nvt.return_value = {"nvt_signal": 100.0}

        agent = ExternalDataAgent()
        with patch.object(agent, '_save_signal_to_db'):
            result = agent.collect_all()

        assert "timestamp" in result
        assert "sources" in result
        assert "external_signal" in result
        assert "errors" in result
        assert "collection_time_sec" in result

    @patch("agents.external_data._fetch_nvt_signal")
    @patch("agents.external_data._run_script")
    @patch("agents.external_data.load_user_feedback", return_value=[])
    @patch("agents.external_data.load_performance_review", return_value={"available": False})
    def test_collect_all_error_isolation(self, mock_perf, mock_fb, mock_run, mock_nvt):
        """하나가 실패해도 나머지는 정상."""
        def side_effect(script, *args, **kwargs):
            if "fear_greed" in script:
                return {"error": "timeout"}
            return {"status": "ok"}

        mock_run.side_effect = side_effect
        mock_nvt.return_value = {"nvt_signal": 100.0}

        agent = ExternalDataAgent()
        with patch.object(agent, '_save_signal_to_db'):
            result = agent.collect_all()

        assert "fear_greed" in result["errors"]

    def test_get_fgi_value(self):
        agent = ExternalDataAgent()
        results = {
            "sources": {
                "fear_greed": {"current": {"value": 25}},
            },
        }
        assert agent.get_fgi_value(results) == 25

    def test_get_fgi_value_default(self):
        agent = ExternalDataAgent()
        assert agent.get_fgi_value({}) == 50


# ============================================================
# ExternalDataAgent - Performance Review
# ============================================================

class TestPerformanceReview:

    @patch("agents.external_data._load_supabase")
    def test_no_data(self, mock_load):
        mock_load.return_value = []
        result = load_performance_review()
        assert result["available"] is False

    @patch("agents.external_data._load_supabase")
    def test_good_performance(self, mock_load):
        mock_load.return_value = [
            {"decision": "buy", "profit_loss": 5.0, "confidence": 0.8,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "sell", "profit_loss": 3.0, "confidence": 0.7,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "buy", "profit_loss": -2.0, "confidence": 0.6,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
        ]
        result = load_performance_review()
        assert result["available"] is True
        assert result["wins"] == 2
        assert result["losses"] == 1
        assert result["win_rate_pct"] > 60

    @patch("agents.external_data._load_supabase")
    def test_losing_streak(self, mock_load):
        mock_load.return_value = [
            {"decision": "buy", "profit_loss": -3.0, "confidence": 0.5,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "sell", "profit_loss": -2.0, "confidence": 0.5,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "buy", "profit_loss": 5.0, "confidence": 0.5,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
        ]
        result = load_performance_review()
        assert result["recent_streak_type"] == "loss"
        assert result["recent_streak"] == 2


# ============================================================
# Orchestrator - Phase Classification
# ============================================================

class TestOrchestratorPhases:

    @patch("agents.orchestrator._load_state")
    def test_classify_phases(self, mock_state):
        mock_state.return_value = {
            "active_agent": "conservative",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        assert orch._classify_phase(10) == "extreme_fear"
        assert orch._classify_phase(20) == "extreme_fear"
        assert orch._classify_phase(25) == "fear"
        assert orch._classify_phase(35) == "fear"
        assert orch._classify_phase(50) == "neutral"
        assert orch._classify_phase(60) == "neutral"
        assert orch._classify_phase(70) == "greed"
        assert orch._classify_phase(80) == "greed"
        assert orch._classify_phase(90) == "extreme_greed"


# ============================================================
# Orchestrator - Danger Score
# ============================================================

class TestOrchestratorDangerScore:

    @patch("agents.orchestrator._load_state")
    def test_zero_danger(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        assert score == 0

    @patch("agents.orchestrator._load_state")
    def test_consecutive_losses(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=3,
        )
        assert score == 30  # max 30

    @patch("agents.orchestrator._load_state")
    def test_consecutive_losses_capped(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=10,
        )
        # 10*10=100, capped at 30
        assert score >= 30

    @patch("agents.orchestrator._load_state")
    def test_crash_adds_danger(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-5,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        assert score == 25  # abs(-5)*5=25

    @patch("agents.orchestrator._load_state")
    def test_kimchi_premium_danger(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=5, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0,
        )
        assert score == 10  # (5-3)*5=10

    @patch("agents.orchestrator._load_state")
    def test_macro_negative_danger(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0, macro_score=-20,
        )
        assert score == 10  # abs(-20)*0.5=10

    @patch("agents.orchestrator._load_state")
    def test_news_negative_danger(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=0,
            kimchi_pct=0, ls_ratio=1.0, btc_ratio=0.2,
            consecutive_losses=0, news_sentiment="negative",
        )
        assert score == 10

    @patch("agents.orchestrator._load_state")
    def test_danger_capped_at_100(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_danger_score(
            fgi=50, rsi=50, price_change_24h=-10,
            kimchi_pct=10, ls_ratio=2.0, btc_ratio=0.8,
            consecutive_losses=5, macro_score=-30, news_sentiment="negative",
        )
        assert score <= 100


# ============================================================
# Orchestrator - Opportunity Score
# ============================================================

class TestOrchestratorOpportunityScore:

    @patch("agents.orchestrator._load_state")
    def test_zero_opportunity(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 0

    @patch("agents.orchestrator._load_state")
    def test_extreme_fear_opportunity(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=10, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # 25-10=15
        assert score == 15

    @patch("agents.orchestrator._load_state")
    def test_rsi_oversold_opportunity(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=50, rsi=25, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=0,
        )
        # (35-25)*1.3=13
        assert score == 13

    @patch("agents.orchestrator._load_state")
    def test_strong_buy_fusion(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="strong_buy", fusion_score=50,
            funding_rate=0, kimchi_pct=0,
        )
        assert score == 20

    @patch("agents.orchestrator._load_state")
    def test_negative_funding_opportunity(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=-0.02, kimchi_pct=0,
        )
        assert score == 10

    @patch("agents.orchestrator._load_state")
    def test_kimchi_discount_opportunity(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=50, rsi=50, price_change_24h=0,
            fusion_signal="neutral", fusion_score=0,
            funding_rate=0, kimchi_pct=-3,
        )
        # abs(-3)*3=9
        assert score == 9

    @patch("agents.orchestrator._load_state")
    def test_opportunity_capped_at_100(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        score = orch._calculate_opportunity_score(
            fgi=5, rsi=20, price_change_24h=5,
            fusion_signal="strong_buy", fusion_score=50,
            funding_rate=-0.02, kimchi_pct=-5,
            macro_score=30, news_sentiment="positive",
        )
        assert score <= 100


# ============================================================
# Orchestrator - Strategy Switching
# ============================================================

class TestOrchestratorSwitching:

    def _make_orchestrator(self, active="moderate"):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": active,
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            orch._learning_data = None
            orch._performance = {}
            return orch

    def test_danger_70_to_conservative(self):
        orch = self._make_orchestrator("aggressive")
        ms = {
            "danger_score": 75, "opportunity_score": 10,
            "fgi": 30, "rsi": 50, "price_change_24h": -6,
            "kimchi_pct": 3, "ls_ratio": 1.0,
            "consecutive_losses": 2, "fusion_signal": "neutral",
            "phase": "fear",
        }
        target = orch._decide_target("aggressive", ms, 75, 10)
        assert target == "conservative"

    def test_danger_50_moderate_to_conservative(self):
        orch = self._make_orchestrator("moderate")
        ms = {
            "danger_score": 55, "opportunity_score": 10,
            "fgi": 35, "rsi": 50, "price_change_24h": -3,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 2, "fusion_signal": "neutral",
            "phase": "fear",
        }
        target = orch._decide_target("moderate", ms, 55, 10)
        assert target == "conservative"

    def test_danger_45_aggressive_to_moderate(self):
        orch = self._make_orchestrator("aggressive")
        ms = {
            "danger_score": 50, "opportunity_score": 10,
            "fgi": 50, "rsi": 50, "price_change_24h": 0,
            "kimchi_pct": 4, "ls_ratio": 1.3,
            "consecutive_losses": 0, "fusion_signal": "neutral",
            "phase": "neutral",
        }
        target = orch._decide_target("aggressive", ms, 50, 10)
        assert target == "moderate"

    def test_opportunity_60_to_aggressive(self):
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 65,
            "fgi": 15, "rsi": 25, "price_change_24h": 2,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "strong_buy",
            "phase": "extreme_fear",
        }
        target = orch._decide_target("conservative", ms, 10, 65)
        assert target == "aggressive"

    def test_opportunity_40_conservative_to_moderate(self):
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 45,
            "fgi": 30, "rsi": 35, "price_change_24h": 1,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "neutral",
            "phase": "fear",
        }
        target = orch._decide_target("conservative", ms, 10, 45)
        assert target == "moderate"

    def test_opportunity_40_moderate_to_aggressive(self):
        orch = self._make_orchestrator("moderate")
        ms = {
            "danger_score": 10, "opportunity_score": 45,
            "fgi": 30, "rsi": 35, "price_change_24h": 1,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "buy",
            "phase": "fear",
        }
        target = orch._decide_target("moderate", ms, 10, 45)
        assert target == "aggressive"

    def test_opportunity_25_conservative_to_moderate(self):
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 30,
            "fgi": 40, "rsi": 40, "price_change_24h": 0,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "neutral",
            "phase": "neutral",
        }
        target = orch._decide_target("conservative", ms, 10, 30)
        assert target == "moderate"

    def test_sideways_aggressive_to_moderate(self):
        orch = self._make_orchestrator("aggressive")
        ms = {
            "danger_score": 10, "opportunity_score": 10,
            "fgi": 50, "rsi": 50, "price_change_24h": 0,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "neutral",
            "phase": "neutral",
        }
        target = orch._decide_target("aggressive", ms, 10, 10)
        assert target == "moderate"

    def test_sideways_conservative_to_moderate(self):
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 10,
            "fgi": 50, "rsi": 50, "price_change_24h": 0,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "neutral",
            "phase": "neutral",
        }
        target = orch._decide_target("conservative", ms, 10, 10)
        assert target == "moderate"

    def test_no_switch_already_optimal(self):
        orch = self._make_orchestrator("moderate")
        ms = {
            "danger_score": 30, "opportunity_score": 30,
            "fgi": 50, "rsi": 50, "price_change_24h": 0,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "neutral",
            "phase": "neutral",
        }
        target = orch._decide_target("moderate", ms, 30, 30)
        assert target is None


# ============================================================
# Orchestrator - FOMO Prevention
# ============================================================

class TestOrchestratorFomo:

    def _make_orchestrator(self, active="conservative"):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": active,
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            orch._learning_data = None
            orch._performance = {}
            return orch

    def test_fomo_block_crash(self):
        """24h -6% 급락 중 → 공격적 전환 차단."""
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 65,
            "fgi": 30, "rsi": 25, "price_change_24h": -6,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "strong_buy",
            "phase": "fear",
        }
        target = orch._decide_target("conservative", ms, 10, 65)
        # FOMO block: price < -5 and not (fgi<=20 and price > -8)
        # fgi=30 > 20 → FOMO blocked
        assert target is None

    def test_fomo_exception_extreme_fear(self):
        """24h -6% 이나 FGI<=20 극공포 + 하락폭 < -8 → 공격적 전환 허용."""
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 65,
            "fgi": 15, "rsi": 25, "price_change_24h": -6,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "strong_buy",
            "phase": "extreme_fear",
        }
        target = orch._decide_target("conservative", ms, 10, 65)
        assert target == "aggressive"

    def test_fomo_no_exception_deep_crash(self):
        """24h -9% 심각한 급락 → FGI 극공포여도 -8 초과 → 차단."""
        orch = self._make_orchestrator("conservative")
        ms = {
            "danger_score": 10, "opportunity_score": 65,
            "fgi": 15, "rsi": 25, "price_change_24h": -9,
            "kimchi_pct": 0, "ls_ratio": 1.0,
            "consecutive_losses": 0, "fusion_signal": "strong_buy",
            "phase": "extreme_fear",
        }
        target = orch._decide_target("conservative", ms, 10, 65)
        # -9 < -8 → exception does NOT apply → fomo blocked
        assert target is None


# ============================================================
# Orchestrator - Cooldown
# ============================================================

class TestOrchestratorCooldown:

    def test_cooldown_active(self):
        """마지막 전환 후 2시간 이내면 쿨다운."""
        kst = timezone(timedelta(hours=9))
        recent_time = (datetime.now(kst) - timedelta(hours=1)).isoformat()

        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": recent_time,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            assert orch._is_on_cooldown() is True

    def test_cooldown_expired(self):
        """마지막 전환 후 3시간 경과 → 쿨다운 해제."""
        kst = timezone(timedelta(hours=9))
        old_time = (datetime.now(kst) - timedelta(hours=3)).isoformat()

        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": old_time,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            assert orch._is_on_cooldown() is False

    def test_cooldown_strengthened_many_switches(self):
        """같은 날 3회 이상 전환 시 쿨다운 4시간."""
        kst = timezone(timedelta(hours=9))
        recent_time = (datetime.now(kst) - timedelta(hours=3)).isoformat()
        today = time.strftime("%Y-%m-%d")

        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": recent_time,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [
                    {"timestamp": f"{today}T01:00:00+09:00"},
                    {"timestamp": f"{today}T05:00:00+09:00"},
                    {"timestamp": f"{today}T09:00:00+09:00"},
                ],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            # 3시간 경과이나 당일 3회 전환 → 4시간 쿨다운 → 아직 쿨다운 중
            assert orch._is_on_cooldown() is True

    def test_no_cooldown_no_history(self):
        """전환 이력 없으면 쿨다운 아님."""
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            assert orch._is_on_cooldown() is False

    def test_emergency_bypasses_cooldown(self):
        """danger>=70 또는 24h -7% → 쿨다운 무시."""
        kst = timezone(timedelta(hours=9))
        recent_time = (datetime.now(kst) - timedelta(minutes=30)).isoformat()

        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "aggressive",
                "last_switch_time": recent_time,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            orch = Orchestrator()
            orch._learning_data = None
            orch._performance = {}

            ms = {
                "danger_score": 75, "opportunity_score": 5,
                "fgi": 20, "rsi": 50, "price_change_24h": -8,
                "kimchi_pct": 0, "ls_ratio": 1.0,
                "consecutive_losses": 3, "fusion_signal": "neutral",
                "phase": "extreme_fear",
            }
            # _evaluate_switch checks is_emergency first
            result = orch._evaluate_switch(ms)
            assert result is not None
            assert result["to"] == "conservative"


# ============================================================
# Orchestrator - DB Learning
# ============================================================

class TestOrchestratorLearning:

    def _make_orchestrator(self):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    def test_no_learning_data(self):
        orch = self._make_orchestrator()
        orch._learning_data = None
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result == {"danger_adjust": 0, "opportunity_adjust": 0}

    def test_bad_aggressive_history(self):
        orch = self._make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 30,
                "total_switches": 5,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == -10

    def test_bad_conservative_history(self):
        orch = self._make_orchestrator()
        orch._learning_data = {
            ("moderate", "conservative"): {
                "success_rate_pct": 35,
                "total_switches": 4,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["danger_adjust"] == -10

    def test_good_history_no_penalty(self):
        orch = self._make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 60,
                "total_switches": 5,
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == 0

    def test_insufficient_data_no_penalty(self):
        orch = self._make_orchestrator()
        orch._learning_data = {
            ("moderate", "aggressive"): {
                "success_rate_pct": 20,
                "total_switches": 2,  # < 3
            },
        }
        result = orch._get_learning_penalty("moderate", "neutral")
        assert result["opportunity_adjust"] == 0

    def test_aggregate_learning(self):
        orch = self._make_orchestrator()
        rows = [
            {"from_agent": "moderate", "to_agent": "aggressive", "outcome": "good", "profit_after_24h": 2.0},
            {"from_agent": "moderate", "to_agent": "aggressive", "outcome": "bad", "profit_after_24h": -1.5},
            {"from_agent": "moderate", "to_agent": "aggressive", "outcome": "good", "profit_after_24h": 1.0},
        ]
        result = orch._aggregate_learning(rows)
        key = ("moderate", "aggressive")
        assert key in result
        assert result[key]["good_count"] == 2
        assert result[key]["bad_count"] == 1
        assert result[key]["success_rate_pct"] == pytest.approx(66.7, abs=0.1)


# ============================================================
# Orchestrator - Auto Emergency
# ============================================================

class TestOrchestratorAutoEmergency:

    def _make_orchestrator(self):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    def test_flash_crash_triggers(self):
        orch = self._make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -12, "cascade_risk": 50,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -5}},
        )
        assert result is not None
        assert "플래시 크래시" in result["reason"]

    def test_cascade_plus_danger_triggers(self):
        orch = self._make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 85, "consecutive_losses": 0},
            drop_context={"price_change_4h": -5, "cascade_risk": 92,
                          "external_bearish_count": 3, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -5}},
        )
        assert result is not None
        assert "캐스케이딩" in result["reason"]

    def test_external_bearish_overload_triggers(self):
        orch = self._make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 0},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 5,
                          "external_bearish_details": ["a", "b", "c", "d", "e"]},
            portfolio={"btc": {"profit_pct": -5}},
        )
        assert result is not None
        assert "외부 약세" in result["reason"]

    def test_portfolio_crisis_triggers(self):
        orch = self._make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 50, "consecutive_losses": 6},
            drop_context={"price_change_4h": -2, "cascade_risk": 30,
                          "external_bearish_count": 2, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": -18}},
        )
        assert result is not None
        assert "연속 손절" in result["reason"]

    def test_no_emergency_normal_conditions(self):
        orch = self._make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 30, "consecutive_losses": 1},
            drop_context={"price_change_4h": -1, "cascade_risk": 20,
                          "external_bearish_count": 1, "external_bearish_details": []},
            portfolio={"btc": {"profit_pct": 3}},
        )
        assert result is None

    def test_multiple_triggers_combined(self):
        orch = self._make_orchestrator()
        result = orch._evaluate_auto_emergency(
            market_state={"danger_score": 85, "consecutive_losses": 6},
            drop_context={"price_change_4h": -12, "cascade_risk": 95,
                          "external_bearish_count": 5,
                          "external_bearish_details": ["a", "b", "c", "d", "e"]},
            portfolio={"btc": {"profit_pct": -20}},
        )
        assert result is not None
        # Multiple reasons combined
        assert " / " in result["reason"]


# ============================================================
# Orchestrator - Override Decision
# ============================================================

class TestOrchestratorOverride:

    def _make_orchestrator(self):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    def test_override_dca_cascade_high(self):
        """DCA + 캐스케이딩 70+ → 매도로 전환."""
        orch = self._make_orchestrator()
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

    def test_override_buy_during_crash(self):
        """신규 매수 + 급락 진행 → 관망으로 전환."""
        orch = self._make_orchestrator()
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

    def test_no_override_normal(self):
        """정상 조건 → 오버라이드 없음."""
        orch = self._make_orchestrator()
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
        assert not hasattr(result, '_orchestrator_override') or not result._orchestrator_override

    def test_no_override_sell(self):
        """매도 결정은 오버라이드하지 않음."""
        orch = self._make_orchestrator()
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


# ============================================================
# Orchestrator - Drop Context
# ============================================================

class TestOrchestratorDropContext:

    def _make_orchestrator(self, dca_history=None):
        with patch("agents.orchestrator._load_state") as mock:
            state = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            if dca_history:
                state["dca_history"] = dca_history
            mock.return_value = state
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    def test_basic_drop_context(self):
        orch = self._make_orchestrator()
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
        port = _make_portfolio()
        dc = orch._build_drop_context(md, ext, port)
        assert "cascade_risk" in dc
        assert "trend_falling" in dc
        assert "dca_already_done" in dc
        assert "external_bearish_count" in dc

    def test_cascade_risk_flash_crash(self):
        orch = self._make_orchestrator()
        candles = [
            {"trade_price": 50000000, "opening_price": 51000000, "candle_acc_trade_volume": 100},
            {"trade_price": 47000000, "opening_price": 50000000, "candle_acc_trade_volume": 300},
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
        port = _make_portfolio()
        dc = orch._build_drop_context(md, ext, port)
        # Many bearish signals → high cascade_risk
        assert dc["cascade_risk"] >= 50
        assert dc["external_bearish_count"] >= 4

    def test_dca_history_tracking(self):
        orch = self._make_orchestrator(dca_history={
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
        port = _make_portfolio()
        dc = orch._build_drop_context(md, ext, port)
        assert dc["dca_already_done"] is True


# ============================================================
# Orchestrator - DCA Tracking
# ============================================================

class TestOrchestratorDcaTracking:

    def _make_orchestrator(self):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
                "dca_history": {},
            }
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    @patch("agents.orchestrator._save_state")
    def test_track_dca_buy(self, mock_save):
        orch = self._make_orchestrator()
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
    def test_track_dca_sell_resets(self, mock_save):
        orch = self._make_orchestrator()
        orch.state["dca_history"]["KRW-BTC"] = {"dca_count": 2, "dca_total_amount": 100000}
        decision = Decision(
            decision="sell", confidence=0.8, reason="Sell",
            buy_score={},
            trade_params={"market": "KRW-BTC", "volume": 0.01},
            external_signal={}, agent_name="test",
        )
        orch._track_dca(decision)
        assert "KRW-BTC" not in orch.state["dca_history"]


# ============================================================
# Orchestrator - Emergency Stop
# ============================================================

class TestOrchestratorEmergencyStop:

    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "true"})
    def test_user_emergency_stop(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()

        result = orch.run({}, {}, {})
        assert result["decision"]["decision"] == "hold"
        assert "EMERGENCY_STOP" in result["decision"]["reason"]

    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false"})
    def test_auto_emergency_blocks(self, mock_state):
        mock_state.return_value = {
            "active_agent": "moderate",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }
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


# ============================================================
# Orchestrator - Feedback & Performance
# ============================================================

class TestOrchestratorFeedbackPerformance:

    def _make_orchestrator(self):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    def test_apply_feedback_conservative(self):
        orch = self._make_orchestrator()
        orch._apply_feedback([{"content": "좀 더 보수적으로 해주세요"}])
        assert orch.state.get("feedback_bias") == "conservative"

    def test_apply_feedback_aggressive(self):
        orch = self._make_orchestrator()
        orch._apply_feedback([{"content": "공격적으로 전환해줘"}])
        assert orch.state.get("feedback_bias") == "aggressive"

    def test_apply_feedback_moderate(self):
        orch = self._make_orchestrator()
        orch._apply_feedback([{"content": "moderate로 가자"}])
        assert orch.state.get("feedback_bias") == "moderate"

    def test_apply_feedback_empty(self):
        orch = self._make_orchestrator()
        orch._apply_feedback([])
        assert "feedback_bias" not in orch.state

    def test_performance_adjustment_losing_streak(self):
        orch = self._make_orchestrator()
        orch._performance = {
            "available": True,
            "win_rate_pct": 30,
            "recent_streak_type": "loss",
            "recent_streak": 4,
        }
        adj = orch._get_performance_adjustment()
        assert adj == 15  # danger에 +15

    def test_performance_adjustment_winning_streak(self):
        orch = self._make_orchestrator()
        orch._performance = {
            "available": True,
            "win_rate_pct": 70,
            "recent_streak_type": "win",
            "recent_streak": 4,
        }
        adj = orch._get_performance_adjustment()
        assert adj == -10  # opportunity에 +10

    def test_performance_adjustment_no_data(self):
        orch = self._make_orchestrator()
        orch._performance = {}
        adj = orch._get_performance_adjustment()
        assert adj == 0


# ============================================================
# Orchestrator - Consecutive Losses
# ============================================================

class TestOrchestratorConsecutiveLosses:

    def _make_orchestrator(self):
        with patch("agents.orchestrator._load_state") as mock:
            mock.return_value = {
                "active_agent": "moderate",
                "last_switch_time": None,
                "last_trade_time": None,
                "consecutive_losses": 0,
                "switch_history": [],
            }
            from agents.orchestrator import Orchestrator
            return Orchestrator()

    def test_no_losses(self):
        orch = self._make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "buy", "profit_loss": 5.0},
            {"decision": "sell", "profit_loss": 3.0},
        ])
        assert count == 0

    def test_consecutive_losses(self):
        orch = self._make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "buy", "profit_loss": -2.0},
            {"decision": "sell", "profit_loss": -3.0},
            {"decision": "buy", "profit_loss": 5.0},
        ])
        assert count == 2

    def test_hold_not_counted(self):
        orch = self._make_orchestrator()
        count = orch._count_consecutive_losses([
            {"decision": "hold", "profit_loss": None},
            {"decision": "buy", "profit_loss": -2.0},
        ])
        assert count == 0

    def test_empty_decisions(self):
        orch = self._make_orchestrator()
        count = orch._count_consecutive_losses([])
        assert count == 0


# ============================================================
# Orchestrator - Full Run Integration (mocked)
# ============================================================

class TestOrchestratorRun:

    @patch("agents.orchestrator._save_state")
    @patch("agents.orchestrator._load_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false", "MAX_TRADE_AMOUNT": "100000"})
    def test_full_run_hold(self, mock_state, mock_save):
        mock_state.return_value = {
            "active_agent": "conservative",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
            "dca_history": {},
        }
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
                with patch.object(orch.active_agent, 'save_buy_score_detail', return_value=None):
                    result = orch.run(md, ext, port)

        assert "decision" in result
        assert "active_agent" in result
        assert "market_state" in result
        # With neutral conditions, conservative should hold
        assert result["decision"]["decision"] == "hold"


# ============================================================
# Edge Cases & Boundary Tests
# ============================================================

class TestEdgeCases:

    def test_decision_with_none_external_signal(self):
        """external_signal이 빈 dict일 때도 to_dict 동작."""
        d = Decision(
            decision="hold", confidence=0.5, reason="test",
            buy_score={}, trade_params={},
            external_signal={}, agent_name="test",
        )
        result = d.to_dict()
        assert result["external_signal_summary"]["total_score"] == 0
        assert result["external_signal_summary"]["fusion_signal"] == "unknown"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_zero_krw_balance(self, conservative):
        """잔고 0일 때 매수 금액 0."""
        amount = conservative._calculate_trade_amount(0)
        assert amount == 0

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_negative_external_bonus(self, conservative):
        """음수 외부 보너스."""
        score = conservative.calculate_buy_score(
            fgi=25, rsi=25, sma_deviation=-6.0,
            news_negative=False, external_bonus=-15,
        )
        assert score["external"]["score"] == -15
        # FGI 25<=30 → 30pts (no extreme bonus since 25 > 15), RSI 25, SMA 25, News 20 = 100
        total_without = 30 + 25 + 25 + 20  # 100
        assert score["total"] == total_without - 15  # 85

    def test_evaluate_sell_at_exact_stop_loss(self, conservative):
        """정확히 손절선(-5.0%)일 때 동작 확인."""
        buy_score = {
            "fgi": {"score": 0}, "rsi": {"score": 0},
            "sma": {"score": 0}, "news": {"score": 0},
        }
        result = conservative.evaluate_sell(
            profit_pct=-5.0, current_fgi=50, current_rsi=50,
            buy_score=buy_score, ai_signal_score=0,
        )
        assert result is not None
        assert result["action"] == "sell"

    def test_evaluate_sell_at_exact_forced_stop(self, conservative):
        """정확히 강제 손절선(-10.0%)일 때."""
        result = conservative.evaluate_sell(
            profit_pct=-10.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["type"] == "forced_stop"

    def test_evaluate_sell_at_exact_target(self, conservative):
        """정확히 목표 수익(15.0%)일 때."""
        result = conservative.evaluate_sell(
            profit_pct=15.0, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result["action"] == "sell"
        assert result["type"] == "target_profit"

    def test_evaluate_sell_slightly_below_stop(self, conservative):
        """손절선 바로 위(-4.9%)면 매도하지 않음."""
        result = conservative.evaluate_sell(
            profit_pct=-4.9, current_fgi=50, current_rsi=50,
            buy_score={}, ai_signal_score=0,
        )
        assert result is None

    @patch("agents.orchestrator._load_state")
    def test_fgi_extraction(self, mock_state):
        """external_data에서 FGI 추출."""
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        ext = {"sources": {"fear_greed": {"current": {"value": 25}}}}
        assert orch._get_fgi(ext) == 25

    @patch("agents.orchestrator._load_state")
    def test_fgi_extraction_missing(self, mock_state):
        """FGI 데이터 없으면 기본값 50."""
        mock_state.return_value = {
            "active_agent": "moderate", "last_switch_time": None,
            "last_trade_time": None, "consecutive_losses": 0, "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch._get_fgi({}) == 50


# ============================================================
# Cross-Agent Comparison Tests
# ============================================================

class TestCrossAgentComparison:
    """세 에이전트 간 임계값과 행동 차이를 검증한다."""

    def test_buy_threshold_ordering(self, conservative, moderate, aggressive):
        assert aggressive.buy_score_threshold < moderate.buy_score_threshold
        assert moderate.buy_score_threshold < conservative.buy_score_threshold

    def test_fgi_threshold_ordering(self, conservative, moderate, aggressive):
        assert conservative.fgi_threshold < moderate.fgi_threshold < aggressive.fgi_threshold

    def test_rsi_threshold_ordering(self, conservative, moderate, aggressive):
        assert conservative.rsi_threshold < moderate.rsi_threshold < aggressive.rsi_threshold

    def test_target_profit_ordering(self, conservative, moderate, aggressive):
        assert aggressive.target_profit_pct < moderate.target_profit_pct < conservative.target_profit_pct

    def test_stop_loss_ordering(self, conservative, moderate, aggressive):
        # More negative = tighter (aggressive has tighter stop)
        assert aggressive.stop_loss_pct > moderate.stop_loss_pct

    def test_trade_ratio_ordering(self, conservative, moderate, aggressive):
        assert conservative.max_trade_ratio < moderate.max_trade_ratio < aggressive.max_trade_ratio

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "1000000"})
    def test_same_conditions_different_decisions(self, conservative, moderate, aggressive):
        """동일 시장 조건에서 세 에이전트의 결정이 전략에 맞는지."""
        md = _make_market_data(rsi=42, sma_deviation=-2.0, fgi=42, ai_score=5)
        port = _make_portfolio()
        ext = _make_external_signal(strategy_bonus=5)

        d_con = conservative.decide(md, ext, port)
        d_mod = moderate.decide(md, ext, port)
        d_agg = aggressive.decide(md, ext, port)

        # Conservative (threshold 70) should hold
        assert d_con.decision == "hold"
        # Aggressive (threshold 45) should buy (FGI 42<=60, RSI 42<=50 partial, SMA 0, News 20, ext 5)
        # FGI: 42<=60 → 30pts, RSI: 42<=50 → partial 15, SMA: -2.0 > -1.0 → 0, News: 20, ext: 5 = 70 >= 45
        assert d_agg.decision == "buy"
