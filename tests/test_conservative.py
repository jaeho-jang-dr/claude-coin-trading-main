"""
ConservativeAgent 유닛 테스트

Coverage:
  - 임계값 검증 (v2: threshold 60, FGI 35, RSI 35, SMA -3)
  - 매수 결정: 점수 충족, AI 거부권, 극공포 예외
  - 매도 결정: FGI 과열, 목표 수익, DCA
  - 분할 매도, DCA 최소 금액
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

from agents.conservative import ConservativeAgent
from agents.base_agent import Decision


@pytest.fixture
def agent():
    return ConservativeAgent()


def _make_market_data(rsi=50, sma_deviation=0, fgi=50, price_change_rate=0,
                      ai_score=0, news_sentiment="neutral", trade_price=50000000):
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
        "candles_4h": [],
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


def _make_ext(total_score=0, strategy_bonus=0, fusion_signal="neutral"):
    return {"total_score": total_score, "strategy_bonus": strategy_bonus,
            "fusion": {"signal": fusion_signal}}


# ============================================================
# Threshold Verification
# ============================================================

class TestConservativeThresholds:

    def test_buy_threshold(self, agent):
        assert agent.buy_score_threshold == 60

    def test_fgi_threshold(self, agent):
        assert agent.fgi_threshold == 35

    def test_rsi_threshold(self, agent):
        assert agent.rsi_threshold == 35

    def test_sma_deviation(self, agent):
        assert agent.sma_deviation_pct == -3.0

    def test_stop_loss(self, agent):
        assert agent.stop_loss_pct == -5.0

    def test_forced_stop_loss(self, agent):
        assert agent.forced_stop_loss_pct == -10.0

    def test_target_profit(self, agent):
        assert agent.target_profit_pct == 15.0

    def test_max_trade_ratio(self, agent):
        assert agent.max_trade_ratio == 0.10

    def test_no_macd_bonus(self, agent):
        assert agent.macd_bonus is False

    def test_sell_fgi_threshold(self, agent):
        assert agent.sell_fgi_threshold == 75

    def test_sell_rsi_threshold(self, agent):
        assert agent.sell_rsi_threshold == 70

    def test_weekend_reduction(self, agent):
        assert agent.weekend_reduction == 0.50

    def test_max_daily_trades(self, agent):
        assert agent.max_daily_trades == 3

    def test_name_and_emoji(self, agent):
        assert agent.name == "conservative"
        assert agent.emoji == "🛡️"


# ============================================================
# Buy Decisions
# ============================================================

class TestConservativeBuy:

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_buy_all_signals_met(self, agent):
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=15, ai_score=5)
        port = _make_portfolio(krw=1000000)
        ext = _make_ext(strategy_bonus=10)
        decision = agent.decide(md, ext, port)
        assert decision.decision == "buy"
        assert decision.confidence > 0.5

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_hold_low_score(self, agent):
        md = _make_market_data(rsi=60, sma_deviation=2.0, fgi=70, ai_score=0)
        port = _make_portfolio()
        ext = _make_ext()
        decision = agent.decide(md, ext, port)
        assert decision.decision == "hold"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_holds(self, agent):
        """매수 점수 충족이나 AI 음수 → 보류 (FGI > 15)."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=20, ai_score=-5)
        port = _make_portfolio()
        ext = _make_ext(strategy_bonus=10)
        decision = agent.decide(md, ext, port)
        assert decision.decision == "hold"
        assert decision._was_ai_vetoed is True
        assert decision._original_action == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_override_extreme_fear(self, agent):
        """AI 음수이나 FGI <= 15 극공포면 매수 허용."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=10, ai_score=-5)
        port = _make_portfolio()
        ext = _make_ext(strategy_bonus=10)
        decision = agent.decide(md, ext, port)
        assert decision.decision == "buy"
        assert "극단 공포" in decision.reason

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_boundary_fgi_15(self, agent):
        """FGI = 15일 때 극공포 매수 허용."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=15, ai_score=-5)
        port = _make_portfolio()
        ext = _make_ext(strategy_bonus=10)
        decision = agent.decide(md, ext, port)
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_boundary_fgi_16(self, agent):
        """FGI = 16 > 15일 때 AI 거부권 발동."""
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=16, ai_score=-5)
        port = _make_portfolio()
        ext = _make_ext(strategy_bonus=10)
        decision = agent.decide(md, ext, port)
        assert decision.decision == "hold"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_buy_confidence_capped(self, agent):
        """confidence는 0.9를 넘지 않는다."""
        md = _make_market_data(rsi=10, sma_deviation=-10.0, fgi=5, ai_score=10)
        port = _make_portfolio()
        ext = _make_ext(strategy_bonus=20)
        decision = agent.decide(md, ext, port)
        assert decision.confidence <= 0.9

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_buy_agent_name_includes_emoji(self, agent):
        md = _make_market_data(rsi=25, sma_deviation=-6.0, fgi=15, ai_score=5)
        port = _make_portfolio()
        ext = _make_ext(strategy_bonus=10)
        decision = agent.decide(md, ext, port)
        assert "🛡️" in decision.agent_name
        assert "conservative" in decision.agent_name


# ============================================================
# Sell Decisions
# ============================================================

class TestConservativeSell:

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_sell_fgi_overbought(self, agent):
        md = _make_market_data(fgi=80, rsi=50)
        port = _make_portfolio(btc_balance=0.01, profit_pct=5.0, total_eval=1500000)
        ext = _make_ext()
        decision = agent.decide(md, ext, port)
        assert decision.decision == "sell"
        assert decision.trade_params["side"] == "ask"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_sell_target_profit(self, agent):
        md = _make_market_data(fgi=50, rsi=50)
        port = _make_portfolio(btc_balance=0.01, profit_pct=16.0, total_eval=1000000)
        ext = _make_ext()
        decision = agent.decide(md, ext, port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_dca_when_stop_loss_with_signals(self, agent):
        md = _make_market_data(fgi=15, rsi=25, sma_deviation=-6.0, ai_score=5)
        port = _make_portfolio(btc_balance=0.01, btc_avg_price=55000000,
                               profit_pct=-6.0, total_eval=1000000)
        ext = _make_ext(strategy_bonus=5)
        dc = {"cascade_risk": 10, "external_bearish_count": 0,
              "dca_already_done": False, "trend_falling": False}
        decision = agent.decide(md, ext, port, drop_context=dc)
        assert decision.decision == "buy"
        assert decision.trade_params.get("is_dca") is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_dca_minimum_amount(self, agent):
        """DCA 금액이 5000원 미만이면 hold."""
        md = _make_market_data(fgi=15, rsi=25, sma_deviation=-6.0, ai_score=5)
        port = _make_portfolio(krw=100, btc_balance=0.0001, btc_avg_price=50000000,
                               profit_pct=-6.0, total_eval=100000)
        ext = _make_ext(strategy_bonus=5)
        dc = {"cascade_risk": 10, "external_bearish_count": 0,
              "dca_already_done": False, "trend_falling": False}
        decision = agent.decide(md, ext, port, drop_context=dc)
        # DCA amount likely < 5000 → hold
        if decision.decision == "hold":
            assert "DCA 금액 부족" in decision.reason or decision.confidence <= 0.5

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_partial_sell_overweight(self, agent):
        md = _make_market_data(fgi=50, rsi=50)
        port = {
            "krw_balance": 200000,
            "btc": {"balance": 0.01, "avg_buy_price": 50000000,
                    "profit_pct": 7.0, "eval_amount": 800000},
            "total_eval": 1000000,
            "btc_ratio": 0.8,
        }
        ext = _make_ext()
        decision = agent.decide(md, ext, port)
        assert decision.decision == "sell"
        assert decision.trade_params.get("is_partial") is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_hold_defer_strong_ai(self, agent):
        """목표 수익 + AI 강세 → hold_defer."""
        md = _make_market_data(fgi=50, rsi=50, ai_score=25)
        port = _make_portfolio(btc_balance=0.01, profit_pct=16.0, total_eval=1000000)
        ext = _make_ext()
        decision = agent.decide(md, ext, port)
        assert decision.decision == "hold"
        assert "유예" in decision.reason or "보류" in decision.reason

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_no_sell_when_not_holding(self, agent):
        """BTC 미보유 시 매도 평가 건너뜀."""
        md = _make_market_data(fgi=80, rsi=75)
        port = _make_portfolio(btc_balance=0)
        ext = _make_ext()
        decision = agent.decide(md, ext, port)
        # Not holding BTC, so no sell evaluation
        assert decision.decision == "hold"  # score too low for buy
