"""
ModerateAgent 유닛 테스트

Coverage:
  - 임계값 검증 (v2: threshold 50, FGI 45, RSI 40, SMA -2)
  - MACD 골든크로스 보너스
  - AI 거부권 (FGI > 20)
  - 매도: 낮은 FGI/RSI 임계
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

from agents.moderate import ModerateAgent


@pytest.fixture
def agent():
    return ModerateAgent()


def _md(rsi=50, sma_deviation=0, fgi=50, ai_score=0,
        macd_histogram=0, signal_cross=False, news_sentiment="neutral",
        trade_price=50000000):
    return {
        "indicators": {
            "rsi_14": rsi, "sma_20": trade_price,
            "sma_20_deviation_pct": sma_deviation,
            "macd": {"histogram": macd_histogram, "signal_cross": signal_cross},
            "bollinger": {},
        },
        "ticker": {"trade_price": trade_price, "signed_change_rate": 0},
        "fear_greed": {"value": fgi},
        "news": {"overall_sentiment": news_sentiment},
        "ai_composite_signal": {"score": ai_score},
        "current_price": trade_price,
        "candles_4h": [],
    }


def _port(krw=1000000, btc_balance=0, profit_pct=0, total_eval=1000000):
    btc_eval = btc_balance * 50000000 if btc_balance > 0 else 0
    return {
        "krw_balance": krw,
        "btc": {"balance": btc_balance, "avg_buy_price": 50000000,
                "profit_pct": profit_pct, "eval_amount": btc_eval},
        "total_eval": total_eval,
        "btc_ratio": btc_eval / total_eval if total_eval > 0 else 0,
    }


def _ext(strategy_bonus=0):
    return {"total_score": 0, "strategy_bonus": strategy_bonus,
            "fusion": {"signal": "neutral"}}


# ============================================================
# Thresholds
# ============================================================

class TestModerateThresholds:

    def test_buy_threshold(self, agent):
        assert agent.buy_score_threshold == 50

    def test_fgi_threshold(self, agent):
        assert agent.fgi_threshold == 45

    def test_rsi_threshold(self, agent):
        assert agent.rsi_threshold == 40

    def test_sma_deviation(self, agent):
        assert agent.sma_deviation_pct == -2.0

    def test_target_profit(self, agent):
        assert agent.target_profit_pct == 10.0

    def test_stop_loss(self, agent):
        assert agent.stop_loss_pct == -5.0

    def test_forced_stop_loss(self, agent):
        assert agent.forced_stop_loss_pct == -10.0

    def test_macd_bonus_enabled(self, agent):
        assert agent.macd_bonus is True

    def test_max_trade_ratio(self, agent):
        assert agent.max_trade_ratio == 0.15

    def test_max_daily_trades(self, agent):
        assert agent.max_daily_trades == 5

    def test_weekend_reduction(self, agent):
        assert agent.weekend_reduction == 0.30

    def test_sell_fgi_threshold(self, agent):
        assert agent.sell_fgi_threshold == 70

    def test_sell_rsi_threshold(self, agent):
        assert agent.sell_rsi_threshold == 65

    def test_name_and_emoji(self, agent):
        assert agent.name == "moderate"
        assert agent.emoji == "⚖️"


# ============================================================
# Buy with MACD Bonus
# ============================================================

class TestModerateBuy:

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_buy_with_macd_golden_cross(self, agent):
        """MACD 골든크로스 보너스로 매수 점수 도달."""
        md = _md(rsi=38, sma_deviation=-4.0, fgi=40, ai_score=5,
                 macd_histogram=1.0, signal_cross=True)
        decision = agent.decide(md, _ext(strategy_bonus=5), _port())
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_buy_without_macd_still_possible(self, agent):
        """MACD 없이도 다른 점수로 매수 가능."""
        md = _md(rsi=30, sma_deviation=-5.0, fgi=30, ai_score=5)
        decision = agent.decide(md, _ext(strategy_bonus=10), _port())
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_macd_no_golden_cross_no_bonus(self, agent):
        """MACD 히스토그램 양수이나 시그널 크로스 없으면 보너스 없음."""
        score = agent.calculate_buy_score(
            fgi=80, rsi=70, sma_deviation=5.0,
            news_negative=True, external_bonus=0,
            macd_golden_cross=False,
        )
        assert "macd" not in score

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_hold_low_score(self, agent):
        md = _md(rsi=60, sma_deviation=2.0, fgi=70, ai_score=0)
        decision = agent.decide(md, _ext(), _port())
        assert decision.decision == "hold"


# ============================================================
# AI Veto (Moderate)
# ============================================================

class TestModerateAiVeto:

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_fgi_above_20(self, agent):
        """AI 음수 + FGI > 20 → 보류."""
        md = _md(rsi=35, sma_deviation=-4.0, fgi=30, ai_score=-3)
        decision = agent.decide(md, _ext(strategy_bonus=10), _port())
        assert decision.decision == "hold"
        assert decision._was_ai_vetoed is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_override_fgi_20(self, agent):
        """FGI <= 20 → AI 거부권 무시."""
        md = _md(rsi=35, sma_deviation=-4.0, fgi=18, ai_score=-3)
        decision = agent.decide(md, _ext(strategy_bonus=10), _port())
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_boundary_fgi_20(self, agent):
        """FGI = 20 → 매수 허용."""
        md = _md(rsi=35, sma_deviation=-4.0, fgi=20, ai_score=-3)
        decision = agent.decide(md, _ext(strategy_bonus=10), _port())
        assert decision.decision == "buy"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_veto_boundary_fgi_21(self, agent):
        """FGI = 21 > 20 → AI 거부권 발동."""
        md = _md(rsi=35, sma_deviation=-4.0, fgi=21, ai_score=-3)
        decision = agent.decide(md, _ext(strategy_bonus=10), _port())
        assert decision.decision == "hold"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_ai_positive_no_veto(self, agent):
        """AI 양수면 거부권 없음."""
        md = _md(rsi=35, sma_deviation=-4.0, fgi=30, ai_score=5)
        decision = agent.decide(md, _ext(strategy_bonus=10), _port())
        assert decision.decision == "buy"


# ============================================================
# Sell Decisions
# ============================================================

class TestModerateSell:

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_sell_lower_fgi_threshold(self, agent):
        """Moderate: 매도 FGI 임계(70)가 Conservative(75)보다 낮음."""
        md = _md(fgi=72, rsi=50)
        port = _port(btc_balance=0.01, profit_pct=3.0, total_eval=1000000)
        decision = agent.decide(md, _ext(), port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_sell_lower_rsi_threshold(self, agent):
        """Moderate: 매도 RSI 임계(65)."""
        md = _md(fgi=50, rsi=66)
        port = _port(btc_balance=0.01, profit_pct=3.0, total_eval=1000000)
        decision = agent.decide(md, _ext(), port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_target_profit_10pct(self, agent):
        """목표 수익 10%."""
        md = _md(fgi=50, rsi=50)
        port = _port(btc_balance=0.01, profit_pct=11.0, total_eval=1000000)
        decision = agent.decide(md, _ext(), port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_forced_stop_at_minus_10(self, agent):
        md = _md(fgi=50, rsi=50)
        port = _port(btc_balance=0.01, profit_pct=-11.0, total_eval=1000000)
        decision = agent.decide(md, _ext(), port)
        assert decision.decision == "sell"

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "100000"})
    def test_dca_with_signals(self, agent):
        """손절선 + 바닥 시그널 → DCA."""
        md = _md(fgi=15, rsi=25, sma_deviation=-6.0, ai_score=5)
        port = _port(btc_balance=0.01, profit_pct=-6.0, total_eval=1000000)
        ext = _ext(strategy_bonus=5)
        dc = {"cascade_risk": 10, "external_bearish_count": 0,
              "dca_already_done": False, "trend_falling": False}
        decision = agent.decide(md, ext, port, drop_context=dc)
        assert decision.decision == "buy"
        assert decision.trade_params.get("is_dca") is True

    @patch.dict(os.environ, {"MAX_TRADE_AMOUNT": "500000"})
    def test_trade_ratio_15pct(self, agent):
        """1회 매매 15%."""
        with patch.object(agent, '_is_weekend', return_value=False):
            amount = agent._calculate_trade_amount(1000000)
            assert amount == 150000
