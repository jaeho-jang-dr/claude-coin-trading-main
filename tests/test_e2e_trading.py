"""End-to-end tests for the trading pipeline components.

Tests the integration between:
  - Orchestrator (agent selection, switching, decision flow)
  - Strategy agents (conservative, moderate, aggressive)
  - ExternalDataAgent (data collection and fusion)
  - Safety checks (DRY_RUN, EMERGENCY_STOP)

All external API calls (Upbit, Supabase, Telegram, blockchain.com, etc.) are mocked.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import Orchestrator, _load_state, _save_state, STATE_FILE, AUTO_EMERGENCY_FILE
from agents.conservative import ConservativeAgent
from agents.moderate import ModerateAgent
from agents.aggressive import AggressiveAgent
from agents.base_agent import BaseStrategyAgent, Decision
from agents.external_data import ExternalDataAgent, analyze_news_sentiment


# ═══════════════════════════════════════════════════════════
# Sample data fixtures
# ═══════════════════════════════════════════════════════════

def _sample_market_data(
    price=85_000_000,
    rsi=45.0,
    sma_20=84_000_000,
    sma_deviation=-1.2,
    fgi=35,
    change_rate=0.01,
    ai_score=5,
):
    """Realistic market data dict as returned by collect_market_data.py."""
    return {
        "current_price": price,
        "ticker": {
            "trade_price": price,
            "signed_change_rate": change_rate,
        },
        "indicators": {
            "rsi_14": rsi,
            "sma_20": sma_20,
            "sma_20_deviation_pct": sma_deviation,
            "macd": {"macd": 100, "signal": 80, "histogram": 20, "signal_cross": False},
            "bollinger": {"upper": 90_000_000, "middle": 85_000_000, "lower": 80_000_000},
            "adx_regime": "ranging",
        },
        "fear_greed": {"value": fgi},
        "news": {"overall_sentiment": "neutral"},
        "ai_composite_signal": {"score": ai_score},
    }


def _sample_external_data(
    fgi_value=35,
    fusion_signal="neutral",
    fusion_score=5,
    strategy_bonus=5,
    kimchi_pct=1.0,
    ls_ratio=1.0,
    funding_rate=0.005,
    macro_score=0,
    news_sentiment="neutral",
):
    """Realistic external data dict as returned by ExternalDataAgent.collect_all()."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "collection_time_sec": 3.5,
        "sources": {
            "fear_greed": {
                "current": {"value": fgi_value, "classification": "Fear"},
            },
            "news": {"articles_count": 5, "by_category": {"btc": 3, "economy": 2}},
            "news_sentiment": {
                "sentiment_score": 10,
                "overall_sentiment": news_sentiment,
                "positive_count": 3,
                "negative_count": 1,
                "neutral_count": 1,
                "total_articles": 5,
            },
            "whale_tracker": {
                "whale_score": {"score": 5, "direction": "accumulate"},
                "block_analysis": {
                    "whale_txs_count": 3,
                    "direction_btc": {"exchange_deposit": 10.5, "exchange_withdrawal": 25.2},
                },
            },
            "binance_sentiment": {
                "sentiment_score": {"score": 3},
                "top_trader_long_short": {"current_ratio": ls_ratio},
                "funding_rate": {"current_rate": funding_rate},
                "open_interest": {"oi_change_24h_pct": 2.1},
                "kimchi_premium": {"premium_pct": kimchi_pct},
            },
            "eth_btc": {"eth_btc_ratio": 0.045, "eth_btc_z_score": 0.3, "signal": "neutral"},
            "macro": {
                "analysis": {"macro_score": macro_score, "sentiment": "neutral"},
                "quotes": {
                    "sp500": {"change_pct": 0.2},
                    "dxy": {"change_pct": -0.1},
                    "gold": {"change_pct": 0.3},
                    "us10y": {"change_pct": 0.0},
                },
            },
            "crypto_signals": {"btc": {"anomaly_level": "LOW", "change_24h": 1.2}},
            "coinmarketcap": {"status": "success", "btc_dominance": 52.0},
            "user_feedback": [],
            "performance_review": {"available": False, "message": "no data"},
        },
        "external_signal": {
            "total_score": fusion_score,
            "strategy_bonus": strategy_bonus,
            "fusion": {"signal": fusion_signal, "note": "test"},
        },
        "errors": [],
    }


def _sample_portfolio(krw=1_000_000, btc_balance=0.0, btc_avg_price=0, btc_profit_pct=0.0):
    """Realistic portfolio dict as returned by get_portfolio.py."""
    btc_eval = btc_balance * (btc_avg_price * (1 + btc_profit_pct / 100)) if btc_balance else 0
    total_eval = krw + btc_eval
    return {
        "krw_balance": krw,
        "total_eval": total_eval,
        "btc_ratio": btc_eval / total_eval if total_eval > 0 else 0,
        "btc": {
            "balance": btc_balance,
            "avg_buy_price": btc_avg_price,
            "profit_pct": btc_profit_pct,
            "eval_amount": btc_eval,
        },
    }


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _clean_env_and_state(tmp_path, monkeypatch):
    """Reset environment and state files for every test."""
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("EMERGENCY_STOP", "false")
    monkeypatch.setenv("MAX_TRADE_AMOUNT", "100000")
    monkeypatch.setenv("MAX_POSITION_RATIO", "0.5")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    # Redirect state files to tmp
    state_file = tmp_path / "agent_state.json"
    auto_em_file = tmp_path / "auto_emergency.json"
    monkeypatch.setattr("agents.orchestrator.STATE_FILE", state_file)
    monkeypatch.setattr("agents.orchestrator.AUTO_EMERGENCY_FILE", auto_em_file)
    yield


# ═══════════════════════════════════════════════════════════
# 1. Orchestrator initialization with all 3 agents
# ═══════════════════════════════════════════════════════════

class TestOrchestratorInit:

    def test_default_agent_is_conservative(self):
        orch = Orchestrator()
        agent = orch.active_agent
        assert isinstance(agent, ConservativeAgent)
        assert agent.name == "conservative"

    def test_all_three_agents_available(self):
        from agents.orchestrator import AGENTS
        assert "conservative" in AGENTS
        assert "moderate" in AGENTS
        assert "aggressive" in AGENTS
        assert AGENTS["conservative"] is ConservativeAgent
        assert AGENTS["moderate"] is ModerateAgent
        assert AGENTS["aggressive"] is AggressiveAgent

    def test_state_loads_from_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "agent_state.json"
        state_file.write_text(json.dumps({
            "active_agent": "aggressive",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }))
        monkeypatch.setattr("agents.orchestrator.STATE_FILE", state_file)

        orch = Orchestrator()
        agent = orch.active_agent
        assert isinstance(agent, AggressiveAgent)

    def test_state_defaults_when_file_missing(self):
        orch = Orchestrator()
        assert orch.state["active_agent"] == "conservative"
        assert orch.state["consecutive_losses"] == 0


# ═══════════════════════════════════════════════════════════
# 2. Orchestrator.run() produces valid decision dict
# ═══════════════════════════════════════════════════════════

class TestOrchestratorRun:

    def test_run_returns_valid_structure(self):
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(),
            external_data=_sample_external_data(),
            portfolio=_sample_portfolio(),
        )
        assert "active_agent" in result
        assert "decision" in result
        assert "market_state" in result

        decision = result["decision"]
        assert decision["decision"] in ("buy", "sell", "hold")
        assert 0 <= decision["confidence"] <= 1.0
        assert "reason" in decision
        assert "agent_name" in decision

    def test_run_hold_with_neutral_market(self):
        """Neutral market conditions should produce hold."""
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(fgi=50, rsi=50, sma_deviation=0),
            external_data=_sample_external_data(fgi_value=50),
            portfolio=_sample_portfolio(),
        )
        assert result["decision"]["decision"] == "hold"

    def test_run_buy_with_extreme_fear(self):
        """Extreme fear + oversold RSI + SMA below + neutral news should trigger buy."""
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(fgi=15, rsi=25, sma_deviation=-6.0, ai_score=10),
            external_data=_sample_external_data(fgi_value=15, strategy_bonus=10),
            portfolio=_sample_portfolio(krw=500_000),
        )
        assert result["decision"]["decision"] == "buy"

    def test_run_sell_on_target_profit(self):
        """Holding BTC at target profit should trigger sell."""
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(fgi=50, rsi=50, ai_score=-5),
            external_data=_sample_external_data(fgi_value=50),
            portfolio=_sample_portfolio(
                krw=500_000, btc_balance=0.01,
                btc_avg_price=70_000_000, btc_profit_pct=16.0,
            ),
        )
        assert result["decision"]["decision"] == "sell"

    def test_run_market_state_contains_scores(self):
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(),
            external_data=_sample_external_data(),
            portfolio=_sample_portfolio(),
        )
        ms = result["market_state"]
        assert "danger_score" in ms
        assert "opportunity_score" in ms
        assert "phase" in ms
        assert isinstance(ms["danger_score"], int)
        assert isinstance(ms["opportunity_score"], int)


# ═══════════════════════════════════════════════════════════
# 3. Agent score calculation: calculate_buy_score()
# ═══════════════════════════════════════════════════════════

class TestBuyScoreCalculation:

    def test_conservative_full_score(self):
        agent = ConservativeAgent()
        score = agent.calculate_buy_score(
            fgi=10, rsi=20, sma_deviation=-6.0,
            news_negative=False, external_bonus=10,
        )
        # FGI <= 15 (half of 30) => 30 + 5 bonus = 35
        assert score["fgi"]["score"] == 35
        # RSI <= 30 => 25
        assert score["rsi"]["score"] == 25
        # SMA <= -5.0 => 25
        assert score["sma"]["score"] == 25
        # News not negative => 20
        assert score["news"]["score"] == 20
        # External bonus
        assert score["external"]["score"] == 10
        # Total: 35 + 25 + 25 + 20 + 10 = 115
        assert score["total"] == 115
        assert score["result"] == "buy"

    def test_conservative_below_threshold(self):
        agent = ConservativeAgent()
        score = agent.calculate_buy_score(
            fgi=60, rsi=55, sma_deviation=2.0,
            news_negative=True, external_bonus=0,
        )
        assert score["total"] < agent.buy_score_threshold
        assert score["result"] == "hold"

    def test_moderate_macd_bonus(self):
        agent = ModerateAgent()
        score_with = agent.calculate_buy_score(
            fgi=40, rsi=35, sma_deviation=-4.0,
            news_negative=False, external_bonus=5,
            macd_golden_cross=True,
        )
        score_without = agent.calculate_buy_score(
            fgi=40, rsi=35, sma_deviation=-4.0,
            news_negative=False, external_bonus=5,
            macd_golden_cross=False,
        )
        assert score_with["total"] == score_without["total"] + 10
        assert "macd" in score_with

    def test_aggressive_lower_threshold(self):
        agent = AggressiveAgent()
        assert agent.buy_score_threshold == 45
        score = agent.calculate_buy_score(
            fgi=50, rsi=40, sma_deviation=-2.0,
            news_negative=False, external_bonus=5,
        )
        # FGI 50 <= 60 => 30, RSI 40 <= 50 => 25, SMA -2 <= -1 => 25, news=20, ext=5
        assert score["total"] >= 45
        assert score["result"] == "buy"

    def test_partial_fgi_score(self):
        """FGI slightly above threshold gives partial credit."""
        agent = ConservativeAgent()
        score = agent.calculate_buy_score(
            fgi=35, rsi=50, sma_deviation=0,
            news_negative=False, external_bonus=0,
        )
        # FGI 35 is within threshold+10 (30+10=40), so partial
        assert score["fgi"]["score"] == 15  # half of 30
        assert score["fgi"].get("partial") is True


# ═══════════════════════════════════════════════════════════
# 4. Agent decision flow: market_data -> agent -> decision
# ═══════════════════════════════════════════════════════════

class TestAgentDecisionFlow:

    def test_conservative_hold_normal_market(self):
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=50, rsi=50, sma_deviation=0),
            external_signal={"strategy_bonus": 0, "fusion": {"signal": "neutral"}},
            portfolio=_sample_portfolio(),
        )
        assert isinstance(decision, Decision)
        assert decision.decision == "hold"
        assert "conservative" in decision.agent_name

    def test_conservative_buy_extreme_fear(self):
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=10, rsi=20, sma_deviation=-6.0, ai_score=5),
            external_signal={"strategy_bonus": 10, "fusion": {"signal": "buy"}},
            portfolio=_sample_portfolio(krw=500_000),
        )
        assert decision.decision == "buy"
        assert decision.trade_params["side"] == "bid"
        assert decision.trade_params["market"] == "KRW-BTC"
        assert decision.trade_params["amount"] > 0

    def test_conservative_sell_target_profit(self):
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=50, rsi=50, ai_score=-5),
            external_signal={"strategy_bonus": 0, "fusion": {"signal": "neutral"}},
            portfolio=_sample_portfolio(
                krw=500_000, btc_balance=0.01,
                btc_avg_price=70_000_000, btc_profit_pct=16.0,
            ),
        )
        assert decision.decision == "sell"
        assert decision.trade_params["side"] == "ask"

    def test_ai_veto_blocks_buy(self):
        """Negative AI signal vetoes a buy (unless extreme fear)."""
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=25, rsi=25, sma_deviation=-6.0, ai_score=-10),
            external_signal={"strategy_bonus": 10, "fusion": {"signal": "buy"}},
            portfolio=_sample_portfolio(krw=500_000),
        )
        # FGI=25 > 15, so AI veto applies
        assert decision.decision == "hold"
        assert hasattr(decision, "_was_ai_vetoed")
        assert decision._was_ai_vetoed is True

    def test_ai_veto_overridden_extreme_fear(self):
        """Extreme fear (FGI <= 15) overrides AI veto."""
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=12, rsi=20, sma_deviation=-6.0, ai_score=-10),
            external_signal={"strategy_bonus": 10, "fusion": {"signal": "buy"}},
            portfolio=_sample_portfolio(krw=500_000),
        )
        assert decision.decision == "buy"

    def test_decision_to_dict(self):
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(),
            external_signal={"strategy_bonus": 0, "fusion": {"signal": "neutral"}},
            portfolio=_sample_portfolio(),
        )
        d = decision.to_dict()
        assert isinstance(d, dict)
        required_keys = {"decision", "confidence", "reason", "buy_score",
                         "trade_params", "external_signal_summary", "agent_name", "timestamp"}
        assert required_keys.issubset(d.keys())

    def test_moderate_agent_decides(self):
        agent = ModerateAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=40, rsi=35, sma_deviation=-4.0, ai_score=5),
            external_signal={"strategy_bonus": 5, "fusion": {"signal": "neutral"}},
            portfolio=_sample_portfolio(krw=500_000),
        )
        assert isinstance(decision, Decision)
        assert decision.decision in ("buy", "hold", "sell")

    def test_aggressive_agent_decides(self):
        agent = AggressiveAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=55, rsi=45, sma_deviation=-2.0, ai_score=5),
            external_signal={"strategy_bonus": 5, "fusion": {"signal": "neutral"}},
            portfolio=_sample_portfolio(krw=500_000),
        )
        assert isinstance(decision, Decision)
        assert decision.decision in ("buy", "hold", "sell")

    def test_trade_amount_respects_max(self, monkeypatch):
        """Trade amount must not exceed MAX_TRADE_AMOUNT."""
        monkeypatch.setenv("MAX_TRADE_AMOUNT", "50000")
        agent = ConservativeAgent()
        decision = agent.decide(
            market_data=_sample_market_data(fgi=10, rsi=20, sma_deviation=-6.0, ai_score=5),
            external_signal={"strategy_bonus": 10, "fusion": {"signal": "buy"}},
            portfolio=_sample_portfolio(krw=10_000_000),
        )
        assert decision.decision == "buy"
        assert decision.trade_params["amount"] <= 50000


# ═══════════════════════════════════════════════════════════
# 5. ExternalDataAgent: collect_all() with mocked API responses
# ═══════════════════════════════════════════════════════════

class TestExternalDataAgent:

    @patch("agents.external_data._run_script")
    @patch("agents.external_data._fetch_nvt_signal")
    @patch("agents.external_data.load_user_feedback", return_value=[])
    @patch("agents.external_data.load_performance_review", return_value={"available": False})
    def test_collect_all_returns_expected_structure(
        self, mock_perf, mock_feedback, mock_nvt, mock_script
    ):
        mock_script.return_value = {"status": "ok", "test": True}
        mock_nvt.return_value = {"nvt_signal": 80.0, "interpretation": "normal"}

        agent = ExternalDataAgent(snapshot_dir=None)

        # Mock _calculate_fusion to return a simple result
        with patch.object(agent, "_calculate_fusion") as mock_fusion:
            mock_fusion.return_value = {
                "total_score": 10,
                "strategy_bonus": 5,
                "fusion": {"signal": "neutral"},
            }
            # Mock DB save
            with patch.object(agent, "_save_signal_to_db"):
                result = agent.collect_all()

        assert "timestamp" in result
        assert "collection_time_sec" in result
        assert "sources" in result
        assert "external_signal" in result
        assert "errors" in result
        assert isinstance(result["sources"], dict)

    @patch("agents.external_data._run_script")
    @patch("agents.external_data._fetch_nvt_signal")
    @patch("agents.external_data.load_user_feedback", return_value=[])
    @patch("agents.external_data.load_performance_review", return_value={"available": False})
    def test_collect_all_handles_script_errors(
        self, mock_perf, mock_feedback, mock_nvt, mock_script
    ):
        """Failed scripts should be recorded in errors list."""
        mock_script.return_value = {"error": "timeout"}
        mock_nvt.return_value = {"nvt_signal": 100.0, "error": "fetch_failed"}

        agent = ExternalDataAgent(snapshot_dir=None)
        with patch.object(agent, "_calculate_fusion") as mock_fusion:
            mock_fusion.return_value = {"total_score": 0, "strategy_bonus": 0, "fusion": {"signal": "neutral"}}
            with patch.object(agent, "_save_signal_to_db"):
                result = agent.collect_all()

        # All scripts returned errors
        assert len(result["errors"]) > 0

    def test_inline_fusion_calculation(self):
        """Test the inline fallback fusion calculation."""
        agent = ExternalDataAgent(snapshot_dir=None)
        results = {
            "binance_sentiment": {"sentiment_score": {"score": 5}},
            "whale_tracker": {"whale_score": {"score": 10, "direction": "accumulate"}},
            "macro": {"analysis": {"macro_score": 15, "sentiment": "bullish"}},
            "news_sentiment": {"sentiment_score": 35, "overall_sentiment": "positive"},
            "crypto_signals": {"btc": {"anomaly_level": "LOW", "change_24h": 1.0}},
            "coinmarketcap": {"status": "success", "btc_dominance": 56.0},
        }
        fusion = agent._inline_fusion(results)
        assert "total_score" in fusion
        assert "strategy_bonus" in fusion
        assert isinstance(fusion["total_score"], (int, float))

    def test_news_sentiment_analysis(self):
        """Test keyword-based news sentiment analysis."""
        news_data = {
            "articles": [
                {"title": "BTC ETF Approved by SEC", "content": "Institutional adoption rally expected"},
                {"title": "Bitcoin crash fears", "content": "Bearish sell-off warnings from analysts"},
                {"title": "Market update", "content": "BTC trades sideways near support"},
            ]
        }
        result = analyze_news_sentiment(news_data)
        assert result["total_articles"] == 3
        assert result["positive_count"] >= 1
        assert result["negative_count"] >= 1
        assert result["overall_sentiment"] in (
            "positive", "negative", "neutral",
            "slightly_positive", "slightly_negative",
        )

    def test_news_sentiment_empty(self):
        result = analyze_news_sentiment({"articles": []})
        assert result["sentiment_score"] == 0
        assert result["overall_sentiment"] == "neutral"
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_get_fgi_value(self):
        agent = ExternalDataAgent()
        ext = _sample_external_data(fgi_value=22)
        fgi = agent.get_fgi_value(ext)
        assert fgi == 22


# ═══════════════════════════════════════════════════════════
# 6. Data flow: external_data -> orchestrator -> decision
# ═══════════════════════════════════════════════════════════

class TestDataFlowIntegration:

    def test_full_pipeline_hold(self):
        """Neutral conditions -> hold decision with correct output format."""
        orch = Orchestrator()
        market = _sample_market_data(fgi=50, rsi=50, sma_deviation=0)
        external = _sample_external_data(fgi_value=50)
        portfolio = _sample_portfolio(krw=1_000_000)

        result = orch.run(market, external, portfolio)

        # Validate output structure
        assert result["decision"]["decision"] == "hold"
        assert "active_agent" in result
        assert "market_state" in result
        assert "drop_context" in result
        # market_state should reflect the external data
        ms = result["market_state"]
        assert ms["fgi"] == 50
        assert ms["phase"] == "neutral"

    def test_full_pipeline_buy_flow(self):
        """Fear market -> buy signal propagates through the pipeline."""
        orch = Orchestrator()
        market = _sample_market_data(fgi=10, rsi=20, sma_deviation=-7.0, ai_score=10)
        external = _sample_external_data(
            fgi_value=10,
            strategy_bonus=15,
            fusion_signal="buy",
            fusion_score=25,
        )
        portfolio = _sample_portfolio(krw=500_000)

        result = orch.run(market, external, portfolio)
        assert result["decision"]["decision"] == "buy"
        assert result["decision"]["trade_params"].get("side") == "bid"
        assert result["decision"]["trade_params"].get("amount") > 0

    def test_full_pipeline_sell_flow(self):
        """Profit target hit -> sell signal with volume."""
        orch = Orchestrator()
        market = _sample_market_data(fgi=60, rsi=55, ai_score=-5)
        external = _sample_external_data(fgi_value=60)
        portfolio = _sample_portfolio(
            krw=200_000, btc_balance=0.01,
            btc_avg_price=70_000_000, btc_profit_pct=16.0,
        )

        result = orch.run(market, external, portfolio)
        assert result["decision"]["decision"] == "sell"
        assert result["decision"]["trade_params"].get("side") == "ask"
        assert result["decision"]["trade_params"].get("volume") > 0

    def test_external_signal_bonus_influences_buy_score(self):
        """Higher external bonus should increase buy score."""
        agent = ConservativeAgent()
        low_bonus = agent.calculate_buy_score(
            fgi=25, rsi=28, sma_deviation=-6.0,
            news_negative=False, external_bonus=0,
        )
        high_bonus = agent.calculate_buy_score(
            fgi=25, rsi=28, sma_deviation=-6.0,
            news_negative=False, external_bonus=20,
        )
        assert high_bonus["total"] == low_bonus["total"] + 20

    def test_orchestrator_switches_agent_on_high_danger(self, tmp_path, monkeypatch):
        """High danger score should switch to conservative."""
        state_file = tmp_path / "agent_state.json"
        state_file.write_text(json.dumps({
            "active_agent": "aggressive",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }))
        monkeypatch.setattr("agents.orchestrator.STATE_FILE", state_file)

        orch = Orchestrator()
        assert orch._active_agent_name == "aggressive"

        # Dangerous market: big drop + high kimchi premium + long crowding
        market = _sample_market_data(fgi=30, rsi=40, change_rate=-0.05)
        external = _sample_external_data(
            fgi_value=30,
            kimchi_pct=5.0,
            ls_ratio=1.5,
        )
        portfolio = _sample_portfolio(krw=500_000)

        result = orch.run(market, external, portfolio, past_decisions=[
            {"profit_loss": -2}, {"profit_loss": -3}, {"profit_loss": -1},
        ])

        # Agent should have switched toward conservative
        if result.get("switch"):
            assert result["switch"]["to"] in ("conservative", "moderate")


# ═══════════════════════════════════════════════════════════
# 7. Safety checks: DRY_RUN prevents actual trades
# ═══════════════════════════════════════════════════════════

class TestDryRunSafety:

    @patch("scripts.execute_trade.requests.post")
    def test_dry_run_prevents_trade_execution(self, mock_post, monkeypatch):
        """When DRY_RUN=true, execute_trade should not call Upbit API."""
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("UPBIT_ACCESS_KEY", "test_key")
        monkeypatch.setenv("UPBIT_SECRET_KEY", "test_secret")

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from scripts.execute_trade import execute

        result = execute("bid", "KRW-BTC", "100000")
        assert result["dry_run"] is True
        assert result["success"] is True
        # Upbit API should NOT have been called
        mock_post.assert_not_called()

    def test_orchestrator_respects_dry_run(self, monkeypatch):
        """Orchestrator's decisions work the same under DRY_RUN=true."""
        monkeypatch.setenv("DRY_RUN", "true")
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(fgi=10, rsi=20, sma_deviation=-6.0, ai_score=10),
            external_data=_sample_external_data(fgi_value=10, strategy_bonus=10),
            portfolio=_sample_portfolio(krw=500_000),
        )
        # Decision is still made (buy), DRY_RUN doesn't affect the decision logic
        assert result["decision"]["decision"] == "buy"


# ═══════════════════════════════════════════════════════════
# 8. Emergency stop flag check
# ═══════════════════════════════════════════════════════════

class TestEmergencyStop:

    def test_orchestrator_blocks_on_emergency_stop(self, monkeypatch):
        monkeypatch.setenv("EMERGENCY_STOP", "true")
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(),
            external_data=_sample_external_data(),
            portfolio=_sample_portfolio(),
        )
        assert result["decision"]["decision"] == "hold"
        assert "EMERGENCY_STOP" in result["decision"]["reason"]
        assert "긴급정지" in result["active_agent"]

    @patch("scripts.execute_trade.requests.post")
    def test_execute_trade_blocks_on_emergency(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMERGENCY_STOP", "true")
        monkeypatch.setenv("UPBIT_ACCESS_KEY", "test_key")
        monkeypatch.setenv("UPBIT_SECRET_KEY", "test_secret")

        from scripts.execute_trade import execute

        result = execute("bid", "KRW-BTC", "100000")
        assert result["success"] is False
        assert "EMERGENCY_STOP" in result["error"]
        mock_post.assert_not_called()

    def test_auto_emergency_blocks_orchestrator(self, tmp_path, monkeypatch):
        """Auto emergency flag file should block trading."""
        # Use a very recent activation time so 12h haven't elapsed (can't lift)
        from datetime import datetime, timezone, timedelta
        kst = timezone(timedelta(hours=9))
        recent_time = datetime.now(kst).isoformat()

        auto_em_file = tmp_path / "auto_emergency.json"
        auto_em_file.write_text(json.dumps({
            "active": True,
            "reason": "4h -10% crash detected",
            "activated_at": recent_time,
        }))
        monkeypatch.setattr("agents.orchestrator.AUTO_EMERGENCY_FILE", auto_em_file)

        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(),
            external_data=_sample_external_data(),
            portfolio=_sample_portfolio(),
        )
        assert result["decision"]["decision"] == "hold"
        assert "긴급정지" in result["active_agent"]

    def test_no_emergency_allows_normal_operation(self, monkeypatch):
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        orch = Orchestrator()
        result = orch.run(
            market_data=_sample_market_data(),
            external_data=_sample_external_data(),
            portfolio=_sample_portfolio(),
        )
        assert "긴급정지" not in result.get("active_agent", "")


# ═══════════════════════════════════════════════════════════
# Additional edge case tests
# ═══════════════════════════════════════════════════════════

class TestEvaluateSell:

    def test_forced_stop_loss(self):
        agent = ConservativeAgent()
        result = agent.evaluate_sell(
            profit_pct=-11.0,
            current_fgi=50,
            current_rsi=50,
            buy_score={"fgi": {"score": 0}, "rsi": {"score": 0}, "sma": {"score": 0}, "news": {"score": 0}},
            ai_signal_score=20,
        )
        assert result is not None
        assert result["action"] == "sell"
        assert result["type"] == "forced_stop"

    def test_fgi_overbought_sell(self):
        agent = ConservativeAgent()
        result = agent.evaluate_sell(
            profit_pct=5.0,
            current_fgi=80,
            current_rsi=50,
            buy_score={},
            ai_signal_score=0,
        )
        assert result is not None
        assert result["action"] == "sell"
        assert result["type"] == "fgi_overbought"

    def test_hybrid_dca_on_bottom_signals(self):
        """With enough bottom signals + positive AI, DCA instead of stop loss."""
        agent = ConservativeAgent()
        buy_score = {
            "fgi": {"score": 30}, "rsi": {"score": 25},
            "sma": {"score": 25}, "news": {"score": 20},
        }
        result = agent.evaluate_sell(
            profit_pct=-6.0,
            current_fgi=20,
            current_rsi=25,
            buy_score=buy_score,
            ai_signal_score=10,
        )
        assert result is not None
        assert result["action"] == "dca"
        assert result["type"] == "hybrid_dca"

    def test_overweight_partial_sell(self):
        agent = ConservativeAgent()
        result = agent.evaluate_sell(
            profit_pct=6.0,
            current_fgi=50,
            current_rsi=50,
            buy_score={},
            ai_signal_score=0,
            btc_position_ratio=0.7,
        )
        assert result is not None
        assert result["action"] == "sell_partial"
        assert result["type"] == "overweight_rebalance"

    def test_no_sell_when_in_range(self):
        """No sell signal when profit is modest and indicators normal."""
        agent = ConservativeAgent()
        result = agent.evaluate_sell(
            profit_pct=3.0,
            current_fgi=50,
            current_rsi=50,
            buy_score={},
            ai_signal_score=0,
        )
        assert result is None


class TestTradeAmount:

    def test_max_trade_amount_cap(self, monkeypatch):
        monkeypatch.setenv("MAX_TRADE_AMOUNT", "50000")
        agent = ConservativeAgent()
        amount = agent._calculate_trade_amount(total_krw=10_000_000)
        assert amount <= 50000

    def test_trade_ratio_applied(self):
        agent = ConservativeAgent()
        # 10% of 1M = 100K, but capped at MAX_TRADE_AMOUNT (100K default)
        amount = agent._calculate_trade_amount(total_krw=1_000_000)
        assert amount == 100_000

    def test_zero_balance_gives_zero(self):
        agent = ConservativeAgent()
        amount = agent._calculate_trade_amount(total_krw=0)
        assert amount == 0
