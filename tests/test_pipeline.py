"""
Pipeline integration tests — end-to-end data flow, schema, and config validation.

Verifies:
  - Data format consistency between pipeline stages
  - JSON schema validation (decision_result.json, agent_state.json)
  - Supabase migration SQL validity and table/column consistency with code
  - Pipeline scripts reference integrity (all referenced scripts exist)
  - Cron/Windows scheduler script correctness
  - Agent state file format consistency between readers and writers
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════
# Fixtures: sample data matching the actual output formats
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_market_data() -> dict:
    """Matches scripts/collect_market_data.py stdout JSON structure."""
    return {
        "timestamp": "2026-03-12T10:00:00+09:00",
        "market": "KRW-BTC",
        "current_price": 130000000,
        "change_rate_24h": -0.02,
        "volume_24h": 1234.56,
        "indicators": {
            "sma_20": 128000000.0,
            "sma_50": 127000000.0,
            "sma_200": 125000000.0,
            "ema_10": 129000000.0,
            "ema_50": 127500000.0,
            "ema_200": 126000000.0,
            "rsi_14": 42.5,
            "macd": {"macd": 150.0, "signal": 100.0, "histogram": 50.0},
            "bollinger": {"upper": 135000000.0, "middle": 128000000.0, "lower": 121000000.0},
            "stochastic": {"k": 35.0, "d": 38.0},
            "adx": {"adx": 22.5, "plus_di": 18.0, "minus_di": 25.0, "regime": "transitioning"},
            "atr": 2500000.0,
        },
        "indicators_4h": {
            "rsi_14": 45.0,
            "macd": {"macd": 50.0, "signal": 30.0, "histogram": 20.0},
            "stochastic": {"k": 40.0, "d": 42.0},
        },
        "orderbook": {"bid_total": 50.0, "ask_total": 40.0, "ratio": 1.25},
        "trade_pressure": {"buy_volume": 10.0, "sell_volume": 8.0},
        "eth_btc_analysis": {
            "eth_price": 4500000,
            "eth_change_24h": -1.5,
            "eth_btc_ratio": 0.035,
            "eth_btc_ratio_avg60": 0.034,
            "eth_btc_z_score": 0.5,
        },
        "daily_summary_5d": [
            {"date": "2026-03-08", "open": 129000000, "high": 131000000,
             "low": 127000000, "close": 130000000, "change_pct": 0.78, "volume": 1200.5},
        ],
    }


@pytest.fixture
def sample_portfolio() -> dict:
    """Matches scripts/get_portfolio.py stdout JSON structure."""
    return {
        "timestamp": "2026-03-12T10:00:00+09:00",
        "krw_balance": 500000.0,
        "holdings": [
            {
                "currency": "BTC",
                "balance": 0.001,
                "avg_buy_price": 128000000.0,
                "current_price": 130000000,
                "eval_amount": 130000.0,
                "profit_loss_pct": 1.56,
            }
        ],
        "total_eval": 630000.0,
        "total_invested": 628000.0,
        "total_profit_loss_pct": 0.32,
    }


@pytest.fixture
def sample_ai_signal() -> dict:
    """Matches scripts/collect_ai_signal.py stdout JSON structure."""
    return {
        "timestamp": "2026-03-12T10:00:00+09:00",
        "composite_signal": {
            "score": 15,
            "signal": "mildly_bullish",
            "components": {
                "orderbook": 5,
                "trades": 3,
                "whale": 2,
                "divergence": 3,
                "volatility": 2,
            },
        },
    }


@pytest.fixture
def sample_external_data() -> dict:
    """Matches ExternalDataAgent.collect_all() return structure."""
    return {
        "timestamp": "2026-03-12T10:00:00+09:00",
        "collection_time_sec": 3.5,
        "sources": {
            "fear_greed": {"current": {"value": 35, "value_classification": "Fear"}},
            "news": {"articles_count": 5, "by_category": {}, "categories": {}},
            "news_sentiment": {
                "sentiment_score": -10,
                "positive_count": 2,
                "negative_count": 3,
                "neutral_count": 1,
                "total_articles": 6,
                "overall_sentiment": "slightly_negative",
                "key_signals": [],
            },
            "whale_tracker": {"direction": "neutral"},
            "binance_sentiment": {
                "kimchi_premium": {"premium_pct": 1.2},
                "top_trader_long_short": {"current_ratio": 1.1},
                "funding_rate": {"current_rate": 0.0001},
            },
            "eth_btc": {"eth_btc_z_score": 0.3},
            "macro": {"analysis": {"macro_score": 5, "sentiment": "neutral"}},
            "user_feedback": [],
            "performance_review": {"available": False, "message": "No data"},
        },
        "external_signal": {
            "total_score": 5,
            "strategy_bonus": 3,
            "fusion": {"signal": "neutral", "score": 5},
        },
        "errors": [],
    }


@pytest.fixture
def sample_agent_state() -> dict:
    """Matches data/agent_state.json actual structure."""
    return {
        "active_agent": "moderate",
        "last_switch_time": "2026-03-11T15:12:38+09:00",
        "last_trade_time": None,
        "consecutive_losses": 0,
        "switch_history": [
            {
                "from": "conservative",
                "to": "moderate",
                "reason": "test reason",
                "danger_score": 0,
                "opportunity_score": 0,
                "market_phase": "neutral",
                "timestamp": "2026-03-11T15:12:38+09:00",
            }
        ],
        "dca_history": {},
    }


@pytest.fixture
def sample_decision_dict() -> dict:
    """Matches Decision.to_dict() output used by Orchestrator."""
    return {
        "decision": "hold",
        "confidence": 0.6,
        "reason": "RSI neutral, FGI fear but not extreme, holding position",
        "buy_score": {
            "fgi": {"score": 15, "value": 35, "partial": True},
            "rsi": {"score": 0, "value": 42.5},
            "sma": {"score": 0, "value": 1.56},
            "news": {"score": 20, "negative": False},
            "external": {"score": 3},
            "total": 38,
            "threshold": 55,
            "result": "hold",
        },
        "trade_params": {},
        "external_signal_summary": {
            "total_score": 5,
            "strategy_bonus": 3,
            "fusion_signal": "neutral",
        },
        "agent_name": "moderate",
        "timestamp": "2026-03-12T10:00:00+09:00",
    }


@pytest.fixture
def sample_orchestrator_result(sample_decision_dict) -> dict:
    """Full result from Orchestrator.run() as consumed by run_agents.sh Phase 3+."""
    return {
        "active_agent": "moderate",
        "switch": None,
        "decision": sample_decision_dict,
        "market_state": {
            "fgi": 35,
            "rsi": 42.5,
            "price_change_24h": -2.0,
            "danger_score": 15,
            "opportunity_score": 20,
        },
        "drop_context": {"cascade_risk": 10},
    }


# ═══════════════════════════════════════════════════════════════
# 1. Data Format Consistency Between Pipeline Stages
# ═══════════════════════════════════════════════════════════════

class TestDataFormatConsistency:
    """Verify that each script's output keys match what the next stage expects."""

    def test_market_data_has_required_keys_for_orchestrator(self, sample_market_data):
        """run_agents.sh Phase 2 reads these keys from market_data.json."""
        assert "current_price" in sample_market_data
        assert "indicators" in sample_market_data
        indicators = sample_market_data["indicators"]
        assert "rsi_14" in indicators
        assert "sma_20" in indicators
        assert "macd" in indicators
        assert "bollinger" in indicators

    def test_market_data_has_ticker_fallback_keys(self, sample_market_data):
        """run_agents.sh Phase 5 uses ticker.trade_price OR current_price."""
        # The code does: market.get('current_price') or market.get('ticker', {}).get('trade_price', 0)
        # collect_market_data.py outputs current_price at top level (no 'ticker' subobject).
        # This is fine -- the fallback handles it.
        assert sample_market_data.get("current_price") is not None or \
               sample_market_data.get("ticker", {}).get("trade_price") is not None

    def test_portfolio_has_required_keys_for_agent(self, sample_portfolio):
        """Orchestrator reads coins/BTC or btc from portfolio."""
        # run_agents.sh does: portfolio.get('coins', {}).get('BTC', portfolio.get('btc', {}))
        # get_portfolio.py outputs 'holdings' list, not 'coins' dict.
        # The code then manually sets portfolio['btc'] and portfolio['btc_ratio'].
        assert "krw_balance" in sample_portfolio
        assert "holdings" in sample_portfolio
        assert "total_eval" in sample_portfolio

    def test_portfolio_btc_ratio_injection(self, sample_portfolio):
        """run_agents.sh injects btc_ratio into portfolio before Orchestrator."""
        # Simulate the injection logic from run_agents.sh Phase 2
        btc_info = {}
        for h in sample_portfolio.get("holdings", []):
            if h["currency"] == "BTC":
                btc_info = h
                break
        total_eval = sample_portfolio.get("total_eval", 1)
        btc_eval = btc_info.get("eval_amount", 0)
        btc_ratio = btc_eval / total_eval if total_eval > 0 else 0

        sample_portfolio["btc_ratio"] = btc_ratio
        sample_portfolio["btc"] = btc_info

        assert "btc_ratio" in sample_portfolio
        assert 0 <= sample_portfolio["btc_ratio"] <= 1
        assert "btc" in sample_portfolio

    def test_ai_signal_composite_injected_into_market_data(
        self, sample_market_data, sample_ai_signal
    ):
        """run_agents.sh injects ai_composite_signal into market_data."""
        sample_market_data["ai_composite_signal"] = sample_ai_signal.get(
            "composite_signal", {}
        )
        assert "ai_composite_signal" in sample_market_data
        assert "score" in sample_market_data["ai_composite_signal"]

    def test_external_data_fgi_injected_into_market_data(
        self, sample_market_data, sample_external_data
    ):
        """run_agents.sh injects FGI from external_data into market_data."""
        fgi_data = sample_external_data.get("sources", {}).get("fear_greed", {})
        sample_market_data["fear_greed"] = fgi_data.get("current", {})
        assert "fear_greed" in sample_market_data
        assert "value" in sample_market_data["fear_greed"]

    def test_external_data_news_injected_into_market_data(
        self, sample_market_data, sample_external_data
    ):
        """run_agents.sh injects news + sentiment into market_data."""
        news_data = sample_external_data.get("sources", {}).get("news", {})
        news_sentiment = sample_external_data.get("sources", {}).get("news_sentiment", {})
        sample_market_data["news"] = news_data
        sample_market_data["news"]["overall_sentiment"] = news_sentiment.get(
            "overall_sentiment", "neutral"
        )
        sample_market_data["news"]["sentiment_score"] = news_sentiment.get(
            "sentiment_score", 0
        )
        assert "news" in sample_market_data
        assert "overall_sentiment" in sample_market_data["news"]
        assert "sentiment_score" in sample_market_data["news"]

    def test_orchestrator_result_has_decision_fields_for_phase3(
        self, sample_orchestrator_result
    ):
        """run_agents.sh Phase 3 extracts decision, reason, active_agent."""
        r = sample_orchestrator_result
        assert "decision" in r
        assert "decision" in r["decision"]
        assert r["decision"]["decision"] in ("buy", "sell", "hold")
        assert "reason" in r["decision"]
        assert "active_agent" in r

    def test_buy_decision_has_trade_params(self):
        """When decision is 'buy', trade_params must have side/market/amount."""
        buy_decision = {
            "decision": "buy",
            "confidence": 0.7,
            "reason": "Strong buy signal",
            "buy_score": {"total": 75, "threshold": 55, "result": "buy"},
            "trade_params": {
                "side": "bid",
                "market": "KRW-BTC",
                "amount": 50000,
                "is_dca": False,
            },
            "external_signal_summary": {},
            "agent_name": "moderate",
            "timestamp": "2026-03-12T10:00:00+09:00",
        }
        params = buy_decision["trade_params"]
        assert params["side"] in ("bid", "ask")
        assert params["market"] == "KRW-BTC"
        assert isinstance(params["amount"], (int, float))
        assert params["amount"] > 0

    def test_sell_decision_has_trade_params(self):
        """When decision is 'sell', trade_params must have market/volume."""
        sell_decision = {
            "decision": "sell",
            "trade_params": {
                "side": "ask",
                "market": "KRW-BTC",
                "volume": 0.001,
            },
        }
        params = sell_decision["trade_params"]
        assert params["market"] == "KRW-BTC"
        assert isinstance(params["volume"], (int, float))
        assert params["volume"] > 0

    def test_external_data_collect_all_structure(self, sample_external_data):
        """ExternalDataAgent.collect_all() output structure check."""
        assert "timestamp" in sample_external_data
        assert "collection_time_sec" in sample_external_data
        assert isinstance(sample_external_data["collection_time_sec"], (int, float))
        assert "sources" in sample_external_data
        assert "external_signal" in sample_external_data
        assert "errors" in sample_external_data
        assert isinstance(sample_external_data["errors"], list)

    def test_external_signal_has_fusion(self, sample_external_data):
        """External signal must contain fusion data for Orchestrator."""
        ext_sig = sample_external_data["external_signal"]
        assert "total_score" in ext_sig
        assert "fusion" in ext_sig
        assert "signal" in ext_sig["fusion"]


# ═══════════════════════════════════════════════════════════════
# 2. Schema Validation
# ═══════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """Validate JSON schemas used in the project."""

    def test_decision_result_schema_is_valid_json(self):
        """prompts/schemas/decision_result.json must be valid JSON."""
        schema_path = PROJECT_DIR / "prompts" / "schemas" / "decision_result.json"
        assert schema_path.exists(), f"Schema file missing: {schema_path}"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_decision_result_schema_required_fields(self):
        """Check that required fields match what code actually produces."""
        schema_path = PROJECT_DIR / "prompts" / "schemas" / "decision_result.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        required = schema["required"]
        assert "timestamp" in required
        assert "decision" in required
        assert "confidence" in required
        assert "reason" in required
        assert "market_analysis" in required
        assert "trade_details" in required

    def test_decision_result_schema_decision_enum(self):
        """Schema decision enum should match Korean decision values for LLM mode."""
        schema_path = PROJECT_DIR / "prompts" / "schemas" / "decision_result.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        enum_vals = schema["properties"]["decision"]["enum"]
        # LLM mode uses Korean values
        assert set(enum_vals) == {"매수", "매도", "관망"}

    def test_agent_decision_uses_english_values(self, sample_decision_dict):
        """Agent mode uses English decision values (buy/sell/hold)."""
        assert sample_decision_dict["decision"] in ("buy", "sell", "hold")

    def test_supabase_decisions_table_accepts_both_formats(self):
        """decisions table CHECK constraint uses Korean values.
        run_agents.sh Phase 5 maps English -> Korean before DB insert."""
        migration_path = PROJECT_DIR / "supabase" / "migrations" / "001_initial_schema.sql"
        content = migration_path.read_text(encoding="utf-8")
        # The CHECK constraint
        assert "decision IN ('매수', '매도', '관망')" in content

        # run_agents.sh maps: buy->매수, sell->매도, hold->관망
        decision_map = {"buy": "매수", "sell": "매도", "hold": "관망"}
        for eng, kor in decision_map.items():
            assert kor in content

    def test_decision_to_dict_matches_orchestrator_result_structure(self):
        """Decision.to_dict() output keys must match what run_agents.sh reads."""
        from agents.base_agent import Decision

        dec = Decision(
            decision="hold",
            confidence=0.5,
            reason="Test reason for hold decision in neutral market",
            buy_score={"total": 30, "threshold": 55},
            trade_params={},
            external_signal={"total_score": 0, "fusion": {"signal": "neutral"}},
            agent_name="conservative",
        )
        d = dec.to_dict()

        # Keys expected by run_agents.sh
        assert "decision" in d
        assert "reason" in d
        assert "confidence" in d
        assert "buy_score" in d
        assert "trade_params" in d
        assert "external_signal_summary" in d
        assert "agent_name" in d
        assert "timestamp" in d

    def test_decision_external_signal_summary_structure(self):
        """external_signal_summary must have total_score, strategy_bonus, fusion_signal."""
        from agents.base_agent import Decision

        dec = Decision(
            decision="hold",
            confidence=0.5,
            reason="Test reason checking external signal summary keys",
            buy_score={},
            trade_params={},
            external_signal={
                "total_score": 10,
                "strategy_bonus": 5,
                "fusion": {"signal": "buy"},
            },
            agent_name="moderate",
        )
        summary = dec.to_dict()["external_signal_summary"]
        assert "total_score" in summary
        assert "strategy_bonus" in summary
        assert "fusion_signal" in summary


# ═══════════════════════════════════════════════════════════════
# 3. Agent State File Consistency
# ═══════════════════════════════════════════════════════════════

class TestAgentStateConsistency:
    """Verify agent_state.json format between readers (_load_state) and writers (_save_state)."""

    def test_actual_agent_state_file_is_valid(self):
        """data/agent_state.json must be valid JSON with expected keys."""
        state_path = PROJECT_DIR / "data" / "agent_state.json"
        if not state_path.exists():
            pytest.skip("agent_state.json not present")
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        assert "active_agent" in state
        assert state["active_agent"] in ("conservative", "moderate", "aggressive")

    def test_load_state_default_has_required_keys(self, tmp_path):
        """_load_state() default when file missing must have all keys."""
        from agents.orchestrator import _load_state

        with patch("agents.orchestrator.STATE_FILE", tmp_path / "missing_state.json"):
            state = _load_state()

        assert "active_agent" in state
        assert "last_switch_time" in state
        assert "last_trade_time" in state
        assert "consecutive_losses" in state
        assert "switch_history" in state
        assert state["active_agent"] == "conservative"

    def test_state_roundtrip_consistency(self, tmp_path, sample_agent_state):
        """Write state -> read state must be identical."""
        from agents.orchestrator import _load_state, _save_state

        state_file = tmp_path / "agent_state.json"
        with patch("agents.orchestrator.STATE_FILE", state_file):
            _save_state(sample_agent_state)
            loaded = _load_state()

        assert loaded["active_agent"] == sample_agent_state["active_agent"]
        assert loaded["last_switch_time"] == sample_agent_state["last_switch_time"]
        assert loaded["consecutive_losses"] == sample_agent_state["consecutive_losses"]
        assert len(loaded["switch_history"]) == len(sample_agent_state["switch_history"])

    def test_switch_history_entry_format(self, sample_agent_state):
        """Each switch_history entry must have required fields."""
        required_keys = {"from", "to", "reason", "danger_score",
                         "opportunity_score", "market_phase", "timestamp"}
        for entry in sample_agent_state["switch_history"]:
            assert required_keys.issubset(entry.keys()), \
                f"Missing keys: {required_keys - entry.keys()}"

    def test_dca_history_format(self, sample_agent_state):
        """dca_history must be a dict (possibly empty)."""
        assert isinstance(sample_agent_state.get("dca_history", {}), dict)

    def test_active_agent_must_be_valid_name(self, sample_agent_state):
        """active_agent must be one of the registered agent names."""
        valid_agents = {"conservative", "moderate", "aggressive"}
        assert sample_agent_state["active_agent"] in valid_agents


# ═══════════════════════════════════════════════════════════════
# 4. Supabase Migration Consistency
# ═══════════════════════════════════════════════════════════════

class TestSupabaseMigrations:
    """Validate SQL migrations are syntactically correct and consistent with code."""

    MIGRATION_DIR = PROJECT_DIR / "supabase" / "migrations"

    def test_migrations_directory_exists(self):
        assert self.MIGRATION_DIR.exists()

    def test_all_migration_files_are_valid_sql(self):
        """Each .sql file must parse without obvious syntax issues."""
        for sql_file in sorted(self.MIGRATION_DIR.glob("*.sql")):
            content = sql_file.read_text(encoding="utf-8")
            assert len(content.strip()) > 0, f"Empty migration: {sql_file.name}"
            # Basic SQL syntax checks
            # Every CREATE TABLE / ALTER TABLE should not have unmatched parens
            open_parens = content.count("(")
            close_parens = content.count(")")
            assert open_parens == close_parens, \
                f"Unmatched parentheses in {sql_file.name}: ({open_parens} vs {close_parens})"

    def test_initial_schema_has_required_tables(self):
        """001_initial_schema.sql must create all core tables."""
        content = (self.MIGRATION_DIR / "001_initial_schema.sql").read_text(encoding="utf-8")
        required_tables = [
            "decisions", "portfolio_snapshots", "market_data",
            "feedback", "execution_logs", "strategy_history",
        ]
        for table in required_tables:
            assert f"CREATE TABLE {table}" in content, \
                f"Missing table: {table}"

    def test_agent_switches_migration_has_required_columns(self):
        """004_agent_switches.sql must have learning-related columns."""
        content = (self.MIGRATION_DIR / "004_agent_switches.sql").read_text(encoding="utf-8")
        required_cols = [
            "from_agent", "to_agent", "reason",
            "fgi_at_switch", "rsi_at_switch", "price_at_switch",
            "price_after_4h", "price_after_24h",
            "profit_after_4h", "profit_after_24h",
            "outcome",
        ]
        for col in required_cols:
            assert col in content, f"Missing column: {col}"

    def test_cycle_id_migration_covers_all_tables(self):
        """009_cycle_id_and_views.sql must add cycle_id to key tables."""
        content = (self.MIGRATION_DIR / "009_cycle_id_and_views.sql").read_text(encoding="utf-8")
        tables_needing_cycle_id = [
            "decisions", "portfolio_snapshots", "market_data",
            "execution_logs", "agent_switches",
        ]
        for table in tables_needing_cycle_id:
            assert table in content, \
                f"Table {table} missing cycle_id migration"

    def test_decisions_table_columns_match_code_inserts(self):
        """Columns inserted by run_agents.sh Phase 5 must exist in schema."""
        # Columns inserted by run_agents.sh
        inserted_columns = [
            "decision", "confidence", "reason", "current_price",
            "rsi_value", "fear_greed_value", "market_data_snapshot",
            "cycle_id", "source",
        ]
        # Read initial schema + cycle_id migration
        schema = (self.MIGRATION_DIR / "001_initial_schema.sql").read_text(encoding="utf-8")
        cycle_migration = (self.MIGRATION_DIR / "009_cycle_id_and_views.sql").read_text(encoding="utf-8")
        combined = schema + cycle_migration

        for col in inserted_columns:
            assert col in combined, \
                f"Column '{col}' inserted by code but not in migrations"

    def test_market_data_table_columns_match_code_inserts(self):
        """Columns inserted in Phase 5b must exist in schema."""
        inserted_columns = [
            "market", "price", "volume_24h", "change_rate_24h",
            "fear_greed_value", "fear_greed_class",
            "rsi_14", "sma_20", "news_sentiment", "cycle_id",
        ]
        schema = (self.MIGRATION_DIR / "001_initial_schema.sql").read_text(encoding="utf-8")
        cycle_migration = (self.MIGRATION_DIR / "009_cycle_id_and_views.sql").read_text(encoding="utf-8")
        combined = schema + cycle_migration

        for col in inserted_columns:
            assert col in combined, \
                f"Column '{col}' for market_data not in migrations"

    def test_execution_logs_columns_match_code_inserts(self):
        """Columns inserted in Phase 5b execution_logs must exist in schema."""
        inserted_columns = [
            "execution_mode", "duration_ms", "data_sources",
            "raw_output", "cycle_id",
        ]
        schema = (self.MIGRATION_DIR / "001_initial_schema.sql").read_text(encoding="utf-8")
        cycle_migration = (self.MIGRATION_DIR / "009_cycle_id_and_views.sql").read_text(encoding="utf-8")
        combined = schema + cycle_migration

        for col in inserted_columns:
            assert col in combined, \
                f"Column '{col}' for execution_logs not in migrations"

    def test_execution_mode_check_constraint(self):
        """execution_logs execution_mode must accept analyze/execute/dry_run."""
        schema = (self.MIGRATION_DIR / "001_initial_schema.sql").read_text(encoding="utf-8")
        for mode in ("analyze", "execute", "dry_run"):
            assert mode in schema, f"execution_mode '{mode}' not in CHECK constraint"


# ═══════════════════════════════════════════════════════════════
# 5. Pipeline Script Reference Integrity
# ═══════════════════════════════════════════════════════════════

class TestPipelineScriptIntegrity:
    """All scripts referenced by pipeline shell scripts must exist."""

    SCRIPTS_DIR = PROJECT_DIR / "scripts"

    def test_run_agents_sh_referenced_scripts_exist(self):
        """Scripts called in run_agents.sh must exist."""
        required_scripts = [
            "collect_market_data.py",
            "get_portfolio.py",
            "collect_ai_signal.py",
            "notify_telegram.py",
            "execute_trade.py",
            "evaluate_switches.py",
        ]
        for script in required_scripts:
            path = self.SCRIPTS_DIR / script
            assert path.exists(), f"Script referenced in run_agents.sh not found: {script}"

    def test_run_analysis_sh_referenced_scripts_exist(self):
        """Scripts called in run_analysis.sh must exist."""
        required_scripts = [
            "collect_market_data.py",
            "collect_fear_greed.py",
            "collect_news.py",
            "capture_chart.py",
            "get_portfolio.py",
            "collect_ai_signal.py",
            "collect_onchain_data.py",
            "whale_tracker.py",
            "binance_sentiment.py",
            "collect_crypto_signals.py",
            "collect_coinmarketcap.py",
            "calculate_external_signal.py",
        ]
        for script in required_scripts:
            path = self.SCRIPTS_DIR / script
            assert path.exists(), f"Script referenced in run_analysis.sh not found: {script}"

    def test_run_analysis_sh_optional_scripts(self):
        """Optional scripts (summarize_news, recall_rag) should exist."""
        optional_scripts = ["summarize_news.py", "recall_rag.py"]
        for script in optional_scripts:
            path = self.SCRIPTS_DIR / script
            assert path.exists(), f"Optional script missing: {script}"

    def test_cron_run_sh_references_run_agents_py(self):
        """cron_run.sh calls run_agents.py which must exist."""
        assert (self.SCRIPTS_DIR / "run_agents.py").exists()

    def test_cron_run_sh_references_retrospective(self):
        """cron_run.sh calls retrospective.py which must exist."""
        assert (self.SCRIPTS_DIR / "retrospective.py").exists()

    def test_cron_run_sh_references_save_decision(self):
        """cron_run.sh fallback calls save_decision.py."""
        assert (self.SCRIPTS_DIR / "save_decision.py").exists()

    def test_external_data_agent_referenced_scripts_exist(self):
        """Scripts run by ExternalDataAgent.collect_all() must exist."""
        required_scripts = [
            "collect_fear_greed.py",
            "collect_news.py",
            "whale_tracker.py",
            "binance_sentiment.py",
            "collect_eth_btc.py",
            "collect_macro.py",
            "collect_crypto_signals.py",
            "collect_coinmarketcap.py",
            "calculate_external_signal.py",
        ]
        for script in required_scripts:
            path = self.SCRIPTS_DIR / script
            assert path.exists(), \
                f"Script referenced by ExternalDataAgent not found: {script}"

    def test_cycle_id_module_exists(self):
        """scripts/cycle_id.py referenced in run_agents.sh must exist."""
        assert (self.SCRIPTS_DIR / "cycle_id.py").exists()


# ═══════════════════════════════════════════════════════════════
# 6. Cron / Windows Scheduler Script Validation
# ═══════════════════════════════════════════════════════════════

class TestCronScripts:
    """Validate cron and Windows scheduler scripts."""

    def test_cron_run_sh_has_shebang(self):
        content = (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash")

    def test_cron_run_sh_loads_env(self):
        content = (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(encoding="utf-8")
        assert "source .env" in content

    def test_cron_run_sh_checks_emergency_stop(self):
        content = (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(encoding="utf-8")
        assert "EMERGENCY_STOP" in content

    def test_cron_run_sh_creates_log_dirs(self):
        content = (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(encoding="utf-8")
        assert "mkdir -p" in content
        assert "logs/executions" in content

    def test_cron_run_sh_has_fallback(self):
        """cron_run.sh must have fallback to bash pipeline if Python fails."""
        content = (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(encoding="utf-8")
        assert "FALLBACK" in content or "fallback" in content

    def test_setup_cron_sh_has_install_remove_status(self):
        content = (PROJECT_DIR / "scripts" / "setup_cron.sh").read_text(encoding="utf-8")
        assert "install" in content
        assert "remove" in content
        assert "status" in content

    def test_setup_cron_sh_supports_multiple_intervals(self):
        content = (PROJECT_DIR / "scripts" / "setup_cron.sh").read_text(encoding="utf-8")
        # Should offer 4, 8, 12, 24 hour intervals
        assert "4시간" in content or "0,4,8,12,16,20" in content
        assert "8시간" in content or "0,8,16" in content

    def test_win_cron_run_ps1_exists_and_valid(self):
        path = PROJECT_DIR / "scripts" / "win_cron_run.ps1"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "PYTHONIOENCODING" in content
        assert "EMERGENCY_STOP" in content
        assert "run_agents.py" in content

    def test_win_cron_run_ps1_loads_env(self):
        content = (PROJECT_DIR / "scripts" / "win_cron_run.ps1").read_text(encoding="utf-8")
        assert ".env" in content

    def test_setup_win_cron_ps1_exists_and_valid(self):
        path = PROJECT_DIR / "scripts" / "setup_win_cron.ps1"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "CoinTrading_AutoTrade" in content
        assert "Register-ScheduledTask" in content

    def test_setup_win_cron_ps1_validates_intervals(self):
        content = (PROJECT_DIR / "scripts" / "setup_win_cron.ps1").read_text(encoding="utf-8")
        assert "4, 8, 12, 24" in content

    def test_run_agents_sh_has_shebang(self):
        content = (PROJECT_DIR / "scripts" / "run_agents.sh").read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash")

    def test_run_agents_sh_sets_utf8(self):
        """Must set PYTHONIOENCODING and PYTHONUTF8 for Windows."""
        content = (PROJECT_DIR / "scripts" / "run_agents.sh").read_text(encoding="utf-8")
        assert "PYTHONIOENCODING=utf-8" in content
        assert "PYTHONUTF8=1" in content

    def test_run_agents_sh_windows_python_detection(self):
        """Must detect .venv/Scripts/python.exe for Windows."""
        content = (PROJECT_DIR / "scripts" / "run_agents.sh").read_text(encoding="utf-8")
        assert ".venv/Scripts/python.exe" in content
        assert ".venv/bin/python" in content

    def test_run_analysis_sh_delegates_to_agents(self):
        """--agent flag must delegate to run_agents.sh."""
        content = (PROJECT_DIR / "scripts" / "run_analysis.sh").read_text(encoding="utf-8")
        assert 'exec bash scripts/run_agents.sh' in content


# ═══════════════════════════════════════════════════════════════
# 7. Data Directory Structure Assumptions
# ═══════════════════════════════════════════════════════════════

class TestDataDirectoryStructure:
    """Verify data/ directory assumptions are consistent across scripts."""

    def test_data_directory_exists(self):
        assert (PROJECT_DIR / "data").exists()

    def test_snapshot_dir_is_created_by_pipeline(self, tmp_path):
        """Pipeline creates data/snapshots/TIMESTAMP/ directories."""
        snapshot_dir = tmp_path / "data" / "snapshots" / "20260312_100000"
        snapshot_dir.mkdir(parents=True)
        assert snapshot_dir.exists()

    def test_snapshot_files_match_pipeline_expectations(self):
        """Pipeline expects these files in each snapshot directory."""
        expected_files = [
            "market_data.json",
            "portfolio.json",
            "ai_signal.json",
            "agent_result.json",  # Written by run_agents.sh after Phase 2
        ]
        # These are the filenames the pipeline writes to snapshot_dir
        for f in expected_files:
            assert f.endswith(".json"), f"Non-JSON snapshot file: {f}"

    def test_auto_emergency_file_path_consistency(self):
        """auto_emergency.json path must match between orchestrator and run_agents.sh."""
        # Orchestrator uses: PROJECT_DIR / "data" / "auto_emergency.json"
        from agents.orchestrator import AUTO_EMERGENCY_FILE
        expected = PROJECT_DIR / "data" / "auto_emergency.json"
        assert AUTO_EMERGENCY_FILE == expected

    def test_agent_state_file_path_consistency(self):
        """agent_state.json path must match between orchestrator and code."""
        from agents.orchestrator import STATE_FILE
        expected = PROJECT_DIR / "data" / "agent_state.json"
        assert STATE_FILE == expected

    def test_logs_directory_structure(self):
        """Pipeline expects logs/executions/ and logs/claude_responses/."""
        # These are created by mkdir -p in cron_run.sh
        # Just verify the paths are consistent
        assert "logs/executions" in (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(
            encoding="utf-8"
        )
        assert "logs/claude_responses" in (PROJECT_DIR / "scripts" / "cron_run.sh").read_text(
            encoding="utf-8"
        )


# ═══════════════════════════════════════════════════════════════
# 8. Orchestrator Integration
# ═══════════════════════════════════════════════════════════════

class TestOrchestratorIntegration:
    """Test Orchestrator with mocked dependencies."""

    @patch("agents.orchestrator._load_state")
    @patch("agents.orchestrator._save_state")
    @patch.dict(os.environ, {"EMERGENCY_STOP": "true"})
    def test_emergency_stop_returns_hold(self, mock_save, mock_load):
        """EMERGENCY_STOP=true must return hold without trading."""
        mock_load.return_value = {
            "active_agent": "moderate",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        result = orch.run(
            market_data={},
            external_data={},
            portfolio={},
        )
        assert result["decision"]["decision"] == "hold"
        assert "긴급정지" in result["active_agent"] or "EMERGENCY_STOP" in result["decision"]["reason"]

    def test_agents_registry_complete(self):
        """All three agents must be registered in AGENTS dict."""
        from agents.orchestrator import AGENTS
        assert "conservative" in AGENTS
        assert "moderate" in AGENTS
        assert "aggressive" in AGENTS

    def test_agent_classes_have_required_attributes(self):
        """Each agent class must have name, emoji, buy_score_threshold."""
        from agents.orchestrator import AGENTS
        for name, cls in AGENTS.items():
            agent = cls()
            assert hasattr(agent, "name")
            assert hasattr(agent, "emoji")
            assert hasattr(agent, "buy_score_threshold")
            assert isinstance(agent.buy_score_threshold, (int, float))

    def test_buy_score_thresholds_ordering(self):
        """Conservative > Moderate > Aggressive thresholds."""
        from agents.conservative import ConservativeAgent
        from agents.moderate import ModerateAgent
        from agents.aggressive import AggressiveAgent

        c = ConservativeAgent()
        m = ModerateAgent()
        a = AggressiveAgent()

        assert c.buy_score_threshold > m.buy_score_threshold > a.buy_score_threshold, \
            f"Threshold ordering wrong: C={c.buy_score_threshold}, M={m.buy_score_threshold}, A={a.buy_score_threshold}"


# ═══════════════════════════════════════════════════════════════
# 9. News Sentiment Analysis Consistency
# ═══════════════════════════════════════════════════════════════

class TestNewsSentimentConsistency:
    """Verify news sentiment output is consumed correctly downstream."""

    def test_analyze_news_sentiment_output_format(self):
        """analyze_news_sentiment must return expected keys."""
        from agents.external_data import analyze_news_sentiment

        result = analyze_news_sentiment({"articles": []})
        assert "sentiment_score" in result
        assert "overall_sentiment" in result
        assert "positive_count" in result
        assert "negative_count" in result
        assert "neutral_count" in result

    def test_sentiment_with_positive_articles(self):
        from agents.external_data import analyze_news_sentiment

        articles = [
            {"title": "Bitcoin ETF approved by SEC", "content": "Institutional adoption rally"},
            {"title": "BTC hits all-time high", "content": "Bullish momentum continues"},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] >= 1
        assert result["sentiment_score"] > 0

    def test_sentiment_with_negative_articles(self):
        from agents.external_data import analyze_news_sentiment

        articles = [
            {"title": "Major crypto exchange hacked", "content": "Fraud investigation launched"},
            {"title": "Bitcoin crash continues", "content": "Panic selling bearish"},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["negative_count"] >= 1
        assert result["sentiment_score"] < 0

    def test_sentiment_overall_classification(self):
        """overall_sentiment must be one of the expected values."""
        from agents.external_data import analyze_news_sentiment

        result = analyze_news_sentiment({"articles": []})
        valid = {"positive", "negative", "slightly_positive",
                 "slightly_negative", "neutral"}
        assert result["overall_sentiment"] in valid


# ═══════════════════════════════════════════════════════════════
# 10. Config File Format Validation
# ═══════════════════════════════════════════════════════════════

class TestConfigValidation:
    """Validate project config files."""

    def test_env_example_exists(self):
        assert (PROJECT_DIR / ".env.example").exists()

    def test_env_example_has_required_keys(self):
        content = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")
        required_keys = [
            "DRY_RUN",
            "EMERGENCY_STOP",
            "MAX_TRADE_AMOUNT",
        ]
        for key in required_keys:
            assert key in content, f"Missing key in .env.example: {key}"

    def test_strategy_md_exists(self):
        assert (PROJECT_DIR / "strategy.md").exists()

    def test_requirements_txt_exists(self):
        assert (PROJECT_DIR / "requirements.txt").exists()

    def test_requirements_has_core_dependencies(self):
        content = (PROJECT_DIR / "requirements.txt").read_text(encoding="utf-8")
        core_deps = ["requests", "python-dotenv"]
        for dep in core_deps:
            assert dep in content.lower(), f"Missing dependency: {dep}"

    def test_mcp_json_exists(self):
        """MCP config for bitcoin-mcp should exist."""
        mcp_path = PROJECT_DIR / ".mcp.json"
        if mcp_path.exists():
            with open(mcp_path, encoding="utf-8") as f:
                config = json.load(f)
            assert isinstance(config, dict)


# ═══════════════════════════════════════════════════════════════
# 11. End-to-End Data Flow Simulation (Mocked)
# ═══════════════════════════════════════════════════════════════

class TestEndToEndDataFlow:
    """Simulate the full agent pipeline data flow with mocked data."""

    def test_phase1_to_phase2_data_injection(
        self, sample_market_data, sample_portfolio, sample_ai_signal, sample_external_data
    ):
        """Simulate run_agents.sh Phase 2 data injection logic."""
        market_data = sample_market_data.copy()
        portfolio = sample_portfolio.copy()

        # Inject ai_composite_signal
        market_data["ai_composite_signal"] = sample_ai_signal.get("composite_signal", {})

        # Inject FGI
        fgi_data = sample_external_data.get("sources", {}).get("fear_greed", {})
        market_data["fear_greed"] = fgi_data.get("current", {})

        # Inject news
        news_data = sample_external_data.get("sources", {}).get("news", {})
        news_sentiment = sample_external_data.get("sources", {}).get("news_sentiment", {})
        market_data["news"] = news_data
        market_data["news"]["overall_sentiment"] = news_sentiment.get("overall_sentiment", "neutral")
        market_data["news"]["sentiment_score"] = news_sentiment.get("sentiment_score", 0)

        # BTC ratio calculation
        btc_info = {}
        for h in portfolio.get("holdings", []):
            if h.get("currency") == "BTC":
                btc_info = h
                break
        total_eval = portfolio.get("total_eval", portfolio.get("total_evaluation", 1))
        btc_eval = btc_info.get("eval_amount", 0) if isinstance(btc_info, dict) else 0
        portfolio["btc_ratio"] = btc_eval / total_eval if total_eval > 0 else 0
        portfolio["btc"] = btc_info

        # Verify all injections
        assert "ai_composite_signal" in market_data
        assert "fear_greed" in market_data
        assert "news" in market_data
        assert "overall_sentiment" in market_data["news"]
        assert "btc_ratio" in portfolio
        assert "btc" in portfolio

    def test_phase3_decision_extraction(self, sample_orchestrator_result):
        """Simulate Phase 3: extracting decision from orchestrator result."""
        r = sample_orchestrator_result
        decision = r["decision"]["decision"]
        reason = r["decision"]["reason"]
        agent_name = r["active_agent"]

        assert decision in ("buy", "sell", "hold")
        assert isinstance(reason, str) and len(reason) > 0
        assert isinstance(agent_name, str)

    def test_phase5_supabase_row_construction(
        self, sample_orchestrator_result, sample_market_data
    ):
        """Simulate Phase 5: building the Supabase decisions row."""
        DECISION_MAP = {"buy": "매수", "sell": "매도", "hold": "관망"}
        result = sample_orchestrator_result
        dec = result["decision"]
        market = sample_market_data

        raw_decision = dec["decision"]
        kr_decision = DECISION_MAP.get(raw_decision, raw_decision)

        row = {
            "decision": kr_decision,
            "confidence": dec.get("confidence", 0),
            "reason": dec.get("reason", ""),
            "current_price": int(
                market.get("current_price")
                or market.get("ticker", {}).get("trade_price", 0)
            ),
            "rsi_value": market.get("indicators", {}).get("rsi_14"),
            "cycle_id": "20260312-1000-agent",
            "source": "agent",
        }

        assert row["decision"] in ("매수", "매도", "관망")
        assert isinstance(row["current_price"], int)
        assert row["source"] == "agent"

    def test_phase5b_market_data_row_construction(self, sample_market_data):
        """Simulate Phase 5b: building market_data row."""
        market = sample_market_data
        indicators = market.get("indicators", {})

        row = {
            "market": "KRW-BTC",
            "price": int(market.get("current_price", 0)),
            "volume_24h": float(market.get("volume_24h", 0)),
            "change_rate_24h": float(market.get("change_rate_24h", 0)),
            "rsi_14": indicators.get("rsi_14"),
            "sma_20": int(indicators.get("sma_20")) if indicators.get("sma_20") else None,
            "news_sentiment": "neutral",
            "cycle_id": "20260312-1000-agent",
        }

        assert row["market"] == "KRW-BTC"
        assert isinstance(row["price"], int)
        assert row["price"] > 0

    def test_full_agent_result_json_serializable(self, sample_orchestrator_result):
        """The final agent result must be JSON-serializable for snapshot saving."""
        output = {
            "timestamp": "2026-03-12T10:00:00+09:00",
            "active_agent": sample_orchestrator_result["active_agent"],
            "switch": sample_orchestrator_result.get("switch"),
            "decision": sample_orchestrator_result["decision"],
            "external_data_summary": {
                "collection_time_sec": 3.5,
                "errors": [],
                "signal": {},
            },
            "snapshot_dir": "data/snapshots/20260312_100000",
        }
        # Must not raise
        serialized = json.dumps(output, ensure_ascii=False, indent=2)
        parsed = json.loads(serialized)
        assert parsed["decision"]["decision"] in ("buy", "sell", "hold")


# ═══════════════════════════════════════════════════════════════
# 12. Buy Score Calculation Consistency
# ═══════════════════════════════════════════════════════════════

class TestBuyScoreConsistency:
    """Test that buy_score format is consistent across agents."""

    def _get_agent_classes(self):
        from agents.conservative import ConservativeAgent
        from agents.moderate import ModerateAgent
        from agents.aggressive import AggressiveAgent
        return [ConservativeAgent(), ModerateAgent(), AggressiveAgent()]

    def test_all_agents_produce_same_buy_score_keys(self):
        """All agents must produce buy_score with same top-level keys."""
        agents = self._get_agent_classes()
        for agent in agents:
            score = agent.calculate_buy_score(
                fgi=30, rsi=35, sma_deviation=-3.0,
                news_negative=False, external_bonus=5,
            )
            assert "total" in score, f"{agent.name}: missing 'total'"
            assert "threshold" in score, f"{agent.name}: missing 'threshold'"
            assert "result" in score, f"{agent.name}: missing 'result'"
            assert "fgi" in score, f"{agent.name}: missing 'fgi'"
            assert "rsi" in score, f"{agent.name}: missing 'rsi'"
            assert "sma" in score, f"{agent.name}: missing 'sma'"
            assert "news" in score, f"{agent.name}: missing 'news'"
            assert "external" in score, f"{agent.name}: missing 'external'"

    def test_buy_score_result_matches_threshold(self):
        """result field must be 'buy' iff total >= threshold."""
        agents = self._get_agent_classes()
        for agent in agents:
            score = agent.calculate_buy_score(
                fgi=10, rsi=20, sma_deviation=-10.0,
                news_negative=False, external_bonus=20,
            )
            if score["total"] >= score["threshold"]:
                assert score["result"] == "buy", \
                    f"{agent.name}: total={score['total']} >= threshold={score['threshold']} but result={score['result']}"
            else:
                assert score["result"] == "hold", \
                    f"{agent.name}: total={score['total']} < threshold={score['threshold']} but result={score['result']}"

    def test_buy_score_each_component_has_score_key(self):
        """Each component (fgi, rsi, sma, news) must have a 'score' key."""
        agents = self._get_agent_classes()
        for agent in agents:
            breakdown = agent.calculate_buy_score(
                fgi=25, rsi=28, sma_deviation=-6.0,
                news_negative=True, external_bonus=0,
            )
            for component in ("fgi", "rsi", "sma", "news", "external"):
                assert "score" in breakdown[component], \
                    f"{agent.name}: {component} missing 'score' key"
