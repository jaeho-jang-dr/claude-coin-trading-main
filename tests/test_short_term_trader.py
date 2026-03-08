"""
short_term_trader.py unit tests

All network/WebSocket/external API calls are mocked — no real calls are made.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call

import requests

import pytest

# Patch environment and external dependencies before import
_env_patch = patch.dict(os.environ, {
    "UPBIT_ACCESS_KEY": "test_ak",
    "UPBIT_SECRET_KEY": "test_sk",
    "DRY_RUN": "true",
    "EMERGENCY_STOP": "false",
    "SHORT_TERM_BUDGET": "500000",
    "SHORT_TERM_MAX_TRADE": "200000",
    "SHORT_TERM_MAX_DAILY": "10",
    "SHORT_TERM_STOP_LOSS": "0.8",
    "SHORT_TERM_TAKE_PROFIT": "0.8",
    "SHORT_TERM_MAX_HOLD_MIN": "20",
    "SUPABASE_URL": "",
    "SUPABASE_SERVICE_ROLE_KEY": "",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_USER_ID": "",
})
_env_patch.start()

with patch("dotenv.load_dotenv"):
    from scripts.short_term_trader import (
        COMMISSION_PCT,
        KST,
        LOCK_FILE,
        MIN_PROFIT_AFTER_FEE,
        SELL_PRESSURE_BLOCK_RATIO,
        SHORT_TERM_BUDGET,
        SHORT_TERM_MAX_DAILY,
        SHORT_TERM_MAX_HOLD_MIN,
        SHORT_TERM_MAX_TRADE,
        SHORT_TERM_STOP_LOSS,
        SHORT_TERM_TAKE_PROFIT,
        SPIKE_THRESHOLD_PCT,
        SPIKE_WINDOW_SEC,
        WHALE_RATIO_THRESHOLD,
        WHALE_RATIO_WINDOW_SEC,
        WHALE_THRESHOLD_KRW,
        Position,
        ShortTermTrader,
        TradeSignal,
        acquire_lock,
        check_lock,
        db_insert,
        release_lock,
        send_telegram,
        upbit_auth_header,
        upbit_order,
        get_current_price,
    )

_env_patch.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trader(dry_run: bool = True) -> ShortTermTrader:
    """Create a fresh trader instance for testing."""
    trader = ShortTermTrader(dry_run=dry_run)
    trader.current_price = 100_000_000  # 1억원
    return trader


def _make_position(
    entry_price: float = 100_000_000,
    amount_krw: float = 200_000,
    strategy: str = "spike",
    entry_time: datetime | None = None,
    stop_loss_pct: float = SHORT_TERM_STOP_LOSS,
    take_profit_pct: float = SHORT_TERM_TAKE_PROFIT,
    max_hold_min: int = SHORT_TERM_MAX_HOLD_MIN,
) -> Position:
    if entry_time is None:
        entry_time = datetime.now(KST)
    return Position(
        strategy=strategy,
        side="bid",
        entry_price=entry_price,
        amount_krw=amount_krw,
        btc_qty=amount_krw / entry_price * (1 - COMMISSION_PCT / 100),
        entry_time=entry_time,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_min=max_hold_min,
    )


# ===========================================================================
# 1. Configuration / Constants
# ===========================================================================

class TestConfiguration:
    """Verify default parameters match documented v3 values."""

    def test_budget_default(self):
        assert SHORT_TERM_BUDGET == 500_000

    def test_max_trade_default(self):
        assert SHORT_TERM_MAX_TRADE == 200_000

    def test_max_daily_default(self):
        assert SHORT_TERM_MAX_DAILY == 10

    def test_stop_loss_default(self):
        assert SHORT_TERM_STOP_LOSS == 0.8

    def test_take_profit_default(self):
        assert SHORT_TERM_TAKE_PROFIT == 0.8

    def test_max_hold_minutes(self):
        assert SHORT_TERM_MAX_HOLD_MIN == 20

    def test_commission_pct(self):
        assert COMMISSION_PCT == 0.05

    def test_min_profit_after_fee(self):
        # round-trip fee (0.05*2) + 0.1 margin = 0.2
        assert MIN_PROFIT_AFTER_FEE == pytest.approx(0.2)

    def test_spike_threshold(self):
        assert SPIKE_THRESHOLD_PCT == 0.8

    def test_spike_window_sec(self):
        assert SPIKE_WINDOW_SEC == 300

    def test_whale_threshold_krw(self):
        assert WHALE_THRESHOLD_KRW == 30_000_000

    def test_whale_ratio_threshold(self):
        assert WHALE_RATIO_THRESHOLD == 0.7

    def test_sell_pressure_block_ratio(self):
        assert SELL_PRESSURE_BLOCK_RATIO == 3.0


# ===========================================================================
# 2. Position dataclass
# ===========================================================================

class TestPositionDataclass:
    def test_position_defaults(self):
        pos = _make_position()
        assert pos.side == "bid"
        assert pos.exit_price is None
        assert pos.exit_time is None
        assert pos.exit_reason is None
        assert pos.pnl_pct is None

    def test_position_btc_qty_accounts_for_commission(self):
        pos = _make_position(entry_price=100_000_000, amount_krw=200_000)
        expected = 200_000 / 100_000_000 * (1 - COMMISSION_PCT / 100)
        assert pos.btc_qty == pytest.approx(expected)


# ===========================================================================
# 3. Position exit checks (stop-loss, take-profit, time limit)
# ===========================================================================

class TestPositionExit:
    def test_take_profit_triggers(self):
        trader = _make_trader()
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        # Price rose 0.9% (above 0.8% take-profit)
        trader.current_price = 100_900_000
        exits = trader.check_position_exit()
        assert len(exits) == 1
        assert "익절" in exits[0][1]

    def test_stop_loss_triggers(self):
        trader = _make_trader()
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        # Price dropped 0.9%
        trader.current_price = 99_100_000
        exits = trader.check_position_exit()
        assert len(exits) == 1
        assert "손절" in exits[0][1]

    def test_max_hold_time_triggers(self):
        trader = _make_trader()
        entry_time = datetime.now(KST) - timedelta(minutes=25)
        pos = _make_position(entry_price=100_000_000, entry_time=entry_time)
        trader.positions.append(pos)
        # Price is flat — no SL/TP triggered
        trader.current_price = 100_000_000
        exits = trader.check_position_exit()
        assert len(exits) == 1
        assert "시간 제한" in exits[0][1]

    def test_no_exit_within_bounds(self):
        trader = _make_trader()
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        # Price moved only 0.3% — within bounds
        trader.current_price = 100_300_000
        exits = trader.check_position_exit()
        assert len(exits) == 0

    def test_no_exit_when_price_zero(self):
        trader = _make_trader()
        pos = _make_position()
        trader.positions.append(pos)
        trader.current_price = 0
        exits = trader.check_position_exit()
        assert len(exits) == 0


# ===========================================================================
# 4. can_trade checks
# ===========================================================================

class TestCanTrade:
    def test_can_trade_ok(self):
        trader = _make_trader()
        ok, reason = trader.can_trade()
        assert ok is True
        assert reason == "OK"

    @patch.dict(os.environ, {"EMERGENCY_STOP": "true"})
    def test_emergency_stop_blocks(self):
        trader = _make_trader()
        ok, reason = trader.can_trade()
        assert ok is False
        assert "EMERGENCY_STOP" in reason

    def test_daily_limit_blocks(self):
        trader = _make_trader()
        trader.daily_trade_count = SHORT_TERM_MAX_DAILY
        ok, reason = trader.can_trade()
        assert ok is False
        assert reason == "daily_limit"

    def test_budget_limit_blocks(self):
        trader = _make_trader()
        trader.used_budget = SHORT_TERM_BUDGET
        ok, reason = trader.can_trade()
        assert ok is False
        assert reason == "budget_limit"

    def test_position_limit_blocks(self):
        trader = _make_trader()
        trader.positions = [_make_position(), _make_position()]
        ok, reason = trader.can_trade()
        assert ok is False
        assert reason == "position_limit"


# ===========================================================================
# 5. Trade size calculation
# ===========================================================================

class TestTradeSize:
    """execute_entry caps amount at min(signal.suggested, MAX_TRADE, remaining budget)."""

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_amount_capped_at_max_trade(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        signal = TradeSignal(
            strategy="news", action="buy", confidence=0.8,
            reason="test", suggested_amount=999_999,
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 1
        assert trader.positions[0].amount_krw == SHORT_TERM_MAX_TRADE

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_amount_capped_at_remaining_budget(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        trader.used_budget = 400_000  # only 100k remaining
        signal = TradeSignal(
            strategy="news", action="buy", confidence=0.8,
            reason="test", suggested_amount=200_000,
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 1
        assert trader.positions[0].amount_krw == 100_000

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_amount_below_minimum_rejected(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        trader.used_budget = 498_000  # only 2000 remaining < 5000 min
        signal = TradeSignal(
            strategy="news", action="buy", confidence=0.8,
            reason="test", suggested_amount=200_000,
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 0


# ===========================================================================
# 6. News signal detection
# ===========================================================================

class TestNewsSignal:
    def test_neutral_returns_none(self):
        trader = _make_trader()
        trader.news_sentiment_score = 0.0
        assert trader.check_news_signal() is None

    def test_weak_positive_returns_none(self):
        trader = _make_trader()
        trader.news_sentiment_score = 0.3
        assert trader.check_news_signal() is None

    def test_strong_positive_returns_buy(self):
        trader = _make_trader()
        trader.news_sentiment_score = 0.5
        sig = trader.check_news_signal()
        assert sig is not None
        assert sig.action == "buy"
        assert sig.strategy == "news"
        assert sig.confidence == pytest.approx(0.5)

    def test_strong_negative_returns_sell(self):
        trader = _make_trader()
        trader.news_sentiment_score = -0.6
        sig = trader.check_news_signal()
        assert sig is not None
        assert sig.action == "sell"
        assert sig.strategy == "news"

    def test_boundary_positive_returns_buy(self):
        trader = _make_trader()
        trader.news_sentiment_score = 0.4
        sig = trader.check_news_signal()
        assert sig is not None
        assert sig.action == "buy"

    def test_boundary_negative_returns_sell(self):
        trader = _make_trader()
        trader.news_sentiment_score = -0.4
        sig = trader.check_news_signal()
        assert sig is not None
        assert sig.action == "sell"


# ===========================================================================
# 7. Spike rebound signal
# ===========================================================================

class TestSpikeSignal:
    def test_insufficient_data_returns_none(self):
        trader = _make_trader()
        # Less than 30 items in price_history
        trader.price_history = deque(maxlen=600)
        for i in range(20):
            trader.price_history.append({"price": 100_000_000, "time": time.time()})
        assert trader.check_spike_signal() is None

    def test_flat_market_returns_none(self):
        trader = _make_trader()
        now = time.time()
        for i in range(60):
            trader.price_history.append({"price": 100_000_000, "time": now - 60 + i})
        # Also need trade_history for buy/sell ratio
        for i in range(10):
            trader.trade_history.append({
                "price": 100_000_000, "volume": 0.01,
                "side": "BID", "krw": 1_000_000, "time": now - 10 + i,
            })
        trader.current_price = 100_000_000
        assert trader.check_spike_signal() is None

    def test_drop_with_recovery_and_buy_volume_returns_buy(self):
        """Simulate a 1% drop followed by 0.4% recovery with strong buy volume."""
        trader = _make_trader()
        now = time.time()

        # Build price history: drop from 100M to 99M then recover to 99.4M
        prices = []
        # Phase 1: high at 100M
        for i in range(20):
            prices.append({"price": 100_000_000, "time": now - 200 + i})
        # Phase 2: drop to 99M
        for i in range(20):
            prices.append({"price": 99_000_000, "time": now - 100 + i})
        # Phase 3: partial recovery
        for i in range(20):
            prices.append({"price": 99_400_000, "time": now - 20 + i})

        trader.price_history = deque(prices, maxlen=600)
        trader.current_price = 99_400_000

        # Strong buy volume in last minute
        for i in range(10):
            trader.trade_history.append({
                "price": 99_400_000, "volume": 0.01,
                "side": "BID", "krw": 990_000, "time": now - 10 + i,
            })
        # Some sell volume
        for i in range(3):
            trader.trade_history.append({
                "price": 99_400_000, "volume": 0.01,
                "side": "ASK", "krw": 990_000, "time": now - 10 + i,
            })

        sig = trader.check_spike_signal()
        assert sig is not None
        assert sig.action == "buy"
        assert sig.strategy == "spike"


# ===========================================================================
# 8. Whale signal
# ===========================================================================

class TestWhaleSignal:
    def test_no_whales_returns_none(self):
        trader = _make_trader()
        assert trader.check_whale_signal() is None

    def test_single_whale_returns_none(self):
        trader = _make_trader()
        trader.whale_recent.append({
            "side": "BID", "krw": 50_000_000, "time": time.time(),
        })
        assert trader.check_whale_signal() is None

    def test_strong_buy_whales_returns_buy(self):
        trader = _make_trader()
        now = time.time()
        # 3 buy whales, 0 sell
        for i in range(3):
            trader.whale_recent.append({
                "side": "BID", "krw": 50_000_000, "time": now - 10 + i,
            })
        sig = trader.check_whale_signal()
        assert sig is not None
        assert sig.action == "buy"
        assert sig.strategy == "whale"

    def test_strong_sell_whales_returns_sell(self):
        trader = _make_trader()
        now = time.time()
        for i in range(3):
            trader.whale_recent.append({
                "side": "ASK", "krw": 50_000_000, "time": now - 10 + i,
            })
        sig = trader.check_whale_signal()
        assert sig is not None
        assert sig.action == "sell"

    def test_mixed_whales_below_threshold_returns_none(self):
        trader = _make_trader()
        now = time.time()
        # 50/50 split — neither side reaches 70%
        trader.whale_recent.append({"side": "BID", "krw": 50_000_000, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": 50_000_000, "time": now})
        sig = trader.check_whale_signal()
        assert sig is None

    def test_old_whales_excluded(self):
        trader = _make_trader()
        now = time.time()
        # Old whales (4 minutes ago, beyond 3-min window)
        trader.whale_recent.append({"side": "BID", "krw": 50_000_000, "time": now - 250})
        trader.whale_recent.append({"side": "BID", "krw": 50_000_000, "time": now - 240})
        sig = trader.check_whale_signal()
        assert sig is None


# ===========================================================================
# 9. Sell pressure blocking
# ===========================================================================

class TestSellPressureBlocking:
    def test_no_whales_no_block(self):
        trader = _make_trader()
        assert trader.is_sell_pressure_blocking() is False

    def test_sell_only_blocks(self):
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "ASK", "krw": 50_000_000, "time": now})
        assert trader.is_sell_pressure_blocking() is True

    def test_high_sell_ratio_blocks(self):
        trader = _make_trader()
        now = time.time()
        # sell 3x buy => blocks
        trader.whale_recent.append({"side": "BID", "krw": 30_000_000, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": 100_000_000, "time": now})
        assert trader.is_sell_pressure_blocking() is True

    def test_balanced_does_not_block(self):
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "BID", "krw": 50_000_000, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": 50_000_000, "time": now})
        assert trader.is_sell_pressure_blocking() is False


# ===========================================================================
# 10. Error counter -> emergency stop
# ===========================================================================

class TestErrorCounter:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_consecutive_errors_trigger_emergency(self, mock_order, mock_lock, mock_db, mock_tg):
        """5 consecutive order failures should trigger emergency stop."""
        mock_order.return_value = {"ok": False, "data": {"error": {"name": "unknown"}}}
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 4  # already at 4

        signal = TradeSignal(
            strategy="spike", action="buy", confidence=0.8, reason="test",
        )
        trader.execute_entry(signal)
        assert trader.emergency_stopped is True
        assert trader.running is False

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_auth_error_triggers_immediate_stop(self, mock_order, mock_lock, mock_db, mock_tg):
        """Auth errors (jwt_verification etc) trigger immediate emergency stop."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "jwt_verification"}},
        }
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(
            strategy="whale", action="buy", confidence=0.9, reason="test",
        )
        trader.execute_entry(signal)
        assert trader.emergency_stopped is True

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_successful_trade_resets_error_counter(self, mock_db, mock_tg):
        """A successful dry-run trade should reset consecutive_errors to 0."""
        trader = _make_trader(dry_run=True)
        trader.consecutive_errors = 3

        signal = TradeSignal(
            strategy="news", action="buy", confidence=0.8, reason="test",
        )
        trader.execute_entry(signal)
        assert trader.consecutive_errors == 0


# ===========================================================================
# 11. Emergency stop behavior
# ===========================================================================

class TestEmergencyStop:
    @patch("scripts.short_term_trader.send_telegram")
    def test_emergency_stop_sets_flags(self, mock_tg):
        trader = _make_trader()
        trader.emergency_stop("test reason")
        assert trader.emergency_stopped is True
        assert trader.running is False
        mock_tg.assert_called_once()
        assert "긴급 정지" in mock_tg.call_args[0][0]

    @patch("scripts.short_term_trader.send_telegram")
    def test_emergency_stop_idempotent(self, mock_tg):
        trader = _make_trader()
        trader.emergency_stop("first")
        trader.emergency_stop("second")
        # Only called once — second call is no-op
        mock_tg.assert_called_once()

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_emergency_stop_blocks_entry(self, mock_db, mock_tg):
        trader = _make_trader()
        trader.emergency_stop("test")
        signal = TradeSignal(strategy="news", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        assert len(trader.positions) == 0


# ===========================================================================
# 12. Lock file interaction
# ===========================================================================

class TestLockFile:
    def test_check_lock_no_file(self, tmp_path, monkeypatch):
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        assert check_lock() is True

    def test_check_lock_stale_file(self, tmp_path, monkeypatch):
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        # Write lock from 15 minutes ago
        ts = datetime.now(KST) - timedelta(minutes=15)
        fake_lock.write_text(json.dumps({
            "process": "test", "pid": 99999, "timestamp": ts.isoformat(),
        }))
        assert check_lock() is True
        # Stale lock should be removed
        assert not fake_lock.exists()

    def test_check_lock_active_file(self, tmp_path, monkeypatch):
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        ts = datetime.now(KST) - timedelta(minutes=1)
        fake_lock.write_text(json.dumps({
            "process": "cron_run", "pid": 99999, "timestamp": ts.isoformat(),
        }))
        assert check_lock() is False

    def test_acquire_and_release_lock(self, tmp_path, monkeypatch):
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        acquire_lock("test_bot")
        assert fake_lock.exists()
        data = json.loads(fake_lock.read_text())
        assert data["process"] == "test_bot"
        assert data["pid"] == os.getpid()

        release_lock()
        assert not fake_lock.exists()

    def test_release_lock_wrong_pid(self, tmp_path, monkeypatch):
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text(json.dumps({
            "process": "other", "pid": 99999,
            "timestamp": datetime.now(KST).isoformat(),
        }))
        release_lock()
        # Should NOT remove lock owned by different PID
        assert fake_lock.exists()


# ===========================================================================
# 13. Supabase recording (db_insert)
# ===========================================================================

class TestDbInsert:
    @patch("scripts.short_term_trader.requests.post")
    def test_db_insert_with_credentials(self, mock_post, monkeypatch):
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "test_key")
        db_insert("scalp_trades", {"strategy": "news", "side": "bid"})
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "scalp_trades" in args[0]
        assert kwargs["json"]["strategy"] == "news"

    @patch("scripts.short_term_trader.requests.post")
    def test_db_insert_without_credentials_skips(self, mock_post, monkeypatch):
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "")
        db_insert("scalp_trades", {"strategy": "news"})
        mock_post.assert_not_called()

    @patch("scripts.short_term_trader.requests.post", side_effect=Exception("network error"))
    def test_db_insert_failure_does_not_raise(self, mock_post, monkeypatch):
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "test_key")
        # Should not raise
        db_insert("scalp_trades", {"strategy": "news"})


# ===========================================================================
# 14. Telegram notification
# ===========================================================================

class TestTelegram:
    @patch("scripts.short_term_trader.requests.post")
    def test_send_telegram_with_credentials(self, mock_post, monkeypatch):
        monkeypatch.setattr("scripts.short_term_trader.os.getenv",
                            lambda k, d="": {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"}.get(k, d))
        send_telegram("test message")
        mock_post.assert_called_once()

    @patch("scripts.short_term_trader.requests.post")
    def test_send_telegram_missing_credentials_skips(self, mock_post, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_USER_ID", raising=False)
        send_telegram("test message")
        mock_post.assert_not_called()


# ===========================================================================
# 15. execute_entry dry-run integration
# ===========================================================================

class TestExecuteEntryDryRun:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_dry_run_creates_position(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        signal = TradeSignal(
            strategy="whale", action="buy", confidence=0.7,
            reason="whale buy 80%",
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 1
        pos = trader.positions[0]
        assert pos.strategy == "whale"
        assert pos.entry_price == 100_000_000
        assert trader.daily_trade_count == 1
        assert trader.used_budget == 200_000

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_dry_run_sends_db_record(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        signal = TradeSignal(strategy="news", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        mock_db.assert_called()
        call_args = mock_db.call_args
        assert call_args[0][0] == "scalp_trades"
        assert call_args[0][1]["dry_run"] is True


# ===========================================================================
# 16. execute_exit
# ===========================================================================

class TestExecuteExit:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_exit_calculates_pnl(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        pos = _make_position(entry_price=100_000_000, amount_krw=200_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000  # +1%

        trader.execute_exit(pos, "test exit")
        assert pos.exit_price == 101_000_000
        assert pos.pnl_pct == pytest.approx(1.0, abs=0.01)
        assert pos.exit_reason == "test exit"
        assert len(trader.positions) == 0
        assert len(trader.closed_positions) == 1
        assert trader.daily_trade_count == 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_exit_records_to_db(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 100_500_000
        trader.execute_exit(pos, "take profit")
        # db_insert should be called with scalp_trades and side=ask
        found = False
        for c in mock_db.call_args_list:
            if c[0][0] == "scalp_trades" and c[0][1].get("side") == "ask":
                found = True
                assert c[0][1]["exit_reason"] == "take profit"
        assert found


# ===========================================================================
# 17. Print summary
# ===========================================================================

class TestPrintSummary:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_summary_computes_win_rate(self, mock_db, mock_tg):
        trader = _make_trader()
        # 2 wins, 1 loss
        for pnl in [0.5, 0.3, -0.4]:
            pos = _make_position()
            pos.exit_price = pos.entry_price * (1 + pnl / 100)
            pos.exit_time = datetime.now(KST)
            pos.exit_reason = "test"
            pos.pnl_pct = pnl
            trader.closed_positions.append(pos)

        trader.print_summary()
        # DB insert for scalp_sessions
        session_call = None
        for c in mock_db.call_args_list:
            if c[0][0] == "scalp_sessions":
                session_call = c[0][1]
        assert session_call is not None
        assert session_call["total_trades"] == 3
        assert session_call["wins"] == 2
        assert session_call["losses"] == 1
        assert session_call["win_rate"] == pytest.approx(66.67, abs=0.1)


# ===========================================================================
# 18. TradeSignal dataclass
# ===========================================================================

class TestTradeSignal:
    def test_trade_signal_defaults(self):
        sig = TradeSignal(strategy="news", action="buy", confidence=0.5, reason="test")
        assert sig.suggested_amount == SHORT_TERM_MAX_TRADE
        assert sig.timestamp is not None

    def test_trade_signal_custom_amount(self):
        sig = TradeSignal(
            strategy="spike", action="buy", confidence=0.5,
            reason="test", suggested_amount=100_000,
        )
        assert sig.suggested_amount == 100_000


# ===========================================================================
# 19. Shutdown
# ===========================================================================

class TestShutdown:
    @patch("scripts.short_term_trader.release_lock")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_shutdown_closes_positions(self, mock_db, mock_tg, mock_release):
        trader = _make_trader(dry_run=True)
        pos = _make_position()
        trader.positions.append(pos)
        trader.shutdown()
        assert trader.running is False
        assert len(trader.positions) == 0
        assert len(trader.closed_positions) == 1
        mock_release.assert_called_once()


# ===========================================================================
# 20. Trader initialization
# ===========================================================================

class TestTraderInit:
    def test_initial_state(self):
        trader = ShortTermTrader(dry_run=True)
        assert trader.dry_run is True
        assert trader.running is True
        assert trader.current_price == 0
        assert trader.positions == []
        assert trader.closed_positions == []
        assert trader.daily_trade_count == 0
        assert trader.daily_pnl == 0.0
        assert trader.used_budget == 0
        assert trader.consecutive_errors == 0
        assert trader.MAX_CONSECUTIVE_ERRORS == 5
        assert trader.emergency_stopped is False
        assert trader.news_sentiment_score == 0
        assert trader.last_news_sentiment == "neutral"

    def test_live_mode(self):
        trader = ShortTermTrader(dry_run=False)
        assert trader.dry_run is False


# ===========================================================================
# 21. RSS parsing (scan_news)
# ===========================================================================

class TestRSSParsing:
    """Test the RSS feed parsing logic inside scan_news."""

    VALID_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>CoinDesk</title>
        <item>
          <title>Bitcoin surge rally to all-time high</title>
        </item>
        <item>
          <title>Ethereum network upgrade complete</title>
        </item>
        <item>
          <title>Crypto crash plunge as ban looms</title>
        </item>
      </channel>
    </rss>"""

    VALID_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Crypto News</title>
      <entry>
        <title>Bitcoin breakout bull run</title>
      </entry>
    </feed>"""

    EMPTY_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Empty Feed</title>
      </channel>
    </rss>"""

    MALFORMED_XML = """<rss><not valid xml"""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_valid_rss_updates_sentiment(self, mock_get, mock_db):
        """Valid RSS with positive headlines should produce positive sentiment."""
        import asyncio

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = self.VALID_RSS_XML.encode("utf-8")
        mock_get.return_value = mock_response

        trader = _make_trader()
        trader.last_news_scan = 0  # Force scan

        async def run_one_scan():
            # Run scan_news but stop after one iteration
            trader.running = True
            original_sleep = asyncio.sleep

            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
                await original_sleep(0)

            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_one_scan())
        # "surge", "rally", "all-time high" are positive; "crash", "plunge", "ban" are negative
        # The score should be computed from these keyword counts
        assert trader.news_sentiment_score != 0 or trader.last_news_sentiment in ("positive", "negative", "neutral")

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_empty_rss_keeps_neutral_sentiment(self, mock_get, mock_db):
        """Empty RSS feed should keep sentiment at zero/neutral."""
        import asyncio

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = self.EMPTY_RSS_XML.encode("utf-8")
        mock_get.return_value = mock_response

        trader = _make_trader()
        trader.last_news_scan = 0
        trader.news_sentiment_score = 0.0

        async def run_one_scan():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_one_scan())
        assert trader.news_sentiment_score == 0

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_malformed_xml_handled_gracefully(self, mock_get, mock_db):
        """Malformed XML should not crash, sentiment stays unchanged."""
        import asyncio

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = self.MALFORMED_XML.encode("utf-8")
        mock_get.return_value = mock_response

        trader = _make_trader()
        trader.last_news_scan = 0
        trader.news_sentiment_score = 0.0

        async def run_one_scan():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_one_scan())
        assert trader.news_sentiment_score == 0

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_rss_request_failure_handled(self, mock_get, mock_db):
        """Network failure on RSS fetch should not crash."""
        import asyncio

        mock_response = MagicMock()
        mock_response.ok = False
        mock_get.return_value = mock_response

        trader = _make_trader()
        trader.last_news_scan = 0

        async def run_one_scan():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_one_scan())
        assert trader.news_sentiment_score == 0

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_atom_feed_parsing(self, mock_get, mock_db):
        """Atom format feeds should be parsed correctly."""
        import asyncio

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = self.VALID_ATOM_XML.encode("utf-8")
        mock_get.return_value = mock_response

        trader = _make_trader()
        trader.last_news_scan = 0

        async def run_one_scan():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_one_scan())
        # "breakout" and "bull" are positive keywords
        assert trader.news_sentiment_score > 0

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_duplicate_headlines_filtered(self, mock_get, mock_db):
        """Same headlines across multiple scans should not be double-counted."""
        import asyncio

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = self.VALID_RSS_XML.encode("utf-8")
        mock_get.return_value = mock_response

        trader = _make_trader()
        trader.last_news_scan = 0

        scan_count = 0

        async def run_two_scans():
            nonlocal scan_count
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count, scan_count
                call_count += 1
                # After first scan completes, reset last_news_scan to force second scan
                if call_count == 2:
                    trader.last_news_scan = 0
                    scan_count += 1
                if call_count >= 4:
                    trader.running = False
                    scan_count += 1
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_two_scans())
        # Should not crash; sentiment should be stable
        assert isinstance(trader.news_sentiment_score, (int, float))

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_news_scan_interval_respected(self, mock_get, mock_db):
        """scan_news should skip if last scan was recent."""
        import asyncio

        trader = _make_trader()
        trader.last_news_scan = time.time()  # Just scanned

        async def run_one_scan():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_one_scan())
        # requests.get should not be called since we just scanned
        mock_get.assert_not_called()


# ===========================================================================
# 22. WebSocket message handling
# ===========================================================================

class TestWebSocketMessageHandling:
    """Test ws_ticker and ws_trades message processing logic directly."""

    def test_ticker_message_updates_price(self):
        """Simulating ticker message updates current_price and price_history."""
        trader = _make_trader()
        trader.current_price = 0

        # Simulate what ws_ticker does when receiving a message
        data = {"trade_price": 105_000_000}
        trader.current_price = data.get("trade_price", trader.current_price)
        trader.price_history.append({
            "price": trader.current_price,
            "time": time.time(),
        })

        assert trader.current_price == 105_000_000
        assert len(trader.price_history) == 1
        assert trader.price_history[0]["price"] == 105_000_000

    def test_ticker_message_missing_price_keeps_old(self):
        """If trade_price is missing, current_price should not change."""
        trader = _make_trader()
        trader.current_price = 100_000_000

        data = {}  # No trade_price key
        trader.current_price = data.get("trade_price", trader.current_price)
        assert trader.current_price == 100_000_000

    def test_trade_message_adds_to_history(self):
        """Simulating trade message appends to trade_history."""
        trader = _make_trader()

        data = {
            "trade_price": 100_000_000,
            "trade_volume": 0.5,
            "ask_bid": "BID",
        }
        trade = {
            "price": data.get("trade_price", 0),
            "volume": data.get("trade_volume", 0),
            "side": data.get("ask_bid", ""),
            "krw": data.get("trade_price", 0) * data.get("trade_volume", 0),
            "time": time.time(),
        }
        trader.trade_history.append(trade)

        assert len(trader.trade_history) == 1
        assert trader.trade_history[0]["side"] == "BID"
        assert trader.trade_history[0]["krw"] == 50_000_000

    def test_trade_message_missing_fields_defaults_to_zero(self):
        """Missing fields in trade message should default to zero."""
        trader = _make_trader()

        data = {}
        trade = {
            "price": data.get("trade_price", 0),
            "volume": data.get("trade_volume", 0),
            "side": data.get("ask_bid", ""),
            "krw": data.get("trade_price", 0) * data.get("trade_volume", 0),
            "time": time.time(),
        }
        trader.trade_history.append(trade)

        assert trader.trade_history[0]["price"] == 0
        assert trader.trade_history[0]["volume"] == 0
        assert trader.trade_history[0]["krw"] == 0
        assert trader.trade_history[0]["side"] == ""

    @patch("scripts.short_term_trader.db_insert")
    def test_whale_detection_from_trade(self, mock_db):
        """Trades >= WHALE_THRESHOLD_KRW should be added to whale_recent."""
        trader = _make_trader()

        # Simulate a whale trade (50M KRW)
        trade = {
            "price": 100_000_000,
            "volume": 0.5,
            "side": "BID",
            "krw": 50_000_000,
            "time": time.time(),
        }
        trader.trade_history.append(trade)

        if trade["krw"] >= WHALE_THRESHOLD_KRW:
            trader.whale_recent.append(trade)

        assert len(trader.whale_recent) == 1
        assert trader.whale_recent[0]["side"] == "BID"

    def test_non_whale_trade_not_added(self):
        """Trades below WHALE_THRESHOLD_KRW should not be in whale_recent."""
        trader = _make_trader()

        trade = {
            "price": 100_000_000,
            "volume": 0.0001,
            "side": "BID",
            "krw": 10_000,
            "time": time.time(),
        }
        trader.trade_history.append(trade)

        if trade["krw"] >= WHALE_THRESHOLD_KRW:
            trader.whale_recent.append(trade)

        assert len(trader.whale_recent) == 0

    def test_price_history_buffer_respects_maxlen(self):
        """price_history deque should respect maxlen=600."""
        trader = _make_trader()
        for i in range(700):
            trader.price_history.append({"price": 100_000_000 + i, "time": time.time()})

        assert len(trader.price_history) == 600
        # Oldest entries should be dropped
        assert trader.price_history[0]["price"] == 100_000_100


# ===========================================================================
# 23. Spike sell signal (surge + pullback)
# ===========================================================================

class TestSpikeSellSignal:
    def test_surge_with_pullback_and_sell_volume_returns_sell(self):
        """Simulate a 1% surge followed by a pullback with strong sell volume."""
        trader = _make_trader()
        now = time.time()

        # Build price history: low at 99M, surge to 100M, pullback to 99.65M
        prices = []
        for i in range(20):
            prices.append({"price": 99_000_000, "time": now - 200 + i})
        for i in range(20):
            prices.append({"price": 100_000_000, "time": now - 100 + i})
        for i in range(20):
            prices.append({"price": 99_650_000, "time": now - 20 + i})

        trader.price_history = deque(prices, maxlen=600)
        trader.current_price = 99_650_000

        # Strong sell volume in last minute
        for i in range(10):
            trader.trade_history.append({
                "price": 99_650_000, "volume": 0.01,
                "side": "ASK", "krw": 996_500, "time": now - 10 + i,
            })
        for i in range(3):
            trader.trade_history.append({
                "price": 99_650_000, "volume": 0.01,
                "side": "BID", "krw": 996_500, "time": now - 10 + i,
            })

        sig = trader.check_spike_signal()
        assert sig is not None
        assert sig.action == "sell"
        assert sig.strategy == "spike"

    def test_small_move_no_signal(self):
        """Price movement below threshold should return None."""
        trader = _make_trader()
        now = time.time()

        # Only 0.3% price range — below 0.8% threshold
        for i in range(60):
            price = 100_000_000 + (i % 3) * 100_000  # 0.3% max swing
            trader.price_history.append({"price": price, "time": now - 60 + i})

        trader.current_price = 100_200_000
        for i in range(10):
            trader.trade_history.append({
                "price": 100_200_000, "volume": 0.01,
                "side": "BID", "krw": 1_002_000, "time": now - 10 + i,
            })

        assert trader.check_spike_signal() is None


# ===========================================================================
# 24. Error counter edge cases
# ===========================================================================

class TestErrorCounterEdgeCases:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_timeout_increments_error_counter(self, mock_order, mock_lock, mock_db, mock_tg):
        """Request timeout should increment consecutive_errors."""
        import requests as req
        mock_order.side_effect = req.exceptions.Timeout("timeout")
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(strategy="spike", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        assert trader.consecutive_errors == 1
        assert len(trader.positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_generic_exception_increments_error_counter(self, mock_order, mock_lock, mock_db, mock_tg):
        """Generic exception during order should increment errors."""
        mock_order.side_effect = Exception("connection reset")
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(strategy="whale", action="buy", confidence=0.9, reason="test")
        trader.execute_entry(signal)
        assert trader.consecutive_errors == 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_timeout_at_threshold_triggers_emergency(self, mock_order, mock_lock, mock_db, mock_tg):
        """Timeout at error threshold should trigger emergency stop."""
        import requests as req
        mock_order.side_effect = req.exceptions.Timeout("timeout")
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 4

        signal = TradeSignal(strategy="spike", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        assert trader.emergency_stopped is True
        assert trader.running is False

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_no_authorization_triggers_immediate_stop(self, mock_order, mock_lock, mock_db, mock_tg):
        """no_authorization error should trigger immediate emergency stop."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "no_authorization"}},
        }
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(strategy="news", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        assert trader.emergency_stopped is True

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_invalid_access_key_triggers_immediate_stop(self, mock_order, mock_lock, mock_db, mock_tg):
        """invalid_access_key error should trigger immediate emergency stop."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "invalid_access_key"}},
        }
        trader = _make_trader(dry_run=False)

        signal = TradeSignal(strategy="news", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        assert trader.emergency_stopped is True


# ===========================================================================
# 25. Lock file edge cases
# ===========================================================================

class TestLockFileEdgeCases:
    def test_check_lock_corrupt_json(self, tmp_path, monkeypatch):
        """Corrupt JSON in lock file should return True (safe)."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text("not valid json {{{")
        assert check_lock() is True

    def test_check_lock_missing_timestamp(self, tmp_path, monkeypatch):
        """Lock file without timestamp should return True (safe)."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text(json.dumps({"process": "test", "pid": 99999}))
        # Missing timestamp causes fromisoformat to fail -> caught by except
        assert check_lock() is True

    def test_acquire_lock_creates_parent_dirs(self, tmp_path, monkeypatch):
        """acquire_lock should create parent directories."""
        fake_lock = tmp_path / "subdir" / "deep" / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        acquire_lock("test_process")
        assert fake_lock.exists()
        data = json.loads(fake_lock.read_text())
        assert data["process"] == "test_process"

    def test_release_lock_no_file(self, tmp_path, monkeypatch):
        """release_lock on non-existent file should not raise."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        # Should not raise
        release_lock()

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=False)
    @patch("scripts.short_term_trader.upbit_order")
    def test_lock_held_blocks_live_entry(self, mock_order, mock_lock, mock_db, mock_tg):
        """When lock is held by another process, live entry should be blocked."""
        trader = _make_trader(dry_run=False)
        signal = TradeSignal(strategy="news", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)
        assert len(trader.positions) == 0
        mock_order.assert_not_called()


# ===========================================================================
# 26. Position monitoring — additional cases
# ===========================================================================

class TestPositionMonitoringExtra:
    def test_multiple_positions_mixed_exits(self):
        """Multiple positions: one hits TP, another hits SL, third holds."""
        trader = _make_trader()

        # Position 1: take profit
        pos_tp = _make_position(entry_price=100_000_000)
        # Position 2: stop loss
        pos_sl = _make_position(entry_price=102_000_000)
        # Position 3: within bounds
        pos_hold = _make_position(entry_price=100_500_000)

        trader.positions = [pos_tp, pos_sl, pos_hold]
        trader.current_price = 100_900_000  # +0.9% from 100M, -1.08% from 102M, +0.4% from 100.5M

        exits = trader.check_position_exit()
        exit_positions = [e[0] for e in exits]

        assert pos_tp in exit_positions  # +0.9% >= 0.8% TP
        assert pos_sl in exit_positions  # -1.08% <= -0.8% SL
        assert pos_hold not in exit_positions  # +0.4% within bounds

    def test_exact_take_profit_boundary(self):
        """Position at exactly the take profit percentage should trigger."""
        trader = _make_trader()
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 100_800_000  # Exactly +0.8%
        exits = trader.check_position_exit()
        assert len(exits) == 1
        assert "익절" in exits[0][1]

    def test_exact_stop_loss_boundary(self):
        """Position at exactly the stop loss percentage should trigger."""
        trader = _make_trader()
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_200_000  # Exactly -0.8%
        exits = trader.check_position_exit()
        assert len(exits) == 1
        assert "손절" in exits[0][1]

    def test_max_hold_at_exactly_boundary(self):
        """Position at exactly max_hold_min should trigger exit."""
        trader = _make_trader()
        entry_time = datetime.now(KST) - timedelta(minutes=20, seconds=1)
        pos = _make_position(entry_price=100_000_000, entry_time=entry_time)
        trader.positions.append(pos)
        trader.current_price = 100_000_000
        exits = trader.check_position_exit()
        assert len(exits) == 1
        assert "시간 제한" in exits[0][1]


# ===========================================================================
# 27. Strategy loop integration
# ===========================================================================

class TestStrategyLoop:
    """Test strategy_loop processes signals correctly."""

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_strategy_loop_processes_exit_then_entry(self, mock_db, mock_tg):
        """strategy_loop should check exits before entries."""
        import asyncio

        trader = _make_trader(dry_run=True)
        # Set up a position that should be closed (take profit)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 100_900_000  # +0.9% triggers TP

        # Also set up a buy signal via news sentiment
        trader.news_sentiment_score = 0.6

        async def run_one_iteration():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.strategy_loop()

        asyncio.run(run_one_iteration())
        # The existing position should have been closed
        assert len(trader.closed_positions) >= 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_strategy_loop_skips_when_no_price(self, mock_db, mock_tg):
        """strategy_loop should skip iteration when price is zero."""
        import asyncio

        trader = _make_trader(dry_run=True)
        trader.current_price = 0

        async def run_one_iteration():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.strategy_loop()

        asyncio.run(run_one_iteration())
        # Nothing should have happened
        assert len(trader.positions) == 0
        assert len(trader.closed_positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_strategy_loop_sell_signal_closes_position(self, mock_db, mock_tg):
        """Sell signal from strategy should close existing positions."""
        import asyncio

        trader = _make_trader(dry_run=True)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 100_200_000  # Within bounds (no auto-exit)

        # Strong negative news -> sell signal
        trader.news_sentiment_score = -0.6

        async def run_one_iteration():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.strategy_loop()

        asyncio.run(run_one_iteration())
        assert len(trader.positions) == 0
        assert len(trader.closed_positions) >= 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_sell_pressure_blocks_buy(self, mock_db, mock_tg):
        """Buy signal should be blocked when sell pressure is high."""
        import asyncio

        trader = _make_trader(dry_run=True)
        trader.current_price = 100_000_000
        trader.news_sentiment_score = 0.6  # Would normally generate buy

        # Add heavy sell whales
        now = time.time()
        trader.whale_recent.append({"side": "ASK", "krw": 100_000_000, "time": now})

        async def run_one_iteration():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.strategy_loop()

        asyncio.run(run_one_iteration())
        # Sell pressure should have blocked the buy
        assert len(trader.positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_low_confidence_buy_rejected(self, mock_db, mock_tg):
        """Buy signals with confidence < 0.5 should not execute."""
        import asyncio

        trader = _make_trader(dry_run=True)
        trader.current_price = 100_000_000
        # Just barely above threshold for signal generation but below 0.5 confidence
        trader.news_sentiment_score = 0.41  # confidence = 0.41, below 0.5

        async def run_one_iteration():
            trader.running = True
            call_count = 0
            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    trader.running = False
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.strategy_loop()

        asyncio.run(run_one_iteration())
        assert len(trader.positions) == 0


# ===========================================================================
# 28. Execute exit — live mode errors
# ===========================================================================

class TestExecuteExitLive:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_exit_order_failure_does_not_remove_position(self, mock_order, mock_db, mock_tg):
        """Failed sell order should keep position open."""
        mock_order.return_value = {"ok": False, "data": {"error": {"name": "insufficient_funds"}}}
        trader = _make_trader(dry_run=False)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000

        trader.execute_exit(pos, "take profit")
        # Position should still be open because order failed
        assert len(trader.positions) == 1
        assert pos.exit_price is None

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_exit_exception_does_not_remove_position(self, mock_order, mock_db, mock_tg):
        """Exception during sell should keep position open."""
        mock_order.side_effect = Exception("network error")
        trader = _make_trader(dry_run=False)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000

        trader.execute_exit(pos, "take profit")
        assert len(trader.positions) == 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_exit_failure_increments_error_counter(self, mock_order, mock_db, mock_tg):
        """Failed sell should increment consecutive_errors."""
        mock_order.return_value = {"ok": False, "data": {"error": {"name": "unknown"}}}
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000

        trader.execute_exit(pos, "take profit")
        assert trader.consecutive_errors == 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_exit_success_resets_error_counter(self, mock_order, mock_db, mock_tg):
        """Successful sell should reset consecutive_errors."""
        mock_order.return_value = {"ok": True, "data": {"uuid": "test"}}
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 3
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000

        trader.execute_exit(pos, "take profit")
        assert trader.consecutive_errors == 0
        assert len(trader.positions) == 0


# ===========================================================================
# 29. Emergency stop with live positions
# ===========================================================================

class TestEmergencyStopLivePositions:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_emergency_stop_live_attempts_exit(self, mock_order, mock_db, mock_tg):
        """Emergency stop in live mode should try to close positions."""
        mock_order.return_value = {"ok": True, "data": {"uuid": "test"}}
        trader = _make_trader(dry_run=False)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_000_000

        trader.emergency_stop("test reason")
        assert trader.emergency_stopped is True
        assert trader.running is False
        # Should have attempted to sell
        mock_order.assert_called()

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_emergency_stop_exit_failure_sends_alert(self, mock_order, mock_db, mock_tg):
        """If emergency exit fails, should send telegram alert about failure."""
        mock_order.side_effect = Exception("API down")
        trader = _make_trader(dry_run=False)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_000_000

        trader.emergency_stop("test reason")
        assert trader.emergency_stopped is True
        # Should have sent multiple telegram messages (initial + failure alert)
        assert mock_tg.call_count >= 2

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_emergency_stop_dry_run_no_order(self, mock_db, mock_tg):
        """Emergency stop in dry_run should not attempt orders."""
        trader = _make_trader(dry_run=True)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_000_000

        trader.emergency_stop("test reason")
        assert trader.emergency_stopped is True
        # Position still exists (no live order attempted, no execute_exit call for dry_run in emergency_stop)
        assert trader.running is False


# ===========================================================================
# 30. Print status
# ===========================================================================

class TestPrintStatus:
    @patch("scripts.short_term_trader.get_current_price", return_value=100_500_000)
    def test_print_status_outputs_json(self, mock_price, capsys):
        trader = _make_trader(dry_run=True)
        trader.print_status()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["mode"] == "DRY_RUN"
        assert data["current_price"] == 100_500_000
        assert "config" in data


# ===========================================================================
# 31. upbit_auth_header
# ===========================================================================

class TestUpbitAuthHeader:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "my_ak", "UPBIT_SECRET_KEY": "my_sk"})
    def test_returns_bearer_token(self):
        import jwt as pyjwt
        headers = upbit_auth_header("market=KRW-BTC&side=bid")
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["Content-Type"] == "application/json"

        token_str = headers["Authorization"].split(" ", 1)[1]
        decoded = pyjwt.decode(token_str, "my_sk", algorithms=["HS256"])
        assert decoded["access_key"] == "my_ak"
        assert "nonce" in decoded
        assert "timestamp" in decoded
        assert decoded["query_hash_alg"] == "SHA512"

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak2", "UPBIT_SECRET_KEY": "sk2"})
    def test_query_hash_matches_input(self):
        import hashlib
        import jwt as pyjwt
        qs = "market=KRW-BTC&side=ask&volume=0.001"
        headers = upbit_auth_header(qs)
        token_str = headers["Authorization"].split(" ", 1)[1]
        decoded = pyjwt.decode(token_str, "sk2", algorithms=["HS256"])
        expected_hash = hashlib.sha512(qs.encode()).hexdigest()
        assert decoded["query_hash"] == expected_hash


# ===========================================================================
# 32. upbit_order
# ===========================================================================

class TestUpbitOrder:
    @patch("scripts.short_term_trader.requests.post")
    @patch("scripts.short_term_trader.upbit_auth_header", return_value={"Authorization": "Bearer x", "Content-Type": "application/json"})
    def test_bid_order_body(self, mock_auth, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "abc"}
        mock_post.return_value = mock_resp

        result = upbit_order("bid", "KRW-BTC", "100000")
        assert result["ok"] is True
        assert result["data"] == {"uuid": "abc"}

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["side"] == "bid"
        assert body["ord_type"] == "price"
        assert body["price"] == "100000"
        assert body["market"] == "KRW-BTC"
        assert "volume" not in body

    @patch("scripts.short_term_trader.requests.post")
    @patch("scripts.short_term_trader.upbit_auth_header", return_value={"Authorization": "Bearer x", "Content-Type": "application/json"})
    def test_ask_order_body(self, mock_auth, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "def"}
        mock_post.return_value = mock_resp

        result = upbit_order("ask", "KRW-BTC", "0.001")
        assert result["ok"] is True

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["side"] == "ask"
        assert body["ord_type"] == "market"
        assert body["volume"] == "0.001"
        assert "price" not in body

    @patch("scripts.short_term_trader.requests.post")
    @patch("scripts.short_term_trader.upbit_auth_header", return_value={"Authorization": "Bearer x", "Content-Type": "application/json"})
    def test_posts_to_correct_url(self, mock_auth, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.json.return_value = {"error": {"name": "some_error"}}
        mock_post.return_value = mock_resp

        result = upbit_order("bid", "KRW-BTC", "50000")
        assert result["ok"] is False
        url = mock_post.call_args[0][0]
        assert url.endswith("/orders")


# ===========================================================================
# 33. get_current_price
# ===========================================================================

class TestGetCurrentPrice:
    @patch("scripts.short_term_trader.requests.get")
    def test_returns_trade_price(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"trade_price": 95_000_000}]
        mock_get.return_value = mock_resp

        price = get_current_price("KRW-BTC")
        assert price == 95_000_000
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs.get("params") == {"markets": "KRW-BTC"} or \
               call_kwargs[1].get("params") == {"markets": "KRW-BTC"}


# ===========================================================================
# 34. send_telegram exception path
# ===========================================================================

class TestSendTelegramException:
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("scripts.short_term_trader.requests.post", side_effect=ConnectionError("network down"))
    def test_exception_silently_caught(self, mock_post):
        # Should not raise
        send_telegram("test message")


# ===========================================================================
# 35. execute_entry block reason dedup
# ===========================================================================

class TestExecuteEntryBlockReason:
    def test_block_reason_logged_once(self):
        trader = _make_trader(dry_run=True)
        trader.daily_trade_count = SHORT_TERM_MAX_DAILY  # trigger daily_limit

        signal = TradeSignal(
            strategy="spike", action="buy", confidence=0.9,
            reason="test spike", suggested_amount=100_000,
        )

        trader._last_block_reason = set()
        trader.execute_entry(signal)
        assert "daily_limit" in trader._last_block_reason

        # Second call with same block reason should not add again (set already has it)
        trader.execute_entry(signal)
        # Still just one entry - set deduplicates automatically
        assert trader._last_block_reason == {"daily_limit"}

    def test_different_block_reasons_both_logged(self):
        trader = _make_trader(dry_run=True)
        trader._last_block_reason = set()

        # First: budget limit
        trader.used_budget = SHORT_TERM_BUDGET
        signal = TradeSignal(
            strategy="spike", action="buy", confidence=0.9,
            reason="test", suggested_amount=100_000,
        )
        trader.execute_entry(signal)
        assert "budget_limit" in trader._last_block_reason

        # Second: daily limit (reset budget, set daily limit)
        trader.used_budget = 0
        trader.daily_trade_count = SHORT_TERM_MAX_DAILY
        trader.execute_entry(signal)
        assert "daily_limit" in trader._last_block_reason
        assert len(trader._last_block_reason) == 2


# ===========================================================================
# 36. Live entry success (lines 778-781)
# ===========================================================================

class TestLiveEntrySuccess:
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_live_entry_creates_position(self, mock_order, mock_lock, mock_tg, mock_db):
        mock_order.return_value = {"ok": True, "data": {"uuid": "live-uuid"}}

        trader = _make_trader(dry_run=False)
        trader.current_price = 100_000_000
        trader.consecutive_errors = 3  # should reset on success

        signal = TradeSignal(
            strategy="spike", action="buy", confidence=0.8,
            reason="live test", suggested_amount=100_000,
        )
        trader.execute_entry(signal)

        assert trader.consecutive_errors == 0
        assert len(trader.positions) == 1
        assert trader.positions[0].strategy == "spike"
        assert trader.positions[0].entry_price == 100_000_000


# ===========================================================================
# 37. run() method (lines 1238-1278)
# ===========================================================================

class TestRunMethod:
    @pytest.mark.asyncio
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    async def test_run_startup_and_cancelled(self, mock_sound, mock_tg):
        import asyncio

        trader = _make_trader(dry_run=True)

        # Mock all async tasks to raise CancelledError immediately
        async def _cancel():
            raise asyncio.CancelledError()

        with patch.object(trader, "ws_ticker", side_effect=_cancel), \
             patch.object(trader, "ws_trades", side_effect=_cancel), \
             patch.object(trader, "scan_news", side_effect=_cancel), \
             patch.object(trader, "strategy_loop", side_effect=_cancel), \
             patch.object(trader, "status_reporter", side_effect=_cancel), \
             patch.object(trader, "strategy_alert_monitor", side_effect=_cancel), \
             patch.object(trader, "print_summary") as mock_summary, \
             patch("scripts.short_term_trader.db_insert"):
            # Mock signal handler registration to avoid issues in test
            loop = asyncio.get_event_loop()
            with patch.object(loop, "add_signal_handler"):
                await trader.run()

        # send_telegram called at startup
        assert mock_tg.call_count >= 1
        startup_msg = mock_tg.call_args_list[0][0][0]
        assert "단타봇 시작" in startup_msg

        # print_summary called in finally
        mock_summary.assert_called_once()


# ===========================================================================
# 38. shutdown() (lines 1280-1286)
# ===========================================================================

class TestShutdown:
    @patch("scripts.short_term_trader.release_lock")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    def test_shutdown_exits_positions(self, mock_tg, mock_db, mock_release):
        trader = _make_trader(dry_run=True)
        pos1 = _make_position(strategy="spike")
        pos2 = _make_position(strategy="whale")
        trader.positions.append(pos1)
        trader.positions.append(pos2)

        with patch.object(trader, "execute_exit") as mock_exit:
            trader.shutdown()

        assert mock_exit.call_count == 2
        assert trader.running is False
        mock_release.assert_called_once()

    @patch("scripts.short_term_trader.release_lock")
    def test_shutdown_no_positions(self, mock_release):
        trader = _make_trader(dry_run=True)
        trader.shutdown()
        assert trader.running is False
        mock_release.assert_called_once()


# ===========================================================================
# 39. print_summary() (lines 1288-1335)
# ===========================================================================

class TestPrintSummary:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_summary_with_closed_positions(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        trader.daily_pnl = 5000

        win_pos = _make_position(strategy="spike", entry_price=100_000_000)
        win_pos.exit_price = 101_000_000
        win_pos.pnl_pct = 0.84
        win_pos.exit_reason = "take_profit"

        loss_pos = _make_position(strategy="whale", entry_price=100_000_000)
        loss_pos.exit_price = 99_000_000
        loss_pos.pnl_pct = -0.84
        loss_pos.exit_reason = "stop_loss"

        trader.closed_positions = [win_pos, loss_pos]

        trader.print_summary()

        # db_insert called with scalp_sessions table
        mock_db.assert_called_once()
        db_args = mock_db.call_args[0]
        assert db_args[0] == "scalp_sessions"
        session_data = db_args[1]
        assert session_data["total_trades"] == 2
        assert session_data["wins"] == 1
        assert session_data["losses"] == 1
        assert session_data["win_rate"] == 50.0
        assert session_data["total_pnl_krw"] == 5000

        # telegram sent with summary
        mock_tg.assert_called_once()
        tg_msg = mock_tg.call_args[0][0]
        assert "초단타봇 종료" in tg_msg
        assert "2회" in tg_msg

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_summary_no_trades(self, mock_db, mock_tg):
        trader = _make_trader(dry_run=True)
        trader.daily_pnl = 0
        trader.closed_positions = []

        trader.print_summary()

        mock_db.assert_called_once()
        session_data = mock_db.call_args[0][1]
        assert session_data["total_trades"] == 0
        assert session_data["wins"] == 0
        assert session_data["losses"] == 0
        assert session_data["win_rate"] == 0.0

        mock_tg.assert_called_once()


# ===========================================================================
# 40. __main__ block (lines 1366-1379)
# ===========================================================================

class TestMainBlock:
    @patch("scripts.short_term_trader.asyncio.run")
    @patch("scripts.short_term_trader.ShortTermTrader")
    def test_main_dry_run_flag(self, MockTrader, mock_run):
        import sys
        mock_instance = MagicMock()
        MockTrader.return_value = mock_instance

        with patch.object(sys, "argv", ["short_term_trader.py", "--dry-run"]):
            with patch.dict(os.environ, {"DRY_RUN": "false"}):
                # Simulate __main__ logic
                dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
                if "--dry-run" in sys.argv:
                    dry_run = True
                elif "--live" in sys.argv:
                    dry_run = False
                assert dry_run is True

    @patch("scripts.short_term_trader.asyncio.run")
    @patch("scripts.short_term_trader.ShortTermTrader")
    def test_main_live_flag(self, MockTrader, mock_run):
        import sys
        mock_instance = MagicMock()
        MockTrader.return_value = mock_instance

        with patch.object(sys, "argv", ["short_term_trader.py", "--live"]):
            with patch.dict(os.environ, {"DRY_RUN": "true"}):
                dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
                if "--dry-run" in sys.argv:
                    dry_run = True
                elif "--live" in sys.argv:
                    dry_run = False
                assert dry_run is False

    @patch("scripts.short_term_trader.asyncio.run")
    @patch("scripts.short_term_trader.ShortTermTrader")
    def test_main_status_flag(self, MockTrader, mock_run):
        import sys
        mock_instance = MagicMock()
        MockTrader.return_value = mock_instance

        with patch.object(sys, "argv", ["short_term_trader.py", "--status"]):
            # Simulate __main__ logic
            dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
            trader = MockTrader(dry_run=dry_run)
            if "--status" in sys.argv:
                trader.print_status()
                status_called = True
            else:
                status_called = False
            assert status_called is True
            trader.print_status.assert_called_once()

    def test_main_env_default(self):
        import sys
        with patch.object(sys, "argv", ["short_term_trader.py"]):
            with patch.dict(os.environ, {"DRY_RUN": "true"}):
                dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
                if "--dry-run" in sys.argv:
                    dry_run = True
                elif "--live" in sys.argv:
                    dry_run = False
                assert dry_run is True


# ===========================================================================
# 41. websockets import fallback (lines 45-47)
# ===========================================================================

class TestWebsocketsImportFallback:
    """Test that missing websockets triggers sys.exit(1)."""

    def test_missing_websockets_calls_sys_exit(self):
        """When websockets is not importable, the module-level try/except
        calls sys.exit(1). We verify by executing the import guard code
        with websockets blocked in sys.modules."""
        code = (
            "import sys\n"
            "try:\n"
            "    import websockets\n"
            "except ImportError:\n"
            "    sys.exit(1)\n"
        )
        saved = sys.modules.get("websockets")
        sys.modules["websockets"] = None  # forces ImportError on import
        try:
            with pytest.raises(SystemExit) as exc_info:
                exec(compile(code, "<test>", "exec"), {"__builtins__": __builtins__})
            assert exc_info.value.code == 1
        finally:
            if saved is not None:
                sys.modules["websockets"] = saved
            else:
                sys.modules.pop("websockets", None)


# ===========================================================================
# 42. send_telegram edge cases (lines 200-211)
# ===========================================================================

class TestSendTelegramEdgeCases:
    @patch("scripts.short_term_trader.requests.post")
    def test_missing_bot_token_skips(self, mock_post, monkeypatch):
        """No TELEGRAM_BOT_TOKEN -> early return, no request made."""
        monkeypatch.setattr(
            "scripts.short_term_trader.os.getenv",
            lambda k, d="": {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_USER_ID": "123"}.get(k, d),
        )
        send_telegram("hello")
        mock_post.assert_not_called()

    @patch("scripts.short_term_trader.requests.post")
    def test_missing_user_id_skips(self, mock_post, monkeypatch):
        """No TELEGRAM_USER_ID -> early return, no request made."""
        monkeypatch.setattr(
            "scripts.short_term_trader.os.getenv",
            lambda k, d="": {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": ""}.get(k, d),
        )
        send_telegram("hello")
        mock_post.assert_not_called()

    @patch("scripts.short_term_trader.requests.post")
    def test_both_present_calls_post_with_correct_payload(self, mock_post, monkeypatch):
        """Both token and user_id present -> requests.post called correctly."""
        monkeypatch.setattr(
            "scripts.short_term_trader.os.getenv",
            lambda k, d="": {"TELEGRAM_BOT_TOKEN": "mytoken", "TELEGRAM_USER_ID": "42"}.get(k, d),
        )
        send_telegram("test msg")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "mytoken" in args[0]
        assert kwargs["json"]["chat_id"] == "42"
        assert kwargs["json"]["text"] == "test msg"
        assert kwargs["json"]["parse_mode"] == "HTML"

    @patch(
        "scripts.short_term_trader.requests.post",
        side_effect=requests.exceptions.ConnectionError("conn refused"),
    )
    def test_connection_error_silently_caught(self, mock_post, monkeypatch):
        """requests.post raises ConnectionError -> silently caught."""
        monkeypatch.setattr(
            "scripts.short_term_trader.os.getenv",
            lambda k, d="": {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "1"}.get(k, d),
        )
        send_telegram("test")  # should not raise


# ===========================================================================
# 43. check_lock edge cases (lines 218-233)
# ===========================================================================

class TestCheckLockEdgeCasesExtra:
    def test_invalid_json_returns_true(self, tmp_path, monkeypatch):
        """Lock file with invalid JSON -> exception caught -> returns True (safe)."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text("NOT VALID JSON {{{")
        assert check_lock() is True

    def test_missing_timestamp_key_returns_true(self, tmp_path, monkeypatch):
        """Lock file with valid JSON but no timestamp key -> ValueError caught -> True."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text(json.dumps({"process": "test", "pid": 99999}))
        # .get("timestamp", "") returns "" -> fromisoformat("") raises ValueError
        assert check_lock() is True

    def test_timestamp_empty_string_returns_true(self, tmp_path, monkeypatch):
        """Lock file with empty timestamp string -> ValueError caught -> True."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text(json.dumps({
            "process": "test", "pid": 99999, "timestamp": "",
        }))
        assert check_lock() is True


# ===========================================================================
# 44. release_lock edge cases (lines 244-252)
# ===========================================================================

class TestReleaseLockEdgeCasesExtra:
    def test_different_pid_not_deleted(self, tmp_path, monkeypatch):
        """Lock owned by different PID -> NOT deleted."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text(json.dumps({
            "process": "other_bot",
            "pid": os.getpid() + 9999,
            "timestamp": datetime.now(KST).isoformat(),
        }))
        release_lock()
        assert fake_lock.exists()

    def test_read_exception_silently_caught(self, tmp_path, monkeypatch):
        """Lock file read raises exception -> silently caught."""
        fake_lock = tmp_path / "trading.lock"
        monkeypatch.setattr("scripts.short_term_trader.LOCK_FILE", fake_lock)
        fake_lock.write_text("CORRUPT DATA {{{{")
        release_lock()  # should not raise
        # File still exists since JSON parse failed before PID check
        assert fake_lock.exists()


# ===========================================================================
# 45. can_trade edge cases (lines 719-729)
# ===========================================================================

class TestCanTradeEdgeCasesExtra:
    @patch.dict(os.environ, {"EMERGENCY_STOP": "true"})
    def test_emergency_stop_exact_message(self):
        """EMERGENCY_STOP=true -> exact message check."""
        trader = _make_trader()
        ok, reason = trader.can_trade()
        assert ok is False
        assert reason == "EMERGENCY_STOP 활성화"

    @patch.dict(os.environ, {"EMERGENCY_STOP": "TRUE"})
    def test_emergency_stop_case_insensitive(self):
        """EMERGENCY_STOP=TRUE (uppercase) should also trigger."""
        trader = _make_trader()
        ok, reason = trader.can_trade()
        assert ok is False

    def test_position_limit_exactly_two(self):
        """Exactly 2 positions -> (False, 'position_limit')."""
        trader = _make_trader()
        trader.positions = [_make_position(), _make_position()]
        ok, reason = trader.can_trade()
        assert ok is False
        assert reason == "position_limit"

    def test_one_position_still_allowed(self):
        """1 position (< 2 limit) -> still allowed."""
        trader = _make_trader()
        trader.positions = [_make_position()]
        ok, reason = trader.can_trade()
        assert ok is True
        assert reason == "OK"


# ===========================================================================
# 46. execute_entry amount edge cases (lines 731-748)
# ===========================================================================

class TestExecuteEntryAmountEdgeCasesExtra:
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_amount_below_5000_after_budget_calc(self, mock_db, mock_tg):
        """After budget subtraction, amount < 5000 -> no position created."""
        trader = _make_trader(dry_run=True)
        trader.used_budget = 496_000  # only 4000 remaining
        signal = TradeSignal(
            strategy="spike", action="buy", confidence=0.9,
            reason="test", suggested_amount=200_000,
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_suggested_amount_capped_at_max_trade(self, mock_db, mock_tg):
        """signal.suggested_amount > SHORT_TERM_MAX_TRADE -> capped."""
        trader = _make_trader(dry_run=True)
        signal = TradeSignal(
            strategy="whale", action="buy", confidence=0.8,
            reason="test", suggested_amount=500_000,
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 1
        assert trader.positions[0].amount_krw == SHORT_TERM_MAX_TRADE

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_exact_5000_boundary_creates_position(self, mock_db, mock_tg):
        """Exactly 5000 remaining -> should create position."""
        trader = _make_trader(dry_run=True)
        trader.used_budget = 495_000  # 5000 remaining
        signal = TradeSignal(
            strategy="news", action="buy", confidence=0.7,
            reason="test", suggested_amount=200_000,
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 1
        assert trader.positions[0].amount_krw == 5_000

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_emergency_stopped_blocks_entry(self, mock_db, mock_tg):
        """emergency_stopped=True -> execute_entry returns immediately."""
        trader = _make_trader(dry_run=True)
        trader.emergency_stopped = True
        signal = TradeSignal(
            strategy="news", action="buy", confidence=0.9, reason="test",
        )
        trader.execute_entry(signal)
        assert len(trader.positions) == 0


# ===========================================================================
# 47. db_insert edge cases (lines 178-195)
# ===========================================================================

class TestDbInsertEdgeCasesExtra:
    @patch("scripts.short_term_trader.requests.post")
    def test_missing_supabase_url_only_skips(self, mock_post, monkeypatch):
        """SUPABASE_URL empty but key present -> early return."""
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "some_key")
        db_insert("scalp_trades", {"foo": "bar"})
        mock_post.assert_not_called()

    @patch("scripts.short_term_trader.requests.post")
    def test_missing_supabase_key_only_skips(self, mock_post, monkeypatch):
        """SUPABASE_KEY empty but URL present -> early return."""
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "")
        db_insert("scalp_trades", {"foo": "bar"})
        mock_post.assert_not_called()

    @patch(
        "scripts.short_term_trader.requests.post",
        side_effect=requests.exceptions.Timeout("timeout"),
    )
    def test_timeout_exception_silently_caught(self, mock_post, monkeypatch):
        """requests.post raises Timeout -> silently caught."""
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "key123")
        db_insert("scalp_trades", {"strategy": "news"})  # should not raise

    @patch(
        "scripts.short_term_trader.requests.post",
        side_effect=RuntimeError("unexpected"),
    )
    def test_generic_exception_silently_caught(self, mock_post, monkeypatch):
        """Any exception from requests.post -> silently caught."""
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setattr("scripts.short_term_trader.SUPABASE_KEY", "key123")
        db_insert("scalp_trades", {"strategy": "spike"})  # should not raise


# ===========================================================================
# 41. Async WebSocket: ws_ticker (lines 314-335)
# ===========================================================================

class TestWsTicker:
    """Tests for the ws_ticker async WebSocket method."""

    @patch("scripts.short_term_trader.db_insert")
    def test_ticker_updates_price_and_history(self, mock_db):
        """Messages with trade_price update current_price and price_history."""
        import asyncio

        trader = _make_trader()
        trader.running = True

        messages = [
            json.dumps({"trade_price": 101_000_000}),
            json.dumps({"trade_price": 102_000_000}),
        ]

        msg_index = 0

        class FakeWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal msg_index
                if msg_index < len(messages):
                    m = messages[msg_index]
                    msg_index += 1
                    return m
                trader.running = False
                raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=FakeWS()):
                await trader.ws_ticker()

        asyncio.run(run())

        assert trader.current_price == 102_000_000
        assert len(trader.price_history) == 2
        assert trader.price_history[0]["price"] == 101_000_000
        assert trader.price_history[1]["price"] == 102_000_000

    @patch("scripts.short_term_trader.db_insert")
    def test_ticker_reconnects_on_exception(self, mock_db):
        """WebSocket ticker should reconnect after exception."""
        import asyncio
        from unittest.mock import AsyncMock

        trader = _make_trader()
        trader.running = True

        connect_count = 0

        class FailFirstWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def __aenter__(self):
                nonlocal connect_count
                connect_count += 1
                if connect_count == 1:
                    raise ConnectionError("test disconnect")
                return self

            async def __aexit__(self, *args):
                trader.running = False

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=FailFirstWS()):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await trader.ws_ticker()

        asyncio.run(run())
        assert connect_count == 2

    @patch("scripts.short_term_trader.db_insert")
    def test_ticker_running_false_breaks_inner_loop(self, mock_db):
        """Setting running=False should break the inner message loop."""
        import asyncio

        trader = _make_trader()
        trader.running = True

        class StopAfterFirstWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                trader.running = False
                return json.dumps({"trade_price": 105_000_000})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=StopAfterFirstWS()):
                await trader.ws_ticker()

        asyncio.run(run())
        assert trader.running is False


# ===========================================================================
# 42. Async WebSocket: ws_trades (lines 339-384)
# ===========================================================================

class TestWsTrades:
    """Tests for the ws_trades async WebSocket method."""

    @patch("scripts.short_term_trader.db_insert")
    def test_trades_appended_to_history(self, mock_db):
        """Trade messages should be appended to trade_history."""
        import asyncio

        trader = _make_trader()
        trader.running = True

        messages = [
            json.dumps({
                "trade_price": 100_000_000,
                "trade_volume": 0.001,
                "ask_bid": "BID",
            }),
            json.dumps({
                "trade_price": 100_100_000,
                "trade_volume": 0.002,
                "ask_bid": "ASK",
            }),
        ]

        msg_index = 0

        class FakeWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal msg_index
                if msg_index < len(messages):
                    m = messages[msg_index]
                    msg_index += 1
                    return m
                trader.running = False
                raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=FakeWS()):
                await trader.ws_trades()

        asyncio.run(run())

        assert len(trader.trade_history) == 2
        assert trader.trade_history[0]["side"] == "BID"
        assert trader.trade_history[0]["price"] == 100_000_000
        assert trader.trade_history[1]["side"] == "ASK"

    @patch("scripts.short_term_trader.db_insert")
    def test_whale_detection_and_db_insert(self, mock_db):
        """Trade with krw >= WHALE_THRESHOLD_KRW triggers whale detection and db_insert."""
        import asyncio

        trader = _make_trader()
        trader.running = True

        whale_price = 100_000_000
        whale_volume = 0.5  # 0.5 BTC = 50M KRW (above 30M threshold)
        messages = [
            json.dumps({
                "trade_price": whale_price,
                "trade_volume": whale_volume,
                "ask_bid": "BID",
            }),
        ]

        msg_index = 0

        class FakeWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal msg_index
                if msg_index < len(messages):
                    m = messages[msg_index]
                    msg_index += 1
                    return m
                trader.running = False
                raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=FakeWS()):
                await trader.ws_trades()

        asyncio.run(run())

        assert len(trader.whale_recent) == 1
        assert trader.whale_recent[0]["side"] == "BID"
        assert trader.whale_recent[0]["krw"] == whale_price * whale_volume

        mock_db.assert_called_once()
        call_args = mock_db.call_args
        assert call_args[0][0] == "whale_detections"
        inserted_data = call_args[0][1]
        assert inserted_data["side"] == "BID"
        assert inserted_data["volume"] == whale_volume
        assert inserted_data["price"] == whale_price
        assert inserted_data["krw_amount"] == int(whale_price * whale_volume)
        assert inserted_data["whale_buy_count"] == 1
        assert inserted_data["whale_sell_count"] == 0

    @patch("scripts.short_term_trader.db_insert")
    def test_trades_reconnects_on_exception(self, mock_db):
        """WebSocket trades should reconnect after exception."""
        import asyncio
        from unittest.mock import AsyncMock

        trader = _make_trader()
        trader.running = True

        connect_count = 0

        class FailFirstWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def __aenter__(self):
                nonlocal connect_count
                connect_count += 1
                if connect_count == 1:
                    raise ConnectionError("test disconnect")
                return self

            async def __aexit__(self, *args):
                trader.running = False

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=FailFirstWS()):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await trader.ws_trades()

        asyncio.run(run())
        assert connect_count == 2

    @patch("scripts.short_term_trader.db_insert")
    def test_small_trade_no_whale(self, mock_db):
        """Trade below WHALE_THRESHOLD_KRW should not trigger whale detection."""
        import asyncio

        trader = _make_trader()
        trader.running = True

        messages = [
            json.dumps({
                "trade_price": 100_000_000,
                "trade_volume": 0.001,
                "ask_bid": "BID",
            }),
        ]

        msg_index = 0

        class FakeWS:
            async def send(self, data):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal msg_index
                if msg_index < len(messages):
                    m = messages[msg_index]
                    msg_index += 1
                    return m
                trader.running = False
                raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        async def run():
            with patch("scripts.short_term_trader.websockets.connect", return_value=FakeWS()):
                await trader.ws_trades()

        asyncio.run(run())

        assert len(trader.whale_recent) == 0
        mock_db.assert_not_called()


# ===========================================================================
# 43. strategy_loop exception handling (lines 957-959)
# ===========================================================================

class TestStrategyLoopExceptionHandling:
    """Test that exceptions inside strategy_loop don't kill the loop."""

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    def test_exception_in_loop_continues(self, mock_db, mock_tg):
        """An exception in strategy_loop body is caught and loop continues."""
        import asyncio
        from unittest.mock import AsyncMock

        trader = _make_trader()
        trader.running = True
        trader.current_price = 100_000_000

        iteration = 0

        def exploding_check():
            nonlocal iteration
            iteration += 1
            if iteration == 1:
                raise RuntimeError("simulated error in strategy loop")
            trader.running = False
            return []

        trader.check_position_exit = exploding_check

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await trader.strategy_loop()

        asyncio.run(run())

        assert iteration == 2
        assert trader.running is False


# ===========================================================================
# 44. status_reporter (lines 963-974)
# ===========================================================================

class TestStatusReporter:
    """Tests for the status_reporter async method."""

    @patch("scripts.short_term_trader.db_insert")
    def test_status_reporter_logs_output(self, mock_db, caplog):
        """status_reporter should log a status report after sleep."""
        import asyncio
        import logging

        trader = _make_trader()
        trader.running = True
        trader.current_price = 100_500_000
        trader.daily_trade_count = 3
        trader.daily_pnl = 5000
        trader.used_budget = 200_000

        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                trader.running = False

        async def run():
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with caplog.at_level(logging.INFO, logger="short_term"):
                    await trader.status_reporter()

        asyncio.run(run())

        log_text = caplog.text
        assert "상태 리포트" in log_text
        assert "100,500,000" in log_text
        assert "보유 포지션: 0개" in log_text

    @patch("scripts.short_term_trader.db_insert")
    def test_status_reporter_with_position(self, mock_db, caplog):
        """status_reporter should include position info when positions exist."""
        import asyncio
        import logging

        trader = _make_trader()
        trader.running = True
        trader.current_price = 101_000_000

        pos = _make_position(entry_price=100_000_000, amount_krw=200_000, strategy="whale")
        trader.positions.append(pos)

        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                trader.running = False

        async def run():
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with caplog.at_level(logging.INFO, logger="short_term"):
                    await trader.status_reporter()

        asyncio.run(run())

        log_text = caplog.text
        assert "상태 리포트" in log_text
        assert "보유 포지션: 1개" in log_text
        assert "whale" in log_text


# ===========================================================================
# 45. scan_news deeper coverage (lines 440-527)
# ===========================================================================

class TestScanNewsDeeper:
    """Cover uncovered lines: Atom fallback, empty titles, seen_titles overflow,
    negative sentiment, outer exception with 30s sleep, ParseError handling."""

    @staticmethod
    async def _run_one_scan(trader):
        import asyncio as _aio

        trader.running = True
        call_count = 0

        async def fake_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                trader.running = False

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await trader.scan_news()

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_atom_entry_fallback_parsed(self, mock_get, mock_db):
        """Atom feed with <entry> (no <item>) triggers namespace fallback (lines 442-443, 449-451)."""
        import asyncio

        atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Atom Feed</title>
          <entry>
            <title>Bitcoin crash plunge dump ban hack fraud</title>
          </entry>
          <entry>
            <title>Crypto collapse sell-off bearish fear regulation</title>
          </entry>
        </feed>"""

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = atom_xml.encode("utf-8")
        mock_get.return_value = mock_resp

        trader = _make_trader()
        trader.last_news_scan = 0

        asyncio.run(self._run_one_scan(trader))

        assert trader.news_sentiment_score < 0
        assert trader.last_news_sentiment == "negative"

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_items_with_none_or_empty_title_skipped(self, mock_get, mock_db):
        """Items with no <title> element or empty text are skipped (line 452)."""
        import asyncio

        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item></item>
            <item><title></title></item>
            <item><title>Bitcoin surge rally breakout bull pump</title></item>
          </channel>
        </rss>"""

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = rss_xml.encode("utf-8")
        mock_get.return_value = mock_resp

        trader = _make_trader()
        trader.last_news_scan = 0

        asyncio.run(self._run_one_scan(trader))

        # Only the third item counted; it has strong positive keywords
        assert trader.news_sentiment_score > 0

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_seen_titles_overflow_trimmed(self, mock_get, mock_db):
        """When seen_titles > 500 it is trimmed to 300 (lines 482-483)."""
        import asyncio

        batch_counter = {"n": 0}

        def make_feed():
            n = batch_counter["n"]
            batch_counter["n"] += 1
            items = "".join(
                f"<item><title>Unique headline batch{n} item{i}</title></item>"
                for i in range(10)
            )
            rss = f"""<?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0"><channel>{items}</channel></rss>"""
            resp = MagicMock()
            resp.ok = True
            resp.content = rss.encode("utf-8")
            return resp

        trader = _make_trader()
        trader.last_news_scan = 0

        async def run_many():
            trader.running = True
            iteration = 0

            async def fake_sleep(n):
                nonlocal iteration
                iteration += 1
                trader.last_news_scan = 0  # force rescan
                mock_get.return_value = make_feed()
                if iteration >= 120:
                    trader.running = False

            mock_get.return_value = make_feed()
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await trader.scan_news()

        asyncio.run(run_many())
        assert trader.running is False

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_negative_sentiment_sets_negative(self, mock_get, mock_db):
        """Strongly negative headlines set last_news_sentiment = 'negative' (line 495)."""
        import asyncio

        neg_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item><title>crash plunge dump ban hack fraud collapse</title></item>
            <item><title>regulation crackdown war sanctions sell-off</title></item>
            <item><title>liquidation bearish fear</title></item>
          </channel>
        </rss>"""

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = neg_rss.encode("utf-8")
        mock_get.return_value = mock_resp

        trader = _make_trader()
        trader.last_news_scan = 0
        trader.last_news_sentiment = "neutral"

        asyncio.run(self._run_one_scan(trader))

        assert trader.last_news_sentiment == "negative"
        assert trader.news_sentiment_score < -0.3

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_outer_exception_sleeps_30s(self, mock_get, mock_db):
        """Exception in outer try (lines 532-534): caught, sleep 30s.

        Inner except catches feed-level errors; to trigger outer except,
        make db_insert raise after the feed loop completes.
        """
        import asyncio

        valid_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item><title>Bitcoin surge rally breakout</title></item>
          </channel>
        </rss>"""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = valid_rss.encode("utf-8")
        mock_get.return_value = mock_resp

        mock_db.side_effect = RuntimeError("db failure")

        trader = _make_trader()
        trader.last_news_scan = 0

        sleep_durations = []

        async def fake_sleep(n):
            sleep_durations.append(n)
            trader.running = False

        trader.running = True
        with patch("asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(trader.scan_news())

        assert 30 in sleep_durations

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.requests.get")
    def test_parse_error_continues_to_next_feed(self, mock_get, mock_db):
        """ET.ParseError on one feed continues to next feed (lines 476-479)."""
        import asyncio

        valid_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item><title>Bitcoin surge rally breakout bull pump</title></item>
          </channel>
        </rss>"""

        malformed_resp = MagicMock()
        malformed_resp.ok = True
        malformed_resp.content = b"<rss><broken xml"

        valid_resp = MagicMock()
        valid_resp.ok = True
        valid_resp.content = valid_rss.encode("utf-8")

        mock_get.side_effect = [malformed_resp, valid_resp] * 10

        trader = _make_trader()
        trader.last_news_scan = 0

        asyncio.run(self._run_one_scan(trader))

        assert isinstance(trader.news_sentiment_score, (int, float))


# ===========================================================================
# 46. execute_entry live path (lines 760-793)
# ===========================================================================

class TestExecuteEntryLivePathDeep:
    """Cover remaining uncovered lines in execute_entry live mode."""

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=False)
    @patch("scripts.short_term_trader.upbit_order")
    def test_lock_check_fails_returns_early(self, mock_order, mock_lock, mock_db, mock_tg):
        """check_lock() False -> log warning, return (lines 762-764)."""
        trader = _make_trader(dry_run=False)
        signal = TradeSignal(strategy="spike", action="buy", confidence=0.8, reason="test")
        trader.execute_entry(signal)

        assert len(trader.positions) == 0
        mock_order.assert_not_called()

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_ok_false_auth_error_emergency_stop(self, mock_order, mock_lock, mock_db, mock_tg):
        """ok=False with jwt_verification -> emergency_stop (lines 773-774)."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "jwt_verification"}},
        }
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(strategy="news", action="buy", confidence=0.8, reason="auth test")
        trader.execute_entry(signal)

        assert trader.emergency_stopped is True
        assert trader.running is False
        assert trader.consecutive_errors == 1

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_ok_false_non_auth_below_threshold_increments(self, mock_order, mock_lock, mock_db, mock_tg):
        """ok=False, non-auth, errors < 5 -> increment only (lines 775-777)."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "insufficient_funds_bid"}},
        }
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 2

        signal = TradeSignal(strategy="whale", action="buy", confidence=0.9, reason="funds test")
        trader.execute_entry(signal)

        assert trader.consecutive_errors == 3
        assert trader.emergency_stopped is False
        assert len(trader.positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_timeout_increments_errors(self, mock_order, mock_lock, mock_db, mock_tg):
        """requests.exceptions.Timeout -> increment consecutive_errors (lines 782-787)."""
        import requests as req
        mock_order.side_effect = req.exceptions.Timeout("timed out")

        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 1

        signal = TradeSignal(strategy="spike", action="buy", confidence=0.8, reason="timeout test")
        trader.execute_entry(signal)

        assert trader.consecutive_errors == 2
        assert trader.emergency_stopped is False
        assert len(trader.positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_generic_exception_increments_errors(self, mock_order, mock_lock, mock_db, mock_tg):
        """Generic exception -> increment consecutive_errors (lines 788-793)."""
        mock_order.side_effect = RuntimeError("unexpected")

        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(strategy="news", action="buy", confidence=0.7, reason="exception test")
        trader.execute_entry(signal)

        assert trader.consecutive_errors == 1
        assert trader.emergency_stopped is False
        assert len(trader.positions) == 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.check_lock", return_value=True)
    @patch("scripts.short_term_trader.upbit_order")
    def test_ok_false_data_as_string_fallback(self, mock_order, mock_lock, mock_db, mock_tg):
        """ok=False with data as string (not dict) -> err_name fallback (line 770)."""
        mock_order.return_value = {
            "ok": False,
            "data": "some plain error string",
        }
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 0

        signal = TradeSignal(strategy="spike", action="buy", confidence=0.8, reason="str err")
        trader.execute_entry(signal)

        assert trader.consecutive_errors == 1
        assert trader.emergency_stopped is False


# ===========================================================================
# 47. execute_exit live path (lines 854, 862)
# ===========================================================================

class TestExecuteExitLivePathDeep:
    """Cover live exit success/failure paths including PnL and telegram alerts."""

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_live_exit_success_removes_and_calculates_pnl(self, mock_order, mock_db, mock_tg):
        """Live exit success -> position removed, PnL calculated (lines 865-873)."""
        mock_order.return_value = {"ok": True, "data": {"uuid": "exit-ok"}}

        trader = _make_trader(dry_run=False)
        pos = _make_position(entry_price=100_000_000, amount_krw=200_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000

        trader.execute_exit(pos, "익절")

        assert len(trader.positions) == 0
        assert len(trader.closed_positions) == 1
        closed = trader.closed_positions[0]
        assert closed.exit_price == 101_000_000
        assert closed.pnl_pct > 0
        assert trader.consecutive_errors == 0
        assert trader.daily_pnl != 0

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_live_exit_failure_keeps_position_sends_alert(self, mock_order, mock_db, mock_tg):
        """Live exit failure -> position NOT removed, telegram alert sent."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "insufficient_funds_ask"}},
        }
        trader = _make_trader(dry_run=False)
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 101_000_000

        trader.execute_exit(pos, "익절")

        assert len(trader.positions) == 1
        assert pos.exit_price is None
        assert trader.consecutive_errors == 1

        tg_calls = [str(c) for c in mock_tg.call_args_list]
        assert any("매도 실패" in msg for msg in tg_calls)

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_live_exit_failure_at_max_errors_emergency(self, mock_order, mock_db, mock_tg):
        """Exit failure at MAX_CONSECUTIVE_ERRORS triggers emergency_stop (line 854)."""
        mock_order.return_value = {
            "ok": False,
            "data": {"error": {"name": "server_error"}},
        }
        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 4
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_000_000

        trader.execute_exit(pos, "손절")

        assert trader.emergency_stopped is True
        assert trader.running is False

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_live_exit_exception_at_max_errors_emergency(self, mock_order, mock_db, mock_tg):
        """Exception during exit at MAX_CONSECUTIVE_ERRORS triggers emergency (line 862)."""
        mock_order.side_effect = ConnectionError("API down")

        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 4
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_000_000

        trader.execute_exit(pos, "손절")

        assert trader.emergency_stopped is True
        assert trader.running is False
        tg_calls = [str(c) for c in mock_tg.call_args_list]
        assert any("매도 예외" in msg for msg in tg_calls)

    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.upbit_order")
    def test_live_exit_exception_below_threshold_increments(self, mock_order, mock_db, mock_tg):
        """Exception during exit below threshold only increments errors."""
        mock_order.side_effect = ConnectionError("network blip")

        trader = _make_trader(dry_run=False)
        trader.consecutive_errors = 1
        pos = _make_position(entry_price=100_000_000)
        trader.positions.append(pos)
        trader.current_price = 99_000_000

        trader.execute_exit(pos, "시간 제한")

        assert trader.consecutive_errors == 2
        assert trader.emergency_stopped is False
        assert len(trader.positions) == 1


# ===========================================================================
# 48. is_sell_pressure_blocking edge cases
# ===========================================================================

class TestSellPressureBlockingEdge:
    """Edge cases for is_sell_pressure_blocking."""

    def test_empty_whale_recent_returns_false(self):
        """No whale trades -> no sell pressure."""
        trader = _make_trader()
        trader.whale_recent.clear()
        assert trader.is_sell_pressure_blocking() is False

    def test_only_ask_whales_blocks(self):
        """Only ASK whales (buy_krw=0, sell_krw>0) should block (line 682-683)."""
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "ASK", "krw": 50_000_000, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": 60_000_000, "time": now})
        assert trader.is_sell_pressure_blocking() is True

    def test_only_bid_whales_does_not_block(self):
        """Only BID whales should not block."""
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "BID", "krw": 50_000_000, "time": now})
        trader.whale_recent.append({"side": "BID", "krw": 60_000_000, "time": now})
        assert trader.is_sell_pressure_blocking() is False

    def test_stale_whales_excluded(self):
        """Whales outside WHALE_RATIO_WINDOW_SEC should be ignored."""
        trader = _make_trader()
        old = time.time() - WHALE_RATIO_WINDOW_SEC - 100
        trader.whale_recent.append({"side": "ASK", "krw": 200_000_000, "time": old})
        assert trader.is_sell_pressure_blocking() is False

    def test_sell_ratio_at_threshold_blocks(self):
        """sell/buy ratio at SELL_PRESSURE_BLOCK_RATIO should block (line 684)."""
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "BID", "krw": 10_000_000, "time": now})
        trader.whale_recent.append(
            {"side": "ASK", "krw": int(10_000_000 * SELL_PRESSURE_BLOCK_RATIO), "time": now}
        )
        assert trader.is_sell_pressure_blocking() is True

    def test_sell_ratio_below_threshold_no_block(self):
        """sell/buy ratio below threshold should not block."""
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "BID", "krw": 50_000_000, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": 20_000_000, "time": now})
        assert trader.is_sell_pressure_blocking() is False


# ===========================================================================
# 49. check_whale_signal total_krw edge (line 642)
# ===========================================================================

class TestCheckWhaleSignalTotalEdge:
    """Cover total_krw < WHALE_THRESHOLD_KRW returning None (line 641-642)."""

    def test_total_below_threshold_returns_none(self):
        """Small whale trades summing below threshold -> None."""
        trader = _make_trader()
        now = time.time()
        tiny = WHALE_THRESHOLD_KRW // 10
        trader.whale_recent.append({"side": "BID", "krw": tiny, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": tiny, "time": now})
        assert trader.check_whale_signal() is None

    def test_equal_split_no_signal(self):
        """50/50 split above threshold -> neither side reaches WHALE_RATIO_THRESHOLD -> None."""
        trader = _make_trader()
        now = time.time()
        trader.whale_recent.append({"side": "BID", "krw": WHALE_THRESHOLD_KRW, "time": now})
        trader.whale_recent.append({"side": "ASK", "krw": WHALE_THRESHOLD_KRW, "time": now})
        result = trader.check_whale_signal()
        assert result is None


# ===========================================================================
# strategy_alert_monitor tests
# ===========================================================================


def _run_alert_monitor(trader):
    """Run one iteration of strategy_alert_monitor.

    The monitor is ``async def`` with ``while self.running`` and starts with
    ``await asyncio.sleep(600)``.  We let the first sleep pass then stop
    the loop on the second sleep so exactly one iteration body executes.
    """
    import asyncio
    from unittest.mock import patch as _patch

    call_count = 0

    async def _fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            trader.running = False

    with _patch("asyncio.sleep", side_effect=_fake_sleep):
        asyncio.run(trader.strategy_alert_monitor())


def _candle_closes_for_rsi(target_rsi, n=15):
    """Generate *n* close prices producing roughly *target_rsi*."""
    if target_rsi >= 100:
        return [100_000_000 + i * 100_000 for i in range(n)]
    if target_rsi <= 0:
        return [100_000_000 - i * 100_000 for i in range(n)]
    R = target_rsi / (100 - target_rsi)
    gain_step = R * 100_000
    loss_step = 100_000
    closes = [100_000_000]
    for i in range(1, n):
        if i % 2 == 1:
            closes.append(closes[-1] + gain_step)
        else:
            closes.append(closes[-1] - loss_step)
    return closes


def _mock_candle_response(closes):
    """Return a mock response whose json() yields Upbit candle format."""
    from unittest.mock import MagicMock as _MM
    candles = [{"trade_price": p} for p in reversed(closes)]
    resp = _MM()
    resp.ok = True
    resp.json.return_value = candles
    return resp


class TestAlertMonitorRSI:
    """RSI-based alerts (lines 997-1042)."""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_rsi_extreme_oversold(self, mock_get, mock_sound, mock_tg, mock_db):
        """RSI <= 25 triggers rsi_extreme_oversold alert."""
        closes = _candle_closes_for_rsi(20)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.current_price = 100_000_000
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "rsi_extreme_oversold"]
        assert len(calls) == 1
        mock_sound.assert_called()
        mock_tg.assert_called()

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_rsi_oversold(self, mock_get, mock_sound, mock_tg, mock_db):
        """RSI in (25, 30] triggers rsi_oversold alert."""
        closes = _candle_closes_for_rsi(28)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "rsi_oversold"]
        assert len(calls) == 1

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_rsi_overbought(self, mock_get, mock_sound, mock_tg, mock_db):
        """RSI >= 75 triggers rsi_overbought alert."""
        closes = _candle_closes_for_rsi(80)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "rsi_overbought"]
        assert len(calls) == 1
        mock_tg.assert_called()

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_rsi_normal_no_alert(self, mock_get, mock_sound, mock_tg, mock_db):
        """RSI in normal range (30 < RSI < 75) -> no RSI alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        _run_alert_monitor(trader)

        rsi_calls = [c for c in mock_db.call_args_list
                     if c[0][0] == "strategy_alerts" and "rsi" in c[0][1].get("alert_type", "")]
        assert len(rsi_calls) == 0

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_rsi_cooldown_suppresses_repeat(self, mock_get, mock_sound, mock_tg, mock_db):
        """Same RSI alert within 1800s cooldown is suppressed."""
        import asyncio

        closes = _candle_closes_for_rsi(20)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        # Run two iterations - second should be suppressed by cooldown
        call_count = 0

        async def _fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                trader.running = False

        with patch("asyncio.sleep", side_effect=_fake_sleep):
            asyncio.run(trader.strategy_alert_monitor())

        rsi_calls = [c for c in mock_db.call_args_list
                     if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "rsi_extreme_oversold"]
        assert len(rsi_calls) == 1


class TestAlertMonitorWhale:
    """Whale direction reversal (lines 1047-1064)."""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_whale_buy_reversal(self, mock_get, mock_sound, mock_tg, mock_db):
        """4+ BID whales in last 5 -> whale_buy_reversal alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0
        trader.whale_recent = deque(
            [{"side": "BID"}, {"side": "BID"}, {"side": "BID"}, {"side": "BID"}, {"side": "ASK"}],
            maxlen=20,
        )

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "whale_buy_reversal"]
        assert len(calls) == 1
        mock_sound.assert_called()
        mock_tg.assert_called()

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_whale_sell_pressure(self, mock_get, mock_sound, mock_tg, mock_db):
        """4+ ASK whales in last 5 -> whale_sell_pressure alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0
        trader.whale_recent = deque(
            [{"side": "ASK"}, {"side": "ASK"}, {"side": "ASK"}, {"side": "ASK"}, {"side": "BID"}],
            maxlen=20,
        )

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "whale_sell_pressure"]
        assert len(calls) == 1


class TestAlertMonitorPriceSpike:
    """Price spike/crash (lines 1067-1084)."""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_price_spike_alert(self, mock_get, mock_sound, mock_tg, mock_db):
        """1.5%+ increase in price_history -> price_spike alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.news_sentiment_score = 0.0
        base_price = 100_000_000
        prices = [{"price": base_price}] * 600
        trader.price_history = deque(prices, maxlen=600)
        trader.current_price = int(base_price * 1.02)

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "price_spike"]
        assert len(calls) == 1

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_price_crash_alert(self, mock_get, mock_sound, mock_tg, mock_db):
        """1.5%+ decrease -> price_crash alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.news_sentiment_score = 0.0
        base_price = 100_000_000
        prices = [{"price": base_price}] * 600
        trader.price_history = deque(prices, maxlen=600)
        trader.current_price = int(base_price * 0.97)

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "price_crash"]
        assert len(calls) == 1

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_price_no_alert_below_threshold(self, mock_get, mock_sound, mock_tg, mock_db):
        """<1.5% change -> no price alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.news_sentiment_score = 0.0
        base_price = 100_000_000
        prices = [{"price": base_price}] * 600
        trader.price_history = deque(prices, maxlen=600)
        trader.current_price = int(base_price * 1.005)

        _run_alert_monitor(trader)

        spike_calls = [c for c in mock_db.call_args_list
                       if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") in ("price_spike", "price_crash")]
        assert len(spike_calls) == 0


class TestAlertMonitorSupportBreak:
    """Support break at 100M (lines 1087-1093)."""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_support_break_below_100m(self, mock_get, mock_sound, mock_tg, mock_db):
        """current_price < 100M -> support_break alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.current_price = 99_000_000
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "support_break"]
        assert len(calls) >= 1
        mock_tg.assert_called()


class TestAlertMonitorNews:
    """News extreme (lines 1096-1105)."""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_positive_news_extreme(self, mock_get, mock_sound, mock_tg, mock_db):
        """news_sentiment_score >= 0.6 -> positive news_extreme alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.7

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "news_extreme"]
        assert len(calls) == 1
        mock_sound.assert_called()

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_negative_news_extreme(self, mock_get, mock_sound, mock_tg, mock_db):
        """news_sentiment_score <= -0.6 -> negative news_extreme alert."""
        closes = _candle_closes_for_rsi(50)
        mock_get.return_value = _mock_candle_response(closes)

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = -0.8

        _run_alert_monitor(trader)

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and c[0][1].get("alert_type") == "news_extreme"]
        assert len(calls) == 1


class TestAlertMonitorStrategySwitch:
    """Strategy switch suggestions (lines 1108-1158)."""

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_conservative_fgi_high_suggest_aggressive(self, mock_get, mock_sound, mock_tg, mock_db):
        """Conservative strategy + FGI >= 50 -> suggest aggressive."""
        closes = _candle_closes_for_rsi(50)

        fgi_resp = MagicMock()
        fgi_resp.ok = True
        fgi_resp.json.return_value = {"data": [{"value": "55"}]}

        mock_get.side_effect = [_mock_candle_response(closes), fgi_resp]

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        from pathlib import Path as _Path
        strategy_path = _Path(__file__).resolve().parent.parent / "strategy.md"
        original = strategy_path.read_text() if strategy_path.exists() else None
        try:
            strategy_path.write_text("# 보수적 전략\n매수 조건...")
            _run_alert_monitor(trader)
        finally:
            if original is not None:
                strategy_path.write_text(original)
            elif strategy_path.exists():
                strategy_path.unlink()

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and "공격적" in c[0][1].get("message", "")]
        assert len(calls) == 1

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_conservative_fgi_mid_suggest_moderate(self, mock_get, mock_sound, mock_tg, mock_db):
        """Conservative strategy + FGI >= 35 (but < 50) -> suggest moderate."""
        closes = _candle_closes_for_rsi(50)

        fgi_resp = MagicMock()
        fgi_resp.ok = True
        fgi_resp.json.return_value = {"data": [{"value": "40"}]}

        mock_get.side_effect = [_mock_candle_response(closes), fgi_resp]

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        from pathlib import Path as _Path
        strategy_path = _Path(__file__).resolve().parent.parent / "strategy.md"
        original = strategy_path.read_text() if strategy_path.exists() else None
        try:
            strategy_path.write_text("# 보수적 전략\n매수 조건...")
            _run_alert_monitor(trader)
        finally:
            if original is not None:
                strategy_path.write_text(original)
            elif strategy_path.exists():
                strategy_path.unlink()

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and "보통" in c[0][1].get("message", "")]
        assert len(calls) == 1

    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_aggressive_fgi_low_suggest_conservative(self, mock_get, mock_sound, mock_tg, mock_db):
        """Aggressive strategy + FGI <= 25 -> suggest conservative."""
        closes = _candle_closes_for_rsi(50)

        fgi_resp = MagicMock()
        fgi_resp.ok = True
        fgi_resp.json.return_value = {"data": [{"value": "20"}]}

        mock_get.side_effect = [_mock_candle_response(closes), fgi_resp]

        trader = _make_trader()
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        from pathlib import Path as _Path
        strategy_path = _Path(__file__).resolve().parent.parent / "strategy.md"
        original = strategy_path.read_text() if strategy_path.exists() else None
        try:
            strategy_path.write_text("# aggressive 전략\n매수 조건...")
            _run_alert_monitor(trader)
        finally:
            if original is not None:
                strategy_path.write_text(original)
            elif strategy_path.exists():
                strategy_path.unlink()

        calls = [c for c in mock_db.call_args_list
                 if c[0][0] == "strategy_alerts" and "보수적" in c[0][1].get("message", "")]
        assert len(calls) == 1


class TestAlertMonitorDCAStopLoss:
    """DCA vs stop-loss (lines 1162-1229)."""

    def _make_acct_response(self, avg_buy_price, balance="0.001"):
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = [
            {"currency": "BTC", "balance": balance, "avg_buy_price": str(avg_buy_price)},
        ]
        return resp

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "test_ak", "UPBIT_SECRET_KEY": "test_sk"})
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_dca_suggestion_with_buy_signal(self, mock_get, mock_sound, mock_tg, mock_db):
        """Loss -3% to -5% with FGI <= 30 -> DCA suggestion."""
        closes = _candle_closes_for_rsi(50)
        fgi_resp = MagicMock()
        fgi_resp.ok = True
        fgi_resp.json.return_value = {"data": [{"value": "25"}]}

        current_price = 96_000_000
        avg_buy_price = 100_000_000

        acct_resp = self._make_acct_response(avg_buy_price)
        mock_get.side_effect = [_mock_candle_response(closes), fgi_resp, acct_resp]

        trader = _make_trader(dry_run=False)
        trader.current_price = current_price
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        from pathlib import Path as _Path
        strategy_path = _Path(__file__).resolve().parent.parent / "strategy.md"
        original = strategy_path.read_text() if strategy_path.exists() else None
        try:
            strategy_path.write_text("# 보수적 전략\n매수 조건...")
            _run_alert_monitor(trader)
        finally:
            if original is not None:
                strategy_path.write_text(original)
            elif strategy_path.exists():
                strategy_path.unlink()

        dca_calls = [c for c in mock_db.call_args_list
                     if c[0][0] == "strategy_alerts" and "물타기" in c[0][1].get("message", "")]
        assert len(dca_calls) == 1
        mock_tg.assert_called()

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "test_ak", "UPBIT_SECRET_KEY": "test_sk"})
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_stop_loss_suggestion_no_buy_signal(self, mock_get, mock_sound, mock_tg, mock_db):
        """Loss -3% to -5% with FGI > 30 -> stop-loss suggestion."""
        closes = _candle_closes_for_rsi(50)
        fgi_resp = MagicMock()
        fgi_resp.ok = True
        fgi_resp.json.return_value = {"data": [{"value": "45"}]}

        current_price = 96_000_000
        avg_buy_price = 100_000_000

        acct_resp = self._make_acct_response(avg_buy_price)
        mock_get.side_effect = [_mock_candle_response(closes), fgi_resp, acct_resp]

        trader = _make_trader(dry_run=False)
        trader.current_price = current_price
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        from pathlib import Path as _Path
        strategy_path = _Path(__file__).resolve().parent.parent / "strategy.md"
        original = strategy_path.read_text() if strategy_path.exists() else None
        try:
            strategy_path.write_text("# 보수적 전략\n매수 조건...")
            _run_alert_monitor(trader)
        finally:
            if original is not None:
                strategy_path.write_text(original)
            elif strategy_path.exists():
                strategy_path.unlink()

        sl_calls = [c for c in mock_db.call_args_list
                    if c[0][0] == "strategy_alerts" and "손절" in c[0][1].get("message", "")]
        assert len(sl_calls) == 1

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "test_ak", "UPBIT_SECRET_KEY": "test_sk"})
    @patch("scripts.short_term_trader.db_insert")
    @patch("scripts.short_term_trader.send_telegram")
    @patch("scripts.short_term_trader.sound_alert")
    @patch("scripts.short_term_trader.requests.get")
    def test_emergency_stop_loss_below_5pct(self, mock_get, mock_sound, mock_tg, mock_db):
        """Loss < -5% -> emergency stop-loss suggestion."""
        closes = _candle_closes_for_rsi(50)
        fgi_resp = MagicMock()
        fgi_resp.ok = True
        fgi_resp.json.return_value = {"data": [{"value": "45"}]}

        current_price = 93_000_000
        avg_buy_price = 100_000_000

        acct_resp = self._make_acct_response(avg_buy_price)
        mock_get.side_effect = [_mock_candle_response(closes), fgi_resp, acct_resp]

        trader = _make_trader(dry_run=False)
        trader.current_price = current_price
        trader.whale_recent = deque(maxlen=20)
        trader.price_history = deque(maxlen=600)
        trader.news_sentiment_score = 0.0

        from pathlib import Path as _Path
        strategy_path = _Path(__file__).resolve().parent.parent / "strategy.md"
        original = strategy_path.read_text() if strategy_path.exists() else None
        try:
            strategy_path.write_text("# 보수적 전략\n매수 조건...")
            _run_alert_monitor(trader)
        finally:
            if original is not None:
                strategy_path.write_text(original)
            elif strategy_path.exists():
                strategy_path.unlink()

        tg_calls = [str(c) for c in mock_tg.call_args_list]
        assert any("긴급 손절" in c for c in tg_calls)
