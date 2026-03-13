"""
execute_trade.py comprehensive unit tests.

Covers:
  - EMERGENCY_STOP (buy blocked, sell allowed for liquidation)
  - Auto emergency (auto_emergency.json)
  - DRY_RUN mode
  - MAX_TRADE_AMOUNT enforcement (buy only, sell bypasses)
  - Lock file: creation, stale detection, dead PID, corrupt JSON, release
  - Order execution: success, API errors, non-JSON response, network timeout
  - check_open_orders_and_cancel
  - _record_trade_to_db
  - make_auth_header

All network calls are mocked - no real API calls are made.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import requests

with patch("dotenv.load_dotenv"):
    from scripts.execute_trade import (
        LOCK_TIMEOUT_SECONDS,
        KST,
        acquire_lock,
        execute,
        release_lock,
        make_auth_header,
        check_open_orders_and_cancel,
        _record_trade_to_db,
    )


# ── Helpers ──────────────────────────────────────────────

def _write_lock(lock_path, pid, age_seconds=0, bad_json=False):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if bad_json:
        lock_path.write_text("{corrupted!!")
        return
    ts = datetime.now(KST) - timedelta(seconds=age_seconds)
    lock_path.write_text(json.dumps({
        "process": "test",
        "pid": pid,
        "timestamp": ts.isoformat(),
    }))


def _set_env(monkeypatch, **kwargs):
    defaults = {
        "EMERGENCY_STOP": "false",
        "DRY_RUN": "false",
        "MAX_TRADE_AMOUNT": "100000",
        "UPBIT_ACCESS_KEY": "test_access",
        "UPBIT_SECRET_KEY": "test_secret",
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_lock_and_project(tmp_path, monkeypatch):
    """Redirect LOCK_FILE and PROJECT_DIR to temp for every test."""
    lock = tmp_path / "data" / "trading.lock"
    import scripts.execute_trade as mod
    monkeypatch.setattr(mod, "LOCK_FILE", lock)
    monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "data" / "trading.lock"


@pytest.fixture
def auto_em_path(tmp_path):
    p = tmp_path / "data" / "auto_emergency.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════════════════
# EMERGENCY_STOP
# ══════════════════════════════════════════════════════════════

class TestEmergencyStop:
    def test_blocks_buy(self, monkeypatch):
        monkeypatch.setenv("EMERGENCY_STOP", "true")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "EMERGENCY_STOP" in result["error"]

    @patch("scripts.execute_trade.requests.post")
    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    def test_allows_sell_for_liquidation(self, mock_cancel, mock_post, monkeypatch):
        """EMERGENCY_STOP allows sells for position liquidation."""
        _set_env(monkeypatch, EMERGENCY_STOP="true")
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "sell-uuid", "side": "ask"}
        mock_post.return_value = mock_resp

        result = execute("ask", "KRW-BTC", "0.001")
        assert result["success"] is True
        assert result["dry_run"] is False

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("EMERGENCY_STOP", "TRUE")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False


# ══════════════════════════════════════════════════════════════
# Auto emergency (auto_emergency.json)
# ══════════════════════════════════════════════════════════════

class TestAutoEmergency:
    def test_blocks_buy_when_active(self, monkeypatch, auto_em_path):
        _set_env(monkeypatch)
        auto_em_path.write_text(json.dumps({"active": True, "reason": "급락"}))

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "자동긴급정지" in result["error"]
        assert "급락" in result["error"]

    @patch("scripts.execute_trade.requests.post")
    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    def test_allows_sell_when_active(self, mock_cancel, mock_post, monkeypatch, auto_em_path):
        """Auto emergency allows sells (liquidation)."""
        _set_env(monkeypatch)
        auto_em_path.write_text(json.dumps({"active": True, "reason": "급락"}))
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "uuid"}
        mock_post.return_value = mock_resp

        result = execute("ask", "KRW-BTC", "0.001")
        assert result["success"] is True

    def test_inactive_does_not_block(self, monkeypatch, auto_em_path):
        _set_env(monkeypatch, DRY_RUN="true")
        auto_em_path.write_text(json.dumps({"active": False}))

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_missing_file_does_not_block(self, monkeypatch):
        _set_env(monkeypatch, DRY_RUN="true")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True

    def test_corrupt_file_does_not_block(self, monkeypatch, auto_em_path):
        _set_env(monkeypatch, DRY_RUN="true")
        auto_em_path.write_text("{bad json!!")

        result = execute("bid", "KRW-BTC", "50000")
        # Corrupt file should be handled gracefully (pass in except)
        assert result["success"] is True


# ══════════════════════════════════════════════════════════════
# DRY_RUN
# ══════════════════════════════════════════════════════════════

class TestDryRun:
    @patch("scripts.execute_trade.requests.post")
    def test_returns_success_no_api_call(self, mock_post, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is True
        mock_post.assert_not_called()

    def test_default_is_true(self, monkeypatch):
        """DRY_RUN defaults to 'true' when not set."""
        monkeypatch.delenv("DRY_RUN", raising=False)
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["dry_run"] is True

    def test_dry_run_for_sell(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        result = execute("ask", "KRW-BTC", "0.001")
        assert result["dry_run"] is True
        assert result["side"] == "ask"


# ══════════════════════════════════════════════════════════════
# MAX_TRADE_AMOUNT
# ══════════════════════════════════════════════════════════════

class TestMaxTradeAmount:
    def test_bid_exceeds_max(self, monkeypatch):
        _set_env(monkeypatch, MAX_TRADE_AMOUNT="100000")
        result = execute("bid", "KRW-BTC", "200000")
        assert result["success"] is False
        assert "상한 초과" in result["error"]

    def test_bid_at_exact_max(self, monkeypatch):
        _set_env(monkeypatch, DRY_RUN="true", MAX_TRADE_AMOUNT="100000")
        # Execute goes to DRY_RUN check first (step 2), which is before max check (step 3)
        # So with DRY_RUN=false we test max. With DRY_RUN=true, it returns before max check.
        _set_env(monkeypatch, DRY_RUN="false", MAX_TRADE_AMOUNT="100000")
        # 100000 == max, int(float("100000")) == 100000, NOT > 100000
        # So it should pass the max check and proceed to order
        # We need to mock the order execution
        with patch("scripts.execute_trade.requests.post") as mock_post, \
             patch("scripts.execute_trade.check_open_orders_and_cancel"):
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {"uuid": "ok"}
            mock_post.return_value = mock_resp
            result = execute("bid", "KRW-BTC", "100000")
        assert result["success"] is True

    @patch("scripts.execute_trade.requests.post")
    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    def test_sell_ignores_max_amount(self, mock_cancel, mock_post, monkeypatch):
        """MAX_TRADE_AMOUNT only applies to buys, not sells."""
        _set_env(monkeypatch, MAX_TRADE_AMOUNT="100000")
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "sell"}
        mock_post.return_value = mock_resp

        result = execute("ask", "KRW-BTC", "999.0")
        assert result["success"] is True

    def test_float_amount_string(self, monkeypatch):
        _set_env(monkeypatch, MAX_TRADE_AMOUNT="100000")
        result = execute("bid", "KRW-BTC", "200000.50")
        assert result["success"] is False
        assert "상한 초과" in result["error"]


# ══════════════════════════════════════════════════════════════
# Lock file
# ══════════════════════════════════════════════════════════════

class TestAcquireLock:
    def test_fresh_creation(self, lock_path):
        assert not lock_path.exists()
        acquire_lock()
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    def test_stale_by_timeout(self, lock_path):
        _write_lock(lock_path, pid=os.getpid(), age_seconds=LOCK_TIMEOUT_SECONDS + 60)
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    def test_stale_by_dead_pid(self, lock_path):
        _write_lock(lock_path, pid=99999999, age_seconds=5)
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    def test_active_lock_raises_timeout(self, lock_path, monkeypatch):
        fake_pid = 77777
        _write_lock(lock_path, pid=fake_pid, age_seconds=5)

        original_kill = os.kill
        def fake_kill(pid, sig):
            if pid == fake_pid and sig == 0:
                return
            return original_kill(pid, sig)

        monkeypatch.setattr(os, "kill", fake_kill)
        with pytest.raises((TimeoutError, RuntimeError)):
            acquire_lock(timeout=1)

    def test_corrupt_json_removed(self, lock_path):
        _write_lock(lock_path, pid=0, bad_json=True)
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    def test_pid_zero_treated_as_stale(self, lock_path):
        _write_lock(lock_path, pid=0, age_seconds=5)
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()


class TestReleaseLock:
    def test_release_own_pid(self, lock_path):
        _write_lock(lock_path, pid=os.getpid())
        release_lock()
        assert not lock_path.exists()

    def test_does_not_release_other_pid(self, lock_path):
        _write_lock(lock_path, pid=1)
        release_lock()
        assert lock_path.exists()

    def test_no_file_no_error(self, lock_path):
        """Releasing when no lock file exists should not raise."""
        release_lock()


# ══════════════════════════════════════════════════════════════
# Order execution
# ══════════════════════════════════════════════════════════════

class TestOrderExecution:
    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_successful_buy(self, mock_post, mock_cancel, monkeypatch):
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "uuid": "test-uuid", "side": "bid", "price": "50000",
            "state": "wait", "market": "KRW-BTC",
        }
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["response"]["uuid"] == "test-uuid"
        assert "_latency_ms" in result

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_successful_sell(self, mock_post, mock_cancel, monkeypatch):
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "sell-uuid", "side": "ask"}
        mock_post.return_value = mock_resp

        result = execute("ask", "KRW-BTC", "0.001")
        assert result["success"] is True
        assert result["side"] == "ask"

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_buy_order_body(self, mock_post, mock_cancel, monkeypatch):
        """Buy order uses ord_type=price and price field."""
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "u"}
        mock_post.return_value = mock_resp

        execute("bid", "KRW-BTC", "50000")
        body = mock_post.call_args[1]["json"]
        assert body["ord_type"] == "price"
        assert body["price"] == "50000"
        assert body["side"] == "bid"

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_sell_order_body(self, mock_post, mock_cancel, monkeypatch):
        """Sell order uses ord_type=market and volume field."""
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "u"}
        mock_post.return_value = mock_resp

        execute("ask", "KRW-BTC", "0.001")
        body = mock_post.call_args[1]["json"]
        assert body["ord_type"] == "market"
        assert body["volume"] == "0.001"
        assert body["side"] == "ask"

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_insufficient_balance(self, mock_post, mock_cancel, monkeypatch):
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.json.return_value = {
            "error": {"name": "insufficient_funds_bid", "message": "잔고 부족"}
        }
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "insufficient_funds_bid" in result["error"]

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_non_json_response(self, mock_post, mock_cancel, monkeypatch):
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_resp.text = "<html>Server Error</html>"
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "raw_response" in result["response"]

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_network_timeout(self, mock_post, mock_cancel, monkeypatch):
        _set_env(monkeypatch)
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "주문 요청 실패" in result["error"]

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_connection_error(self, mock_post, mock_cancel, monkeypatch):
        _set_env(monkeypatch)
        mock_post.side_effect = requests.exceptions.ConnectionError("DNS")

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "주문 요청 실패" in result["error"]

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_lock_released_after_execution(self, mock_post, mock_cancel, monkeypatch, lock_path):
        """Lock file is released after order execution (success or failure)."""
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "u"}
        mock_post.return_value = mock_resp

        execute("bid", "KRW-BTC", "50000")
        assert not lock_path.exists()

    @patch("scripts.execute_trade.check_open_orders_and_cancel")
    @patch("scripts.execute_trade.requests.post")
    def test_lock_released_on_failure(self, mock_post, mock_cancel, monkeypatch, lock_path):
        _set_env(monkeypatch)
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        execute("bid", "KRW-BTC", "50000")
        assert not lock_path.exists()


# ══════════════════════════════════════════════════════════════
# make_auth_header
# ══════════════════════════════════════════════════════════════

class TestMakeAuthHeader:
    def test_missing_keys_raises(self, monkeypatch):
        monkeypatch.delenv("UPBIT_ACCESS_KEY", raising=False)
        monkeypatch.delenv("UPBIT_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="환경변수"):
            make_auth_header("test_query")

    def test_returns_bearer_token(self, monkeypatch):
        monkeypatch.setenv("UPBIT_ACCESS_KEY", "acc")
        monkeypatch.setenv("UPBIT_SECRET_KEY", "sec")
        headers = make_auth_header("market=KRW-BTC")
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["Content-Type"] == "application/json"

    def test_query_hash_is_sha512(self, monkeypatch):
        import hashlib
        monkeypatch.setenv("UPBIT_ACCESS_KEY", "acc")
        monkeypatch.setenv("UPBIT_SECRET_KEY", "sec")

        with patch("scripts.execute_trade.jwt.encode") as mock_jwt:
            mock_jwt.return_value = "token"
            make_auth_header("test_query=123")

            payload = mock_jwt.call_args[0][0]
            expected_hash = hashlib.sha512(b"test_query=123").hexdigest()
            assert payload["query_hash"] == expected_hash
            assert payload["query_hash_alg"] == "SHA512"


# ══════════════════════════════════════════════════════════════
# check_open_orders_and_cancel
# ══════════════════════════════════════════════════════════════

class TestCheckOpenOrders:
    @patch("scripts.execute_trade.time.sleep")
    @patch("scripts.execute_trade.requests.delete")
    @patch("scripts.execute_trade.requests.get")
    @patch("scripts.execute_trade.make_auth_header")
    def test_cancels_same_side_orders(self, mock_auth, mock_get, mock_delete, mock_sleep, monkeypatch):
        mock_auth.return_value = {"Authorization": "Bearer x"}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {"uuid": "order-1", "side": "bid"},
            {"uuid": "order-2", "side": "ask"},
        ]
        mock_get.return_value = mock_resp

        check_open_orders_and_cancel("KRW-BTC", "bid")

        # Should only cancel order-1 (same side)
        assert mock_delete.call_count == 1

    @patch("scripts.execute_trade.requests.get")
    @patch("scripts.execute_trade.make_auth_header")
    def test_api_failure_does_not_raise(self, mock_auth, mock_get, monkeypatch):
        """API failure in check is non-fatal (caught by except)."""
        mock_auth.return_value = {"Authorization": "Bearer x"}
        mock_get.side_effect = requests.exceptions.Timeout("timeout")

        # Should not raise
        check_open_orders_and_cancel("KRW-BTC", "bid")

    @patch("scripts.execute_trade.requests.get")
    @patch("scripts.execute_trade.make_auth_header")
    def test_no_open_orders(self, mock_auth, mock_get, monkeypatch):
        mock_auth.return_value = {"Authorization": "Bearer x"}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        # Should complete without error
        check_open_orders_and_cancel("KRW-BTC", "bid")


# ══════════════════════════════════════════════════════════════
# _record_trade_to_db
# ══════════════════════════════════════════════════════════════

class TestRecordTradeToDb:
    @patch("scripts.execute_trade.requests.post")
    def test_skips_when_no_supabase_url(self, mock_post, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

        _record_trade_to_db({"side": "bid", "success": True})
        mock_post.assert_not_called()

    @patch("scripts.execute_trade.requests.post")
    def test_records_successful_trade(self, mock_post, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key123")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        result = {
            "side": "bid",
            "success": True,
            "market": "KRW-BTC",
            "amount": "50000",
            "dry_run": False,
            "response": {"uuid": "order-123", "price": "50000", "volume": "0.001"},
            "_exec_started": "2026-03-13T10:00:00+09:00",
            "_exec_completed": "2026-03-13T10:00:01+09:00",
            "_latency_ms": 1000,
        }
        _record_trade_to_db(result, source="agent")

        mock_post.assert_called_once()
        row = mock_post.call_args[1]["json"]
        assert row["decision"] == "매수"
        assert row["execution_status"] == "success"
        assert row["source"] == "agent"
        assert row["order_uuid"] == "order-123"

    @patch("scripts.execute_trade.requests.post")
    def test_records_failed_trade(self, mock_post, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key123")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        result = {
            "side": "ask",
            "success": False,
            "market": "KRW-BTC",
            "amount": "0.001",
            "dry_run": False,
            "error": "잔고 부족",
        }
        _record_trade_to_db(result, source="manual")

        row = mock_post.call_args[1]["json"]
        assert row["decision"] == "매도"
        assert row["execution_status"] == "failed"
        assert row["execution_error"] == "잔고 부족"

    @patch("scripts.execute_trade.requests.post")
    def test_retries_without_dry_run_column(self, mock_post, monkeypatch):
        """If DB rejects dry_run column, retries without it."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key123")

        # First call fails with dry_run error, second succeeds
        fail_resp = MagicMock()
        fail_resp.ok = False
        fail_resp.text = "dry_run column does not exist"

        ok_resp = MagicMock()
        ok_resp.ok = True

        mock_post.side_effect = [fail_resp, ok_resp]

        _record_trade_to_db(
            {"side": "bid", "success": True, "market": "KRW-BTC",
             "amount": "50000", "dry_run": True},
            source="manual",
        )

        assert mock_post.call_count == 2
        # Second call should not have dry_run
        second_row = mock_post.call_args_list[1][1]["json"]
        assert "dry_run" not in second_row

    @patch("scripts.execute_trade.requests.post")
    def test_db_failure_does_not_raise(self, mock_post, monkeypatch):
        """DB write failure should not propagate (try/except)."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key123")
        mock_post.side_effect = Exception("Connection refused")

        # Should not raise
        _record_trade_to_db(
            {"side": "bid", "success": True, "market": "KRW-BTC", "amount": "50000"},
            source="manual",
        )


# ══════════════════════════════════════════════════════════════
# Safety mechanism integration (order of checks)
# ══════════════════════════════════════════════════════════════

class TestSafetyOrder:
    """Verify the correct order of safety checks:
    1a. EMERGENCY_STOP (buy only)
    1b. auto_emergency.json (buy only)
    2. DRY_RUN
    3. MAX_TRADE_AMOUNT (buy only)
    4. Lock + execute
    """

    def test_emergency_before_dry_run(self, monkeypatch):
        """EMERGENCY_STOP checked before DRY_RUN."""
        monkeypatch.setenv("EMERGENCY_STOP", "true")
        monkeypatch.setenv("DRY_RUN", "true")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "EMERGENCY_STOP" in result["error"]

    def test_dry_run_before_max_amount(self, monkeypatch):
        """DRY_RUN returns before MAX_TRADE_AMOUNT check."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("MAX_TRADE_AMOUNT", "100")
        # Amount exceeds max, but DRY_RUN returns first
        result = execute("bid", "KRW-BTC", "999999")
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_auto_emergency_before_dry_run(self, monkeypatch, auto_em_path):
        """auto_emergency.json checked before DRY_RUN."""
        _set_env(monkeypatch, DRY_RUN="true")
        auto_em_path.write_text(json.dumps({"active": True, "reason": "test"}))

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "자동긴급정지" in result["error"]

    def test_result_contains_all_fields(self, monkeypatch):
        """Every result has required fields."""
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        result = execute("bid", "KRW-BTC", "50000")

        for key in ["success", "dry_run", "side", "market", "amount", "timestamp"]:
            assert key in result, f"Missing key: {key}"
