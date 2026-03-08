"""
execute_trade.py unit tests

All network calls are mocked - no real API calls are made.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# We need to patch load_dotenv before importing the module so it doesn't
# try to load a real .env file during import.
with patch("dotenv.load_dotenv"):
    from scripts.execute_trade import (
        LOCK_TIMEOUT_SECONDS,
        KST,
        acquire_lock,
        execute,
        release_lock,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_lock(lock_path: Path, pid: int, age_seconds: float = 0, bad_json: bool = False):
    """Write a lock file with given pid and age."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if bad_json:
        lock_path.write_text("{corrupted json!!")
        return
    ts = datetime.now(KST) - timedelta(seconds=age_seconds)
    lock_path.write_text(json.dumps({
        "process": "test",
        "pid": pid,
        "timestamp": ts.isoformat(),
    }))


def _set_env(monkeypatch, **kwargs):
    """Set environment variables for tests."""
    defaults = {
        "EMERGENCY_STOP": "false",
        "DRY_RUN": "false",
        "MAX_TRADE_AMOUNT": "100000",
        "UPBIT_ACCESS_KEY": "test_access_key",
        "UPBIT_SECRET_KEY": "test_secret_key",
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_lock_file(tmp_path, monkeypatch):
    """Redirect LOCK_FILE to a temp directory for every test."""
    lock = tmp_path / "data" / "trading.lock"
    import scripts.execute_trade as mod
    monkeypatch.setattr(mod, "LOCK_FILE", lock)


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "data" / "trading.lock"


# ===========================================================================
# 1. EMERGENCY_STOP
# ===========================================================================

class TestEmergencyStop:
    def test_emergency_stop_blocks_trade(self, monkeypatch):
        monkeypatch.setenv("EMERGENCY_STOP", "true")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "EMERGENCY_STOP" in result["error"]


# ===========================================================================
# 2. DRY_RUN
# ===========================================================================

class TestDryRun:
    @patch("scripts.execute_trade.requests.post")
    def test_dry_run_returns_success_no_api_call(self, mock_post, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is True
        mock_post.assert_not_called()


# ===========================================================================
# 3. MAX_TRADE_AMOUNT exceeded
# ===========================================================================

class TestMaxTradeAmount:
    def test_bid_exceeds_max_amount(self, monkeypatch):
        _set_env(monkeypatch, MAX_TRADE_AMOUNT="100000")
        result = execute("bid", "KRW-BTC", "200000")
        assert result["success"] is False
        assert "상한 초과" in result["error"]

    # 4. MAX_TRADE_AMOUNT ok (under DRY_RUN to avoid real call)
    def test_bid_within_max_amount_dry_run(self, monkeypatch):
        _set_env(monkeypatch, MAX_TRADE_AMOUNT="100000", DRY_RUN="true")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is True


# ===========================================================================
# 5-10. Lock file tests
# ===========================================================================

class TestAcquireLock:
    # 5. Fresh creation
    def test_fresh_lock_creation(self, lock_path):
        assert not lock_path.exists()
        acquire_lock()
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    # 6. Stale by timeout
    def test_stale_lock_by_timeout(self, lock_path):
        _write_lock(lock_path, pid=os.getpid(), age_seconds=LOCK_TIMEOUT_SECONDS + 60)
        # Should auto-remove stale lock and succeed
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    # 7. Stale by dead PID
    def test_stale_lock_by_dead_pid(self, lock_path):
        dead_pid = 99999999  # Almost certainly not running
        _write_lock(lock_path, pid=dead_pid, age_seconds=5)
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    # 8. Active lock (live process, recent timestamp)
    def test_active_lock_raises(self, lock_path, monkeypatch):
        fake_pid = 77777
        _write_lock(lock_path, pid=fake_pid, age_seconds=5)
        # Make os.kill think the process is alive
        original_kill = os.kill

        def fake_kill(pid, sig):
            if pid == fake_pid and sig == 0:
                return  # pretend process exists
            return original_kill(pid, sig)

        monkeypatch.setattr(os, "kill", fake_kill)
        with pytest.raises(RuntimeError, match="다른 매매 프로세스 실행 중"):
            acquire_lock()

    # 9. Corrupt JSON
    def test_corrupt_json_lock(self, lock_path):
        _write_lock(lock_path, pid=0, bad_json=True)
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    # 10. pid=0 treated as stale (not os.kill(0,0))
    def test_pid_zero_treated_as_stale(self, lock_path):
        _write_lock(lock_path, pid=0, age_seconds=5)
        # pid=0 → pid_alive=False → age>0 doesn't matter since not pid_alive
        # But age=5 < 120 so the condition is: age > 120 OR not pid_alive
        # not pid_alive=True → stale lock removed
        acquire_lock()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()


# ===========================================================================
# 11-12. release_lock tests
# ===========================================================================

class TestReleaseLock:
    # 11. Own PID → deletes
    def test_release_own_pid(self, lock_path):
        _write_lock(lock_path, pid=os.getpid(), age_seconds=0)
        release_lock()
        assert not lock_path.exists()

    # 12. Different PID → does NOT delete
    def test_release_different_pid(self, lock_path):
        _write_lock(lock_path, pid=1, age_seconds=0)
        release_lock()
        assert lock_path.exists()


# ===========================================================================
# 13-16. Order execution tests (mock requests.post)
# ===========================================================================

class TestOrderExecution:
    # 13. Successful order
    @patch("scripts.execute_trade.requests.post")
    def test_successful_order(self, mock_post, monkeypatch):
        _set_env(monkeypatch)
        order_response = {
            "uuid": "test-uuid",
            "side": "bid",
            "ord_type": "price",
            "price": "50000",
            "state": "wait",
            "market": "KRW-BTC",
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = order_response
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["response"]["uuid"] == "test-uuid"
        mock_post.assert_called_once()

    # 14. Failed order - insufficient balance
    @patch("scripts.execute_trade.requests.post")
    def test_insufficient_balance(self, mock_post, monkeypatch):
        _set_env(monkeypatch)
        error_response = {
            "error": {
                "name": "insufficient_funds_bid",
                "message": "잔고가 부족합니다.",
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.json.return_value = error_response
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "insufficient_funds_bid" in result["error"]

    # 15. Non-JSON response
    @patch("scripts.execute_trade.requests.post")
    def test_non_json_response(self, mock_post, monkeypatch):
        _set_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_resp.text = "<html>Server Error</html>"
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert result["response"]["raw_response"] == "<html>Server Error</html>"

    # 16. Network timeout
    @patch("scripts.execute_trade.requests.post")
    def test_network_timeout(self, mock_post, monkeypatch):
        _set_env(monkeypatch)
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "주문 요청 실패" in result["error"]
