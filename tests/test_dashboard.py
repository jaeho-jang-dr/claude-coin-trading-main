"""
dashboard.py unit tests

All external dependencies are mocked - no real server launch or subprocess calls.
Uses Flask's test_client() for request testing.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Patch load_dotenv and qrcode before importing to avoid side effects
with patch("dotenv.load_dotenv"):
    from scripts.dashboard import app, run_script, get_local_ip, DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Flask app creation
# ---------------------------------------------------------------------------

class TestAppCreation:
    def test_app_exists(self):
        assert app is not None

    def test_app_is_flask_instance(self):
        from flask import Flask
        assert isinstance(app, Flask)

    def test_app_name(self):
        assert app.name == "scripts.dashboard"


# ---------------------------------------------------------------------------
# 2. Route registration
# ---------------------------------------------------------------------------

class TestRouteRegistration:
    def test_index_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/" in rules

    def test_qr_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/qr" in rules

    def test_api_market_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/market" in rules

    def test_api_portfolio_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/portfolio" in rules

    def test_api_fgi_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/fgi" in rules

    def test_api_status_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/status" in rules

    def test_api_toggle_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/toggle/<key>" in rules

    def test_api_analyze_route_registered(self):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/analyze" in rules


# ---------------------------------------------------------------------------
# 3. Index page
# ---------------------------------------------------------------------------

class TestIndexPage:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert b"<!DOCTYPE html>" in resp.data

    def test_index_contains_title(self, client):
        resp = client.get("/")
        assert b"Crypto Trading Bot" in resp.data or b"Crypto Bot" in resp.data

    def test_index_contains_portfolio_section(self, client):
        resp = client.get("/")
        assert b"Portfolio" in resp.data

    def test_index_contains_indicators_section(self, client):
        resp = client.get("/")
        assert b"Indicators" in resp.data

    def test_index_contains_controls_section(self, client):
        resp = client.get("/")
        assert b"Controls" in resp.data

    def test_index_contains_strategy_section(self, client):
        resp = client.get("/")
        assert b"Strategy Summary" in resp.data


# ---------------------------------------------------------------------------
# 4. QR page
# ---------------------------------------------------------------------------

class TestQRPage:
    @patch("scripts.dashboard.get_local_ip", return_value="192.168.1.100")
    @patch("sys.argv", ["dashboard.py"])
    def test_qr_returns_200(self, mock_ip, client):
        mock_qr_img = MagicMock()
        mock_qr_img.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG_FAKE"))
        with patch("qrcode.make", return_value=mock_qr_img):
            with patch.object(Path, "read_text", return_value="https://example.com/remote"):
                resp = client.get("/qr")
        assert resp.status_code == 200

    @patch("scripts.dashboard.get_local_ip", return_value="192.168.1.100")
    @patch("sys.argv", ["dashboard.py"])
    def test_qr_contains_qr_html(self, mock_ip, client):
        mock_qr_img = MagicMock()
        mock_qr_img.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG_FAKE"))
        with patch("qrcode.make", return_value=mock_qr_img):
            with patch.object(Path, "read_text", return_value="https://example.com/remote"):
                resp = client.get("/qr")
        assert b"QR" in resp.data

    @patch("scripts.dashboard.get_local_ip", return_value="192.168.1.100")
    @patch("sys.argv", ["dashboard.py"])
    def test_qr_without_remote_url(self, mock_ip, client):
        """When remote_url.txt doesn't exist, QR page should still render."""
        mock_qr_img = MagicMock()
        mock_qr_img.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG_FAKE"))
        with patch("qrcode.make", return_value=mock_qr_img):
            with patch.object(Path, "read_text", side_effect=FileNotFoundError):
                resp = client.get("/qr")
        assert resp.status_code == 200

    @patch("scripts.dashboard.get_local_ip", return_value="192.168.1.100")
    @patch("sys.argv", ["dashboard.py"])
    def test_qr_pending_url_excluded(self, mock_ip, client):
        """When remote_url.txt contains 'pending', only Dashboard QR is shown."""
        mock_qr_img = MagicMock()
        mock_qr_img.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG_FAKE"))
        with patch("qrcode.make", return_value=mock_qr_img) as mock_make:
            with patch.object(Path, "read_text", return_value="pending..."):
                resp = client.get("/qr")
        assert resp.status_code == 200
        # qrcode.make should be called only once (Dashboard only, no Claude Code Remote)
        assert mock_make.call_count == 1


# ---------------------------------------------------------------------------
# 5. Status endpoint
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    @patch("scripts.dashboard.load_dotenv")
    def test_status_returns_200(self, mock_dotenv, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    @patch("scripts.dashboard.load_dotenv")
    def test_status_returns_json(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert isinstance(data, dict)

    @patch("scripts.dashboard.load_dotenv")
    def test_status_contains_required_fields(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert "dry_run" in data
        assert "emergency_stop" in data
        assert "max_trade_amount" in data
        assert "max_daily_trades" in data

    @patch.dict(os.environ, {"DRY_RUN": "true", "EMERGENCY_STOP": "false",
                              "MAX_TRADE_AMOUNT": "100000", "MAX_DAILY_TRADES": "6"})
    @patch("scripts.dashboard.load_dotenv")
    def test_status_dry_run_true(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["dry_run"] is True

    @patch.dict(os.environ, {"DRY_RUN": "false", "EMERGENCY_STOP": "true",
                              "MAX_TRADE_AMOUNT": "500000", "MAX_DAILY_TRADES": "10"})
    @patch("scripts.dashboard.load_dotenv")
    def test_status_emergency_stop_true(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["emergency_stop"] is True
        assert data["dry_run"] is False
        assert data["max_trade_amount"] == 500000
        assert data["max_daily_trades"] == 10


# ---------------------------------------------------------------------------
# 6. Environment variable display in status
# ---------------------------------------------------------------------------

class TestEnvVarDisplay:
    @patch.dict(os.environ, {"DRY_RUN": "true", "EMERGENCY_STOP": "false"}, clear=False)
    @patch("scripts.dashboard.load_dotenv")
    def test_dry_run_shown_as_true(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["dry_run"] is True

    @patch.dict(os.environ, {"DRY_RUN": "false", "EMERGENCY_STOP": "false"}, clear=False)
    @patch("scripts.dashboard.load_dotenv")
    def test_dry_run_shown_as_false(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["dry_run"] is False

    @patch.dict(os.environ, {"EMERGENCY_STOP": "true"}, clear=False)
    @patch("scripts.dashboard.load_dotenv")
    def test_emergency_stop_shown_as_true(self, mock_dotenv, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["emergency_stop"] is True

    @patch.dict(os.environ, {}, clear=False)
    @patch("scripts.dashboard.load_dotenv")
    def test_default_values_when_env_missing(self, mock_dotenv, client):
        """When env vars are not set, defaults are used."""
        # Remove keys if present to test defaults
        env_copy = os.environ.copy()
        for key in ["DRY_RUN", "EMERGENCY_STOP", "MAX_TRADE_AMOUNT", "MAX_DAILY_TRADES"]:
            env_copy.pop(key, None)
        with patch.dict(os.environ, env_copy, clear=True):
            resp = client.get("/api/status")
            data = resp.get_json()
            assert data["dry_run"] is True  # default "true"
            assert data["emergency_stop"] is False  # default "false"
            assert data["max_trade_amount"] == 100000  # default
            assert data["max_daily_trades"] == 3  # default


# ---------------------------------------------------------------------------
# 7. Error handling - 404 for unknown routes
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_route_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_unknown_api_route_returns_404(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_invalid_toggle_key_returns_400(self, client):
        resp = client.get("/api/toggle/invalid_key")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# 8. API endpoints with mocked run_script
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    @patch("scripts.dashboard.run_script", return_value={"current_price": 130000000, "change_rate_24h": 0.025})
    def test_api_market(self, mock_run, client):
        resp = client.get("/api/market")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["current_price"] == 130000000
        mock_run.assert_called_once_with("collect_market_data.py")

    @patch("scripts.dashboard.run_script", return_value={"krw_balance": 500000, "total_eval": 1000000})
    def test_api_portfolio(self, mock_run, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["krw_balance"] == 500000
        mock_run.assert_called_once_with("get_portfolio.py")

    @patch("scripts.dashboard.run_script", return_value={"current": {"value": 25, "classification": "Extreme Fear"}})
    def test_api_fgi(self, mock_run, client):
        resp = client.get("/api/fgi")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["current"]["value"] == 25
        mock_run.assert_called_once_with("collect_fear_greed.py")

    @patch("scripts.dashboard.run_script")
    def test_api_analyze(self, mock_run, client):
        mock_run.side_effect = [
            {"current_price": 130000000, "indicators": {"rsi_14": 45.2}},
            {"current": {"value": 30}},
            {"krw_balance": 500000},
        ]
        resp = client.get("/api/analyze")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "분석 완료"
        assert data["price"] == 130000000
        assert data["rsi"] == 45.2
        assert data["fgi"] == 30
        assert data["krw_balance"] == 500000

    @patch("scripts.dashboard.run_script", side_effect=Exception("script crashed"))
    def test_api_analyze_error(self, mock_run, client):
        resp = client.get("/api/analyze")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# 9. Toggle endpoint
# ---------------------------------------------------------------------------

class TestToggleEndpoint:
    @patch.dict(os.environ, {"EMERGENCY_STOP": "false"}, clear=False)
    def test_toggle_emergency_stop(self, client, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EMERGENCY_STOP=false\nDRY_RUN=true\n")
        with patch("scripts.dashboard.PROJECT_ROOT", tmp_path):
            resp = client.get("/api/toggle/emergency_stop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["value"] == "true"
        assert "긴급정지" in data["message"]

    @patch.dict(os.environ, {"DRY_RUN": "true"}, clear=False)
    def test_toggle_dry_run(self, client, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EMERGENCY_STOP=false\nDRY_RUN=true\n")
        with patch("scripts.dashboard.PROJECT_ROOT", tmp_path):
            resp = client.get("/api/toggle/dry_run")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["value"] == "false"
        assert "DRY_RUN" in data["message"]


# ---------------------------------------------------------------------------
# 10. run_script helper
# ---------------------------------------------------------------------------

class TestRunScript:
    @patch("subprocess.run")
    def test_run_script_success(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "ok"}',
        )
        result = run_script("test_script.py")
        assert result == {"result": "ok"}

    @patch("subprocess.run")
    def test_run_script_failure(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stderr="some error",
        )
        result = run_script("test_script.py")
        assert "error" in result
        assert result["error"] == "some error"

    @patch("subprocess.run", side_effect=Exception("timeout"))
    def test_run_script_exception(self, mock_subprocess):
        result = run_script("test_script.py")
        assert "error" in result
        assert "timeout" in result["error"]


# ---------------------------------------------------------------------------
# 11. get_local_ip helper
# ---------------------------------------------------------------------------

class TestGetLocalIP:
    @patch("socket.socket")
    def test_get_local_ip_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.getsockname.return_value = ("192.168.1.50", 12345)
        mock_socket_cls.return_value = mock_sock
        assert get_local_ip() == "192.168.1.50"

    @patch("socket.socket", side_effect=Exception("network error"))
    def test_get_local_ip_fallback(self, mock_socket_cls):
        assert get_local_ip() == "127.0.0.1"
