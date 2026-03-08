"""
web_server.py unit tests

Tests the DashboardHandler and helper functions without launching a real server.
All external dependencies (Supabase, subprocess, file I/O) are mocked.
"""

import http.server
import io
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Patch load_dotenv and sys.stdout/stderr reconfigure before import
with patch("dotenv.load_dotenv"):
    with patch("sys.stdout") as mock_stdout, patch("sys.stderr") as mock_stderr:
        mock_stdout.reconfigure = MagicMock()
        mock_stderr.reconfigure = MagicMock()
        from scripts.web_server import (
            DashboardHandler,
            _get_active_strategy,
            _update_strategy,
            api_decisions,
            api_status,
            get_local_ip,
            read_env,
            supabase_get,
            update_env_value,
        )


# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------

class FakeRequest(io.BytesIO):
    """Simulates a socket-like request object for the handler."""

    def makefile(self, *args, **kwargs):
        return self


def make_handler(method, path, body=None, headers=None):
    """Create a DashboardHandler for testing without a real socket.

    Returns the handler after processing the request. Response is captured
    in handler.wfile (a BytesIO).
    """
    if body is None:
        body = b""
    elif isinstance(body, str):
        body = body.encode("utf-8")

    # Build raw HTTP request
    request_line = f"{method} {path} HTTP/1.1\r\n"
    header_lines = "Host: localhost\r\n"
    if headers:
        for k, v in headers.items():
            header_lines += f"{k}: {v}\r\n"
    if method == "POST":
        header_lines += f"Content-Length: {len(body)}\r\n"
        header_lines += "Content-Type: application/json\r\n"
    header_lines += "\r\n"

    raw = (request_line + header_lines).encode("utf-8") + body
    request = FakeRequest(raw)

    # Capture response
    response = io.BytesIO()

    # Create handler with mocked server and client address
    with patch.object(DashboardHandler, "__init__", lambda self, *a, **kw: None):
        handler = DashboardHandler()
        handler.request = request
        handler.client_address = ("127.0.0.1", 12345)
        handler.server = MagicMock()
        handler.rfile = io.BytesIO(body)
        handler.wfile = response
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.command = method
        handler.path = path
        handler.headers = http.client.HTTPMessage()
        if method == "POST":
            handler.headers["Content-Length"] = str(len(body))
            handler.headers["Content-Type"] = "application/json"
        if headers:
            for k, v in headers.items():
                handler.headers[k] = v

        handler.close_connection = True
        handler._headers_buffer = []

        # Use a list to capture responses
        handler._response_code = None
        handler._response_headers = {}
        handler._response_body = None

    return handler


def call_get(path):
    """Perform a GET request and return (status_code, parsed_json_or_raw)."""
    handler = make_handler("GET", path)
    # Override _json_response to capture output
    captured = {}

    original_json_response = handler._json_response

    def capture_json(data, code=200):
        captured["code"] = code
        captured["data"] = data

    handler._json_response = capture_json

    # For API routes, call do_GET logic manually
    handler.do_GET()
    if "code" in captured:
        return captured["code"], captured["data"]
    return None, None


def call_post(path, body=None):
    """Perform a POST request and return (status_code, parsed_json)."""
    handler = make_handler("POST", path, json.dumps(body) if body else "{}")
    captured = {}

    def capture_json(data, code=200):
        captured["code"] = code
        captured["data"] = data

    handler._json_response = capture_json
    handler.do_POST()
    if "code" in captured:
        return captured["code"], captured["data"]
    return None, None


# ---------------------------------------------------------------------------
# Tests: get_local_ip
# ---------------------------------------------------------------------------

class TestGetLocalIP:
    def test_returns_ip_string(self):
        with patch("socket.socket") as mock_sock:
            instance = MagicMock()
            instance.getsockname.return_value = ("192.168.1.100", 0)
            mock_sock.return_value = instance
            ip = get_local_ip()
            assert ip == "192.168.1.100"
            instance.connect.assert_called_once_with(("8.8.8.8", 80))
            instance.close.assert_called_once()

    def test_returns_localhost_on_error(self):
        with patch("socket.socket", side_effect=OSError("no network")):
            ip = get_local_ip()
            assert ip == "127.0.0.1"


# ---------------------------------------------------------------------------
# Tests: read_env
# ---------------------------------------------------------------------------

class TestReadEnv:
    def test_parses_env_file(self, tmp_path):
        env_content = "DRY_RUN=true\nEMERGENCY_STOP=false\nMAX_TRADE_AMOUNT=500000\n"
        env_file = tmp_path / ".env"
        env_file.write_text(env_content, encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            config = read_env()
            assert config["DRY_RUN"] == "true"
            assert config["EMERGENCY_STOP"] == "false"
            assert config["MAX_TRADE_AMOUNT"] == "500000"

    def test_returns_empty_if_no_file(self, tmp_path):
        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            config = read_env()
            assert config == {}

    def test_ignores_comments_and_blanks(self, tmp_path):
        env_content = "# comment\n\nDRY_RUN=true\n  \n"
        env_file = tmp_path / ".env"
        env_file.write_text(env_content, encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            config = read_env()
            assert config == {"DRY_RUN": "true"}


# ---------------------------------------------------------------------------
# Tests: update_env_value
# ---------------------------------------------------------------------------

class TestUpdateEnvValue:
    def test_updates_existing_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DRY_RUN=true\nOTHER=val\n", encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            update_env_value("DRY_RUN", "false")

        content = env_file.read_text(encoding="utf-8")
        assert "DRY_RUN=false" in content
        assert "OTHER=val" in content
        assert os.environ["DRY_RUN"] == "false"

    def test_appends_new_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DRY_RUN=true\n", encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            update_env_value("NEW_KEY", "new_value")

        content = env_file.read_text(encoding="utf-8")
        assert "NEW_KEY=new_value" in content
        assert os.environ["NEW_KEY"] == "new_value"


# ---------------------------------------------------------------------------
# Tests: supabase_get
# ---------------------------------------------------------------------------

class TestSupabaseGet:
    def test_returns_empty_without_credentials(self):
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            result = supabase_get("decisions")
            assert result == []

    def test_returns_data_on_success(self):
        mock_data = [{"id": 1, "decision": "buy"}]
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        }, clear=False):
            with patch("scripts.web_server.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.json.return_value = mock_data
                mock_resp.raise_for_status = MagicMock()
                mock_get.return_value = mock_resp

                result = supabase_get("decisions", "select=*&limit=5")
                assert result == mock_data
                mock_get.assert_called_once()
                call_url = mock_get.call_args[0][0]
                assert "decisions" in call_url
                assert "select=*&limit=5" in call_url

    def test_returns_error_on_exception(self):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        }, clear=False):
            with patch("scripts.web_server.requests.get", side_effect=Exception("timeout")):
                result = supabase_get("decisions")
                assert "error" in result
                assert "timeout" in result["error"]

    def test_sends_auth_headers(self):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "my-secret-key",
        }, clear=False):
            with patch("scripts.web_server.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.json.return_value = []
                mock_resp.raise_for_status = MagicMock()
                mock_get.return_value = mock_resp

                supabase_get("table")
                headers = mock_get.call_args[1]["headers"]
                assert headers["apikey"] == "my-secret-key"
                assert headers["Authorization"] == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# Tests: _get_active_strategy
# ---------------------------------------------------------------------------

class TestGetActiveStrategy:
    def test_reads_strategy_from_file(self, tmp_path):
        strat = tmp_path / "strategy.md"
        strat.write_text("# Strategy\n## 활성 전략: 보수적\nsome content\n", encoding="utf-8")
        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            assert _get_active_strategy() == "보수적"

    def test_returns_unknown_if_no_file(self, tmp_path):
        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            assert _get_active_strategy() == "unknown"

    def test_returns_unknown_if_no_marker(self, tmp_path):
        strat = tmp_path / "strategy.md"
        strat.write_text("# Strategy\nno active marker here\n", encoding="utf-8")
        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            assert _get_active_strategy() == "unknown"


# ---------------------------------------------------------------------------
# Tests: _update_strategy
# ---------------------------------------------------------------------------

class TestUpdateStrategy:
    def test_changes_active_strategy_marker(self, tmp_path):
        strat = tmp_path / "strategy.md"
        strat.write_text(
            "## 활성 전략: 보수적\n"
            "## 보수적 전략 ← 현재 활성\n"
            "## 보통 전략\n"
            "## 공격적 전략\n",
            encoding="utf-8",
        )
        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            _update_strategy("moderate", "보통 (moderate)")

        content = strat.read_text(encoding="utf-8")
        assert "## 활성 전략: 보통 (moderate)" in content
        assert "보통 전략 ← 현재 활성" in content
        # Old marker removed
        assert "보수적 전략 ← 현재 활성" not in content

    def test_aggressive_strategy(self, tmp_path):
        strat = tmp_path / "strategy.md"
        strat.write_text(
            "## 활성 전략: 보수적\n"
            "## 보수적 전략 ← 현재 활성\n"
            "## 공격적 전략\n",
            encoding="utf-8",
        )
        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            _update_strategy("aggressive", "공격적 (aggressive)")

        content = strat.read_text(encoding="utf-8")
        assert "공격적 전략 ← 현재 활성" in content


# ---------------------------------------------------------------------------
# Tests: api_status
# ---------------------------------------------------------------------------

class TestApiStatus:
    def test_returns_all_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "DRY_RUN=true\nEMERGENCY_STOP=false\n"
            "MAX_TRADE_AMOUNT=200000\nMAX_DAILY_TRADES=5\n"
            "MAX_POSITION_RATIO=0.3\n",
            encoding="utf-8",
        )
        strat = tmp_path / "strategy.md"
        strat.write_text("## 활성 전략: 보수적\n", encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            result = api_status()

        assert result["dry_run"] == "true"
        assert result["emergency_stop"] == "false"
        assert result["max_trade_amount"] == "200000"
        assert result["max_daily_trades"] == "5"
        assert result["max_position_ratio"] == "0.3"
        assert result["strategy"] == "보수적"
        assert "server_time" in result

    def test_uses_defaults_when_env_missing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            result = api_status()

        assert result["dry_run"] == "true"
        assert result["emergency_stop"] == "false"
        assert result["max_trade_amount"] == "100000"
        assert result["max_daily_trades"] == "3"


# ---------------------------------------------------------------------------
# Tests: api_decisions
# ---------------------------------------------------------------------------

class TestApiDecisions:
    def test_calls_supabase_with_correct_params(self):
        mock_data = [{"id": 1}]
        with patch("scripts.web_server.supabase_get", return_value=mock_data) as mock_sb:
            result = api_decisions()
            assert result == mock_data
            mock_sb.assert_called_once_with(
                "decisions", "select=*&order=created_at.desc&limit=10"
            )


# ---------------------------------------------------------------------------
# Tests: DashboardHandler - API GET routes
# ---------------------------------------------------------------------------

class TestHandlerAPIGet:
    def test_api_portfolio(self):
        code, data = call_get("/api/portfolio")
        # api_portfolio calls subprocess; we only verify routing works
        assert code is not None

    def test_api_status(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DRY_RUN=false\n", encoding="utf-8")
        strat = tmp_path / "strategy.md"
        strat.write_text("## 활성 전략: 보통\n", encoding="utf-8")

        with patch("scripts.web_server.PROJECT_DIR", tmp_path):
            code, data = call_get("/api/status")

        assert code == 200
        assert data["dry_run"] == "false"
        assert data["strategy"] == "보통"

    def test_api_decisions(self):
        mock_decisions = [{"id": 1, "decision": "hold"}]
        with patch("scripts.web_server.supabase_get", return_value=mock_decisions):
            code, data = call_get("/api/decisions")
            assert code == 200
            assert data == mock_decisions

    def test_api_market(self):
        mock_market = {"price": 50000000}
        with patch("scripts.web_server.api_market", return_value=mock_market):
            code, data = call_get("/api/market")
            assert code == 200
            assert data == mock_market

    def test_api_unknown_returns_404(self):
        code, data = call_get("/api/nonexistent")
        assert code == 404
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: DashboardHandler - POST routes
# ---------------------------------------------------------------------------

class TestHandlerPost:
    def test_emergency_stop_toggle_on(self):
        with patch.dict(os.environ, {"EMERGENCY_STOP": "false"}, clear=False):
            with patch("scripts.web_server.update_env_value") as mock_update:
                code, data = call_post("/api/emergency-stop")
                assert code == 200
                assert data["emergency_stop"] == "true"
                mock_update.assert_called_once_with("EMERGENCY_STOP", "true")

    def test_emergency_stop_toggle_off(self):
        with patch.dict(os.environ, {"EMERGENCY_STOP": "true"}, clear=False):
            with patch("scripts.web_server.update_env_value") as mock_update:
                code, data = call_post("/api/emergency-stop")
                assert code == 200
                assert data["emergency_stop"] == "false"
                mock_update.assert_called_once_with("EMERGENCY_STOP", "false")

    def test_dry_run_toggle(self):
        with patch.dict(os.environ, {"DRY_RUN": "true"}, clear=False):
            with patch("scripts.web_server.update_env_value") as mock_update:
                code, data = call_post("/api/dry-run")
                assert code == 200
                assert data["dry_run"] == "false"

    def test_strategy_change_valid(self):
        with patch("scripts.web_server._update_strategy") as mock_update:
            code, data = call_post("/api/strategy", {"strategy": "aggressive"})
            assert code == 200
            assert "공격적" in data["strategy"]
            mock_update.assert_called_once()

    def test_strategy_change_conservative(self):
        with patch("scripts.web_server._update_strategy") as mock_update:
            code, data = call_post("/api/strategy", {"strategy": "conservative"})
            assert code == 200
            assert "보수적" in data["strategy"]

    def test_strategy_change_moderate(self):
        with patch("scripts.web_server._update_strategy") as mock_update:
            code, data = call_post("/api/strategy", {"strategy": "moderate"})
            assert code == 200
            assert "보통" in data["strategy"]

    def test_strategy_change_invalid(self):
        code, data = call_post("/api/strategy", {"strategy": "yolo"})
        assert code == 400
        assert "error" in data

    def test_strategy_change_empty(self):
        code, data = call_post("/api/strategy", {})
        assert code == 400
        assert "error" in data

    def test_run_starts_thread(self):
        with patch("scripts.web_server.threading.Thread") as mock_thread:
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            code, data = call_post("/api/run")
            assert code == 200
            assert data["status"] == "started"
            mock_thread.assert_called_once()
            mock_instance.start.assert_called_once()

    def test_post_unknown_returns_404(self):
        code, data = call_post("/api/nonexistent")
        assert code == 404
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: DashboardHandler - Static file routes
# ---------------------------------------------------------------------------

class TestHandlerStaticRoutes:
    """Test that non-API GET requests fall through to file serving."""

    def test_root_redirects_to_index(self):
        handler = make_handler("GET", "/")
        # Override super().do_GET to just capture the path change
        with patch.object(http.server.SimpleHTTPRequestHandler, "do_GET"):
            handler.do_GET()
            assert handler.path == "/index.html"

    def test_non_api_path_delegates_to_parent(self):
        handler = make_handler("GET", "/remote.html")
        with patch.object(
            http.server.SimpleHTTPRequestHandler, "do_GET"
        ) as mock_parent:
            handler.do_GET()
            mock_parent.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _json_response
# ---------------------------------------------------------------------------

class TestJsonResponse:
    def test_json_response_format(self):
        handler = make_handler("GET", "/test")

        # Mock send_response, send_header, end_headers
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()

        test_data = {"key": "value", "number": 42}
        handler._json_response(test_data, 200)

        handler.send_response.assert_called_once_with(200)
        # Verify Content-Type header
        content_type_call = [
            c for c in handler.send_header.call_args_list
            if c[0][0] == "Content-Type"
        ]
        assert len(content_type_call) == 1
        assert "application/json" in content_type_call[0][0][1]

        # Verify CORS header
        cors_call = [
            c for c in handler.send_header.call_args_list
            if c[0][0] == "Access-Control-Allow-Origin"
        ]
        assert len(cors_call) == 1
        assert cors_call[0][0][1] == "*"

        # Verify body is valid JSON
        body = handler.wfile.getvalue()
        parsed = json.loads(body.decode("utf-8"))
        assert parsed == test_data

    def test_json_response_error_code(self):
        handler = make_handler("GET", "/test")
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()

        handler._json_response({"error": "bad"}, 400)
        handler.send_response.assert_called_once_with(400)

    def test_json_response_unicode(self):
        handler = make_handler("GET", "/test")
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()

        handler._json_response({"msg": "한국어 테스트"})
        body = handler.wfile.getvalue()
        parsed = json.loads(body.decode("utf-8"))
        assert parsed["msg"] == "한국어 테스트"


# ---------------------------------------------------------------------------
# Tests: api_portfolio / api_market (subprocess calls)
# ---------------------------------------------------------------------------

class TestSubprocessAPIs:
    def test_api_portfolio_success(self):
        from scripts.web_server import api_portfolio

        mock_result = MagicMock()
        mock_result.stdout = '{"krw_balance": 1000000}'
        with patch("scripts.web_server.subprocess.run", return_value=mock_result):
            result = api_portfolio()
            assert result == {"krw_balance": 1000000}

    def test_api_portfolio_error(self):
        from scripts.web_server import api_portfolio

        with patch("scripts.web_server.subprocess.run", side_effect=Exception("no python")):
            result = api_portfolio()
            assert "error" in result

    def test_api_market_success(self):
        from scripts.web_server import api_market

        mock_result = MagicMock()
        mock_result.stdout = '{"price": 85000000}'
        with patch("scripts.web_server.subprocess.run", return_value=mock_result):
            result = api_market()
            assert result == {"price": 85000000}

    def test_api_market_error(self):
        from scripts.web_server import api_market

        with patch("scripts.web_server.subprocess.run", side_effect=Exception("timeout")):
            result = api_market()
            assert "error" in result

    def test_api_portfolio_invalid_json(self):
        from scripts.web_server import api_portfolio

        mock_result = MagicMock()
        mock_result.stdout = "not json"
        with patch("scripts.web_server.subprocess.run", return_value=mock_result):
            result = api_portfolio()
            assert "error" in result


# ---------------------------------------------------------------------------
# Tests: POST body parsing edge cases
# ---------------------------------------------------------------------------

class TestPostBodyParsing:
    def test_empty_body_does_not_crash(self):
        """POST with empty body should not raise an exception."""
        handler = make_handler("POST", "/api/strategy", "")
        handler.headers["Content-Length"] = "0"
        captured = {}

        def capture_json(data, code=200):
            captured["code"] = code
            captured["data"] = data

        handler._json_response = capture_json
        handler.do_POST()
        # Empty body means strategy="" which is invalid
        assert captured.get("code") == 400

    def test_malformed_json_body(self):
        """POST with malformed JSON should not crash."""
        handler = make_handler("POST", "/api/strategy", "{bad json")
        handler.headers["Content-Length"] = str(len(b"{bad json"))
        handler.rfile = io.BytesIO(b"{bad json")
        captured = {}

        def capture_json(data, code=200):
            captured["code"] = code
            captured["data"] = data

        handler._json_response = capture_json
        handler.do_POST()
        # data will be {} due to JSONDecodeError catch, strategy="" is invalid
        assert captured.get("code") == 400


# ---------------------------------------------------------------------------
# Tests: _run_analysis (background execution)
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_run_analysis_calls_scripts(self):
        from scripts.web_server import _run_analysis

        with patch("scripts.web_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            _run_analysis()
            # Should call 3 data scripts + 1 telegram notification = 4 calls
            assert mock_run.call_count == 4

    def test_run_analysis_handles_errors(self):
        from scripts.web_server import _run_analysis

        with patch("scripts.web_server.subprocess.run", side_effect=Exception("fail")):
            # Should not raise
            _run_analysis()
