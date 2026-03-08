#!/usr/bin/env python3
"""
Unit tests for notify_telegram.py and get_portfolio.py
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from notify_telegram import escape_md, send_message, send_photo
import get_portfolio


# ============================================================
# notify_telegram.py tests
# ============================================================


class TestEscapeMd:
    """MarkdownV2 escaping tests."""

    def test_escapes_dot(self):
        assert escape_md("price 100.5") == r"price 100\.5"

    def test_escapes_dash(self):
        assert escape_md("BTC-KRW") == r"BTC\-KRW"

    def test_escapes_exclamation(self):
        assert escape_md("alert!") == r"alert\!"

    def test_escapes_parentheses(self):
        assert escape_md("(hello)") == r"\(hello\)"

    def test_escapes_brackets(self):
        assert escape_md("[link]") == r"\[link\]"

    def test_escapes_underscore_and_asterisk(self):
        assert escape_md("_bold_ *italic*") == r"\_bold\_ \*italic\*"

    def test_escapes_hash_plus_equals(self):
        assert escape_md("# heading += 1") == r"\# heading \+\= 1"

    def test_escapes_tilde_backtick_pipe(self):
        assert escape_md("~code~ `x` |y|") == r"\~code\~ \`x\` \|y\|"

    def test_escapes_curly_braces(self):
        assert escape_md("{key}") == r"\{key\}"

    def test_escapes_backslash(self):
        assert escape_md("a\\b") == r"a\\b"

    def test_escapes_greater_than(self):
        assert escape_md("> quote") == r"\> quote"

    def test_no_escape_for_plain_text(self):
        assert escape_md("hello world") == "hello world"

    def test_multiple_special_chars(self):
        result = escape_md("BTC: 100,000.5 KRW (+2.3%)")
        assert r"\." in result
        assert r"\+" in result
        assert r"\(" in result
        assert r"\)" in result


class TestSendMessage:
    """send_message tests."""

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        result = send_message("trade", "BTC Buy", "Bought 0.001 BTC")

        assert result["success"] is True
        assert result["type"] == "trade"
        assert result["title"] == "BTC Buy"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_correct_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("trade", "title", "body")

        call_args = mock_post.call_args
        url = call_args[0][0]
        assert url == "https://api.telegram.org/bottest-token/sendMessage"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_correct_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("trade", "title", "body")

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["chat_id"] == "12345"
        assert payload["parse_mode"] == "MarkdownV2"
        assert "title" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_trade_emoji(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("trade", "BTC Buy", "body")

        payload = mock_post.call_args[1]["json"]
        assert "\U0001f4b0" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_error_emoji(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("error", "API Failed", "timeout")

        payload = mock_post.call_args[1]["json"]
        assert "\U0001f6a8" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_analysis_emoji(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("analysis", "Report", "body")

        payload = mock_post.call_args[1]["json"]
        assert "\U0001f4ca" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_unknown_type_uses_default_emoji(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("unknown_type", "title", "body")

        payload = mock_post.call_args[1]["json"]
        assert "\U0001f4ac" in payload["text"]

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_vars(self):
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            send_message("trade", "title", "body")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": ""})
    def test_empty_user_id(self):
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            send_message("trade", "title", "body")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_api_400_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "Bad Request: chat not found"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="텔레그램 전송 실패"):
            send_message("trade", "title", "body")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_api_500_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="텔레그램 전송 실패"):
            send_message("trade", "title", "body")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    def test_message_contains_timestamp(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("trade", "title", "body")

        payload = mock_post.call_args[1]["json"]
        assert "KST" in payload["text"]


class TestSendPhoto:
    """send_photo tests."""

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake_png_data"))
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        result = send_photo("/tmp/chart.png", "BTC chart")

        assert result["success"] is True
        assert result["type"] == "photo"
        assert result["path"] == "/tmp/chart.png"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake_png_data"))
    def test_correct_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_photo("/tmp/chart.png", "caption")

        url = mock_post.call_args[0][0]
        assert url == "https://api.telegram.org/bottest-token/sendPhoto"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake_png_data"))
    def test_multipart_data(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_photo("/tmp/chart.png", "my caption")

        call_args = mock_post.call_args
        assert call_args[1]["data"]["chat_id"] == "12345"
        assert call_args[1]["data"]["caption"] == "my caption"
        assert "photo" in call_args[1]["files"]

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_vars(self):
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            send_photo("/tmp/chart.png", "caption")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_USER_ID": "12345"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake_png_data"))
    def test_api_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="텔레그램 이미지 전송 실패"):
            send_photo("/tmp/chart.png", "caption")


# ============================================================
# get_portfolio.py tests
# ============================================================


def _make_accounts_response(accounts_data):
    """Helper to create a mock response for /accounts."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = accounts_data
    return mock_resp


def _make_market_all_response(markets):
    """Helper to create a mock response for /market/all."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = [{"market": m} for m in markets]
    return mock_resp


def _make_ticker_response(tickers):
    """Helper to create a mock response for /ticker."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = tickers
    return mock_resp


class TestPortfolioEmpty:
    """Test empty portfolio (KRW only)."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_krw_only(self, mock_get, mock_jwt, capsys):
        accounts = [{"currency": "KRW", "balance": "500000", "avg_buy_price": "0"}]
        mock_get.return_value = _make_accounts_response(accounts)

        get_portfolio.main()

        output = json.loads(capsys.readouterr().out)
        assert output["krw_balance"] == 500000.0
        assert output["holdings"] == []
        assert output["total_eval"] == 500000.0

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_zero_krw(self, mock_get, mock_jwt, capsys):
        accounts = [{"currency": "KRW", "balance": "0", "avg_buy_price": "0"}]
        mock_get.return_value = _make_accounts_response(accounts)

        get_portfolio.main()

        output = json.loads(capsys.readouterr().out)
        assert output["krw_balance"] == 0.0
        assert output["total_eval"] == 0.0
        assert output["total_profit_loss_pct"] == 0.0


class TestPortfolioWithHoldings:
    """Test portfolio with crypto holdings."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_btc_and_eth(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
            {"currency": "ETH", "balance": "0.5", "avg_buy_price": "4000000"},
        ]
        markets_all = [{"market": "KRW-BTC"}, {"market": "KRW-ETH"}]
        tickers = [
            {"market": "KRW-BTC", "trade_price": 90000000},
            {"market": "KRW-ETH", "trade_price": 5000000},
        ]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _make_accounts_response(accounts)
            elif "/market/all" in url:
                resp = MagicMock()
                resp.ok = True
                resp.json.return_value = markets_all
                return resp
            elif "/ticker" in url:
                resp = MagicMock()
                resp.ok = True
                resp.json.return_value = tickers
                return resp
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()

        output = json.loads(capsys.readouterr().out)
        assert output["krw_balance"] == 100000.0
        assert len(output["holdings"]) == 2

        btc = next(h for h in output["holdings"] if h["currency"] == "BTC")
        eth = next(h for h in output["holdings"] if h["currency"] == "ETH")

        # BTC: bought at 80M, now 90M -> +12.5%
        assert btc["current_price"] == 90000000
        assert btc["eval_amount"] == 0.01 * 90000000  # 900000
        assert btc["profit_loss_pct"] == 12.5

        # ETH: bought at 4M, now 5M -> +25%
        assert eth["current_price"] == 5000000
        assert eth["eval_amount"] == 0.5 * 5000000  # 2500000
        assert eth["profit_loss_pct"] == 25.0

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_negative_profit(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "50000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "100000000"},
        ]
        tickers = [{"market": "KRW-BTC", "trade_price": 80000000}]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _make_accounts_response(accounts)
            elif "/market/all" in url:
                return _make_market_all_response(["KRW-BTC"])
            elif "/ticker" in url:
                return _make_ticker_response(tickers)
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()

        output = json.loads(capsys.readouterr().out)
        btc = output["holdings"][0]
        # Bought at 100M, now 80M -> -20%
        assert btc["profit_loss_pct"] == -20.0


class TestPortfolioProfitLossCalc:
    """Verify profit/loss calculation: (current - avg) / avg * 100."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_exact_calculation(self, mock_get, mock_jwt, capsys):
        avg_price = 50000000
        cur_price = 55000000
        expected_pct = round((cur_price - avg_price) / avg_price * 100, 2)  # 10.0

        accounts = [
            {"currency": "KRW", "balance": "0", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.1", "avg_buy_price": str(avg_price)},
        ]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _make_accounts_response(accounts)
            elif "/market/all" in url:
                return _make_market_all_response(["KRW-BTC"])
            elif "/ticker" in url:
                return _make_ticker_response([{"market": "KRW-BTC", "trade_price": cur_price}])
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()

        output = json.loads(capsys.readouterr().out)
        assert output["holdings"][0]["profit_loss_pct"] == expected_pct


class TestPortfolioTotalEval:
    """Verify total_eval = KRW + sum(holdings eval)."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_total_evaluation(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "200000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
            {"currency": "ETH", "balance": "1.0", "avg_buy_price": "3000000"},
        ]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _make_accounts_response(accounts)
            elif "/market/all" in url:
                return _make_market_all_response(["KRW-BTC", "KRW-ETH"])
            elif "/ticker" in url:
                return _make_ticker_response([
                    {"market": "KRW-BTC", "trade_price": 90000000},
                    {"market": "KRW-ETH", "trade_price": 4000000},
                ])
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()

        output = json.loads(capsys.readouterr().out)
        btc_eval = 0.01 * 90000000   # 900000
        eth_eval = 1.0 * 4000000     # 4000000
        expected_total = 200000 + btc_eval + eth_eval  # 5100000
        assert output["total_eval"] == expected_total


class TestPortfolioAuth:
    """Verify JWT auth header construction."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "my_access", "UPBIT_SECRET_KEY": "my_secret"})
    @patch("get_portfolio.requests.get")
    @patch("get_portfolio.jwt.encode", return_value="encoded-jwt-token")
    def test_jwt_header(self, mock_jwt, mock_get, capsys):
        accounts = [{"currency": "KRW", "balance": "100", "avg_buy_price": "0"}]
        mock_get.return_value = _make_accounts_response(accounts)

        get_portfolio.main()

        # Verify jwt.encode was called with correct key
        mock_jwt.assert_called_once()
        call_args = mock_jwt.call_args
        payload = call_args[0][0]
        assert payload["access_key"] == "my_access"
        assert call_args[0][1] == "my_secret"
        assert call_args[1]["algorithm"] == "HS256" or call_args[0][2] == "HS256"

        # Verify Authorization header
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer encoded-jwt-token"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_keys(self):
        with pytest.raises(KeyError):
            get_portfolio.make_auth_header()
