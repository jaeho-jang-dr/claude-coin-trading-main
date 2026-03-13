"""
get_portfolio.py comprehensive unit tests.

Covers:
  - KRW-only portfolio
  - Portfolio with crypto holdings
  - Profit/loss calculation: (current - avg) / avg * 100
  - Total evaluation: KRW + sum(holdings eval)
  - Negative profit
  - Zero balance edge cases
  - JWT auth header construction
  - Missing API keys
  - Invalid market filtering
  - Ticker API failure handling

All network calls are mocked - no real API calls are made.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import get_portfolio


# ── Helpers ──────────────────────────────────────────────

def _accounts_resp(accounts):
    resp = MagicMock()
    resp.ok = True
    resp.raise_for_status = MagicMock()
    resp.json.return_value = accounts
    return resp


def _market_all_resp(markets):
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = [{"market": m} for m in markets]
    return resp


def _ticker_resp(tickers):
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = tickers
    return resp


def _route_mock(accounts, markets=None, tickers=None):
    """Return a side_effect function that routes by URL."""
    def side_effect(url, **kwargs):
        if "/accounts" in url:
            return _accounts_resp(accounts)
        elif "/market/all" in url:
            return _market_all_resp(markets or [])
        elif "/ticker" in url:
            return _ticker_resp(tickers or [])
        return MagicMock(ok=False)
    return side_effect


# ══════════════════════════════════════════════════════════════
# Empty / KRW-only portfolio
# ══════════════════════════════════════════════════════════════

class TestKRWOnly:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_krw_only(self, mock_get, mock_jwt, capsys):
        accounts = [{"currency": "KRW", "balance": "500000", "avg_buy_price": "0"}]
        mock_get.return_value = _accounts_resp(accounts)

        get_portfolio.main()

        out = json.loads(capsys.readouterr().out)
        assert out["krw_balance"] == 500000.0
        assert out["holdings"] == []
        assert out["total_eval"] == 500000.0
        assert out["total_profit_loss_pct"] == 0.0

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_zero_krw(self, mock_get, mock_jwt, capsys):
        accounts = [{"currency": "KRW", "balance": "0", "avg_buy_price": "0"}]
        mock_get.return_value = _accounts_resp(accounts)

        get_portfolio.main()

        out = json.loads(capsys.readouterr().out)
        assert out["krw_balance"] == 0.0
        assert out["total_eval"] == 0.0


# ══════════════════════════════════════════════════════════════
# Portfolio with holdings
# ══════════════════════════════════════════════════════════════

class TestWithHoldings:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_btc_and_eth(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
            {"currency": "ETH", "balance": "0.5", "avg_buy_price": "4000000"},
        ]
        tickers = [
            {"market": "KRW-BTC", "trade_price": 90000000},
            {"market": "KRW-ETH", "trade_price": 5000000},
        ]
        mock_get.side_effect = _route_mock(accounts, ["KRW-BTC", "KRW-ETH"], tickers)

        get_portfolio.main()

        out = json.loads(capsys.readouterr().out)
        assert len(out["holdings"]) == 2

        btc = next(h for h in out["holdings"] if h["currency"] == "BTC")
        assert btc["current_price"] == 90000000
        assert btc["eval_amount"] == 0.01 * 90000000
        assert btc["profit_loss_pct"] == 12.5

        eth = next(h for h in out["holdings"] if h["currency"] == "ETH")
        assert eth["current_price"] == 5000000
        assert eth["eval_amount"] == 0.5 * 5000000
        assert eth["profit_loss_pct"] == 25.0


# ══════════════════════════════════════════════════════════════
# Profit/loss calculation
# ══════════════════════════════════════════════════════════════

class TestProfitLoss:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_positive_profit(self, mock_get, mock_jwt, capsys):
        avg, cur = 50000000, 55000000
        expected = round((cur - avg) / avg * 100, 2)
        accounts = [
            {"currency": "KRW", "balance": "0", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.1", "avg_buy_price": str(avg)},
        ]
        mock_get.side_effect = _route_mock(
            accounts, ["KRW-BTC"],
            [{"market": "KRW-BTC", "trade_price": cur}],
        )

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        assert out["holdings"][0]["profit_loss_pct"] == expected

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_negative_profit(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "0", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "100000000"},
        ]
        mock_get.side_effect = _route_mock(
            accounts, ["KRW-BTC"],
            [{"market": "KRW-BTC", "trade_price": 80000000}],
        )

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        assert out["holdings"][0]["profit_loss_pct"] == -20.0

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_breakeven(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "0", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "100000000"},
        ]
        mock_get.side_effect = _route_mock(
            accounts, ["KRW-BTC"],
            [{"market": "KRW-BTC", "trade_price": 100000000}],
        )

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        assert out["holdings"][0]["profit_loss_pct"] == 0.0


# ══════════════════════════════════════════════════════════════
# Total evaluation
# ══════════════════════════════════════════════════════════════

class TestTotalEval:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_total_equals_krw_plus_holdings(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "200000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
            {"currency": "ETH", "balance": "1.0", "avg_buy_price": "3000000"},
        ]
        tickers = [
            {"market": "KRW-BTC", "trade_price": 90000000},
            {"market": "KRW-ETH", "trade_price": 4000000},
        ]
        mock_get.side_effect = _route_mock(accounts, ["KRW-BTC", "KRW-ETH"], tickers)

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)

        expected = 200000 + 0.01 * 90000000 + 1.0 * 4000000
        assert out["total_eval"] == expected


# ══════════════════════════════════════════════════════════════
# Zero balance holdings skipped
# ══════════════════════════════════════════════════════════════

class TestZeroBalanceSkipped:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_zero_balance_crypto_excluded(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0", "avg_buy_price": "80000000"},
            {"currency": "ETH", "balance": "0.5", "avg_buy_price": "4000000"},
        ]
        mock_get.side_effect = _route_mock(
            accounts, ["KRW-ETH"],
            [{"market": "KRW-ETH", "trade_price": 5000000}],
        )

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        assert len(out["holdings"]) == 1
        assert out["holdings"][0]["currency"] == "ETH"


# ══════════════════════════════════════════════════════════════
# Invalid market filtering
# ══════════════════════════════════════════════════════════════

class TestMarketFiltering:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_unknown_market_filtered_out(self, mock_get, mock_jwt, capsys):
        """Holdings for unknown markets get 0 current_price."""
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "FAKE", "balance": "100", "avg_buy_price": "1000"},
        ]
        # /market/all does NOT include KRW-FAKE
        mock_get.side_effect = _route_mock(accounts, ["KRW-BTC"], [])

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        assert len(out["holdings"]) == 1
        assert out["holdings"][0]["currency"] == "FAKE"
        assert out["holdings"][0]["current_price"] == 0
        assert out["holdings"][0]["eval_amount"] == 0


# ══════════════════════════════════════════════════════════════
# Ticker API failure
# ══════════════════════════════════════════════════════════════

class TestTickerFailure:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_ticker_api_fails_gracefully(self, mock_get, mock_jwt, capsys):
        """If ticker API returns not ok, holdings keep 0 current_price."""
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
        ]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _accounts_resp(accounts)
            elif "/market/all" in url:
                return _market_all_resp(["KRW-BTC"])
            elif "/ticker" in url:
                resp = MagicMock()
                resp.ok = False
                return resp
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        assert out["holdings"][0]["current_price"] == 0
        assert out["holdings"][0]["eval_amount"] == 0


# ══════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════

class TestAuth:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "my_access", "UPBIT_SECRET_KEY": "my_secret"})
    @patch("get_portfolio.requests.get")
    @patch("get_portfolio.jwt.encode", return_value="encoded-jwt")
    def test_jwt_payload(self, mock_jwt, mock_get, capsys):
        accounts = [{"currency": "KRW", "balance": "100", "avg_buy_price": "0"}]
        mock_get.return_value = _accounts_resp(accounts)

        get_portfolio.main()

        payload = mock_jwt.call_args[0][0]
        assert payload["access_key"] == "my_access"
        assert mock_jwt.call_args[0][1] == "my_secret"

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.requests.get")
    @patch("get_portfolio.jwt.encode", return_value="jwt-token")
    def test_auth_header(self, mock_jwt, mock_get, capsys):
        accounts = [{"currency": "KRW", "balance": "100", "avg_buy_price": "0"}]
        mock_get.return_value = _accounts_resp(accounts)

        get_portfolio.main()

        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer jwt-token"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_keys_raises(self):
        with pytest.raises(KeyError):
            get_portfolio.make_auth_header()


# ══════════════════════════════════════════════════════════════
# Output format
# ══════════════════════════════════════════════════════════════

class TestOutputFormat:
    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_all_top_level_keys(self, mock_get, mock_jwt, capsys):
        accounts = [{"currency": "KRW", "balance": "100000", "avg_buy_price": "0"}]
        mock_get.return_value = _accounts_resp(accounts)

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)

        for key in ["timestamp", "krw_balance", "holdings", "total_eval",
                     "total_invested", "total_profit_loss_pct"]:
            assert key in out

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="jwt")
    @patch("get_portfolio.requests.get")
    def test_holding_fields(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "0", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
        ]
        mock_get.side_effect = _route_mock(
            accounts, ["KRW-BTC"],
            [{"market": "KRW-BTC", "trade_price": 90000000}],
        )

        get_portfolio.main()
        out = json.loads(capsys.readouterr().out)
        h = out["holdings"][0]

        for key in ["currency", "balance", "avg_buy_price", "current_price",
                     "eval_amount", "profit_loss_pct"]:
            assert key in h
