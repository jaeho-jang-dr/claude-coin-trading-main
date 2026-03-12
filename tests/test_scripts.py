#!/usr/bin/env python3
"""
Comprehensive unit tests for scripts/ directory (Team 2).

Covers:
  - collect_market_data.py: API responses, indicator calculations, edge cases
  - collect_fear_greed.py: API response parsing, error handling, edge cases
  - collect_news.py: search results parsing, keyword extraction, edge cases
  - execute_trade.py: safety checks, auto emergency stop, order execution, DB recording
  - get_portfolio.py: balance calculation, profit/loss, API failure handling
  - notify_telegram.py: message formatting, MarkdownV2 escaping, edge cases
  - collect_ai_signal.py: signal analysis, mega whale detection, composite scoring

All external API calls and file I/O are mocked.
"""

import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call

import pytest
import requests

# ---------------------------------------------------------------------------
# Imports: Add scripts directory to path
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))


# ===========================================================================
# PART 1: collect_market_data.py
# ===========================================================================

from collect_market_data import (
    api_get,
    sma,
    ema,
    rsi,
    macd,
    bollinger,
    stochastic,
    calc_adx,
    calc_atr,
    collect_eth_btc_ratio,
    main as market_data_main,
)


class TestSMAEdgeCases:
    """Additional SMA edge cases."""

    def test_period_equals_list_length(self):
        prices = [10.0, 20.0, 30.0]
        assert sma(prices, 3) == 20.0

    def test_period_larger_than_list(self):
        """When period > len, returns None (insufficient data)."""
        prices = [5.0, 15.0]
        result = sma(prices, 10)
        assert result is None

    def test_large_values(self):
        prices = [1e15, 2e15, 3e15]
        assert sma(prices, 3) == pytest.approx(2e15)

    def test_negative_values(self):
        prices = [-10.0, -20.0, -30.0]
        assert sma(prices, 3) == -20.0

    def test_mixed_positive_negative(self):
        prices = [-10.0, 10.0]
        assert sma(prices, 2) == 0.0


class TestEMAEdgeCases:
    """Additional EMA edge cases."""

    def test_increasing_prices_ema_lags(self):
        """EMA should lag behind in a strong uptrend."""
        prices = [float(i) for i in range(1, 21)]
        result = ema(prices, 10)
        # EMA lags, so it should be less than the last price
        assert result < prices[-1]
        assert result > prices[0]

    def test_decreasing_prices_ema_lags(self):
        prices = [float(20 - i) for i in range(20)]
        result = ema(prices, 10)
        assert result > prices[-1]

    def test_period_one(self):
        """Period=1 means k=1, EMA = last price."""
        prices = [100.0, 200.0, 300.0]
        result = ema(prices, 1)
        assert result == pytest.approx(300.0)

    def test_very_large_period(self):
        """Large period -> k very small -> EMA close to first value."""
        prices = [100.0, 200.0]
        result = ema(prices, 1000)
        # k = 2/1001 ~= 0.002, result ~= 200*0.002 + 100*0.998 ~= 100.2
        assert result == pytest.approx(100.2, abs=0.1)


class TestRSIEdgeCases:
    """Additional RSI edge cases."""

    def test_exactly_two_values_no_gain(self):
        """Two identical values with period=1."""
        prices = [100.0, 100.0]
        result = rsi(prices, 1)
        # gains=0, losses=0 -> al=0 -> RSI=100
        assert result == 100.0

    def test_large_gain_followed_by_small_loss(self):
        prices = [100.0] + [110.0] * 14 + [109.0]
        result = rsi(prices, 14)
        assert result > 80.0

    def test_single_large_drop(self):
        prices = [100.0] * 15 + [50.0]
        result = rsi(prices, 14)
        assert result < 20.0

    def test_period_equals_data_length_minus_one(self):
        """Exactly period+1 elements."""
        prices = [100.0, 101.0]
        result = rsi(prices, 1)
        assert result == 100.0  # pure gain

    def test_rsi_with_zero_prices(self):
        """Edge case: prices at zero boundary."""
        prices = [0.001] * 16
        result = rsi(prices, 14)
        assert result == 100.0  # no losses


class TestMACDEdgeCases:
    """Additional MACD edge cases."""

    def test_exactly_26_constant(self):
        prices = [100.0] * 26
        result = macd(prices)
        assert result["macd"] == pytest.approx(0.0)
        assert result["histogram"] == pytest.approx(0.0)

    def test_strong_uptrend_positive_macd(self):
        prices = [100.0 + i * 5 for i in range(50)]
        result = macd(prices)
        assert result["macd"] > 0

    def test_strong_downtrend_negative_macd(self):
        prices = [500.0 - i * 5 for i in range(50)]
        result = macd(prices)
        assert result["macd"] < 0

    def test_single_price(self):
        result = macd([100.0])
        assert result == {"macd": 0, "signal": 0, "histogram": 0}

    def test_empty_prices(self):
        result = macd([])
        assert result == {"macd": 0, "signal": 0, "histogram": 0}


class TestBollingerEdgeCases:
    """Additional Bollinger band edge cases."""

    def test_two_distinct_prices(self):
        prices = [100.0, 200.0]
        result = bollinger(prices, 2)
        mid = 150.0
        var = ((100 - 150) ** 2 + (200 - 150) ** 2) / 2
        sd = var ** 0.5
        assert result["middle"] == pytest.approx(mid)
        assert result["upper"] == pytest.approx(mid + 2 * sd, abs=0.01)
        assert result["lower"] == pytest.approx(mid - 2 * sd, abs=0.01)

    def test_period_one(self):
        """Period=1: middle=last, sd=0."""
        prices = [50.0, 100.0, 200.0]
        result = bollinger(prices, 1)
        assert result["middle"] == pytest.approx(200.0)
        assert result["upper"] == pytest.approx(200.0)
        assert result["lower"] == pytest.approx(200.0)

    def test_wide_spread_prices(self):
        prices = [0.0, 1000.0] * 10
        result = bollinger(prices, 20)
        assert result["upper"] > result["middle"]
        assert result["lower"] < result["middle"]


class TestStochasticEdgeCases:
    """Additional stochastic edge cases."""

    def test_high_equals_low(self):
        """When high==low, %K should be 50.0 (division by zero guard)."""
        n = 20
        highs = [100.0] * n
        lows = [100.0] * n
        closes = [100.0] * n
        result = stochastic(highs, lows, closes, 14)
        assert result["k"] == 50.0

    def test_exactly_period_plus_two(self):
        """Minimum data that allows calculation."""
        n = 16
        highs = [110.0] * n
        lows = [90.0] * n
        closes = [105.0] * n
        result = stochastic(highs, lows, closes, 14)
        assert 0.0 <= result["k"] <= 100.0
        assert 0.0 <= result["d"] <= 100.0


class TestCalcADXEdgeCases:
    """Additional ADX edge cases."""

    def test_exactly_period_plus_one(self):
        """Minimum data for ADX."""
        n = 15
        highs = [100.0 + i for i in range(n)]
        lows = [98.0 + i for i in range(n)]
        closes = [99.0 + i for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        assert "adx" in result
        assert "regime" in result

    def test_flat_market_low_adx(self):
        """Completely flat data -> ADX should be 0 or very low."""
        n = 50
        highs = [100.0] * n
        lows = [100.0] * n
        closes = [100.0] * n
        result = calc_adx(highs, lows, closes, 14)
        assert result["adx"] == 0 or result["regime"] == "ranging"


class TestCalcATREdgeCases:
    """Additional ATR edge cases."""

    def test_increasing_volatility(self):
        """ATR should increase when ranges widen."""
        n = 30
        highs = [100.0 + i * 2 for i in range(n)]
        lows = [100.0 - i * 2 for i in range(n)]
        closes = [100.0 for _ in range(n)]
        result = calc_atr(highs, lows, closes, 14)
        assert result > 0

    def test_single_bar(self):
        assert calc_atr([100.0], [90.0], [95.0], 14) == 0.0


class TestApiGetEdgeCases:
    """Additional api_get edge cases."""

    @patch("collect_market_data.time.sleep")
    def test_429_backoff_increases(self, mock_sleep):
        """Verify exponential backoff wait times: 1, 2, 4s."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.raise_for_status.side_effect = Exception("429")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_429

        with patch("collect_market_data._get_session", return_value=mock_session):
            with pytest.raises(Exception):
                api_get("/test", max_retries=3)

        # sleep is called for each 429 retry: 2^0=1, 2^1=2, 2^2=4
        assert mock_sleep.call_args_list == [call(1), call(2), call(4)]

    def test_max_retries_one(self):
        """Single retry attempt."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("collect_market_data._get_session", return_value=mock_session):
            result = api_get("/test", max_retries=1)
        assert result == {"ok": True}
        assert mock_session.get.call_count == 1

    def test_params_encoding(self):
        """Verify params are encoded into URL."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("collect_market_data._get_session", return_value=mock_session):
            api_get("/ticker", {"markets": "KRW-BTC", "count": "10"})
        called_url = mock_session.get.call_args[0][0]
        assert "markets=KRW-BTC" in called_url
        assert "count=10" in called_url

    def test_no_params(self):
        """URL without params should not have query string."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("collect_market_data._get_session", return_value=mock_session):
            api_get("/test")
        called_url = mock_session.get.call_args[0][0]
        assert "?" not in called_url


class TestMarketDataMainErrors:
    """Test main() error paths."""

    @patch("collect_market_data.api_get")
    def test_api_failure_on_ticker(self, mock_api):
        """API failure on first call should raise."""
        mock_api.side_effect = requests.ConnectionError("Network error")
        with pytest.raises(requests.ConnectionError):
            market_data_main("KRW-BTC")


# ===========================================================================
# PART 2: collect_fear_greed.py
# ===========================================================================

import collect_fear_greed as fgi_mod


class TestFearGreedParsing:
    """FGI response parsing edge cases."""

    @patch("collect_fear_greed.requests.get")
    def test_http_500_error(self, mock_get):
        """HTTP 500 should propagate."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            fgi_mod.main()

    @patch("collect_fear_greed.requests.get")
    def test_timeout(self, mock_get):
        """Request timeout."""
        mock_get.side_effect = requests.Timeout("Connection timed out")
        with pytest.raises(requests.Timeout):
            fgi_mod.main()

    @patch("collect_fear_greed.requests.get")
    def test_single_data_point(self, mock_get):
        """API returns only 1 data point instead of 7."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "value": "45",
                "value_classification": "Fear",
                "timestamp": str(int(datetime(2026, 3, 8, tzinfo=timezone.utc).timestamp())),
            }]
        }
        mock_get.return_value = mock_resp

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            fgi_mod.main()

        result = json.loads(buf.getvalue())
        assert result["current"]["value"] == 45
        assert len(result["history_7d"]) == 1

    @patch("collect_fear_greed.requests.get")
    def test_string_value_converted_to_int(self, mock_get):
        """Values from API are strings, should be converted to int."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "value": "73",
                "value_classification": "Greed",
                "timestamp": str(int(datetime(2026, 3, 8, tzinfo=timezone.utc).timestamp())),
            }]
        }
        mock_get.return_value = mock_resp

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            fgi_mod.main()

        result = json.loads(buf.getvalue())
        assert isinstance(result["current"]["value"], int)
        assert result["current"]["value"] == 73

    @patch("collect_fear_greed.requests.get")
    def test_date_formatting(self, mock_get):
        """Verify date is formatted as YYYY-MM-DD."""
        ts = int(datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "value": "50",
                "value_classification": "Neutral",
                "timestamp": str(ts),
            }]
        }
        mock_get.return_value = mock_resp

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            fgi_mod.main()

        result = json.loads(buf.getvalue())
        assert result["history_7d"][0]["date"] == "2026-03-08"

    @patch("collect_fear_greed.requests.get")
    def test_output_has_timestamp(self, mock_get):
        """Output JSON should have a top-level timestamp."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "value": "50",
                "value_classification": "Neutral",
                "timestamp": "1709856000",
            }]
        }
        mock_get.return_value = mock_resp

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            fgi_mod.main()

        result = json.loads(buf.getvalue())
        assert "timestamp" in result

    @patch("collect_fear_greed.requests.get")
    def test_request_params(self, mock_get):
        """Verify correct request params."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "value": "50",
                "value_classification": "Neutral",
                "timestamp": "1709856000",
            }]
        }
        mock_get.return_value = mock_resp

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            fgi_mod.main()

        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["limit"] == "7"
        assert call_kwargs[1]["params"]["format"] == "json"
        assert call_kwargs[1]["timeout"] == 10


# ===========================================================================
# PART 3: collect_news.py (additional edge cases)
# ===========================================================================

with patch("dotenv.load_dotenv"):
    from scripts.collect_news import (
        fetch_news,
        _build_queries,
        _load_usage,
        _save_usage,
        _budget_queries,
        MONTHLY_LIMIT,
    )


class TestCollectNewsEdgeCases:
    """Additional edge cases for collect_news.py."""

    @patch("scripts.collect_news.requests.post")
    def test_content_none_handled(self, mock_post):
        """Content field being None should not crash."""
        resp_data = {"results": [{"title": "T", "url": "http://x", "content": None}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        articles = fetch_news("key", "query")
        assert articles[0]["content"] == ""

    @patch("scripts.collect_news.requests.post")
    def test_content_exact_500_chars(self, mock_post):
        """Content exactly 500 chars should not be truncated."""
        content = "A" * 500
        resp_data = {"results": [{"title": "T", "url": "http://x", "content": content}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        articles = fetch_news("key", "query")
        assert len(articles[0]["content"]) == 500

    @patch("scripts.collect_news.requests.post")
    def test_score_missing_defaults_zero(self, mock_post):
        """Missing score field defaults to 0."""
        resp_data = {"results": [{"title": "T", "url": "http://x"}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        articles = fetch_news("key", "query")
        assert articles[0]["score"] == 0

    @patch("scripts.collect_news._build_queries")
    def test_budget_exactly_at_limit(self, mock_build):
        """Count exactly at limit returns empty."""
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT}
        result = _budget_queries(usage)
        assert result == []

    @patch("scripts.collect_news._build_queries")
    def test_budget_one_below_limit(self, mock_build):
        """One remaining call."""
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
        ]
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT - 1}
        result = _budget_queries(usage)
        assert len(result) == 1

    @patch("scripts.collect_news.datetime")
    def test_build_queries_friday(self, mock_dt):
        """Friday (weekday=4) should not include weekend_only."""
        mock_dt.now.return_value = datetime(2026, 3, 6, 12, 0)  # Friday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()
        cats = [q["category"] for q in queries]
        assert "crypto_onchain" not in cats

    @patch("scripts.collect_news.datetime")
    def test_build_queries_sunday(self, mock_dt):
        """Sunday (weekday=6) should include weekend_only."""
        mock_dt.now.return_value = datetime(2026, 3, 8, 12, 0)  # Sunday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()
        cats = [q["category"] for q in queries]
        assert "crypto_onchain" in cats


# ===========================================================================
# PART 4: execute_trade.py (additional edge cases)
# ===========================================================================

with patch("dotenv.load_dotenv"):
    from scripts.execute_trade import (
        execute,
        acquire_lock,
        release_lock,
        _record_trade_to_db,
        check_open_orders_and_cancel,
        make_auth_header,
        LOCK_FILE,
        KST,
        LOCK_TIMEOUT_SECONDS,
    )


def _set_trade_env(monkeypatch, **kwargs):
    """Set environment variables for trade tests."""
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


@pytest.fixture
def patch_lock_file(tmp_path, monkeypatch):
    """Redirect LOCK_FILE to temp directory."""
    lock = tmp_path / "data" / "trading.lock"
    import scripts.execute_trade as mod
    monkeypatch.setattr(mod, "LOCK_FILE", lock)
    return lock


class TestAutoEmergencyStop:
    """Auto emergency stop from data/auto_emergency.json."""

    def test_auto_emergency_blocks_buy(self, tmp_path, monkeypatch, patch_lock_file):
        """Auto emergency stop blocks buy orders."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "false")
        import scripts.execute_trade as mod
        monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)

        em_file = tmp_path / "data" / "auto_emergency.json"
        em_file.parent.mkdir(parents=True, exist_ok=True)
        em_file.write_text(json.dumps({"active": True, "reason": "4h -10% crash"}))

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False
        assert "자동긴급정지" in result["error"]

    def test_auto_emergency_allows_sell(self, tmp_path, monkeypatch, patch_lock_file):
        """Auto emergency stop allows sell (liquidation) orders."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "true")
        import scripts.execute_trade as mod
        monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)

        em_file = tmp_path / "data" / "auto_emergency.json"
        em_file.parent.mkdir(parents=True, exist_ok=True)
        em_file.write_text(json.dumps({"active": True, "reason": "crash"}))

        result = execute("ask", "KRW-BTC", "0.001")
        # Should pass emergency check, then hit DRY_RUN
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_auto_emergency_inactive(self, tmp_path, monkeypatch, patch_lock_file):
        """Inactive auto emergency should not block."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "true")
        import scripts.execute_trade as mod
        monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)

        em_file = tmp_path / "data" / "auto_emergency.json"
        em_file.parent.mkdir(parents=True, exist_ok=True)
        em_file.write_text(json.dumps({"active": False, "reason": "resolved"}))

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True

    def test_auto_emergency_bad_json(self, tmp_path, monkeypatch, patch_lock_file):
        """Corrupt auto_emergency.json should not block."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "true")
        import scripts.execute_trade as mod
        monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)

        em_file = tmp_path / "data" / "auto_emergency.json"
        em_file.parent.mkdir(parents=True, exist_ok=True)
        em_file.write_text("{corrupt json!!!")

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True

    def test_auto_emergency_no_file(self, tmp_path, monkeypatch, patch_lock_file):
        """No auto_emergency.json should not block."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "true")
        import scripts.execute_trade as mod
        monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)

        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True


class TestExecuteTradeAmountEdgeCases:
    """Edge cases for trade amount validation."""

    def test_sell_not_limited_by_max_amount(self, monkeypatch, patch_lock_file):
        """Ask (sell) side should not be limited by MAX_TRADE_AMOUNT."""
        _set_trade_env(monkeypatch, MAX_TRADE_AMOUNT="100000", DRY_RUN="true")
        result = execute("ask", "KRW-BTC", "999999")
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_bid_exact_max_amount(self, monkeypatch, patch_lock_file):
        """Bid at exactly MAX_TRADE_AMOUNT should succeed."""
        _set_trade_env(monkeypatch, MAX_TRADE_AMOUNT="100000", DRY_RUN="true")
        result = execute("bid", "KRW-BTC", "100000")
        assert result["success"] is True

    def test_bid_one_over_max(self, monkeypatch, patch_lock_file):
        """Bid at MAX_TRADE_AMOUNT + 1 should fail."""
        _set_trade_env(monkeypatch, MAX_TRADE_AMOUNT="100000")
        result = execute("bid", "KRW-BTC", "100001")
        assert result["success"] is False
        assert "상한 초과" in result["error"]

    def test_bid_float_amount(self, monkeypatch, patch_lock_file):
        """Float amount should be handled properly."""
        _set_trade_env(monkeypatch, MAX_TRADE_AMOUNT="100000", DRY_RUN="true")
        result = execute("bid", "KRW-BTC", "50000.5")
        assert result["success"] is True

    def test_emergency_stop_case_insensitive(self, monkeypatch, patch_lock_file):
        """EMERGENCY_STOP should be case-insensitive."""
        monkeypatch.setenv("EMERGENCY_STOP", "True")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False

        monkeypatch.setenv("EMERGENCY_STOP", "TRUE")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is False

    def test_dry_run_case_insensitive(self, monkeypatch, patch_lock_file):
        """DRY_RUN should be case-insensitive."""
        monkeypatch.setenv("EMERGENCY_STOP", "false")
        monkeypatch.setenv("DRY_RUN", "True")
        result = execute("bid", "KRW-BTC", "50000")
        assert result["success"] is True
        assert result["dry_run"] is True


class TestMakeAuthHeader:
    """Test auth header construction."""

    def test_auth_header_structure(self, monkeypatch):
        monkeypatch.setenv("UPBIT_ACCESS_KEY", "test_ak")
        monkeypatch.setenv("UPBIT_SECRET_KEY", "test_sk")
        header = make_auth_header("market=KRW-BTC")
        assert "Authorization" in header
        assert header["Authorization"].startswith("Bearer ")
        assert header["Content-Type"] == "application/json"

    def test_auth_header_missing_keys(self, monkeypatch):
        monkeypatch.delenv("UPBIT_ACCESS_KEY", raising=False)
        monkeypatch.delenv("UPBIT_SECRET_KEY", raising=False)
        with pytest.raises(KeyError):
            make_auth_header("test")


class TestCheckOpenOrders:
    """Test open order cancellation."""

    @patch("scripts.execute_trade.requests.delete")
    @patch("scripts.execute_trade.requests.get")
    def test_cancels_same_direction_orders(self, mock_get, mock_delete, monkeypatch):
        _set_trade_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {"side": "bid", "uuid": "order-1"},
            {"side": "ask", "uuid": "order-2"},
        ]
        mock_get.return_value = mock_resp
        mock_delete.return_value = MagicMock(ok=True)

        check_open_orders_and_cancel("KRW-BTC", "bid")

        # Should only cancel bid orders
        mock_delete.assert_called_once()

    @patch("scripts.execute_trade.requests.get")
    def test_handles_api_failure_gracefully(self, mock_get, monkeypatch):
        _set_trade_env(monkeypatch)
        mock_get.side_effect = requests.ConnectionError("timeout")

        # Should not raise
        check_open_orders_and_cancel("KRW-BTC", "bid")


class TestRecordTradeToDb:
    """Test DB recording function."""

    @patch("scripts.execute_trade.requests.post")
    def test_records_successful_trade(self, mock_post, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        result = {
            "success": True,
            "dry_run": False,
            "side": "bid",
            "market": "KRW-BTC",
            "amount": "50000",
            "response": {"uuid": "test-uuid", "price": "50000", "volume": "0.001"},
            "_exec_started": "2026-03-08T10:00:00+09:00",
            "_exec_completed": "2026-03-08T10:00:01+09:00",
            "_latency_ms": 1000,
        }
        _record_trade_to_db(result, source="agent")
        mock_post.assert_called_once()

        posted_data = mock_post.call_args[1]["json"]
        assert posted_data["decision"] == "매수"
        assert posted_data["execution_status"] == "success"
        assert posted_data["source"] == "agent"
        assert posted_data["order_uuid"] == "test-uuid"

    @patch("scripts.execute_trade.requests.post")
    def test_records_failed_trade(self, mock_post, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        result = {
            "success": False,
            "dry_run": False,
            "side": "ask",
            "market": "KRW-BTC",
            "amount": "0.001",
            "error": "insufficient funds",
        }
        _record_trade_to_db(result, source="manual")
        mock_post.assert_called_once()

        posted_data = mock_post.call_args[1]["json"]
        assert posted_data["decision"] == "매도"
        assert posted_data["execution_status"] == "failed"
        assert posted_data["execution_error"] == "insufficient funds"

    def test_no_db_env_does_not_crash(self, monkeypatch):
        """Missing SUPABASE env vars should silently return."""
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

        result = {"success": True, "side": "bid", "market": "KRW-BTC"}
        _record_trade_to_db(result)  # Should not raise

    @patch("scripts.execute_trade.requests.post")
    def test_db_failure_does_not_crash(self, mock_post, monkeypatch):
        """DB API failure should be silently handled."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")

        mock_post.side_effect = requests.ConnectionError("DB down")

        result = {"success": True, "side": "bid", "market": "KRW-BTC", "amount": "50000"}
        _record_trade_to_db(result)  # Should not raise


class TestExecuteTradeOrderBody:
    """Verify correct order body construction."""

    @patch("scripts.execute_trade.requests.post")
    def test_bid_order_body(self, mock_post, monkeypatch, patch_lock_file):
        """Buy order uses ord_type='price' and price field."""
        _set_trade_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "test"}
        mock_post.return_value = mock_resp

        execute("bid", "KRW-BTC", "50000")
        posted_body = mock_post.call_args[1]["json"]
        assert posted_body["ord_type"] == "price"
        assert posted_body["price"] == "50000"
        assert posted_body["side"] == "bid"

    @patch("scripts.execute_trade.requests.post")
    def test_ask_order_body(self, mock_post, monkeypatch, patch_lock_file):
        """Sell order uses ord_type='market' and volume field."""
        _set_trade_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "test"}
        mock_post.return_value = mock_resp

        execute("ask", "KRW-BTC", "0.001")
        posted_body = mock_post.call_args[1]["json"]
        assert posted_body["ord_type"] == "market"
        assert posted_body["volume"] == "0.001"
        assert posted_body["side"] == "ask"


class TestExecuteTradeLatency:
    """Verify latency tracking in execution result."""

    @patch("scripts.execute_trade.requests.post")
    def test_latency_recorded(self, mock_post, monkeypatch, patch_lock_file):
        _set_trade_env(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"uuid": "test"}
        mock_post.return_value = mock_resp

        result = execute("bid", "KRW-BTC", "50000")
        assert "_latency_ms" in result
        assert result["_latency_ms"] is not None
        assert isinstance(result["_latency_ms"], int)
        assert "_exec_started" in result
        assert "_exec_completed" in result


# ===========================================================================
# PART 5: get_portfolio.py (additional edge cases)
# ===========================================================================

import get_portfolio


def _make_accounts(accounts_data):
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = accounts_data
    return mock_resp


class TestPortfolioZeroBalance:
    """Holdings with zero balance are filtered out."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_zero_balance_filtered(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0", "avg_buy_price": "80000000"},
            {"currency": "ETH", "balance": "0.5", "avg_buy_price": "4000000"},
        ]
        markets_all = [{"market": "KRW-ETH"}]
        tickers = [{"market": "KRW-ETH", "trade_price": 5000000}]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _make_accounts(accounts)
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
        assert len(output["holdings"]) == 1
        assert output["holdings"][0]["currency"] == "ETH"


class TestPortfolioMarketAllFailure:
    """When /market/all fails, ticker is not called."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_market_all_failure(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "BTC", "balance": "0.01", "avg_buy_price": "80000000"},
        ]

        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            if "/accounts" in url:
                return _make_accounts(accounts)
            elif "/market/all" in url:
                resp = MagicMock()
                resp.ok = False
                return resp
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()
        output = json.loads(capsys.readouterr().out)
        # Holdings exist but current_price remains 0
        assert len(output["holdings"]) == 1
        assert output["holdings"][0]["current_price"] == 0


class TestPortfolioUnknownMarket:
    """Unknown market is filtered out from ticker call."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_unknown_market_filtered(self, mock_get, mock_jwt, capsys):
        accounts = [
            {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
            {"currency": "UNKNOWNCOIN", "balance": "100", "avg_buy_price": "1000"},
        ]

        def side_effect(url, **kwargs):
            if "/accounts" in url:
                return _make_accounts(accounts)
            elif "/market/all" in url:
                resp = MagicMock()
                resp.ok = True
                resp.json.return_value = [{"market": "KRW-BTC"}]  # No KRW-UNKNOWNCOIN
                return resp
            return MagicMock(ok=False)

        mock_get.side_effect = side_effect

        get_portfolio.main()
        output = json.loads(capsys.readouterr().out)
        assert output["holdings"][0]["current_price"] == 0


class TestPortfolioTotalProfitLoss:
    """Verify total_profit_loss_pct calculation."""

    @patch.dict(os.environ, {"UPBIT_ACCESS_KEY": "ak", "UPBIT_SECRET_KEY": "sk"})
    @patch("get_portfolio.jwt.encode", return_value="fake-jwt")
    @patch("get_portfolio.requests.get")
    def test_no_invested_zero_pct(self, mock_get, mock_jwt, capsys):
        """If total_invested is 0, profit_loss_pct should be 0."""
        accounts = [{"currency": "KRW", "balance": "500000", "avg_buy_price": "0"}]
        mock_get.return_value = _make_accounts(accounts)

        get_portfolio.main()
        output = json.loads(capsys.readouterr().out)
        assert output["total_profit_loss_pct"] == 0.0


# ===========================================================================
# PART 6: notify_telegram.py (additional edge cases)
# ===========================================================================

from notify_telegram import escape_md, send_message, send_photo


class TestEscapeMdAdditional:
    """Additional MarkdownV2 escaping tests."""

    def test_empty_string(self):
        assert escape_md("") == ""

    def test_only_special_chars(self):
        result = escape_md("_*[]()~`>#+-=|{}.!\\")
        # Each char should be escaped
        for ch in "_*[]()~`>#+-=|{}.!\\":
            assert f"\\{ch}" in result

    def test_unicode_preserved(self):
        assert escape_md("비트코인 가격") == "비트코인 가격"

    def test_numbers_not_escaped(self):
        assert escape_md("12345") == "12345"

    def test_mixed_unicode_and_special(self):
        result = escape_md("BTC: 100,000.5 원")
        assert "\\." in result
        assert "원" in result

    def test_newlines_preserved(self):
        result = escape_md("line1\nline2")
        assert "\n" in result

    def test_url_escaped(self):
        result = escape_md("https://example.com/path?q=1")
        assert "\\." in result
        assert "\\?" not in result  # ? is not in the escape list


class TestSendMessageAdditional:
    """Additional send_message tests."""

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_timeout_propagated(self, mock_post):
        """Request timeout should propagate."""
        mock_post.side_effect = requests.Timeout("timed out")
        with pytest.raises(requests.Timeout):
            send_message("trade", "title", "body")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_status_emoji(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("status", "System Status", "All OK")
        payload = mock_post.call_args[1]["json"]
        assert "\U0001f4cb" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_special_chars_in_title_escaped(self, mock_post):
        """Special chars in title should be escaped for MarkdownV2."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("trade", "BTC +2.5% (100,000 KRW)", "body")
        payload = mock_post.call_args[1]["json"]
        # Title should have escaped special chars
        assert "\\." in payload["text"]
        assert "\\+" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_body_with_special_chars(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_message("trade", "Title", "RSI: 28.5 (oversold)")
        payload = mock_post.call_args[1]["json"]
        assert "\\." in payload["text"]
        assert "\\(" in payload["text"]


class TestSendPhotoAdditional:
    """Additional send_photo tests."""

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"PNG"))
    def test_timeout_setting(self, mock_post):
        """Send photo should use 30s timeout."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        send_photo("/tmp/chart.png", "caption")
        assert mock_post.call_args[1]["timeout"] == 30

    def test_file_not_found(self):
        """Non-existent file should raise."""
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        os.environ["TELEGRAM_USER_ID"] = "123"
        with pytest.raises(FileNotFoundError):
            send_photo("/nonexistent/path/chart.png", "caption")


# ===========================================================================
# PART 7: collect_ai_signal.py (additional edge cases)
# ===========================================================================

from collect_ai_signal import (
    calc_rsi as ai_calc_rsi,
    trend_pct as ai_trend_pct,
    analyze_orderbook,
    analyze_whale_trades,
    analyze_volatility,
    analyze_volume,
    compute_composite_score,
    main as ai_signal_main,
)


def _make_candle(trade_price, high=None, low=None, opening=None, volume=1000,
                 date="2026-03-08T00:00:00"):
    if high is None:
        high = trade_price * 1.01
    if low is None:
        low = trade_price * 0.99
    if opening is None:
        opening = trade_price
    return {
        "trade_price": trade_price,
        "high_price": high,
        "low_price": low,
        "opening_price": opening,
        "candle_acc_trade_volume": volume,
        "candle_date_time_kst": date,
    }


class TestAISignalCalcRSIEdge:
    """Additional calc_rsi edge cases."""

    def test_empty_candles(self):
        assert ai_calc_rsi([], 14) is None

    def test_single_candle(self):
        candles = [_make_candle(100)]
        assert ai_calc_rsi(candles, 14) is None

    def test_flat_prices(self):
        candles = [_make_candle(100) for _ in range(20)]
        result = ai_calc_rsi(candles, 14)
        assert result == 100.0  # no losses


class TestAISignalTrendPctEdge:
    """Additional trend_pct edge cases."""

    def test_empty_candles(self):
        assert ai_trend_pct([], 5) == 0.0

    def test_single_candle(self):
        candles = [_make_candle(100)]
        assert ai_trend_pct(candles, 5) == 0.0

    def test_first_price_zero(self):
        """Division by zero guard when first price is 0."""
        candles = [_make_candle(100), _make_candle(50), _make_candle(50),
                   _make_candle(50), _make_candle(0)]
        result = ai_trend_pct(candles, 5)
        assert result == 0.0  # first=0 -> returns 0.0


class TestAnalyzeWhaleMegaDetection:
    """Mega whale detection (50M+ KRW)."""

    @patch("collect_ai_signal.api_get")
    def test_mega_whale_detected(self, mock_api):
        # 1 BTC at 50M = 50M KRW -> mega whale
        trades = [
            {"ask_bid": "BID", "trade_volume": 1.0, "trade_price": 50_000_000,
             "trade_time_utc": "12:00:00"},
        ]
        mock_api.return_value = trades
        result = analyze_whale_trades("KRW-BTC")
        assert result["mega_whale_count"] == 1
        assert result["mega_whale_buy"] == 1
        assert result["mega_whale_sell"] == 0

    @patch("collect_ai_signal.api_get")
    def test_no_whale_trades(self, mock_api):
        """All small trades -> no whales."""
        trades = [
            {"ask_bid": "BID", "trade_volume": 0.00001, "trade_price": 50_000_000,
             "trade_time_utc": "12:00:00"},
        ] * 100
        mock_api.return_value = trades
        result = analyze_whale_trades("KRW-BTC")
        assert result["whale_trades_count"] == 0
        assert result["mega_whale_count"] == 0
        assert result["whale_signal"] == "neutral"

    @patch("collect_ai_signal.api_get")
    def test_empty_trades(self, mock_api):
        """No trades at all."""
        mock_api.return_value = []
        result = analyze_whale_trades("KRW-BTC")
        assert result["total_trades"] == 0
        assert result["buy_count"] == 0
        assert result["sell_count"] == 0

    @patch("collect_ai_signal.api_get")
    def test_sell_pressure(self, mock_api):
        trades = [
            {"ask_bid": "ASK", "trade_volume": 0.01, "trade_price": 50_000_000,
             "trade_time_utc": "12:00:00"},
        ] * 80 + [
            {"ask_bid": "BID", "trade_volume": 0.01, "trade_price": 50_000_000,
             "trade_time_utc": "12:00:00"},
        ] * 20
        mock_api.return_value = trades
        result = analyze_whale_trades("KRW-BTC")
        assert result["trade_pressure"] == "sell"
        assert result["buy_ratio_pct"] == 20.0


class TestCompositeScoreInterpretation:
    """Test interpretation thresholds."""

    def _base_inputs(self):
        return {
            "orderbook": {"imbalance_ratio": 1.0},
            "whale": {"buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
                       "cvd_trend": "neutral"},
            "multi_tf": {"divergence_type": None, "trend_alignment": "mixed"},
            "volatility": {"regime": "normal"},
            "volume": {},
            "daily": [_make_candle(50_000_000) for _ in range(20)],
        }

    def test_strong_buy_at_30(self):
        """Score >= 30 -> strong_buy."""
        inputs = self._base_inputs()
        inputs["orderbook"]["imbalance_ratio"] = 2.0  # +15
        inputs["whale"]["buy_ratio_pct"] = 75  # +20 (capped)
        result = compute_composite_score(
            inputs["orderbook"], inputs["whale"], inputs["multi_tf"],
            inputs["volatility"], inputs["volume"], inputs["daily"],
        )
        assert result["score"] >= 30
        assert result["interpretation"] == "strong_buy"

    def test_weak_buy_between_10_and_29(self):
        """Score 10-29 -> weak_buy."""
        inputs = self._base_inputs()
        inputs["orderbook"]["imbalance_ratio"] = 1.3  # +15
        result = compute_composite_score(
            inputs["orderbook"], inputs["whale"], inputs["multi_tf"],
            inputs["volatility"], inputs["volume"], inputs["daily"],
        )
        assert 10 <= result["score"] < 30
        assert result["interpretation"] == "weak_buy"

    def test_strong_sell_at_minus_30(self):
        """Score <= -30 -> strong_sell."""
        inputs = self._base_inputs()
        inputs["orderbook"]["imbalance_ratio"] = 0.3  # -15 (capped)
        inputs["whale"]["buy_ratio_pct"] = 20  # -20
        result = compute_composite_score(
            inputs["orderbook"], inputs["whale"], inputs["multi_tf"],
            inputs["volatility"], inputs["volume"], inputs["daily"],
        )
        assert result["score"] <= -30
        assert result["interpretation"] == "strong_sell"

    def test_weak_sell_between_minus29_and_minus10(self):
        """Score -29 to -10 -> weak_sell."""
        inputs = self._base_inputs()
        inputs["orderbook"]["imbalance_ratio"] = 0.7  # -15
        result = compute_composite_score(
            inputs["orderbook"], inputs["whale"], inputs["multi_tf"],
            inputs["volatility"], inputs["volume"], inputs["daily"],
        )
        assert -30 < result["score"] <= -10
        assert result["interpretation"] == "weak_sell"

    def test_neutral_between_minus9_and_9(self):
        """Score -9 to 9 -> neutral."""
        inputs = self._base_inputs()
        result = compute_composite_score(
            inputs["orderbook"], inputs["whale"], inputs["multi_tf"],
            inputs["volatility"], inputs["volume"], inputs["daily"],
        )
        assert -10 < result["score"] < 10
        assert result["interpretation"] == "neutral"


class TestCompositeScoreVolumeAnomaly:
    """Test volume anomaly scoring in composite score."""

    def test_high_volume_positive_price(self):
        """Volume spike with positive candle -> positive score."""
        orderbook = {"imbalance_ratio": 1.0}
        whale = {"buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
                 "cvd_trend": "neutral"}
        multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        volatility = {"regime": "normal"}

        daily = [_make_candle(50_000_000, volume=1000) for _ in range(20)]
        # Override first candle: high volume + positive change
        daily[0] = _make_candle(
            55_000_000, high=56_000_000, low=49_000_000,
            opening=50_000_000, volume=5000,
        )

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily,
        )
        vol_comp = [c for c in result["components"] if c["name"] == "volume_anomaly"]
        assert len(vol_comp) == 1
        assert vol_comp[0]["score"] > 0

    def test_high_volume_negative_price(self):
        """Volume spike with negative candle -> negative score."""
        orderbook = {"imbalance_ratio": 1.0}
        whale = {"buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
                 "cvd_trend": "neutral"}
        multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        volatility = {"regime": "normal"}

        daily = [_make_candle(50_000_000, volume=1000) for _ in range(20)]
        daily[0] = _make_candle(
            45_000_000, high=51_000_000, low=44_000_000,
            opening=50_000_000, volume=5000,
        )

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily,
        )
        vol_comp = [c for c in result["components"] if c["name"] == "volume_anomaly"]
        assert len(vol_comp) == 1
        assert vol_comp[0]["score"] < 0

    def test_normal_volume_no_component(self):
        """Normal volume (ratio < 1.5) should not add volume_anomaly component."""
        orderbook = {"imbalance_ratio": 1.0}
        whale = {"buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
                 "cvd_trend": "neutral"}
        multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        volatility = {"regime": "normal"}

        daily = [_make_candle(50_000_000, volume=1000) for _ in range(20)]

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily,
        )
        vol_comp = [c for c in result["components"] if c["name"] == "volume_anomaly"]
        assert len(vol_comp) == 0

    def test_insufficient_daily_data(self):
        """Less than 20 daily candles -> no volume anomaly."""
        orderbook = {"imbalance_ratio": 1.0}
        whale = {"buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
                 "cvd_trend": "neutral"}
        multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        volatility = {"regime": "normal"}

        daily = [_make_candle(50_000_000, volume=5000) for _ in range(5)]

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily,
        )
        vol_comp = [c for c in result["components"] if c["name"] == "volume_anomaly"]
        assert len(vol_comp) == 0


class TestCompositeScoreWhaleWeighting:
    """Test whale direction scoring with mega whale weighting."""

    def test_mega_whale_buy_adds_weight(self):
        """Mega whale buys should increase the whale direction score."""
        orderbook = {"imbalance_ratio": 1.0}
        multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        volatility = {"regime": "normal"}
        daily = [_make_candle(50_000_000) for _ in range(20)]

        whale_normal = {
            "buy_ratio_pct": 50, "whale_buy": 3, "whale_sell": 1,
            "cvd_trend": "neutral",
            "mega_whale_buy": 0, "mega_whale_sell": 0,
        }
        whale_mega = {
            "buy_ratio_pct": 50, "whale_buy": 3, "whale_sell": 1,
            "cvd_trend": "neutral",
            "mega_whale_buy": 2, "mega_whale_sell": 0,
        }

        r_normal = compute_composite_score(
            orderbook, whale_normal, multi_tf, volatility, {}, daily,
        )
        r_mega = compute_composite_score(
            orderbook, whale_mega, multi_tf, volatility, {}, daily,
        )
        # Mega whale should give higher score
        normal_whale_pts = sum(c["score"] for c in r_normal["components"]
                               if c["name"] == "whale_direction")
        mega_whale_pts = sum(c["score"] for c in r_mega["components"]
                             if c["name"] == "whale_direction")
        assert mega_whale_pts >= normal_whale_pts


class TestAnalyzeVolatilityEdge:
    """Additional volatility analysis tests."""

    def test_expanding_regime(self):
        """vol_ratio between 1.0 and 1.5 -> expanding."""
        candles = []
        for i in range(3):
            candles.append(_make_candle(100, high=106, low=100))  # 6%
        for i in range(17):
            candles.append(_make_candle(100, high=105, low=100))  # 5%

        result = analyze_volatility(candles)
        # avg_3 = 6%, avg_20 = (6*3 + 5*17)/20 = 5.15%
        # ratio = 6/5.15 ~= 1.16 -> expanding
        assert result["regime"] == "expanding"

    def test_exact_20_candles(self):
        candles = [_make_candle(100, high=105, low=95) for _ in range(20)]
        result = analyze_volatility(candles)
        assert "error" not in result
        assert result["regime"] in ("normal", "expanding")


class TestAnalyzeVolumeEdge:
    """Additional volume analysis tests."""

    def test_exact_20_candles(self):
        candles = [_make_candle(100, volume=1000,
                                date=f"2026-03-{8-i:02d}T00:00:00")
                   for i in range(20)]
        result = analyze_volume(candles)
        assert "error" not in result
        assert len(result["recent_5d"]) == 5

    def test_high_volume_anomaly(self):
        """Volume ratio 1.5-2.0 -> 'high' anomaly."""
        candles = [_make_candle(100, volume=1700, date="2026-03-08T00:00:00")]
        for i in range(19):
            candles.append(_make_candle(100, volume=1000,
                                        date=f"2026-03-{7-i:02d}T00:00:00"))
        result = analyze_volume(candles)
        # avg = (1700 + 1000*19)/20 = 1035
        # ratio = 1700/1035 ~= 1.64 -> 'high'
        assert result["recent_5d"][0]["anomaly"] == "high"


class TestAISignalMainIntegration:
    """Test main() of collect_ai_signal.py."""

    @patch("collect_ai_signal.api_get")
    @patch("collect_ai_signal.time")
    def test_main_outputs_valid_json(self, mock_time, mock_api, capsys):
        mock_time.sleep = MagicMock()
        mock_time.strftime = MagicMock(return_value="2026-03-08T00:00:00+09:00")

        # Orderbook
        ob = [{
            "total_bid_size": 50.0,
            "total_ask_size": 50.0,
            "orderbook_units": [
                {"bid_price": 50_000_000, "bid_size": 5.0,
                 "ask_price": 50_100_000, "ask_size": 5.0},
            ],
        }]

        # Trades
        trades = [
            {"ask_bid": "BID", "trade_volume": 0.01, "trade_price": 50_000_000,
             "trade_time_utc": "12:00:00"},
        ] * 100 + [
            {"ask_bid": "ASK", "trade_volume": 0.01, "trade_price": 50_000_000,
             "trade_time_utc": "12:00:00"},
        ] * 100

        # Daily candles (30)
        daily = [_make_candle(50_000_000 + i * 10000, volume=1000,
                              date=f"2026-03-{8-i:02d}T00:00:00")
                 for i in range(30)]

        # 4h candles (42)
        h4 = [_make_candle(50_000_000 + i * 5000) for i in range(42)]

        # 1h candles (24)
        h1 = [_make_candle(50_000_000 + i * 2000) for i in range(24)]

        # api_get is called: orderbook, trades, daily(multi_tf), h4, h1, daily(main)
        mock_api.side_effect = [ob, trades, daily, h4, h1, daily]

        ai_signal_main("KRW-BTC")
        output = json.loads(capsys.readouterr().out)

        assert output["market"] == "KRW-BTC"
        assert "ai_composite_signal" in output
        assert "score" in output["ai_composite_signal"]
        assert "interpretation" in output["ai_composite_signal"]
        assert "details" in output
        assert "orderbook_imbalance" in output["details"]
        assert "whale_detection" in output["details"]
        assert "multi_timeframe" in output["details"]
        assert "volatility_regime" in output["details"]
        assert "volume_anomaly" in output["details"]

    @patch("collect_ai_signal.api_get")
    def test_main_api_failure(self, mock_api):
        """API failure in main() should propagate."""
        mock_api.side_effect = requests.ConnectionError("timeout")
        with pytest.raises(requests.ConnectionError):
            ai_signal_main("KRW-BTC")


class TestAnalyzeOrderbookEdge:
    """Additional orderbook analysis tests."""

    @patch("collect_ai_signal.api_get")
    def test_empty_orderbook_units(self, mock_api):
        """Empty orderbook units list."""
        mock_api.return_value = [{
            "total_bid_size": 0.0,
            "total_ask_size": 0.0,
            "orderbook_units": [],
        }]
        result = analyze_orderbook("KRW-BTC")
        assert result["bid_walls"] == []
        assert result["ask_walls"] == []

    @patch("collect_ai_signal.api_get")
    def test_zero_ask_division_guard(self, mock_api):
        """Zero total_ask should not crash (division by near-zero)."""
        mock_api.return_value = [{
            "total_bid_size": 100.0,
            "total_ask_size": 0.0,
            "orderbook_units": [],
        }]
        result = analyze_orderbook("KRW-BTC")
        assert result["imbalance_ratio"] > 0
        assert result["signal"] == "buy"
