"""Unit tests for technical indicator functions in collect_market_data.py."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_market_data import (
    api_get,
    bollinger,
    calc_adx,
    calc_atr,
    collect_eth_btc_ratio,
    ema,
    macd,
    main,
    rsi,
    sma,
    stochastic,
)


# ── SMA ──────────────────────────────────────────────────


class TestSMA:
    def test_basic(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert sma(prices, 5) == 3.0

    def test_window_smaller_than_list(self):
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        # period=3 -> last 3 elements: 30, 40, 50
        assert sma(prices, 3) == 40.0

    def test_single_element(self):
        assert sma([42.0], 1) == 42.0

    def test_known_values(self):
        # 20-period SMA of 1..20 = (1+2+...+20)/20 = 210/20 = 10.5
        prices = [float(i) for i in range(1, 21)]
        assert sma(prices, 20) == 10.5

    def test_all_same(self):
        prices = [100.0] * 10
        assert sma(prices, 5) == 100.0


# ── EMA ──────────────────────────────────────────────────


class TestEMA:
    def test_single_element(self):
        assert ema([50.0], 10) == 50.0

    def test_two_elements(self):
        # k = 2/(10+1) = 2/11
        # value starts at 100, then: 200 * (2/11) + 100 * (9/11)
        k = 2.0 / 11.0
        expected = 200.0 * k + 100.0 * (1 - k)
        assert ema([100.0, 200.0], 10) == pytest.approx(expected)

    def test_manual_calculation_period3(self):
        # period=3, k = 2/4 = 0.5
        prices = [10.0, 20.0, 30.0, 40.0]
        k = 0.5
        v = 10.0
        v = 20.0 * k + v * (1 - k)  # 15.0
        v = 30.0 * k + v * (1 - k)  # 22.5
        v = 40.0 * k + v * (1 - k)  # 31.25
        assert ema(prices, 3) == pytest.approx(v)

    def test_constant_prices(self):
        # EMA of constant series should be the constant
        prices = [50.0] * 20
        assert ema(prices, 10) == pytest.approx(50.0)


# ── RSI ──────────────────────────────────────────────────


class TestRSI:
    def test_insufficient_data(self):
        # < period+1 prices should return 50.0
        prices = [float(i) for i in range(14)]  # 14 elements, need 15
        assert rsi(prices, 14) == 50.0

    def test_all_gains(self):
        # Monotonically increasing -> RSI = 100
        prices = [float(i) for i in range(20)]
        assert rsi(prices, 14) == 100.0

    def test_all_losses(self):
        # Monotonically decreasing -> RSI = 0
        prices = [float(20 - i) for i in range(20)]
        assert rsi(prices, 14) == pytest.approx(0.0, abs=0.01)

    def test_flat_prices(self):
        # No change -> gains=0, losses=0, al=0 -> RSI = 100
        prices = [100.0] * 20
        assert rsi(prices, 14) == 100.0

    def test_known_value(self):
        # Alternating gains and losses: +1, -1, +1, -1...
        # After initial period: avg_gain = avg_loss -> RS=1 -> RSI=50
        prices = [100.0]
        for i in range(1, 30):
            prices.append(prices[-1] + (1 if i % 2 == 1 else -1))
        result = rsi(prices, 14)
        # With Wilder smoothing and alternating, should be close to 50
        assert 45.0 < result < 55.0

    def test_rsi_in_range(self):
        # RSI should always be between 0 and 100
        import random
        random.seed(42)
        prices = [100.0]
        for _ in range(50):
            prices.append(prices[-1] + random.uniform(-5, 5))
        result = rsi(prices, 14)
        assert 0.0 <= result <= 100.0

    def test_exact_period_plus_one(self):
        # Exactly period+1 prices: only initial SMA, no smoothing loop
        prices = [10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 13.5, 14.0, 13.0,
                  14.5, 15.0, 14.0, 15.5, 16.0]
        assert len(prices) == 15  # 14 + 1
        result = rsi(prices, 14)
        assert 0.0 <= result <= 100.0


# ── MACD ─────────────────────────────────────────────────


class TestMACD:
    def test_insufficient_data(self):
        prices = [float(i) for i in range(25)]  # < 26
        result = macd(prices)
        assert result == {"macd": 0, "signal": 0, "histogram": 0}

    def test_exactly_26_prices(self):
        prices = [float(i) for i in range(26)]
        result = macd(prices)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_macd_equals_ema12_minus_ema26(self):
        prices = [100 + i * 0.5 for i in range(50)]
        result = macd(prices)

        # Manually compute EMA12 and EMA26 over all prices
        k12 = 2 / 13
        k26 = 2 / 27
        ema12_val = prices[0]
        ema26_val = prices[0]
        for p in prices[1:]:
            ema12_val = p * k12 + ema12_val * (1 - k12)
            ema26_val = p * k26 + ema26_val * (1 - k26)

        expected_macd = ema12_val - ema26_val
        assert result["macd"] == pytest.approx(expected_macd, abs=0.01)

    def test_histogram_equals_macd_minus_signal(self):
        prices = [100 + i * 0.3 for i in range(50)]
        result = macd(prices)
        assert result["histogram"] == pytest.approx(
            result["macd"] - result["signal"], abs=0.01
        )

    def test_signal_is_9period_ema_of_macd_series(self):
        prices = [100 + i * 0.7 for i in range(60)]
        result = macd(prices)

        # Rebuild MACD series
        k12, k26 = 2 / 13, 2 / 27
        ema12_val = prices[0]
        ema26_val = prices[0]
        macd_series = []
        for p in prices[1:]:
            ema12_val = p * k12 + ema12_val * (1 - k12)
            ema26_val = p * k26 + ema26_val * (1 - k26)
            macd_series.append(ema12_val - ema26_val)

        # Signal = 9-period EMA of macd_series
        k9 = 2 / 10
        s = macd_series[0]
        for v in macd_series[1:]:
            s = v * k9 + s * (1 - k9)

        assert result["signal"] == pytest.approx(s, abs=0.01)

    def test_constant_prices(self):
        prices = [100.0] * 30
        result = macd(prices)
        assert result["macd"] == pytest.approx(0.0, abs=0.01)
        assert result["signal"] == pytest.approx(0.0, abs=0.01)
        assert result["histogram"] == pytest.approx(0.0, abs=0.01)


# ── Bollinger Bands ──────────────────────────────────────


class TestBollinger:
    def test_basic_structure(self):
        prices = [float(i) for i in range(1, 25)]
        result = bollinger(prices, 20)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result

    def test_upper_lower_symmetry(self):
        prices = [float(i) for i in range(1, 25)]
        result = bollinger(prices, 20)
        mid = result["middle"]
        # upper - middle should equal middle - lower
        assert result["upper"] - mid == pytest.approx(mid - result["lower"], abs=0.01)

    def test_known_values(self):
        # All same prices -> SD = 0, upper = middle = lower
        prices = [100.0] * 20
        result = bollinger(prices, 20)
        assert result["upper"] == pytest.approx(100.0)
        assert result["middle"] == pytest.approx(100.0)
        assert result["lower"] == pytest.approx(100.0)

    def test_formula_verification(self):
        prices = [10.0, 12.0, 11.0, 13.0, 14.0]
        period = 5
        result = bollinger(prices, period)

        mid = sum(prices[-period:]) / period  # 12.0
        var = sum((p - mid) ** 2 for p in prices[-period:]) / period
        sd = var ** 0.5

        assert result["middle"] == pytest.approx(mid, abs=0.01)
        assert result["upper"] == pytest.approx(mid + 2 * sd, abs=0.01)
        assert result["lower"] == pytest.approx(mid - 2 * sd, abs=0.01)

    def test_upper_ge_middle_ge_lower(self):
        prices = [100 + i for i in range(25)]
        result = bollinger(prices, 20)
        assert result["upper"] >= result["middle"]
        assert result["middle"] >= result["lower"]


# ── Stochastic ───────────────────────────────────────────


class TestStochastic:
    def test_insufficient_data(self):
        # Need at least period + 2 = 16 elements
        highs = [float(i + 1) for i in range(15)]
        lows = [float(i) for i in range(15)]
        closes = [float(i + 0.5) for i in range(15)]
        result = stochastic(highs, lows, closes, 14)
        assert result == {"k": 50.0, "d": 50.0}

    def test_known_values(self):
        # Construct data where %K is calculable
        # 20 bars: highs all 110, lows all 90, close at 100
        n = 20
        highs = [110.0] * n
        lows = [90.0] * n
        closes = [100.0] * n
        result = stochastic(highs, lows, closes, 14)
        # %K = (100 - 90) / (110 - 90) * 100 = 50.0
        assert result["k"] == pytest.approx(50.0)
        # All %K values are 50 -> %D = 50
        assert result["d"] == pytest.approx(50.0)

    def test_close_at_high(self):
        n = 20
        highs = [110.0] * n
        lows = [90.0] * n
        closes = [110.0] * n  # Close = High
        result = stochastic(highs, lows, closes, 14)
        assert result["k"] == pytest.approx(100.0)
        assert result["d"] == pytest.approx(100.0)

    def test_close_at_low(self):
        n = 20
        highs = [110.0] * n
        lows = [90.0] * n
        closes = [90.0] * n  # Close = Low
        result = stochastic(highs, lows, closes, 14)
        assert result["k"] == pytest.approx(0.0)
        assert result["d"] == pytest.approx(0.0)

    def test_k_and_d_in_range(self):
        import random
        random.seed(123)
        n = 30
        closes = [100.0]
        for _ in range(n - 1):
            closes.append(closes[-1] + random.uniform(-3, 3))
        highs = [c + random.uniform(0, 2) for c in closes]
        lows = [c - random.uniform(0, 2) for c in closes]
        result = stochastic(highs, lows, closes, 14)
        assert 0.0 <= result["k"] <= 100.0
        assert 0.0 <= result["d"] <= 100.0


# ── ADX ──────────────────────────────────────────────────


class TestCalcADX:
    def test_insufficient_data(self):
        highs = [10.0] * 10
        lows = [9.0] * 10
        closes = [9.5] * 10
        result = calc_adx(highs, lows, closes, 14)
        assert result["adx"] == 0
        assert result["regime"] == "unknown"

    def test_trending_data(self):
        # Strong uptrend -> high ADX
        n = 80
        highs = [100.0 + i * 2.0 for i in range(n)]
        lows = [98.0 + i * 2.0 for i in range(n)]
        closes = [99.0 + i * 2.0 for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        assert result["adx"] > 20  # Should be trending or transitioning
        assert result["regime"] in ("trending", "transitioning")
        assert result["plus_di"] > result["minus_di"]  # Uptrend

    def test_ranging_data(self):
        # Sideways oscillation -> low ADX
        n = 80
        highs = [101.0 + (i % 2) * 0.5 for i in range(n)]
        lows = [99.0 - (i % 2) * 0.5 for i in range(n)]
        closes = [100.0 + ((-1) ** i) * 0.3 for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        # Ranging market should have low ADX
        assert result["adx"] < 30  # Not strongly trending

    def test_output_structure(self):
        n = 40
        highs = [100.0 + i for i in range(n)]
        lows = [98.0 + i for i in range(n)]
        closes = [99.0 + i for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result
        assert "regime" in result
        assert result["regime"] in ("trending", "ranging", "transitioning")

    def test_downtrend(self):
        # Strong downtrend -> minus_di > plus_di
        n = 80
        highs = [200.0 - i * 2.0 for i in range(n)]
        lows = [198.0 - i * 2.0 for i in range(n)]
        closes = [199.0 - i * 2.0 for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        assert result["minus_di"] > result["plus_di"]


# ── ATR ──────────────────────────────────────────────────


class TestCalcATR:
    def test_insufficient_data(self):
        highs = [10.0] * 10
        lows = [9.0] * 10
        closes = [9.5] * 10
        assert calc_atr(highs, lows, closes, 14) == 0.0

    def test_known_true_ranges(self):
        # 3 bars, period=2
        # Bar 0: doesn't produce TR (no previous close)
        # Bar 1: TR = max(12-8, |12-10|, |8-10|) = max(4, 2, 2) = 4
        # Bar 2: TR = max(15-9, |15-11|, |9-11|) = max(6, 4, 2) = 6
        # ATR initial (period=2) = (4+6)/2 = 5.0
        highs = [11.0, 12.0, 15.0]
        lows = [9.0, 8.0, 9.0]
        closes = [10.0, 11.0, 14.0]
        result = calc_atr(highs, lows, closes, 2)
        assert result == pytest.approx(5.0, abs=0.01)

    def test_constant_range(self):
        # All bars have same range -> ATR = that range
        n = 20
        highs = [110.0] * n
        lows = [100.0] * n
        closes = [105.0] * n
        result = calc_atr(highs, lows, closes, 14)
        assert result == pytest.approx(10.0, abs=0.1)

    def test_atr_positive(self):
        import random
        random.seed(99)
        n = 30
        closes = [100.0]
        for _ in range(n - 1):
            closes.append(closes[-1] + random.uniform(-2, 2))
        highs = [c + random.uniform(0.5, 3) for c in closes]
        lows = [c - random.uniform(0.5, 3) for c in closes]
        result = calc_atr(highs, lows, closes, 14)
        assert result > 0

    def test_wilder_smoothing(self):
        # 4 bars, period=2
        # Bar 1 TR: max(12-8, |12-10|, |8-10|) = 4
        # Bar 2 TR: max(15-9, |15-11|, |9-11|) = 6
        # Bar 3 TR: max(14-10, |14-14|, |10-14|) = max(4, 0, 4) = 4
        # ATR initial = (4+6)/2 = 5
        # ATR smoothed = (5*(2-1) + 4)/2 = 9/2 = 4.5
        highs = [11.0, 12.0, 15.0, 14.0]
        lows = [9.0, 8.0, 9.0, 10.0]
        closes = [10.0, 11.0, 14.0, 12.0]
        result = calc_atr(highs, lows, closes, 2)
        assert result == pytest.approx(4.5, abs=0.01)


# ── api_get() retry logic ──────────────────────────────


class TestApiGet:
    @patch("collect_market_data.requests.get")
    def test_success_on_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"price": 100}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = api_get("/ticker", {"markets": "KRW-BTC"})
        assert result == {"price": 100}
        assert mock_get.call_count == 1

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.requests.get")
    def test_429_retry_then_success(self, mock_get, mock_sleep):
        # First call returns 429, second succeeds
        mock_429 = MagicMock()
        mock_429.status_code = 429

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"data": "ok"}
        mock_ok.raise_for_status.return_value = None

        mock_get.side_effect = [mock_429, mock_ok]

        result = api_get("/ticker", max_retries=3)
        assert result == {"data": "ok"}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.requests.get")
    def test_429_all_retries_fail(self, mock_get, mock_sleep):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.raise_for_status.side_effect = Exception("429 Too Many Requests")
        mock_get.return_value = mock_429

        # After all retries exhausted, raise_for_status is called and raises
        with pytest.raises(Exception):
            api_get("/ticker", max_retries=3)
        # Loop runs max_retries times, then raise_for_status on last response
        assert mock_get.call_count == 3

    @patch("collect_market_data.requests.get")
    def test_connection_error_returns_exception(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("Connection refused")

        with pytest.raises(req.ConnectionError):
            api_get("/ticker")

    @patch("collect_market_data.requests.get")
    def test_500_error_no_retry(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        mock_get.return_value = mock_resp

        with pytest.raises(Exception, match="500"):
            api_get("/ticker", max_retries=3)
        # 500 is not 429, so raise_for_status is called immediately — only 1 call
        assert mock_get.call_count == 1


# ── collect_eth_btc_ratio() ─────────────────────────────


class TestCollectEthBtcRatio:
    def _make_candle(self, price):
        return {"trade_price": price, "high_price": price + 10, "low_price": price - 10,
                "opening_price": price, "candle_acc_trade_volume": 1.0,
                "candle_date_time_kst": "2026-03-08T00:00:00"}

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_normal_ratio_calculation(self, mock_api, mock_sleep):
        # ETH ticker, BTC ticker, BTC daily, ETH daily
        eth_ticker = [{"trade_price": 5000000, "signed_change_rate": 0.02,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.01,
                        "acc_trade_price_24h": 5e12}]
        # 60 candles each, with a slight variation
        btc_candles = [self._make_candle(100000000 + i * 10000) for i in range(60)]
        eth_candles = [self._make_candle(5000000 + i * 500) for i in range(60)]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]

        result = collect_eth_btc_ratio()

        assert "eth_price" in result
        assert result["eth_price"] == 5000000
        assert "eth_btc_ratio" in result
        assert "eth_btc_z_score" in result
        assert "eth_btc_signal" in result
        assert "eth_rsi_14" in result
        assert "error" not in result

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_eth_api_failure(self, mock_api, mock_sleep):
        mock_api.side_effect = Exception("ETH API down")

        result = collect_eth_btc_ratio()
        assert "error" in result
        assert "ETH" in result["error"]

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_btc_api_failure(self, mock_api, mock_sleep):
        eth_ticker = [{"trade_price": 5000000, "signed_change_rate": 0.02,
                        "acc_trade_price_24h": 1e12}]
        mock_api.side_effect = [eth_ticker, Exception("BTC API down")]

        result = collect_eth_btc_ratio()
        assert "error" in result

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_stdev_zero_fallback(self, mock_api, mock_sleep):
        """All ratios identical -> stdev=0 -> fallback to 0.001."""
        eth_ticker = [{"trade_price": 5000000, "signed_change_rate": 0.02,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.01,
                        "acc_trade_price_24h": 5e12}]
        # All candles at exact same price -> identical ratio
        btc_candles = [self._make_candle(100000000) for _ in range(60)]
        eth_candles = [self._make_candle(5000000) for _ in range(60)]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]

        result = collect_eth_btc_ratio()
        # z_score should be 0 since current ratio == mean ratio
        assert result["eth_btc_z_score"] == pytest.approx(0.0, abs=0.1)
        assert "error" not in result

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_signal_extreme_undervalued(self, mock_api, mock_sleep):
        """z_score < -2 -> 극단적_저평가."""
        eth_ticker = [{"trade_price": 3000000, "signed_change_rate": -0.1,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.01,
                        "acc_trade_price_24h": 5e12}]
        # API returns newest first; reversed() in code makes oldest first
        # After reverse: [3000000, 5M*59] -> last ratio = 5M/100M, but we want last to be low
        # So provide newest (low ETH) first -> after reverse, low is at index 0, high at end
        # Actually: reversed() means first element becomes last. So newest-first input
        # becomes oldest-first after reverse. The LAST element after reverse = first in input = newest.
        # We want the LAST ratio (newest) to be very low.
        btc_candles = [self._make_candle(100000000) for _ in range(60)]
        # Newest first: 3M (newest), then 59x 5M (older)
        eth_prices_newest_first = [3000000] + [5000000] * 59
        eth_candles = [self._make_candle(p) for p in eth_prices_newest_first]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]
        result = collect_eth_btc_ratio()
        assert result["eth_btc_signal"] == "ETH 극단적 저평가"

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_signal_undervalued(self, mock_api, mock_sleep):
        """z_score between -2 and -1 -> 저평가."""
        eth_ticker = [{"trade_price": 4500000, "signed_change_rate": -0.05,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.01,
                        "acc_trade_price_24h": 5e12}]
        # Need z between -2 and -1. Use a gradual decline so stdev is larger.
        # Newest first: gradually declining ETH prices
        btc_candles = [self._make_candle(100000000) for _ in range(60)]
        # Create a spread so stdev is meaningful, newest (4.5M) is moderately below mean
        eth_prices_newest_first = [4500000] + [5000000 + i * 10000 for i in range(59)]
        eth_candles = [self._make_candle(p) for p in eth_prices_newest_first]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]
        result = collect_eth_btc_ratio()
        # z should be negative (below mean)
        assert result["eth_btc_z_score"] < 0
        assert result["eth_btc_signal"] in ("ETH 저평가", "ETH 극단적 저평가")

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_signal_extreme_overvalued(self, mock_api, mock_sleep):
        """z_score > 2 -> 극단적 고평가."""
        eth_ticker = [{"trade_price": 8000000, "signed_change_rate": 0.2,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.01,
                        "acc_trade_price_24h": 5e12}]
        # Newest first: 8M (newest/highest), then 59x 5M (older)
        btc_candles = [self._make_candle(100000000) for _ in range(60)]
        eth_prices_newest_first = [8000000] + [5000000] * 59
        eth_candles = [self._make_candle(p) for p in eth_prices_newest_first]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]
        result = collect_eth_btc_ratio()
        assert result["eth_btc_signal"] == "ETH 극단적 고평가"

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_signal_overvalued(self, mock_api, mock_sleep):
        """z_score between 1 and 2 -> 고평가."""
        eth_ticker = [{"trade_price": 5800000, "signed_change_rate": 0.05,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.01,
                        "acc_trade_price_24h": 5e12}]
        # Newest first: moderately above mean, with spread for larger stdev
        btc_candles = [self._make_candle(100000000) for _ in range(60)]
        eth_prices_newest_first = [5800000] + [5000000 - i * 10000 for i in range(59)]
        eth_candles = [self._make_candle(p) for p in eth_prices_newest_first]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]
        result = collect_eth_btc_ratio()
        assert result["eth_btc_z_score"] > 0
        assert result["eth_btc_signal"] in ("ETH 고평가", "ETH 극단적 고평가")

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_signal_normal(self, mock_api, mock_sleep):
        """z_score near 0 -> 정상 범위."""
        eth_ticker = [{"trade_price": 5000000, "signed_change_rate": 0.0,
                        "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 100000000, "signed_change_rate": 0.0,
                        "acc_trade_price_24h": 5e12}]
        # All same price -> after reversed(), ratio is constant -> z near 0
        btc_candles = [self._make_candle(100000000) for _ in range(60)]
        eth_candles = [self._make_candle(5000000) for _ in range(60)]

        mock_api.side_effect = [eth_ticker, btc_ticker, btc_candles, eth_candles]
        result = collect_eth_btc_ratio()
        assert result["eth_btc_signal"] == "정상 범위"


# ── main() ──────────────────────────────────────────────


class TestMain:
    def _make_candle(self, price, idx=0):
        return {
            "trade_price": price,
            "high_price": price + 100,
            "low_price": price - 100,
            "opening_price": price - 50,
            "candle_acc_trade_volume": 10.0,
            "candle_date_time_kst": f"2026-03-0{min(idx % 9 + 1, 9)}T00:00:00",
        }

    def _make_ticker(self, price=100000000):
        return {
            "trade_price": price,
            "signed_change_rate": 0.01,
            "acc_trade_volume_24h": 5000.0,
        }

    def _make_orderbook(self):
        return {
            "total_bid_size": 100.0,
            "total_ask_size": 80.0,
        }

    def _make_trade(self, ask_bid="BID", volume=0.5):
        return {"ask_bid": ask_bid, "trade_volume": volume}

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_normal_execution(self, mock_api, mock_sleep, mock_eth):
        """All data collected, JSON output to stdout."""
        ticker = [self._make_ticker()]
        daily = [self._make_candle(100000000 + i * 1000, i) for i in range(220)]
        four_h = [self._make_candle(100000000 + i * 100, i) for i in range(42)]
        ob = [self._make_orderbook()]
        trades = [self._make_trade("BID", 0.5)] * 50 + [self._make_trade("ASK", 0.3)] * 50

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {"eth_price": 5000000, "eth_btc_ratio": 0.05}

        import io
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            main("KRW-BTC")

        output = captured.getvalue()
        data = json.loads(output)

        assert data["market"] == "KRW-BTC"
        assert "current_price" in data
        assert "indicators" in data
        assert "indicators_4h" in data
        assert "orderbook" in data
        assert "trade_pressure" in data
        assert "eth_btc_analysis" in data
        assert "daily_summary_5d" in data

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_partial_failure_eth(self, mock_api, mock_sleep, mock_eth):
        """ETH collector returns error -> still outputs valid JSON."""
        ticker = [self._make_ticker()]
        daily = [self._make_candle(100000000 + i * 1000, i) for i in range(220)]
        four_h = [self._make_candle(100000000 + i * 100, i) for i in range(42)]
        ob = [self._make_orderbook()]
        trades = [self._make_trade("BID", 0.5)] * 50 + [self._make_trade("ASK", 0.3)] * 50

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {"error": "ETH 데이터 수집 실패: timeout"}

        import io
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            main("KRW-BTC")

        output = captured.getvalue()
        data = json.loads(output)

        # Should still be valid JSON with an error in eth_btc_analysis
        assert data["eth_btc_analysis"]["error"] == "ETH 데이터 수집 실패: timeout"
        assert "indicators" in data

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_daily_summary_5d_included(self, mock_api, mock_sleep, mock_eth):
        """Verify daily_summary_5d contains last 5 days."""
        ticker = [self._make_ticker()]
        daily = [self._make_candle(100000000 + i * 1000, i) for i in range(220)]
        four_h = [self._make_candle(100000000 + i * 100, i) for i in range(42)]
        ob = [self._make_orderbook()]
        trades = [self._make_trade("BID", 0.5)] * 50 + [self._make_trade("ASK", 0.3)] * 50

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {"eth_price": 5000000}

        import io
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            main("KRW-BTC")

        data = json.loads(captured.getvalue())
        summary = data["daily_summary_5d"]
        assert len(summary) == 5
        for day in summary:
            assert "date" in day
            assert "open" in day
            assert "high" in day
            assert "low" in day
            assert "close" in day
            assert "change_pct" in day
            assert "volume" in day

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_4h_indicators_included(self, mock_api, mock_sleep, mock_eth):
        """Verify 4h candle indicators are calculated."""
        ticker = [self._make_ticker()]
        daily = [self._make_candle(100000000 + i * 1000, i) for i in range(220)]
        four_h = [self._make_candle(100000000 + i * 100, i) for i in range(42)]
        ob = [self._make_orderbook()]
        trades = [self._make_trade("BID", 0.5)] * 50 + [self._make_trade("ASK", 0.3)] * 50

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {"eth_price": 5000000}

        import io
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            main("KRW-BTC")

        data = json.loads(captured.getvalue())
        ind_4h = data["indicators_4h"]
        assert "rsi_14" in ind_4h
        assert "macd" in ind_4h
        assert "stochastic" in ind_4h
        # 42 candles > 14, so rsi should be computed
        assert ind_4h["rsi_14"] is not None
        # 42 candles > 26, so macd should be computed
        assert ind_4h["macd"] is not None
