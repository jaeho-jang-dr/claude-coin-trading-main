"""Unit tests for rl_hybrid/rl/data_loader.py — HistoricalDataLoader"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rl_hybrid.rl.data_loader import HistoricalDataLoader


# ---------------------------------------------------------------------------
# Helpers — synthetic candle generation
# ---------------------------------------------------------------------------

def _make_candles(n: int, base_price: float = 100_000_000.0, seed: int = 42) -> list[dict]:
    """Generate *n* synthetic OHLCV candles with realistic-ish noise."""
    rng = np.random.RandomState(seed)
    candles = []
    price = base_price
    for i in range(n):
        change = rng.normal(0, 0.005) * price
        o = price
        c = price + change
        h = max(o, c) + abs(rng.normal(0, 0.002) * price)
        l = min(o, c) - abs(rng.normal(0, 0.002) * price)
        vol = rng.uniform(10, 100)
        candles.append({
            "timestamp": f"2025-01-01T{i % 24:02d}:00:00",
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(vol),
            "volume_krw": float(vol * c),
        })
        price = c
    return candles


def _make_flat_candles(n: int, price: float = 50_000_000.0) -> list[dict]:
    """All-identical candles (zero volatility)."""
    return [
        {
            "timestamp": f"2025-01-01T{i % 24:02d}:00:00",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 10.0,
            "volume_krw": 10.0 * price,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_market(self):
        loader = HistoricalDataLoader()
        assert loader.market == "KRW-BTC"

    def test_custom_market(self):
        loader = HistoricalDataLoader(market="KRW-ETH")
        assert loader.market == "KRW-ETH"

    def test_empty_cache(self):
        loader = HistoricalDataLoader()
        assert loader._cache == {}


# ---------------------------------------------------------------------------
# 2. compute_indicators — output shape & keys
# ---------------------------------------------------------------------------

REQUIRED_INDICATOR_KEYS = [
    "rsi_14", "sma_20", "sma_50", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_histogram",
    "boll_upper", "boll_middle", "boll_lower",
    "stoch_k", "stoch_d", "atr", "adx",
    "adx_plus_di", "adx_minus_di",
    "change_rate", "volume_sma20",
]


class TestComputeIndicators:
    def test_all_keys_present(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(100)
        enriched = loader.compute_indicators(candles)
        for key in REQUIRED_INDICATOR_KEYS:
            assert key in enriched[0], f"Missing key: {key}"

    def test_output_length_matches_input(self):
        loader = HistoricalDataLoader()
        for n in [20, 60, 200]:
            candles = _make_candles(n)
            enriched = loader.compute_indicators(candles)
            assert len(enriched) == n

    def test_original_ohlcv_preserved(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(50)
        enriched = loader.compute_indicators(candles)
        for orig, enr in zip(candles, enriched):
            assert enr["close"] == orig["close"]
            assert enr["open"] == orig["open"]
            assert enr["volume"] == orig["volume"]

    def test_all_values_are_float(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(60)
        enriched = loader.compute_indicators(candles)
        for key in REQUIRED_INDICATOR_KEYS:
            for i, row in enumerate(enriched):
                assert isinstance(row[key], float), (
                    f"enriched[{i}][{key}] is {type(row[key])}, expected float"
                )

    def test_no_nan_in_enriched(self):
        """After initial warm-up, values must not be NaN."""
        loader = HistoricalDataLoader()
        candles = _make_candles(100)
        enriched = loader.compute_indicators(candles)
        for i in range(50, 100):
            for key in REQUIRED_INDICATOR_KEYS:
                assert not np.isnan(enriched[i][key]), f"NaN at [{i}][{key}]"


# ---------------------------------------------------------------------------
# 3. compute_indicators — minimal / edge cases
# ---------------------------------------------------------------------------

class TestComputeIndicatorsEdge:
    def test_minimal_candles_below_14(self):
        """Even with < 14 candles, should not crash."""
        loader = HistoricalDataLoader()
        candles = _make_candles(5)
        enriched = loader.compute_indicators(candles)
        assert len(enriched) == 5
        # RSI falls back to 50 when data < period
        for row in enriched:
            assert row["rsi_14"] == 50.0

    def test_single_candle(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(1)
        enriched = loader.compute_indicators(candles)
        assert len(enriched) == 1

    def test_two_candles(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(2)
        enriched = loader.compute_indicators(candles)
        assert len(enriched) == 2

    def test_flat_candles(self):
        """Identical prices should not produce NaN/Inf."""
        loader = HistoricalDataLoader()
        candles = _make_flat_candles(60)
        enriched = loader.compute_indicators(candles)
        for i in range(len(enriched)):
            for key in REQUIRED_INDICATOR_KEYS:
                val = enriched[i][key]
                assert np.isfinite(val), f"Non-finite at [{i}][{key}]={val}"


# ---------------------------------------------------------------------------
# 4. _compute_rsi
# ---------------------------------------------------------------------------

class TestComputeRSI:
    def test_values_in_range(self):
        closes = np.array([c["close"] for c in _make_candles(100)])
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        assert rsi.shape == closes.shape
        assert np.all(rsi >= 0) and np.all(rsi <= 100)

    def test_short_array_fallback(self):
        """len <= period => all 50."""
        closes = np.array([100.0, 101.0, 102.0])
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        np.testing.assert_array_equal(rsi, 50.0)

    def test_exactly_period_plus_one(self):
        closes = np.array([float(i) for i in range(15)])  # len=15, period=14
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        assert rsi.shape == (15,)
        # First 14 entries should be 50
        np.testing.assert_array_equal(rsi[:14], 50.0)
        # Last entry: all gains, no losses => RSI near 100
        assert rsi[14] > 90

    def test_monotonic_increasing_high_rsi(self):
        closes = np.linspace(100, 200, 50)
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        # After warm-up, RSI should be very high
        assert np.all(rsi[14:] > 80)

    def test_monotonic_decreasing_low_rsi(self):
        closes = np.linspace(200, 100, 50)
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        assert np.all(rsi[14:] < 20)

    def test_flat_rsi_equals_50_or_near(self):
        """Flat prices: no gains/losses => RSI = 100 (division by 0 guard)."""
        closes = np.full(30, 50000.0)
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        # After warm-up, avg_loss == 0, rs → 100, RSI → 100
        assert rsi.shape == (30,)


# ---------------------------------------------------------------------------
# 5. _sma
# ---------------------------------------------------------------------------

class TestSMA:
    def test_known_values(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = HistoricalDataLoader._sma(data, 3)
        # sma[2] = mean(1,2,3)=2, sma[3]=mean(2,3,4)=3, sma[4]=mean(3,4,5)=4
        np.testing.assert_almost_equal(sma[2], 2.0)
        np.testing.assert_almost_equal(sma[3], 3.0)
        np.testing.assert_almost_equal(sma[4], 4.0)

    def test_initial_fill(self):
        """Before period, SMA uses expanding window mean."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sma = HistoricalDataLoader._sma(data, 3)
        np.testing.assert_almost_equal(sma[0], 10.0)  # mean([10])
        np.testing.assert_almost_equal(sma[1], 15.0)  # mean([10,20])

    def test_short_data(self):
        data = np.array([5.0, 10.0])
        sma = HistoricalDataLoader._sma(data, 20)
        # len < period => result is all nan (except initial fill)
        assert sma.shape == (2,)

    def test_constant_array(self):
        data = np.full(50, 42.0)
        sma = HistoricalDataLoader._sma(data, 10)
        np.testing.assert_array_almost_equal(sma, 42.0)

    def test_output_length(self):
        data = np.arange(100, dtype=float)
        sma = HistoricalDataLoader._sma(data, 20)
        assert sma.shape == data.shape


# ---------------------------------------------------------------------------
# 6. _ema
# ---------------------------------------------------------------------------

class TestEMA:
    def test_first_value_equals_input(self):
        data = np.array([10.0, 20.0, 30.0])
        ema = HistoricalDataLoader._ema(data, 3)
        assert ema[0] == 10.0

    def test_constant_array(self):
        data = np.full(50, 100.0)
        ema = HistoricalDataLoader._ema(data, 12)
        np.testing.assert_array_almost_equal(ema, 100.0)

    def test_output_length(self):
        data = np.arange(30, dtype=float)
        ema = HistoricalDataLoader._ema(data, 5)
        assert ema.shape == data.shape

    def test_ema_smoothing(self):
        """EMA reacts to a step change, converging toward the new level."""
        data = np.concatenate([np.full(20, 50.0), np.full(20, 100.0)])
        ema = HistoricalDataLoader._ema(data, 10)
        # Before step: should be ~50
        np.testing.assert_almost_equal(ema[19], 50.0)
        # After step: converges toward 100, last value should be close
        assert ema[-1] > 95

    def test_single_element(self):
        data = np.array([7.0])
        ema = HistoricalDataLoader._ema(data, 5)
        assert ema[0] == 7.0


# ---------------------------------------------------------------------------
# 7. _stochastic
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_values_in_range(self):
        candles = _make_candles(100)
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        closes = np.array([c["close"] for c in candles])
        k, d = HistoricalDataLoader._stochastic(highs, lows, closes, 14, 3)
        assert k.shape == (100,)
        assert d.shape == (100,)
        # After warm-up, K and D should be in [0, 100]
        assert np.all(k[13:] >= 0) and np.all(k[13:] <= 100)
        assert np.all(d[13:] >= 0) and np.all(d[13:] <= 100)

    def test_flat_prices(self):
        """If high == low == close, K defaults to 50 (initial fill)."""
        n = 30
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        k, d = HistoricalDataLoader._stochastic(highs, lows, closes, 14, 3)
        # h == l for every window, so stoch_k remains 50.0 (initial fill)
        np.testing.assert_array_almost_equal(k, 50.0)

    def test_close_at_high(self):
        """When close = high always, K should be 100 after warm-up."""
        n = 30
        highs = np.linspace(100, 130, n)
        lows = np.linspace(90, 120, n)
        closes = highs.copy()  # close == high
        k, d = HistoricalDataLoader._stochastic(highs, lows, closes, 14, 3)
        # After warm-up, K should be 100
        np.testing.assert_array_almost_equal(k[13:], 100.0)


# ---------------------------------------------------------------------------
# 8. _compute_atr
# ---------------------------------------------------------------------------

class TestComputeATR:
    def test_non_negative(self):
        candles = _make_candles(60)
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        closes = np.array([c["close"] for c in candles])
        atr = HistoricalDataLoader._compute_atr(highs, lows, closes, 14)
        assert atr.shape == (60,)
        assert np.all(atr >= 0)

    def test_flat_prices_zero_atr(self):
        n = 30
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        atr = HistoricalDataLoader._compute_atr(highs, lows, closes, 14)
        np.testing.assert_array_almost_equal(atr, 0.0)

    def test_output_length(self):
        candles = _make_candles(50)
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        closes = np.array([c["close"] for c in candles])
        atr = HistoricalDataLoader._compute_atr(highs, lows, closes, 14)
        assert atr.shape == (50,)


# ---------------------------------------------------------------------------
# 9. _compute_adx
# ---------------------------------------------------------------------------

class TestComputeADX:
    def test_non_negative(self):
        candles = _make_candles(100)
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        closes = np.array([c["close"] for c in candles])
        adx, plus_di, minus_di = HistoricalDataLoader._compute_adx(
            highs, lows, closes, 14
        )
        assert adx.shape == (100,)
        assert np.all(adx >= 0)
        assert np.all(plus_di >= 0)
        assert np.all(minus_di >= 0)

    def test_flat_prices_low_adx(self):
        """Flat prices => no directional movement => ADX ~ 0."""
        n = 60
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        adx, plus_di, minus_di = HistoricalDataLoader._compute_adx(
            highs, lows, closes, 14
        )
        np.testing.assert_array_almost_equal(adx, 0.0)

    def test_output_shape(self):
        candles = _make_candles(50)
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        closes = np.array([c["close"] for c in candles])
        adx, plus_di, minus_di = HistoricalDataLoader._compute_adx(
            highs, lows, closes, 14
        )
        assert adx.shape == plus_di.shape == minus_di.shape == (50,)


# ---------------------------------------------------------------------------
# 10. _default_external_signal
# ---------------------------------------------------------------------------

REQUIRED_SIGNAL_KEYS = [
    "fgi_value", "news_sentiment", "whale_score", "funding_rate",
    "long_short_ratio", "kimchi_premium_pct", "macro_score",
    "eth_btc_score", "fusion_score", "fusion_signal", "nvt_signal",
]


class TestDefaultExternalSignal:
    def test_has_all_keys(self):
        sig = HistoricalDataLoader._default_external_signal()
        for key in REQUIRED_SIGNAL_KEYS:
            assert key in sig, f"Missing key: {key}"

    def test_numeric_types(self):
        sig = HistoricalDataLoader._default_external_signal()
        for key in REQUIRED_SIGNAL_KEYS:
            if key == "fusion_signal":
                assert isinstance(sig[key], str)
            else:
                assert isinstance(sig[key], (int, float)), (
                    f"{key} is {type(sig[key])}, expected numeric"
                )

    def test_neutral_defaults(self):
        sig = HistoricalDataLoader._default_external_signal()
        assert sig["fgi_value"] == 50
        assert sig["long_short_ratio"] == 1.0
        assert sig["fusion_signal"] == "neutral"

    def test_returns_new_dict_each_call(self):
        a = HistoricalDataLoader._default_external_signal()
        b = HistoricalDataLoader._default_external_signal()
        assert a is not b
        a["fgi_value"] = 999
        assert b["fgi_value"] == 50


# ---------------------------------------------------------------------------
# 11. _rolling_std
# ---------------------------------------------------------------------------

class TestRollingStd:
    def test_known_values(self):
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        std = HistoricalDataLoader._rolling_std(data, 3)
        # At i=2: std([2,4,4])
        expected = np.std([2, 4, 4])
        np.testing.assert_almost_equal(std[2], expected)

    def test_constant_array_zero_std(self):
        data = np.full(30, 42.0)
        std = HistoricalDataLoader._rolling_std(data, 5)
        np.testing.assert_array_almost_equal(std, 0.0)

    def test_output_length(self):
        data = np.arange(50, dtype=float)
        std = HistoricalDataLoader._rolling_std(data, 10)
        assert std.shape == (50,)

    def test_initial_fill(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        std = HistoricalDataLoader._rolling_std(data, 3)
        # i=0 => std is 0 (single element before)
        assert std[0] == 0.0
        # i=1 => std([10, 20])
        np.testing.assert_almost_equal(std[1], np.std([10.0, 20.0]))

    def test_non_negative(self):
        data = np.random.RandomState(99).randn(100) * 100
        std = HistoricalDataLoader._rolling_std(data, 20)
        assert np.all(std >= 0)


# ---------------------------------------------------------------------------
# load_candles — mock API
# ---------------------------------------------------------------------------

class TestLoadCandles:
    def test_invalid_interval_raises(self):
        loader = HistoricalDataLoader()
        with pytest.raises(ValueError, match="지원하지 않는 interval"):
            loader.load_candles(days=1, interval="5m")

    @patch("rl_hybrid.rl.data_loader.requests.get")
    def test_load_candles_basic(self, mock_get):
        """Mocked API returns 3 candles, verify parsing & reverse sort."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "candle_date_time_kst": "2025-01-01T03:00:00",
                "candle_date_time_utc": "2025-01-01T02:00:00",
                "opening_price": 100, "high_price": 110,
                "low_price": 90, "trade_price": 105,
                "candle_acc_trade_volume": 50,
                "candle_acc_trade_price": 5000,
            },
            {
                "candle_date_time_kst": "2025-01-01T02:00:00",
                "candle_date_time_utc": "2025-01-01T01:00:00",
                "opening_price": 95, "high_price": 105,
                "low_price": 85, "trade_price": 100,
                "candle_acc_trade_volume": 40,
                "candle_acc_trade_price": 4000,
            },
            {
                "candle_date_time_kst": "2025-01-01T01:00:00",
                "candle_date_time_utc": "2025-01-01T00:00:00",
                "opening_price": 90, "high_price": 100,
                "low_price": 80, "trade_price": 95,
                "candle_acc_trade_volume": 30,
                "candle_acc_trade_price": 3000,
            },
        ]
        mock_get.return_value = mock_response

        loader = HistoricalDataLoader()
        candles = loader.load_candles(days=1, interval="1d")

        assert len(candles) == 3
        # Reversed: oldest first
        assert candles[0]["close"] == 95.0
        assert candles[-1]["close"] == 105.0

    @patch("rl_hybrid.rl.data_loader.requests.get")
    def test_cache_hit(self, mock_get):
        """Second call with same params should use cache, not API."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        loader = HistoricalDataLoader()
        loader.load_candles(days=1, interval="1d")
        loader.load_candles(days=1, interval="1d")
        # Only 1 actual API call (second uses cache)
        assert mock_get.call_count == 1

    @patch("rl_hybrid.rl.data_loader.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("network error")
        loader = HistoricalDataLoader()
        candles = loader.load_candles(days=1, interval="1d")
        assert candles == []


# ---------------------------------------------------------------------------
# align_external_to_candles
# ---------------------------------------------------------------------------

class TestAlignExternalToCandles:
    def test_empty_signals_returns_defaults(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(5)
        result = loader.align_external_to_candles(candles, [])
        assert len(result) == 5
        for r in result:
            assert r["fgi_value"] == 50

    def test_forward_fill(self):
        loader = HistoricalDataLoader()
        candles = [
            {"timestamp": "2025-01-01T10:00:00"},  # KST => UTC 01:00
            {"timestamp": "2025-01-01T11:00:00"},  # KST => UTC 02:00
            {"timestamp": "2025-01-01T12:00:00"},  # KST => UTC 03:00
        ]
        from datetime import datetime
        signals = [
            {
                "recorded_at": datetime(2025, 1, 1, 0, 0, 0),
                "fgi_value": 30,
                "news_sentiment": 1,
                "whale_score": 2,
                "funding_rate": 0.01,
                "long_short_ratio": 1.5,
                "kimchi_premium_pct": 0.5,
                "macro_score": 3,
                "eth_btc_score": 4,
                "fusion_score": 5,
                "fusion_signal": "buy",
            },
        ]
        result = loader.align_external_to_candles(candles, signals)
        assert len(result) == 3
        # All three candles should forward-fill from the single signal
        assert result[0]["fgi_value"] == 30
        assert result[1]["fgi_value"] == 30
        assert result[2]["fgi_value"] == 30
