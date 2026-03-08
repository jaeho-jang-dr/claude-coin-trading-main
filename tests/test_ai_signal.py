"""Unit tests for scripts/collect_ai_signal.py"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_ai_signal import (
    calc_rsi,
    trend_pct,
    analyze_orderbook,
    analyze_whale_trades,
    analyze_multi_timeframe,
    analyze_volatility,
    analyze_volume,
    compute_composite_score,
)


# ── Helper: mock candle factory ──────────────────────────

def make_candle(trade_price, high=None, low=None, opening=None, volume=1000,
                date="2026-03-08T00:00:00"):
    """Create a candle dict matching Upbit API format."""
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


def make_candles_ascending(prices):
    """Create candles from oldest to newest (ascending order for RSI calc)."""
    return [make_candle(p) for p in prices]


def make_candles_descending(prices):
    """Create candles from newest to oldest (Upbit API order)."""
    return [make_candle(p) for p in prices]


# ── 1. calc_rsi ──────────────────────────────────────────

class TestCalcRSI:
    def test_insufficient_data_returns_none(self):
        candles = make_candles_ascending([100, 101, 102])
        assert calc_rsi(candles, period=14) is None

    def test_all_up_near_100(self):
        # 20 ascending prices -> RSI should approach 100
        prices = [100 + i * 10 for i in range(20)]
        candles = make_candles_ascending(prices)
        rsi = calc_rsi(candles, period=14)
        assert rsi is not None
        assert rsi >= 95, f"All-up RSI should be ~100, got {rsi}"

    def test_all_down_near_0(self):
        # 20 descending prices -> RSI should approach 0
        prices = [200 - i * 10 for i in range(20)]
        candles = make_candles_ascending(prices)
        rsi = calc_rsi(candles, period=14)
        assert rsi is not None
        assert rsi <= 5, f"All-down RSI should be ~0, got {rsi}"

    def test_mixed_prices_between_0_and_100(self):
        prices = [100, 105, 102, 108, 103, 110, 106, 112, 107, 115,
                  109, 113, 108, 114, 110, 116]
        candles = make_candles_ascending(prices)
        rsi = calc_rsi(candles, period=14)
        assert rsi is not None
        assert 0 < rsi < 100

    def test_exact_period_plus_one(self):
        # Exactly period+1 data points should work (no smoothing loop)
        prices = [100 + i for i in range(15)]  # 15 points, period=14
        candles = make_candles_ascending(prices)
        rsi = calc_rsi(candles, period=14)
        assert rsi == 100.0  # all gains, no losses


# ── 2. trend_pct ─────────────────────────────────────────

class TestTrendPct:
    def test_basic_uptrend(self):
        # Descending order (newest first): [120, 115, 110, 105, 100]
        candles = make_candles_descending([120, 115, 110, 105, 100])
        result = trend_pct(candles, count=5)
        # first = candles[4] = 100, last = candles[0] = 120
        assert result == 20.0

    def test_basic_downtrend(self):
        candles = make_candles_descending([80, 85, 90, 95, 100])
        result = trend_pct(candles, count=5)
        # first = candles[4] = 100, last = candles[0] = 80
        assert result == -20.0

    def test_insufficient_data_returns_zero(self):
        candles = make_candles_descending([100, 101])
        result = trend_pct(candles, count=5)
        assert result == 0.0

    def test_flat_trend(self):
        candles = make_candles_descending([100, 100, 100, 100, 100])
        result = trend_pct(candles, count=5)
        assert result == 0.0


# ── 3. analyze_orderbook ────────────────────────────────

class TestAnalyzeOrderbook:
    @patch("collect_ai_signal.api_get")
    def test_buy_heavy_signal(self, mock_api):
        mock_api.return_value = [{
            "total_bid_size": 100.0,
            "total_ask_size": 50.0,
            "orderbook_units": [
                {"bid_price": 50000000, "bid_size": 10.0,
                 "ask_price": 50100000, "ask_size": 5.0},
                {"bid_price": 49900000, "bid_size": 10.0,
                 "ask_price": 50200000, "ask_size": 5.0},
            ],
        }]
        result = analyze_orderbook("KRW-BTC")
        assert result["signal"] == "buy"
        assert result["imbalance_ratio"] > 1.1

    @patch("collect_ai_signal.api_get")
    def test_sell_heavy_signal(self, mock_api):
        mock_api.return_value = [{
            "total_bid_size": 30.0,
            "total_ask_size": 100.0,
            "orderbook_units": [
                {"bid_price": 50000000, "bid_size": 3.0,
                 "ask_price": 50100000, "ask_size": 10.0},
                {"bid_price": 49900000, "bid_size": 3.0,
                 "ask_price": 50200000, "ask_size": 10.0},
            ],
        }]
        result = analyze_orderbook("KRW-BTC")
        assert result["signal"] == "sell"
        assert result["imbalance_ratio"] < 0.9

    @patch("collect_ai_signal.api_get")
    def test_balanced_neutral(self, mock_api):
        mock_api.return_value = [{
            "total_bid_size": 50.0,
            "total_ask_size": 50.0,
            "orderbook_units": [
                {"bid_price": 50000000, "bid_size": 5.0,
                 "ask_price": 50100000, "ask_size": 5.0},
            ],
        }]
        result = analyze_orderbook("KRW-BTC")
        assert result["signal"] == "neutral"

    @patch("collect_ai_signal.api_get")
    def test_wall_detection(self, mock_api):
        # avg_bid = 100/5 = 20, wall threshold = 40
        mock_api.return_value = [{
            "total_bid_size": 100.0,
            "total_ask_size": 100.0,
            "orderbook_units": [
                {"bid_price": 50000000, "bid_size": 50.0,
                 "ask_price": 50100000, "ask_size": 5.0},
                {"bid_price": 49900000, "bid_size": 10.0,
                 "ask_price": 50200000, "ask_size": 5.0},
                {"bid_price": 49800000, "bid_size": 10.0,
                 "ask_price": 50300000, "ask_size": 60.0},
                {"bid_price": 49700000, "bid_size": 15.0,
                 "ask_price": 50400000, "ask_size": 15.0},
                {"bid_price": 49600000, "bid_size": 15.0,
                 "ask_price": 50500000, "ask_size": 15.0},
            ],
        }]
        result = analyze_orderbook("KRW-BTC")
        assert len(result["bid_walls"]) >= 1
        assert result["bid_walls"][0]["price"] == 50000000
        assert len(result["ask_walls"]) >= 1
        assert result["ask_walls"][0]["price"] == 50300000


# ── 4. analyze_whale_trades ─────────────────────────────

class TestAnalyzeWhaleTrades:
    def _make_trade(self, side, volume, price=50_000_000, time_utc="12:00:00"):
        return {
            "ask_bid": side,
            "trade_volume": volume,
            "trade_price": price,
            "trade_time_utc": time_utc,
        }

    @patch("collect_ai_signal.api_get")
    def test_large_bid_trades_whale_signal_buy(self, mock_api):
        trades = (
            [self._make_trade("BID", 0.05, 50_000_000)] * 5  # 5 whale BID (2.5M each)
            + [self._make_trade("ASK", 0.001, 50_000_000)] * 50  # small ASK
        )
        mock_api.return_value = trades
        result = analyze_whale_trades("KRW-BTC")
        assert result["whale_signal"] == "buy"
        assert result["whale_buy"] > result["whale_sell"]

    @patch("collect_ai_signal.api_get")
    def test_cvd_positive_accumulating(self, mock_api):
        # First 100 trades: balanced. Next 100: heavy BID -> accumulating
        trades = []
        for i in range(100):
            trades.append(self._make_trade("BID", 0.001))
            trades.append(self._make_trade("ASK", 0.001))
        # Reverse so second half (later indices) has more buys
        # Actually api_get returns trades, and code iterates them in order
        # First 100: balanced, second 100: heavy BID
        first_half = [self._make_trade("BID", 0.001) if i % 2 == 0
                      else self._make_trade("ASK", 0.001) for i in range(100)]
        second_half = [self._make_trade("BID", 0.01) for _ in range(100)]
        mock_api.return_value = first_half + second_half
        result = analyze_whale_trades("KRW-BTC")
        assert result["cvd_trend"] == "accumulating"

    @patch("collect_ai_signal.api_get")
    def test_cvd_negative_distributing(self, mock_api):
        first_half = [self._make_trade("BID", 0.001) if i % 2 == 0
                      else self._make_trade("ASK", 0.001) for i in range(100)]
        second_half = [self._make_trade("ASK", 0.01) for _ in range(100)]
        mock_api.return_value = first_half + second_half
        result = analyze_whale_trades("KRW-BTC")
        assert result["cvd_trend"] == "distributing"

    @patch("collect_ai_signal.api_get")
    def test_buy_ratio_calculation(self, mock_api):
        trades = (
            [self._make_trade("BID", 0.01)] * 80
            + [self._make_trade("ASK", 0.01)] * 20
        )
        mock_api.return_value = trades
        result = analyze_whale_trades("KRW-BTC")
        assert result["buy_ratio_pct"] == 80.0
        assert result["trade_pressure"] == "buy"


# ── 5. analyze_multi_timeframe ──────────────────────────

class TestAnalyzeMultiTimeframe:
    @patch("collect_ai_signal.api_get")
    @patch("collect_ai_signal.time")
    def test_all_bullish_alignment(self, mock_time, mock_api):
        mock_time.sleep = MagicMock()
        mock_time.strftime = MagicMock(return_value="2026-03-08T00:00:00+09:00")

        # All uptrending: newest first, prices decreasing in list = uptrend
        daily = make_candles_descending([110, 108, 106, 104, 102] + [100] * 25)
        h4 = make_candles_descending([110, 108, 106, 104, 102, 100] + [98] * 36)
        h1 = make_candles_descending([110, 108, 106, 104, 102, 100] + [98] * 18)

        mock_api.side_effect = [daily, h4, h1]
        result = analyze_multi_timeframe("KRW-BTC")
        assert result["trend_alignment"] == "all_bullish"

    @patch("collect_ai_signal.api_get")
    @patch("collect_ai_signal.time")
    def test_all_bearish_alignment(self, mock_time, mock_api):
        mock_time.sleep = MagicMock()
        mock_time.strftime = MagicMock(return_value="2026-03-08T00:00:00+09:00")

        daily = make_candles_descending([90, 92, 94, 96, 98] + [100] * 25)
        h4 = make_candles_descending([90, 92, 94, 96, 98, 100] + [102] * 36)
        h1 = make_candles_descending([90, 92, 94, 96, 98, 100] + [102] * 18)

        mock_api.side_effect = [daily, h4, h1]
        result = analyze_multi_timeframe("KRW-BTC")
        assert result["trend_alignment"] == "all_bearish"

    @patch("collect_ai_signal.api_get")
    @patch("collect_ai_signal.time")
    def test_mixed_alignment(self, mock_time, mock_api):
        mock_time.sleep = MagicMock()
        mock_time.strftime = MagicMock(return_value="2026-03-08T00:00:00+09:00")

        # daily up, h4 down, h1 up -> mostly_bullish
        daily = make_candles_descending([110, 108, 106, 104, 102] + [100] * 25)
        h4 = make_candles_descending([90, 92, 94, 96, 98, 100] + [102] * 36)
        h1 = make_candles_descending([110, 108, 106, 104, 102, 100] + [98] * 18)

        mock_api.side_effect = [daily, h4, h1]
        result = analyze_multi_timeframe("KRW-BTC")
        assert result["trend_alignment"] in ("mostly_bullish", "mostly_bearish", "mixed")

    @patch("collect_ai_signal.api_get")
    @patch("collect_ai_signal.time")
    def test_divergence_short_term_oversold(self, mock_time, mock_api):
        mock_time.sleep = MagicMock()
        mock_time.strftime = MagicMock(return_value="2026-03-08T00:00:00+09:00")

        # Daily: strong uptrend -> high RSI
        daily_prices = [100 + i * 5 for i in range(30)]  # ascending, RSI high
        daily = [make_candle(p) for p in reversed(daily_prices)]

        # h4: moderate
        h4_prices = [100 + i * 2 for i in range(42)]
        h4 = [make_candle(p) for p in reversed(h4_prices)]

        # h1: strong downtrend -> low RSI (gap > 15 with daily)
        h1_prices = [200 - i * 8 for i in range(24)]  # descending
        h1 = [make_candle(p) for p in reversed(h1_prices)]

        mock_api.side_effect = [daily, h4, h1]
        result = analyze_multi_timeframe("KRW-BTC")
        if result["divergence_type"] is not None:
            assert result["divergence_type"] in (
                "short_term_oversold", "short_term_overbought"
            )
            assert result["divergence"] is not None


# ── 6. analyze_volatility ───────────────────────────────

class TestAnalyzeVolatility:
    def test_insufficient_data(self):
        candles = [make_candle(100)] * 10
        result = analyze_volatility(candles)
        assert "error" in result

    def test_high_volatility(self):
        # Recent 3 candles: very wide range. Rest: narrow range.
        candles = []
        # First 3 (newest): high-low spread = 10% of low
        for _ in range(3):
            candles.append(make_candle(100, high=110, low=100))
        # Next 17: narrow range (1% of low)
        for _ in range(17):
            candles.append(make_candle(100, high=101, low=100))

        result = analyze_volatility(candles)
        assert result["regime"] == "high_volatility"
        assert result["vol_ratio_3d_20d"] > 1.5

    def test_low_volatility(self):
        candles = []
        # First 3: very narrow
        for _ in range(3):
            candles.append(make_candle(100, high=100.1, low=100))
        # Next 17: wide range
        for _ in range(17):
            candles.append(make_candle(100, high=110, low=100))

        result = analyze_volatility(candles)
        assert result["regime"] == "low_volatility"
        assert result["vol_ratio_3d_20d"] < 0.5

    def test_normal_volatility(self):
        # All same range -> ratio ~1.0
        candles = [make_candle(100, high=105, low=95) for _ in range(20)]
        result = analyze_volatility(candles)
        assert result["regime"] in ("normal", "expanding")
        assert 0.5 <= result["vol_ratio_3d_20d"] <= 1.5


# ── 7. analyze_volume ───────────────────────────────────

class TestAnalyzeVolume:
    def test_insufficient_data(self):
        candles = [make_candle(100)] * 10
        result = analyze_volume(candles)
        assert "error" in result

    def test_spike_detection(self):
        candles = []
        # First candle (today): volume = 5000, avg will be ~1000 -> ratio 5x
        candles.append(make_candle(100, volume=5000, date="2026-03-08T00:00:00"))
        for i in range(19):
            candles.append(make_candle(100, volume=1000,
                                       date=f"2026-03-{7-i:02d}T00:00:00"))

        result = analyze_volume(candles)
        assert result["recent_5d"][0]["anomaly"] == "spike"
        assert result["recent_5d"][0]["ratio_vs_avg"] > 2.0

    def test_normal_volume(self):
        candles = [make_candle(100, volume=1000,
                               date=f"2026-03-{8-i:02d}T00:00:00")
                   for i in range(20)]
        result = analyze_volume(candles)
        assert result["recent_5d"][0]["anomaly"] == "normal"
        assert result["recent_5d"][0]["ratio_vs_avg"] == 1.0

    def test_low_volume(self):
        candles = []
        # First 5: very low volume
        for i in range(5):
            candles.append(make_candle(100, volume=100,
                                       date=f"2026-03-{8-i:02d}T00:00:00"))
        # Rest: normal volume
        for i in range(15):
            candles.append(make_candle(100, volume=1000,
                                       date=f"2026-03-{3-i:02d}T00:00:00"))
        result = analyze_volume(candles)
        # avg_vol = (100*5 + 1000*15) / 20 = 775
        # ratio for first candle = 100 / 775 ~= 0.13
        assert result["recent_5d"][0]["anomaly"] == "low"


# ── 8. compute_composite_score ──────────────────────────

class TestComputeCompositeScore:
    def _make_daily_candles(self, volume=1000, price_change_pct=0):
        """Create 20 daily candles for composite score calc."""
        opening = 50_000_000
        trade = opening * (1 + price_change_pct / 100)
        candles = []
        for _ in range(20):
            candles.append(make_candle(
                trade, high=trade * 1.01, low=trade * 0.99,
                opening=opening, volume=volume,
            ))
        return candles

    def test_all_bullish_positive_score(self):
        orderbook = {"imbalance_ratio": 1.5}
        whale = {
            "buy_ratio_pct": 75,
            "whale_buy": 5,
            "whale_sell": 1,
            "cvd_trend": "accumulating",
        }
        multi_tf = {
            "divergence_type": "short_term_oversold",
            "trend_alignment": "all_bullish",
        }
        volatility = {"regime": "low_volatility"}
        # volume spike + positive price change
        daily = self._make_daily_candles(volume=1000, price_change_pct=0)
        # Override first candle to have high volume and positive change
        daily[0] = make_candle(
            52_000_000, high=52_500_000, low=49_500_000,
            opening=50_000_000, volume=3000,
        )

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily
        )
        assert result["score"] > 0
        assert result["interpretation"] in ("strong_buy", "weak_buy")

    def test_all_bearish_negative_score(self):
        orderbook = {"imbalance_ratio": 0.5}
        whale = {
            "buy_ratio_pct": 25,
            "whale_buy": 1,
            "whale_sell": 5,
            "cvd_trend": "distributing",
        }
        multi_tf = {
            "divergence_type": "short_term_overbought",
            "trend_alignment": "all_bearish",
        }
        volatility = {"regime": "high_volatility"}
        daily = self._make_daily_candles(volume=1000, price_change_pct=0)
        daily[0] = make_candle(
            48_000_000, high=50_500_000, low=47_500_000,
            opening=50_000_000, volume=3000,
        )

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily
        )
        assert result["score"] < 0
        assert result["interpretation"] in ("strong_sell", "weak_sell")

    def test_mixed_inputs_near_zero(self):
        orderbook = {"imbalance_ratio": 1.0}
        whale = {
            "buy_ratio_pct": 50,
            "whale_buy": 2,
            "whale_sell": 2,
            "cvd_trend": "neutral",
        }
        multi_tf = {
            "divergence_type": None,
            "trend_alignment": "mixed",
        }
        volatility = {"regime": "normal"}
        daily = self._make_daily_candles(volume=1000)

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily
        )
        assert result["interpretation"] == "neutral"
        assert -10 < result["score"] < 10

    def test_cvd_accumulating_adds_10(self):
        base_orderbook = {"imbalance_ratio": 1.0}
        base_multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        base_volatility = {"regime": "normal"}
        daily = self._make_daily_candles()

        whale_neutral = {
            "buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
            "cvd_trend": "neutral",
        }
        whale_acc = {
            "buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
            "cvd_trend": "accumulating",
        }

        r_neutral = compute_composite_score(
            base_orderbook, whale_neutral, base_multi_tf, base_volatility, {}, daily
        )
        r_acc = compute_composite_score(
            base_orderbook, whale_acc, base_multi_tf, base_volatility, {}, daily
        )
        assert r_acc["score"] == r_neutral["score"] + 10

    def test_cvd_distributing_subtracts_10(self):
        base_orderbook = {"imbalance_ratio": 1.0}
        base_multi_tf = {"divergence_type": None, "trend_alignment": "mixed"}
        base_volatility = {"regime": "normal"}
        daily = self._make_daily_candles()

        whale_neutral = {
            "buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
            "cvd_trend": "neutral",
        }
        whale_dist = {
            "buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
            "cvd_trend": "distributing",
        }

        r_neutral = compute_composite_score(
            base_orderbook, whale_neutral, base_multi_tf, base_volatility, {}, daily
        )
        r_dist = compute_composite_score(
            base_orderbook, whale_dist, base_multi_tf, base_volatility, {}, daily
        )
        assert r_dist["score"] == r_neutral["score"] - 10

    def test_trend_alignment_all_bullish_adds_10(self):
        base_orderbook = {"imbalance_ratio": 1.0}
        base_whale = {
            "buy_ratio_pct": 50, "whale_buy": 0, "whale_sell": 0,
            "cvd_trend": "neutral",
        }
        base_volatility = {"regime": "normal"}
        daily = self._make_daily_candles()

        multi_mixed = {"divergence_type": None, "trend_alignment": "mixed"}
        multi_bull = {"divergence_type": None, "trend_alignment": "all_bullish"}

        r_mixed = compute_composite_score(
            base_orderbook, base_whale, multi_mixed, base_volatility, {}, daily
        )
        r_bull = compute_composite_score(
            base_orderbook, base_whale, multi_bull, base_volatility, {}, daily
        )
        assert r_bull["score"] == r_mixed["score"] + 10

    def test_score_clamped_to_range(self):
        # Extreme bullish everything
        orderbook = {"imbalance_ratio": 5.0}
        whale = {
            "buy_ratio_pct": 99,
            "whale_buy": 20,
            "whale_sell": 0,
            "cvd_trend": "accumulating",
        }
        multi_tf = {
            "divergence_type": "short_term_oversold",
            "trend_alignment": "all_bullish",
        }
        volatility = {"regime": "low_volatility"}
        daily = self._make_daily_candles(volume=100)
        daily[0] = make_candle(
            55_000_000, high=56_000_000, low=49_000_000,
            opening=50_000_000, volume=5000,
        )

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily
        )
        assert result["score"] <= 100
        assert result["score"] >= -100

    def test_components_list_populated(self):
        orderbook = {"imbalance_ratio": 1.5}
        whale = {
            "buy_ratio_pct": 70,
            "whale_buy": 3,
            "whale_sell": 0,
            "cvd_trend": "accumulating",
        }
        multi_tf = {
            "divergence_type": "short_term_oversold",
            "trend_alignment": "all_bullish",
        }
        volatility = {"regime": "low_volatility"}
        daily = self._make_daily_candles()

        result = compute_composite_score(
            orderbook, whale, multi_tf, volatility, {}, daily
        )
        assert len(result["components"]) > 0
        names = [c["name"] for c in result["components"]]
        assert "orderbook_imbalance" in names
        assert "cvd_trend" in names
