"""
collect_market_data.py comprehensive unit tests.

Covers:
  - Technical indicator calculations (SMA, EMA, RSI, MACD, Bollinger, Stochastic, ADX, ATR)
  - Edge cases: empty lists, None returns, insufficient data
  - api_get with retry and backoff
  - collect_eth_btc_ratio signals
  - main() JSON output structure

All network calls are mocked - no real API calls are made.
"""

import json
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


# ══════════════════════════════════════════════════════════════
# SMA
# ══════════════════════════════════════════════════════════════

class TestSMA:
    def test_basic_average(self):
        assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 5) == 3.0

    def test_uses_last_n_elements(self):
        assert sma([10.0, 20.0, 30.0, 40.0, 50.0], 3) == 40.0

    def test_returns_none_when_insufficient_data(self):
        assert sma([1.0, 2.0], 5) is None

    def test_returns_none_for_empty_list(self):
        assert sma([], 5) is None

    def test_single_element_period_1(self):
        assert sma([42.0], 1) == 42.0

    def test_all_same_values(self):
        assert sma([100.0] * 20, 10) == 100.0

    def test_known_20_period(self):
        prices = [float(i) for i in range(1, 21)]
        assert sma(prices, 20) == 10.5


# ══════════════════════════════════════════════════════════════
# EMA
# ══════════════════════════════════════════════════════════════

class TestEMA:
    def test_single_element_returns_itself(self):
        assert ema([50.0], 10) == 50.0

    def test_constant_prices(self):
        assert ema([50.0] * 20, 10) == pytest.approx(50.0)

    def test_manual_period_3(self):
        prices = [10.0, 20.0, 30.0, 40.0]
        k = 0.5  # 2/(3+1)
        v = 10.0
        v = 20.0 * k + v * (1 - k)
        v = 30.0 * k + v * (1 - k)
        v = 40.0 * k + v * (1 - k)
        assert ema(prices, 3) == pytest.approx(v)

    def test_ema_weighted_toward_recent(self):
        """EMA should be closer to recent prices than SMA."""
        prices = [100.0] * 10 + [200.0] * 10
        ema_val = ema(prices, 10)
        sma_val = sma(prices, 20)
        # EMA should be higher than SMA because it weights recent 200s more
        assert ema_val > sma_val


# ══════════════════════════════════════════════════════════════
# RSI
# ══════════════════════════════════════════════════════════════

class TestRSI:
    def test_insufficient_data_returns_50(self):
        assert rsi([1.0, 2.0, 3.0], 14) == 50.0

    def test_all_gains_returns_100(self):
        prices = [float(i) for i in range(20)]
        assert rsi(prices, 14) == 100.0

    def test_all_losses_returns_near_0(self):
        prices = [float(20 - i) for i in range(20)]
        assert rsi(prices, 14) == pytest.approx(0.0, abs=0.01)

    def test_flat_prices_returns_100(self):
        # No losses → al = 0 → RSI = 100
        assert rsi([100.0] * 20, 14) == 100.0

    def test_alternating_stays_near_50(self):
        prices = [100.0]
        for i in range(1, 30):
            prices.append(prices[-1] + (1 if i % 2 else -1))
        result = rsi(prices, 14)
        assert 40.0 <= result <= 60.0

    def test_always_in_range_0_100(self):
        import random
        random.seed(42)
        prices = [100.0]
        for _ in range(100):
            prices.append(prices[-1] + random.uniform(-10, 10))
        result = rsi(prices, 14)
        assert 0.0 <= result <= 100.0

    def test_exactly_period_plus_one(self):
        """Exactly period+1 data points — only initial average, no smoothing."""
        prices = list(range(15))
        result = rsi([float(p) for p in prices], 14)
        assert 0.0 <= result <= 100.0


# ══════════════════════════════════════════════════════════════
# MACD
# ══════════════════════════════════════════════════════════════

class TestMACD:
    def test_insufficient_data(self):
        result = macd([float(i) for i in range(25)])
        assert result == {"macd": 0, "signal": 0, "histogram": 0}

    def test_histogram_is_macd_minus_signal(self):
        prices = [100 + i * 0.5 for i in range(50)]
        result = macd(prices)
        assert result["histogram"] == pytest.approx(
            result["macd"] - result["signal"], abs=0.01
        )

    def test_constant_prices_zero_macd(self):
        result = macd([100.0] * 30)
        assert result["macd"] == pytest.approx(0.0, abs=0.01)
        assert result["signal"] == pytest.approx(0.0, abs=0.01)

    def test_uptrend_positive_macd(self):
        prices = [100.0 + i * 2.0 for i in range(60)]
        result = macd(prices)
        assert result["macd"] > 0

    def test_downtrend_negative_macd(self):
        prices = [200.0 - i * 2.0 for i in range(60)]
        result = macd(prices)
        assert result["macd"] < 0

    def test_exactly_26_prices(self):
        result = macd([float(i) for i in range(26)])
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result


# ══════════════════════════════════════════════════════════════
# Bollinger Bands
# ══════════════════════════════════════════════════════════════

class TestBollinger:
    def test_insufficient_data(self):
        result = bollinger([1.0, 2.0], 20)
        assert result == {"upper": 0, "middle": 0, "lower": 0}

    def test_constant_prices_bands_collapse(self):
        result = bollinger([100.0] * 20, 20)
        assert result["upper"] == pytest.approx(100.0)
        assert result["middle"] == pytest.approx(100.0)
        assert result["lower"] == pytest.approx(100.0)

    def test_bands_symmetry(self):
        prices = list(range(1, 25))
        result = bollinger([float(p) for p in prices], 20)
        mid = result["middle"]
        assert result["upper"] - mid == pytest.approx(mid - result["lower"], abs=0.01)

    def test_upper_ge_middle_ge_lower(self):
        prices = [100 + i for i in range(25)]
        result = bollinger([float(p) for p in prices], 20)
        assert result["upper"] >= result["middle"] >= result["lower"]

    def test_formula_verification(self):
        prices = [10.0, 12.0, 11.0, 13.0, 14.0]
        period = 5
        result = bollinger(prices, period)
        mid = sum(prices) / period
        var = sum((p - mid) ** 2 for p in prices) / period
        sd = var ** 0.5
        assert result["middle"] == pytest.approx(mid, abs=0.01)
        assert result["upper"] == pytest.approx(mid + 2 * sd, abs=0.01)
        assert result["lower"] == pytest.approx(mid - 2 * sd, abs=0.01)


# ══════════════════════════════════════════════════════════════
# Stochastic
# ══════════════════════════════════════════════════════════════

class TestStochastic:
    def test_insufficient_data(self):
        result = stochastic([10.0] * 15, [9.0] * 15, [9.5] * 15, 14)
        assert result == {"k": 50.0, "d": 50.0}

    def test_close_at_high_returns_100(self):
        n = 20
        result = stochastic([110.0] * n, [90.0] * n, [110.0] * n, 14)
        assert result["k"] == pytest.approx(100.0)

    def test_close_at_low_returns_0(self):
        n = 20
        result = stochastic([110.0] * n, [90.0] * n, [90.0] * n, 14)
        assert result["k"] == pytest.approx(0.0)

    def test_midrange_returns_50(self):
        n = 20
        result = stochastic([110.0] * n, [90.0] * n, [100.0] * n, 14)
        assert result["k"] == pytest.approx(50.0)
        assert result["d"] == pytest.approx(50.0)

    def test_k_and_d_in_range(self):
        import random
        random.seed(123)
        n = 30
        closes = [100.0]
        for _ in range(n - 1):
            closes.append(closes[-1] + random.uniform(-3, 3))
        highs = [c + abs(random.gauss(0, 2)) for c in closes]
        lows = [c - abs(random.gauss(0, 2)) for c in closes]
        result = stochastic(highs, lows, closes, 14)
        assert 0.0 <= result["k"] <= 100.0
        assert 0.0 <= result["d"] <= 100.0


# ══════════════════════════════════════════════════════════════
# ADX
# ══════════════════════════════════════════════════════════════

class TestCalcADX:
    def test_insufficient_data(self):
        result = calc_adx([10.0] * 10, [9.0] * 10, [9.5] * 10, 14)
        assert result["regime"] == "unknown"

    def test_strong_uptrend(self):
        n = 80
        highs = [100.0 + i * 2.0 for i in range(n)]
        lows = [98.0 + i * 2.0 for i in range(n)]
        closes = [99.0 + i * 2.0 for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        assert result["plus_di"] > result["minus_di"]
        assert result["regime"] in ("trending", "transitioning")

    def test_strong_downtrend(self):
        n = 80
        highs = [200.0 - i * 2.0 for i in range(n)]
        lows = [198.0 - i * 2.0 for i in range(n)]
        closes = [199.0 - i * 2.0 for i in range(n)]
        result = calc_adx(highs, lows, closes, 14)
        assert result["minus_di"] > result["plus_di"]

    def test_output_keys(self):
        n = 40
        result = calc_adx(
            [100.0 + i for i in range(n)],
            [98.0 + i for i in range(n)],
            [99.0 + i for i in range(n)],
            14,
        )
        assert set(result.keys()) == {"adx", "plus_di", "minus_di", "regime"}


# ══════════════════════════════════════════════════════════════
# ATR
# ══════════════════════════════════════════════════════════════

class TestCalcATR:
    def test_insufficient_data(self):
        assert calc_atr([10.0] * 5, [9.0] * 5, [9.5] * 5, 14) == 0.0

    def test_constant_range(self):
        n = 20
        result = calc_atr([110.0] * n, [100.0] * n, [105.0] * n, 14)
        assert result == pytest.approx(10.0, abs=0.1)

    def test_always_positive(self):
        import random
        random.seed(99)
        n = 30
        closes = [100.0]
        for _ in range(n - 1):
            closes.append(closes[-1] + random.uniform(-2, 2))
        highs = [c + abs(random.gauss(0, 2)) for c in closes]
        lows = [c - abs(random.gauss(0, 2)) for c in closes]
        assert calc_atr(highs, lows, closes, 14) > 0


# ══════════════════════════════════════════════════════════════
# api_get retry logic
# ══════════════════════════════════════════════════════════════

class TestApiGet:
    @patch("collect_market_data._get_session")
    def test_success_first_try(self, mock_session_fn):
        session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"price": 100}
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp
        mock_session_fn.return_value = session

        result = api_get("/ticker", {"markets": "KRW-BTC"})
        assert result == {"price": 100}

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data._get_session")
    def test_429_retry_then_success(self, mock_session_fn, mock_sleep):
        session = MagicMock()
        mock_429 = MagicMock()
        mock_429.status_code = 429

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"ok": True}
        mock_ok.raise_for_status.return_value = None

        session.get.side_effect = [mock_429, mock_ok]
        mock_session_fn.return_value = session

        result = api_get("/ticker", max_retries=3)
        assert result == {"ok": True}
        mock_sleep.assert_called_once_with(1)

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data._get_session")
    def test_429_all_retries_exhausted(self, mock_session_fn, mock_sleep):
        session = MagicMock()
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.raise_for_status.side_effect = Exception("429")
        session.get.return_value = mock_429
        mock_session_fn.return_value = session

        with pytest.raises(Exception, match="429"):
            api_get("/ticker", max_retries=3)
        assert session.get.call_count == 3

    @patch("collect_market_data._get_session")
    def test_500_raises_immediately(self, mock_session_fn):
        session = MagicMock()
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.raise_for_status.side_effect = Exception("500 Server Error")
        session.get.return_value = mock_500
        mock_session_fn.return_value = session

        with pytest.raises(Exception, match="500"):
            api_get("/ticker", max_retries=3)
        assert session.get.call_count == 1

    @patch("collect_market_data._get_session")
    def test_params_appended_to_url(self, mock_session_fn):
        session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp
        mock_session_fn.return_value = session

        api_get("/ticker", {"markets": "KRW-BTC", "count": "10"})
        url = session.get.call_args[0][0]
        assert "markets=KRW-BTC" in url
        assert "count=10" in url


# ══════════════════════════════════════════════════════════════
# collect_eth_btc_ratio
# ══════════════════════════════════════════════════════════════

class TestCollectEthBtcRatio:
    def _candle(self, price):
        return {
            "trade_price": price,
            "high_price": price + 10,
            "low_price": price - 10,
            "opening_price": price,
            "candle_acc_trade_volume": 1.0,
            "candle_date_time_kst": "2026-03-08T00:00:00",
        }

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_returns_expected_keys(self, mock_api, mock_sleep):
        eth_ticker = [{"trade_price": 5e6, "signed_change_rate": 0.02, "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 1e8, "signed_change_rate": 0.01, "acc_trade_price_24h": 5e12}]
        btc_daily = [self._candle(1e8 + i * 1000) for i in range(60)]
        eth_daily = [self._candle(5e6 + i * 500) for i in range(60)]
        mock_api.side_effect = [eth_ticker, btc_ticker, btc_daily, eth_daily]

        result = collect_eth_btc_ratio()
        expected_keys = {
            "eth_price", "eth_change_24h", "eth_volume_24h_krw",
            "eth_rsi_14", "eth_btc_ratio", "eth_btc_ratio_avg60",
            "eth_btc_ratio_min60", "eth_btc_ratio_max60",
            "eth_btc_z_score", "eth_btc_signal",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert "error" not in result

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_api_failure_returns_error(self, mock_api, mock_sleep):
        mock_api.side_effect = Exception("network error")
        result = collect_eth_btc_ratio()
        assert "error" in result

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_constant_ratio_z_score_zero(self, mock_api, mock_sleep):
        """All same prices -> z_score ~0, signal '정상 범위'."""
        eth_ticker = [{"trade_price": 5e6, "signed_change_rate": 0.0, "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 1e8, "signed_change_rate": 0.0, "acc_trade_price_24h": 5e12}]
        btc_daily = [self._candle(1e8) for _ in range(60)]
        eth_daily = [self._candle(5e6) for _ in range(60)]
        mock_api.side_effect = [eth_ticker, btc_ticker, btc_daily, eth_daily]

        result = collect_eth_btc_ratio()
        assert result["eth_btc_z_score"] == pytest.approx(0.0, abs=0.1)
        assert result["eth_btc_signal"] == "정상 범위"

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_extreme_undervaluation(self, mock_api, mock_sleep):
        eth_ticker = [{"trade_price": 3e6, "signed_change_rate": -0.1, "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 1e8, "signed_change_rate": 0.01, "acc_trade_price_24h": 5e12}]
        btc_daily = [self._candle(1e8) for _ in range(60)]
        eth_daily = [self._candle(3e6)] + [self._candle(5e6)] * 59  # newest first, reversed in code
        mock_api.side_effect = [eth_ticker, btc_ticker, btc_daily, eth_daily]

        result = collect_eth_btc_ratio()
        assert result["eth_btc_signal"] == "ETH 극단적 저평가"

    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_extreme_overvaluation(self, mock_api, mock_sleep):
        eth_ticker = [{"trade_price": 8e6, "signed_change_rate": 0.2, "acc_trade_price_24h": 1e12}]
        btc_ticker = [{"trade_price": 1e8, "signed_change_rate": 0.01, "acc_trade_price_24h": 5e12}]
        btc_daily = [self._candle(1e8) for _ in range(60)]
        eth_daily = [self._candle(8e6)] + [self._candle(5e6)] * 59
        mock_api.side_effect = [eth_ticker, btc_ticker, btc_daily, eth_daily]

        result = collect_eth_btc_ratio()
        assert result["eth_btc_signal"] == "ETH 극단적 고평가"


# ══════════════════════════════════════════════════════════════
# main() - full pipeline JSON output
# ══════════════════════════════════════════════════════════════

class TestMain:
    def _candle(self, price, idx=0):
        return {
            "trade_price": price,
            "high_price": price + 100,
            "low_price": price - 100,
            "opening_price": price - 50,
            "candle_acc_trade_volume": 10.0,
            "candle_date_time_kst": f"2026-03-0{min(idx % 9 + 1, 9)}T00:00:00",
        }

    def _ticker(self, price=1e8):
        return {
            "trade_price": price,
            "signed_change_rate": 0.01,
            "acc_trade_volume_24h": 5000.0,
        }

    def _orderbook(self):
        return {"total_bid_size": 100.0, "total_ask_size": 80.0}

    def _trade(self, side="BID", vol=0.5):
        return {"ask_bid": side, "trade_volume": vol}

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_output_has_all_top_level_keys(self, mock_api, mock_sleep, mock_eth):
        ticker = [self._ticker()]
        daily = [self._candle(1e8 + i * 1000, i) for i in range(220)]
        four_h = [self._candle(1e8 + i * 100, i) for i in range(42)]
        ob = [self._orderbook()]
        trades = [self._trade("BID")] * 50 + [self._trade("ASK")] * 50

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {"eth_price": 5e6}

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            main("KRW-BTC")

        data = json.loads(buf.getvalue())
        for key in [
            "timestamp", "market", "current_price", "change_rate_24h",
            "volume_24h", "indicators", "indicators_4h", "orderbook",
            "trade_pressure", "eth_btc_analysis", "daily_summary_5d",
        ]:
            assert key in data, f"Missing key: {key}"

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_indicators_keys(self, mock_api, mock_sleep, mock_eth):
        ticker = [self._ticker()]
        daily = [self._candle(1e8 + i * 1000, i) for i in range(220)]
        four_h = [self._candle(1e8 + i * 100, i) for i in range(42)]
        ob = [self._orderbook()]
        trades = [self._trade()] * 100

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {}

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            main("KRW-BTC")

        ind = json.loads(buf.getvalue())["indicators"]
        for key in ["sma_20", "sma_50", "sma_200", "ema_10", "ema_50", "ema_200",
                     "rsi_14", "macd", "bollinger", "stochastic", "adx", "atr"]:
            assert key in ind, f"Missing indicator: {key}"

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_daily_summary_has_5_entries(self, mock_api, mock_sleep, mock_eth):
        ticker = [self._ticker()]
        daily = [self._candle(1e8 + i * 1000, i) for i in range(220)]
        four_h = [self._candle(1e8 + i * 100, i) for i in range(42)]
        ob = [self._orderbook()]
        trades = [self._trade()] * 100

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {}

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            main("KRW-BTC")

        summary = json.loads(buf.getvalue())["daily_summary_5d"]
        assert len(summary) == 5
        for day in summary:
            for k in ["date", "open", "high", "low", "close", "change_pct", "volume"]:
                assert k in day

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_trade_pressure_buy_sell(self, mock_api, mock_sleep, mock_eth):
        ticker = [self._ticker()]
        daily = [self._candle(1e8, i) for i in range(220)]
        four_h = [self._candle(1e8, i) for i in range(42)]
        ob = [self._orderbook()]
        trades = ([self._trade("BID", 0.5)] * 60 +
                  [self._trade("ASK", 0.3)] * 40)

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {}

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            main("KRW-BTC")

        tp = json.loads(buf.getvalue())["trade_pressure"]
        assert tp["buy_volume"] == pytest.approx(60 * 0.5)
        assert tp["sell_volume"] == pytest.approx(40 * 0.3)

    @patch("collect_market_data.collect_eth_btc_ratio")
    @patch("collect_market_data.time.sleep")
    @patch("collect_market_data.api_get")
    def test_orderbook_ratio(self, mock_api, mock_sleep, mock_eth):
        ticker = [self._ticker()]
        daily = [self._candle(1e8, i) for i in range(220)]
        four_h = [self._candle(1e8, i) for i in range(42)]
        ob = [{"total_bid_size": 200.0, "total_ask_size": 100.0}]
        trades = [self._trade()] * 100

        mock_api.side_effect = [ticker, daily, four_h, ob, trades]
        mock_eth.return_value = {}

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            main("KRW-BTC")

        data = json.loads(buf.getvalue())
        assert data["orderbook"]["ratio"] == pytest.approx(2.0, abs=0.01)
