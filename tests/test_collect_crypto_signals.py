"""Unit tests for scripts/collect_crypto_signals.py"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_crypto_signals import fetch_signals, classify_anomaly, main


# ── Helper: mock CoinGecko coin data ─────────────────────

def make_coin(symbol, price=50000, change_24h=1.5, volume=1000000,
              market_cap=100000000):
    """Create a CoinGecko /coins/markets item."""
    return {
        "symbol": symbol,
        "current_price": price,
        "price_change_percentage_24h": change_24h,
        "total_volume": volume,
        "market_cap": market_cap,
    }


SAMPLE_COINS = [
    make_coin("btc", price=60000, volume=30_000_000_000, market_cap=1_200_000_000_000),
    make_coin("eth", price=3000, volume=15_000_000_000, market_cap=360_000_000_000),
    make_coin("doge", price=0.15, change_24h=12.0, volume=5_000_000_000, market_cap=20_000_000_000),
    make_coin("shib", price=0.00001, change_24h=25.0, volume=2_000_000_000, market_cap=6_000_000_000),
]


def _mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


def _mock_429():
    mock = MagicMock()
    mock.status_code = 429
    return mock


# ── 1. fetch_signals — successful response ───────────────

class TestFetchSignalsSuccess:
    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_returns_sorted_signals(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_COINS)
        signals = fetch_signals()

        assert len(signals) == 4
        # Should be sorted by vol_mcap_ratio descending
        ratios = [s["vol_mcap_ratio"] for s in signals]
        assert ratios == sorted(ratios, reverse=True)

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_signal_fields(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response([SAMPLE_COINS[0]])
        signals = fetch_signals()

        s = signals[0]
        assert s["symbol"] == "BTC"
        assert s["price"] == 60000
        assert s["change_24h"] == 1.5
        assert s["volume"] == 30_000_000_000
        assert s["market_cap"] == 1_200_000_000_000
        # ratio = 30B / 1.2T * 100 = 2.5
        assert s["vol_mcap_ratio"] == 2.5

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_symbol_uppercased(self, mock_sleep, mock_get):
        coin = make_coin("sol", price=100)
        mock_get.return_value = _mock_response([coin])
        signals = fetch_signals()
        assert signals[0]["symbol"] == "SOL"


# ── 2. fetch_signals — 429 retry logic ───────────────────

class TestFetchSignals429:
    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep, mock_get):
        mock_get.side_effect = [_mock_429(), _mock_response(SAMPLE_COINS[:1])]
        signals = fetch_signals()

        assert len(signals) == 1
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2**0 = 1

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_exits_after_three_429s(self, mock_sleep, mock_get):
        mock_get.side_effect = [_mock_429(), _mock_429(), _mock_429()]
        with pytest.raises(SystemExit) as exc_info:
            fetch_signals()
        assert exc_info.value.code == 1
        assert mock_get.call_count == 3


# ── 3. fetch_signals — network error handling ────────────

class TestFetchSignalsNetworkError:
    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_retries_on_exception_then_succeeds(self, mock_sleep, mock_get):
        import requests as req
        mock_get.side_effect = [
            req.ConnectionError("timeout"),
            _mock_response(SAMPLE_COINS[:1]),
        ]
        signals = fetch_signals()
        assert len(signals) == 1
        assert mock_get.call_count == 2

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_exits_after_three_failures(self, mock_sleep, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("down")
        with pytest.raises(SystemExit) as exc_info:
            fetch_signals()
        assert exc_info.value.code == 1


# ── 4. classify_anomaly — all 4 levels ───────────────────

class TestClassifyAnomaly:
    @pytest.mark.parametrize("ratio,expected", [
        (0.5, "LOW"),
        (1.99, "LOW"),
        (2.0, "LOW"),
        (2.01, "MODERATE"),
        (5.0, "MODERATE"),
        (5.01, "HIGH"),
        (20.0, "HIGH"),
        (20.01, "CRITICAL"),
        (100.0, "CRITICAL"),
    ])
    def test_classification_thresholds(self, ratio, expected):
        assert classify_anomaly(ratio) == expected

    def test_zero_ratio(self):
        assert classify_anomaly(0) == "LOW"

    def test_negative_ratio(self):
        assert classify_anomaly(-1) == "LOW"


# ── 5. main — full output structure ──────────────────────

class TestMainOutput:
    @patch("collect_crypto_signals.fetch_signals")
    def test_output_structure(self, mock_fetch, capsys):
        mock_fetch.return_value = [
            {"symbol": "BTC", "price": 60000, "change_24h": 1.5,
             "volume": 30e9, "market_cap": 1.2e12, "vol_mcap_ratio": 2.5},
            {"symbol": "ETH", "price": 3000, "change_24h": -0.5,
             "volume": 15e9, "market_cap": 360e9, "vol_mcap_ratio": 4.17},
        ]
        main()
        output = json.loads(capsys.readouterr().out)

        assert output["source"] == "coingecko"
        assert "timestamp" in output
        assert output["btc"]["symbol"] == "BTC"
        assert output["btc"]["anomaly_level"] == "MODERATE"
        assert output["eth"]["symbol"] == "ETH"
        assert output["eth"]["anomaly_level"] == "MODERATE"
        assert "anomaly_alerts" in output
        assert "top_volume_tokens" in output

    @patch("collect_crypto_signals.fetch_signals")
    def test_top_volume_tokens_limited_to_10(self, mock_fetch, capsys):
        signals = [
            {"symbol": f"T{i}", "price": i, "change_24h": 0.0,
             "volume": 1000, "market_cap": 10000, "vol_mcap_ratio": 1.0}
            for i in range(20)
        ]
        mock_fetch.return_value = signals
        main()
        output = json.loads(capsys.readouterr().out)
        assert len(output["top_volume_tokens"]) == 10


# ── 6. main — missing BTC/ETH ────────────────────────────

class TestMainMissingTokens:
    @patch("collect_crypto_signals.fetch_signals")
    def test_btc_eth_none_when_absent(self, mock_fetch, capsys):
        mock_fetch.return_value = [
            {"symbol": "DOGE", "price": 0.15, "change_24h": 3.0,
             "volume": 5e9, "market_cap": 20e9, "vol_mcap_ratio": 25.0},
        ]
        main()
        output = json.loads(capsys.readouterr().out)
        assert output["btc"] is None
        assert output["eth"] is None

    @patch("collect_crypto_signals.fetch_signals")
    def test_only_btc_present(self, mock_fetch, capsys):
        mock_fetch.return_value = [
            {"symbol": "BTC", "price": 60000, "change_24h": 1.5,
             "volume": 30e9, "market_cap": 1.2e12, "vol_mcap_ratio": 2.5},
        ]
        main()
        output = json.loads(capsys.readouterr().out)
        assert output["btc"] is not None
        assert output["eth"] is None


# ── 7. main — anomaly_alerts filtering ───────────────────

class TestAnomalyAlerts:
    @patch("collect_crypto_signals.fetch_signals")
    def test_only_ratio_above_5_included(self, mock_fetch, capsys):
        mock_fetch.return_value = [
            {"symbol": "A", "price": 1, "change_24h": 0, "volume": 1000,
             "market_cap": 10000, "vol_mcap_ratio": 6.0},
            {"symbol": "B", "price": 1, "change_24h": 0, "volume": 1000,
             "market_cap": 10000, "vol_mcap_ratio": 4.9},
            {"symbol": "C", "price": 1, "change_24h": 0, "volume": 1000,
             "market_cap": 10000, "vol_mcap_ratio": 21.0},
        ]
        main()
        output = json.loads(capsys.readouterr().out)
        alerts = output["anomaly_alerts"]

        assert alerts["count"] == 2
        symbols = [t["symbol"] for t in alerts["tokens"]]
        assert "A" in symbols
        assert "C" in symbols
        assert "B" not in symbols

    @patch("collect_crypto_signals.fetch_signals")
    def test_anomaly_alerts_capped_at_10(self, mock_fetch, capsys):
        signals = [
            {"symbol": f"X{i}", "price": 1, "change_24h": 0, "volume": 1000,
             "market_cap": 10000, "vol_mcap_ratio": 10.0 + i}
            for i in range(15)
        ]
        mock_fetch.return_value = signals
        main()
        output = json.loads(capsys.readouterr().out)
        assert len(output["anomaly_alerts"]["tokens"]) == 10
        assert output["anomaly_alerts"]["count"] == 15

    @patch("collect_crypto_signals.fetch_signals")
    def test_anomaly_level_assigned(self, mock_fetch, capsys):
        mock_fetch.return_value = [
            {"symbol": "Z", "price": 1, "change_24h": 0, "volume": 1000,
             "market_cap": 10000, "vol_mcap_ratio": 25.0},
        ]
        main()
        output = json.loads(capsys.readouterr().out)
        token = output["anomaly_alerts"]["tokens"][0]
        assert token["anomaly_level"] == "CRITICAL"


# ── 8. Edge cases ────────────────────────────────────────

class TestEdgeCases:
    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_zero_market_cap_ratio_is_zero(self, mock_sleep, mock_get):
        coin = make_coin("abc", volume=1000, market_cap=0)
        mock_get.return_value = _mock_response([coin])
        signals = fetch_signals()
        assert signals[0]["vol_mcap_ratio"] == 0

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_missing_fields_default_gracefully(self, mock_sleep, mock_get):
        coin = {"symbol": None, "current_price": None,
                "price_change_percentage_24h": None,
                "total_volume": None, "market_cap": None}
        mock_get.return_value = _mock_response([coin])
        signals = fetch_signals()
        s = signals[0]
        assert s["symbol"] == ""
        assert s["price"] is None
        assert s["change_24h"] is None
        assert s["volume"] == 0
        assert s["market_cap"] == 0
        assert s["vol_mcap_ratio"] == 0

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_empty_response_returns_empty_list(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response([])
        signals = fetch_signals()
        assert signals == []

    @patch("collect_crypto_signals.fetch_signals")
    def test_empty_signals_main_output(self, mock_fetch, capsys):
        mock_fetch.return_value = []
        main()
        output = json.loads(capsys.readouterr().out)
        assert output["btc"] is None
        assert output["eth"] is None
        assert output["anomaly_alerts"]["count"] == 0
        assert output["anomaly_alerts"]["tokens"] == []
        assert output["top_volume_tokens"] == []

    @patch("collect_crypto_signals.requests.get")
    @patch("collect_crypto_signals.time.sleep")
    def test_missing_symbol_key_entirely(self, mock_sleep, mock_get):
        coin = {"current_price": 100, "total_volume": 500, "market_cap": 10000}
        mock_get.return_value = _mock_response([coin])
        signals = fetch_signals()
        assert signals[0]["symbol"] == ""
