#!/usr/bin/env python3
"""
Tests for collect_onchain_data.py and collect_fear_greed.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Ensure scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import collect_onchain_data as onchain
import collect_fear_greed as fgi


# ============================================================
# collect_onchain_data: collect_funding_rate
# ============================================================

class TestCollectFundingRate:

    def _make_funding_data(self, rate: float, count: int = 10):
        return [{"fundingRate": str(rate), "fundingTime": 1700000000000 + i}
                for i in range(count)]

    @patch("collect_onchain_data.safe_get")
    def test_overheated_long(self, mock_get):
        """rate > 0.01 -> signal '과열_롱'"""
        mock_get.return_value = self._make_funding_data(0.015)
        result = onchain.collect_funding_rate()
        assert result["signal"] == "과열_롱"
        assert "error" not in result
        assert result["current_rate"] == pytest.approx(1.5, abs=0.01)

    @patch("collect_onchain_data.safe_get")
    def test_overheated_short(self, mock_get):
        """rate < -0.01 -> signal '과열_숏'"""
        mock_get.return_value = self._make_funding_data(-0.015)
        result = onchain.collect_funding_rate()
        assert result["signal"] == "과열_숏"
        assert "error" not in result

    @patch("collect_onchain_data.safe_get")
    def test_neutral(self, mock_get):
        """rate near 0 -> signal '중립'"""
        mock_get.return_value = self._make_funding_data(0.0001)
        result = onchain.collect_funding_rate()
        assert result["signal"] == "중립"

    @patch("collect_onchain_data.safe_get")
    def test_api_failure(self, mock_get):
        """API failure returns error dict."""
        mock_get.return_value = None
        result = onchain.collect_funding_rate()
        assert "error" in result


# ============================================================
# collect_onchain_data: collect_long_short_ratio
# ============================================================

class TestCollectLongShortRatio:

    def _make_ls_data(self, long_ratio: float, short_ratio: float):
        return [{"longAccount": str(long_ratio), "shortAccount": str(short_ratio),
                 "timestamp": 1700000000000 + i} for i in range(5)]

    @patch("collect_onchain_data.safe_get")
    def test_extreme_long(self, mock_get):
        """long/short ratio > 2.0 -> '극단적_롱'"""
        mock_get.return_value = self._make_ls_data(0.75, 0.25)
        result = onchain.collect_long_short_ratio()
        assert result["signal"] == "극단적_롱"
        assert result["long_short_ratio"] == 3.0

    @patch("collect_onchain_data.safe_get")
    def test_extreme_short(self, mock_get):
        """long/short ratio < 0.5 -> '극단적_숏'"""
        mock_get.return_value = self._make_ls_data(0.2, 0.8)
        result = onchain.collect_long_short_ratio()
        assert result["signal"] == "극단적_숏"
        assert result["long_short_ratio"] == 0.25

    @patch("collect_onchain_data.safe_get")
    def test_balanced(self, mock_get):
        """long/short ratio ~ 1.0 -> '균형'"""
        mock_get.return_value = self._make_ls_data(0.5, 0.5)
        result = onchain.collect_long_short_ratio()
        assert result["signal"] == "균형"
        assert result["long_short_ratio"] == 1.0


# ============================================================
# collect_onchain_data: collect_open_interest
# ============================================================

class TestCollectOpenInterest:

    @patch("collect_onchain_data.safe_get")
    def test_oi_surge(self, mock_get):
        """OI change > 5% -> '급증'"""
        def side_effect(url, params=None):
            if "openInterestHist" in url:
                return [{"sumOpenInterest": "100"}, {"sumOpenInterest": "105"}]
            return {"openInterest": "110"}  # 10% increase from 100
        mock_get.side_effect = side_effect
        result = onchain.collect_open_interest()
        assert result["signal"] == "급증"
        assert result["oi_change_24h_pct"] == 10.0

    @patch("collect_onchain_data.safe_get")
    def test_oi_plunge(self, mock_get):
        """OI change < -5% -> '급감'"""
        def side_effect(url, params=None):
            if "openInterestHist" in url:
                return [{"sumOpenInterest": "100"}, {"sumOpenInterest": "95"}]
            return {"openInterest": "90"}  # -10% from 100
        mock_get.side_effect = side_effect
        result = onchain.collect_open_interest()
        assert result["signal"] == "급감"
        assert result["oi_change_24h_pct"] == -10.0

    @patch("collect_onchain_data.safe_get")
    def test_oi_stable(self, mock_get):
        """OI change near 0 -> '안정'"""
        def side_effect(url, params=None):
            if "openInterestHist" in url:
                return [{"sumOpenInterest": "100"}, {"sumOpenInterest": "101"}]
            return {"openInterest": "101"}
        mock_get.side_effect = side_effect
        result = onchain.collect_open_interest()
        assert result["signal"] == "안정"

    @patch("collect_onchain_data.safe_get")
    def test_oi_api_failure(self, mock_get):
        """API failure returns error."""
        mock_get.return_value = None
        result = onchain.collect_open_interest()
        assert "error" in result


# ============================================================
# collect_onchain_data: collect_mempool_fees
# ============================================================

class TestCollectMempoolFees:

    @patch("collect_onchain_data.safe_get")
    def test_very_active(self, mock_get):
        """fastestFee > 50 -> '매우_활발'"""
        mock_get.return_value = {
            "fastestFee": 80, "halfHourFee": 60, "hourFee": 40
        }
        result = onchain.collect_mempool_fees()
        assert result["network_activity"] == "매우_활발"
        assert result["fastest_fee_sat_vb"] == 80

    @patch("collect_onchain_data.safe_get")
    def test_quiet(self, mock_get):
        """fastestFee < 5 -> '한산'"""
        mock_get.return_value = {
            "fastestFee": 3, "halfHourFee": 2, "hourFee": 1
        }
        result = onchain.collect_mempool_fees()
        assert result["network_activity"] == "한산"

    @patch("collect_onchain_data.safe_get")
    def test_normal(self, mock_get):
        """5 <= fastestFee <= 20 -> '보통'"""
        mock_get.return_value = {
            "fastestFee": 10, "halfHourFee": 8, "hourFee": 5
        }
        result = onchain.collect_mempool_fees()
        assert result["network_activity"] == "보통"

    @patch("collect_onchain_data.safe_get")
    def test_api_failure(self, mock_get):
        mock_get.return_value = None
        result = onchain.collect_mempool_fees()
        assert "error" in result


# ============================================================
# collect_onchain_data: main() composite signal
# ============================================================

class TestOnchainMainSignal:

    @patch("collect_onchain_data.collect_mempool_fees")
    @patch("collect_onchain_data.collect_open_interest")
    @patch("collect_onchain_data.collect_long_short_ratio")
    @patch("collect_onchain_data.collect_funding_rate")
    @patch("collect_onchain_data.time")
    def test_bullish_signal(self, mock_time, mock_fr, mock_ls, mock_oi, mock_mp, capsys):
        """All sub-signals strongly bullish -> onchain_signal 'bullish'"""
        mock_time.strftime.return_value = "2026-03-08T00:00:00+09:00"
        mock_time.sleep = MagicMock()
        # 과열_숏 -> +2, 극단적_숏 -> +2, 급증 -> +1  => avg = 5/3 = 1.67 > 1
        mock_fr.return_value = {"signal": "과열_숏", "current_rate": -1.5, "avg_rate_10": -1.2}
        mock_ls.return_value = {"signal": "극단적_숏", "long_short_ratio": 0.3, "long_pct": 23, "short_pct": 77}
        mock_oi.return_value = {"signal": "급증", "open_interest_btc": 50000, "oi_change_24h_pct": 8.0}
        mock_mp.return_value = {"network_activity": "활발", "fastest_fee_sat_vb": 30}

        onchain.main()
        output = json.loads(capsys.readouterr().out)
        assert output["onchain_signal"] == "bullish"

    @patch("collect_onchain_data.collect_mempool_fees")
    @patch("collect_onchain_data.collect_open_interest")
    @patch("collect_onchain_data.collect_long_short_ratio")
    @patch("collect_onchain_data.collect_funding_rate")
    @patch("collect_onchain_data.time")
    def test_bearish_signal(self, mock_time, mock_fr, mock_ls, mock_oi, mock_mp, capsys):
        """All sub-signals strongly bearish -> onchain_signal 'bearish'"""
        mock_time.strftime.return_value = "2026-03-08T00:00:00+09:00"
        mock_time.sleep = MagicMock()
        # 과열_롱 -> -2, 극단적_롱 -> -2, 급감 -> -1  => avg = -5/3 = -1.67 < -1
        mock_fr.return_value = {"signal": "과열_롱", "current_rate": 1.5, "avg_rate_10": 1.2}
        mock_ls.return_value = {"signal": "극단적_롱", "long_short_ratio": 3.0, "long_pct": 75, "short_pct": 25}
        mock_oi.return_value = {"signal": "급감", "open_interest_btc": 40000, "oi_change_24h_pct": -8.0}
        mock_mp.return_value = {"network_activity": "한산", "fastest_fee_sat_vb": 3}

        onchain.main()
        output = json.loads(capsys.readouterr().out)
        assert output["onchain_signal"] == "bearish"

    @patch("collect_onchain_data.collect_mempool_fees")
    @patch("collect_onchain_data.collect_open_interest")
    @patch("collect_onchain_data.collect_long_short_ratio")
    @patch("collect_onchain_data.collect_funding_rate")
    @patch("collect_onchain_data.time")
    def test_neutral_signal(self, mock_time, mock_fr, mock_ls, mock_oi, mock_mp, capsys):
        """All neutral -> onchain_signal 'neutral'"""
        mock_time.strftime.return_value = "2026-03-08T00:00:00+09:00"
        mock_time.sleep = MagicMock()
        mock_fr.return_value = {"signal": "중립", "current_rate": 0.01, "avg_rate_10": 0.01}
        mock_ls.return_value = {"signal": "균형", "long_short_ratio": 1.0, "long_pct": 50, "short_pct": 50}
        mock_oi.return_value = {"signal": "안정", "open_interest_btc": 45000, "oi_change_24h_pct": 0.5}
        mock_mp.return_value = {"network_activity": "보통", "fastest_fee_sat_vb": 10}

        onchain.main()
        output = json.loads(capsys.readouterr().out)
        assert output["onchain_signal"] == "neutral"


# ============================================================
# collect_fear_greed: normal response
# ============================================================

def _make_fgi_response(values):
    """Build a mock Alternative.me FGI API response."""
    data = []
    for i, (val, cls) in enumerate(values):
        data.append({
            "value": str(val),
            "value_classification": cls,
            "timestamp": str(int(datetime(2026, 3, 8, tzinfo=timezone.utc).timestamp()) - i * 86400),
        })
    return {"data": data}


class TestCollectFearGreed:

    @patch("collect_fear_greed.requests.get")
    def test_normal_response(self, mock_get):
        """Normal API response with value=25, Extreme Fear."""
        values = [
            (25, "Extreme Fear"),
            (30, "Fear"),
            (28, "Fear"),
            (35, "Fear"),
            (40, "Fear"),
            (22, "Extreme Fear"),
            (20, "Extreme Fear"),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fgi_response(values)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Capture stdout
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert result["current"]["value"] == 25
        assert result["current"]["classification"] == "Extreme Fear"
        assert len(result["history_7d"]) == 7

    @patch("collect_fear_greed.requests.get")
    def test_api_failure(self, mock_get):
        """Connection error -> raises exception (handled by __main__ block)."""
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
        with pytest.raises(requests.exceptions.ConnectionError):
            fgi.main()

    @patch("collect_fear_greed.requests.get")
    def test_7day_history(self, mock_get):
        """Verify history array contains 7 entries with dates."""
        values = [(50 + i, "Neutral") for i in range(7)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fgi_response(values)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert len(result["history_7d"]) == 7
        for entry in result["history_7d"]:
            assert "date" in entry
            assert "value" in entry
            assert "classification" in entry

    @patch("collect_fear_greed.requests.get")
    def test_edge_value_zero(self, mock_get):
        """Edge case: FGI value = 0."""
        values = [(0, "Extreme Fear")] + [(10, "Extreme Fear")] * 6
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fgi_response(values)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())
        assert result["current"]["value"] == 0

    @patch("collect_fear_greed.requests.get")
    def test_edge_value_100(self, mock_get):
        """Edge case: FGI value = 100."""
        values = [(100, "Extreme Greed")] + [(90, "Greed")] * 6
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fgi_response(values)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())
        assert result["current"]["value"] == 100
        assert result["current"]["classification"] == "Extreme Greed"
