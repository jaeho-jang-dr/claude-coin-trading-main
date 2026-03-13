"""
collect_fear_greed.py comprehensive unit tests.

Covers:
  - Normal API response parsing
  - 7-day history structure
  - Edge values (0, 100)
  - Error handling (connection error, HTTP error, malformed JSON)
  - Output JSON structure verification

All network calls are mocked - no real API calls are made.
"""

import json
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import collect_fear_greed as fgi


# ── Helpers ──────────────────────────────────────────────

def _fgi_response(values):
    """Build a fake Alternative.me FGI API response.

    Args:
        values: list of (value, classification) tuples, newest first.
    """
    data = []
    base_ts = int(datetime(2026, 3, 13, tzinfo=timezone.utc).timestamp())
    for i, (val, cls) in enumerate(values):
        data.append({
            "value": str(val),
            "value_classification": cls,
            "timestamp": str(base_ts - i * 86400),
        })
    return {"data": data}


def _mock_response(json_data, ok=True):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if not ok:
        resp.raise_for_status.side_effect = requests.HTTPError("Error")
    return resp


# ══════════════════════════════════════════════════════════════
# Happy path
# ══════════════════════════════════════════════════════════════

class TestFearGreedHappyPath:
    @patch("collect_fear_greed.requests.get")
    def test_normal_response_structure(self, mock_get):
        values = [
            (25, "Extreme Fear"), (30, "Fear"), (28, "Fear"),
            (35, "Fear"), (40, "Fear"), (22, "Extreme Fear"), (20, "Extreme Fear"),
        ]
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert "timestamp" in result
        assert "current" in result
        assert "history_7d" in result

    @patch("collect_fear_greed.requests.get")
    def test_current_value(self, mock_get):
        values = [(42, "Fear")] + [(50, "Neutral")] * 6
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert result["current"]["value"] == 42
        assert result["current"]["classification"] == "Fear"

    @patch("collect_fear_greed.requests.get")
    def test_history_has_7_entries(self, mock_get):
        values = [(50 + i, "Neutral") for i in range(7)]
        mock_get.return_value = _mock_response(_fgi_response(values))

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
    def test_history_dates_are_formatted(self, mock_get):
        values = [(50, "Neutral")] * 7
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        for entry in result["history_7d"]:
            # Should be YYYY-MM-DD format
            date = entry["date"]
            assert len(date) == 10
            assert date[4] == "-" and date[7] == "-"

    @patch("collect_fear_greed.requests.get")
    def test_values_are_integers(self, mock_get):
        values = [(25, "Extreme Fear")] * 7
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert isinstance(result["current"]["value"], int)
        for entry in result["history_7d"]:
            assert isinstance(entry["value"], int)


# ══════════════════════════════════════════════════════════════
# Edge values
# ══════════════════════════════════════════════════════════════

class TestFearGreedEdgeCases:
    @patch("collect_fear_greed.requests.get")
    def test_value_zero(self, mock_get):
        values = [(0, "Extreme Fear")] + [(10, "Extreme Fear")] * 6
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert result["current"]["value"] == 0

    @patch("collect_fear_greed.requests.get")
    def test_value_100(self, mock_get):
        values = [(100, "Extreme Greed")] + [(90, "Greed")] * 6
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        assert result["current"]["value"] == 100
        assert result["current"]["classification"] == "Extreme Greed"

    @patch("collect_fear_greed.requests.get")
    def test_all_classifications(self, mock_get):
        """All five classification levels are preserved correctly."""
        values = [
            (10, "Extreme Fear"),
            (30, "Fear"),
            (50, "Neutral"),
            (70, "Greed"),
            (90, "Extreme Greed"),
            (45, "Fear"),
            (55, "Neutral"),
        ]
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()
        result = json.loads(buf.getvalue())

        classifications = {e["classification"] for e in result["history_7d"]}
        assert "Extreme Fear" in classifications
        assert "Extreme Greed" in classifications
        assert "Neutral" in classifications


# ══════════════════════════════════════════════════════════════
# API call verification
# ══════════════════════════════════════════════════════════════

class TestFearGreedAPICall:
    @patch("collect_fear_greed.requests.get")
    def test_correct_url_and_params(self, mock_get):
        values = [(50, "Neutral")] * 7
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()

        call_args = mock_get.call_args
        assert call_args[0][0] == fgi.FGI_API
        assert call_args[1]["params"]["limit"] == "7"
        assert call_args[1]["params"]["format"] == "json"
        assert call_args[1]["timeout"] == 10

    @patch("collect_fear_greed.requests.get")
    def test_raise_for_status_called(self, mock_get):
        values = [(50, "Neutral")] * 7
        mock_resp = _mock_response(_fgi_response(values))
        mock_get.return_value = mock_resp

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()

        mock_resp.raise_for_status.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════════

class TestFearGreedErrors:
    @patch("collect_fear_greed.requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("DNS failure")
        with pytest.raises(requests.exceptions.ConnectionError):
            fgi.main()

    @patch("collect_fear_greed.requests.get")
    def test_timeout_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            fgi.main()

    @patch("collect_fear_greed.requests.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_response({}, ok=False)
        with pytest.raises(requests.HTTPError):
            fgi.main()

    @patch("collect_fear_greed.requests.get")
    def test_malformed_json_missing_data_key(self, mock_get):
        """API returns JSON without 'data' key."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"unexpected": "response"}
        mock_get.return_value = mock_resp

        with pytest.raises(KeyError):
            fgi.main()

    @patch("collect_fear_greed.requests.get")
    def test_empty_data_array(self, mock_get):
        """API returns empty data array."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        # Accessing history[0] for 'current' should raise IndexError
        with pytest.raises(IndexError):
            fgi.main()


# ══════════════════════════════════════════════════════════════
# Output format
# ══════════════════════════════════════════════════════════════

class TestFearGreedOutputFormat:
    @patch("collect_fear_greed.requests.get")
    def test_output_is_valid_json(self, mock_get):
        values = [(50, "Neutral")] * 7
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()

        # Should not raise
        result = json.loads(buf.getvalue())
        assert isinstance(result, dict)

    @patch("collect_fear_greed.requests.get")
    def test_timestamp_is_iso_format(self, mock_get):
        values = [(50, "Neutral")] * 7
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()

        result = json.loads(buf.getvalue())
        # Should parse as ISO datetime
        datetime.fromisoformat(result["timestamp"])

    @patch("collect_fear_greed.requests.get")
    def test_current_equals_first_history_entry(self, mock_get):
        """'current' should be identical to history_7d[0]."""
        values = [(25, "Extreme Fear")] + [(50, "Neutral")] * 6
        mock_get.return_value = _mock_response(_fgi_response(values))

        buf = io.StringIO()
        with redirect_stdout(buf):
            fgi.main()

        result = json.loads(buf.getvalue())
        assert result["current"] == result["history_7d"][0]
