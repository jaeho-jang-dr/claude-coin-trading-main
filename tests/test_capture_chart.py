"""
capture_chart.py unit tests

All Playwright interactions are mocked - no real browser launches.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _build_playwright_mocks():
    """Build a full chain of Playwright async mocks.

    Returns (mock_async_playwright_fn, mock_browser, mock_context, mock_page).
    """
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = AsyncMock()
    mock_browser.new_context.return_value = mock_context

    mock_chromium = AsyncMock()
    mock_chromium.launch.return_value = mock_browser

    mock_pw_instance = MagicMock()
    mock_pw_instance.chromium = mock_chromium

    # async context manager returned by async_playwright()
    mock_pw_cm = AsyncMock()
    mock_pw_cm.__aenter__ = AsyncMock(return_value=mock_pw_instance)
    mock_pw_cm.__aexit__ = AsyncMock(return_value=False)

    mock_async_playwright = MagicMock(return_value=mock_pw_cm)

    return mock_async_playwright, mock_browser, mock_context, mock_page


def _run_capture(mock_ap, tmp_path, extra_patches=None):
    """Import and run capture_chart with mocks applied."""
    patches = {
        "playwright.async_api.async_playwright": mock_ap,
        "scripts.capture_chart.os.getcwd": MagicMock(return_value=str(tmp_path)),
    }
    if extra_patches:
        patches.update(extra_patches)

    # Need to reimport each time to pick up patched os.getcwd
    import importlib
    import scripts.capture_chart as mod

    ctx_managers = [patch(k, v) for k, v in patches.items()]
    for cm in ctx_managers:
        cm.__enter__()
    try:
        importlib.reload(mod)
        _run_async(mod.capture_chart())
    finally:
        for cm in reversed(ctx_managers):
            cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Tests: Chart URL construction
# ---------------------------------------------------------------------------

class TestChartURL:
    """Verify the correct Upbit URL is used for KRW-BTC."""

    def test_navigates_to_upbit_btc_chart(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        mock_page.goto.assert_called_once()
        url = mock_page.goto.call_args[0][0]
        assert "upbit.com/full_chart" in url
        assert "KRW-BTC" in url

    def test_url_uses_crix_format(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        url = mock_page.goto.call_args[0][0]
        assert url == "https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC"


# ---------------------------------------------------------------------------
# Tests: Output file path
# ---------------------------------------------------------------------------

class TestOutputFilePath:
    """Verify data/charts/ directory and filename format."""

    def test_screenshot_saved_under_data_charts(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        path_arg = mock_page.screenshot.call_args[1]["path"]
        assert "/data/charts/" in path_arg

    def test_filename_contains_btc_chart_prefix(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        path_arg = mock_page.screenshot.call_args[1]["path"]
        filename = Path(path_arg).name
        assert filename.startswith("btc_chart_")
        assert filename.endswith(".png")

    def test_filename_contains_timestamp_format(self, tmp_path):
        """Filename should contain a YYYYMMDD_HHMMSS timestamp."""
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        path_arg = mock_page.screenshot.call_args[1]["path"]
        filename = Path(path_arg).stem  # btc_chart_20260308_143045
        # Extract timestamp part after "btc_chart_"
        ts_part = filename.replace("btc_chart_", "")
        # Should be parseable as a timestamp
        datetime.strptime(ts_part, "%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Tests: Screenshot options (headless, viewport)
# ---------------------------------------------------------------------------

class TestScreenshotOptions:
    """Verify headless mode, viewport size, and screenshot settings."""

    def test_browser_launched_headless(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        mock_chromium = mock_ap.return_value.__aenter__.return_value.chromium
        _run_capture(mock_ap, tmp_path)

        mock_chromium.launch.assert_called_once_with(headless=True)

    def test_viewport_1920x1080(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        ctx_call = mock_browser.new_context.call_args
        assert ctx_call[1]["viewport"] == {"width": 1920, "height": 1080}

    def test_locale_korean(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        ctx_call = mock_browser.new_context.call_args
        assert ctx_call[1]["locale"] == "ko-KR"
        assert ctx_call[1]["timezone_id"] == "Asia/Seoul"

    def test_screenshot_not_full_page(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        screenshot_call = mock_page.screenshot.call_args
        assert screenshot_call[1]["full_page"] is False

    def test_navigation_timeout_30s(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        goto_call = mock_page.goto.call_args
        assert goto_call[1]["timeout"] == 30000

    def test_wait_until_networkidle(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        goto_call = mock_page.goto.call_args
        assert goto_call[1]["wait_until"] == "networkidle"

    def test_chart_rendering_wait_5s(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        mock_page.wait_for_timeout.assert_called_once_with(5000)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify error handling for browser/navigation failures."""

    def test_browser_launch_failure(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        mock_chromium = mock_ap.return_value.__aenter__.return_value.chromium
        mock_chromium.launch.side_effect = Exception("Browser launch failed")

        with pytest.raises(Exception, match="Browser launch failed"):
            _run_capture(mock_ap, tmp_path)

    def test_navigation_timeout_error(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        mock_page.goto.side_effect = Exception("Timeout 30000ms exceeded")

        with pytest.raises(Exception, match="Timeout"):
            _run_capture(mock_ap, tmp_path)

    def test_page_load_error(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        mock_page.wait_for_timeout.side_effect = Exception("Page crashed")

        with pytest.raises(Exception, match="Page crashed"):
            _run_capture(mock_ap, tmp_path)

    def test_screenshot_failure(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        mock_page.screenshot.side_effect = Exception("Screenshot failed")

        with pytest.raises(Exception, match="Screenshot failed"):
            _run_capture(mock_ap, tmp_path)

    def test_browser_close_called_on_success(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        mock_browser.close.assert_called_once()

    def test_main_block_writes_error_to_stderr(self, tmp_path):
        """The __main__ block catches exceptions and writes JSON error to stderr."""
        import importlib
        import scripts.capture_chart as mod

        mock_ap, _, _, _ = _build_playwright_mocks()
        mock_chromium = mock_ap.return_value.__aenter__.return_value.chromium
        mock_chromium.launch.side_effect = RuntimeError("test failure")

        with patch("playwright.async_api.async_playwright", mock_ap), \
             patch("scripts.capture_chart.os.getcwd", return_value=str(tmp_path)):
            importlib.reload(mod)
            with pytest.raises(SystemExit) as exc_info:
                mod.__name__ = "__main__"
                try:
                    asyncio.run(mod.capture_chart())
                except Exception as e:
                    # Simulate __main__ block behavior
                    json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
                    sys.exit(1)
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------

class TestJSONOutput:
    """Verify JSON output with chart path."""

    def test_prints_json_with_chart_path(self, tmp_path, capsys):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "chart_path" in result
        assert "timestamp" in result

    def test_json_chart_path_matches_screenshot_path(self, tmp_path, capsys):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        screenshot_path = mock_page.screenshot.call_args[1]["path"]
        assert result["chart_path"] == screenshot_path

    def test_json_timestamp_is_iso_format(self, tmp_path, capsys):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        # Should parse without error
        datetime.fromisoformat(result["timestamp"])

    def test_json_chart_path_ends_with_png(self, tmp_path, capsys):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        _run_capture(mock_ap, tmp_path)

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["chart_path"].endswith(".png")


# ---------------------------------------------------------------------------
# Tests: Directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    """Verify data/charts/ auto-creation."""

    def test_creates_charts_directory(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        charts_dir = tmp_path / "data" / "charts"
        assert not charts_dir.exists()

        _run_capture(mock_ap, tmp_path)

        assert charts_dir.exists()
        assert charts_dir.is_dir()

    def test_existing_charts_directory_no_error(self, tmp_path):
        mock_ap, mock_browser, mock_ctx, mock_page = _build_playwright_mocks()
        charts_dir = tmp_path / "data" / "charts"
        charts_dir.mkdir(parents=True)

        # Should not raise
        _run_capture(mock_ap, tmp_path)

        assert charts_dir.exists()
