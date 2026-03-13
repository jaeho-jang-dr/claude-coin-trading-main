"""
notify_telegram.py comprehensive unit tests.

Covers:
  - MarkdownV2 escaping (all special characters)
  - send_message: success, correct URL, payload, emoji mapping, timestamps
  - send_message: error handling (missing env vars, API failures)
  - send_photo: success, URL, multipart data
  - send_photo: error handling (missing env vars, API failures)

All network calls are mocked - no real API calls are made.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from notify_telegram import escape_md, send_message, send_photo, EMOJI


# ══════════════════════════════════════════════════════════════
# MarkdownV2 escaping
# ══════════════════════════════════════════════════════════════

class TestEscapeMd:
    """Test all MarkdownV2 special characters are escaped."""

    def test_plain_text_unchanged(self):
        assert escape_md("hello world 123") == "hello world 123"

    def test_dot(self):
        assert escape_md("100.5") == r"100\.5"

    def test_dash(self):
        assert escape_md("BTC-KRW") == r"BTC\-KRW"

    def test_exclamation(self):
        assert escape_md("alert!") == r"alert\!"

    def test_parentheses(self):
        assert escape_md("(hello)") == r"\(hello\)"

    def test_brackets(self):
        assert escape_md("[link]") == r"\[link\]"

    def test_underscore_and_asterisk(self):
        assert escape_md("_bold_ *italic*") == r"\_bold\_ \*italic\*"

    def test_hash_plus_equals(self):
        assert escape_md("# heading += 1") == r"\# heading \+\= 1"

    def test_tilde_backtick_pipe(self):
        assert escape_md("~code~ `x` |y|") == r"\~code\~ \`x\` \|y\|"

    def test_curly_braces(self):
        assert escape_md("{key}") == r"\{key\}"

    def test_backslash(self):
        assert escape_md("a\\b") == r"a\\b"

    def test_greater_than(self):
        assert escape_md("> quote") == r"\> quote"

    def test_combined_financial_text(self):
        result = escape_md("BTC: 100,000.5 KRW (+2.3%)")
        assert r"\." in result
        assert r"\+" in result
        assert r"\(" in result
        assert r"\)" in result

    def test_empty_string(self):
        assert escape_md("") == ""

    def test_korean_text(self):
        assert escape_md("비트코인 매수") == "비트코인 매수"

    def test_all_special_chars_at_once(self):
        special = "_*[]()~`>#+\\-=|{}.!"
        result = escape_md(special)
        # Every character should be escaped with backslash
        assert all(f"\\{c}" in result for c in special if c != "\\")


# ══════════════════════════════════════════════════════════════
# send_message - happy path
# ══════════════════════════════════════════════════════════════

class TestSendMessageHappy:
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_success_response(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        result = send_message("trade", "BTC Buy", "Bought 0.001 BTC")
        assert result == {"success": True, "type": "trade", "title": "BTC Buy"}

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "my-token", "TELEGRAM_USER_ID": "456"})
    @patch("notify_telegram.requests.post")
    def test_correct_url(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "t", "b")
        url = mock_post.call_args[0][0]
        assert url == "https://api.telegram.org/botmy-token/sendMessage"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "789"})
    @patch("notify_telegram.requests.post")
    def test_payload_structure(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "title", "body")
        payload = mock_post.call_args[1]["json"]
        assert payload["chat_id"] == "789"
        assert payload["parse_mode"] == "MarkdownV2"
        assert "title" in payload["text"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_timeout_set(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "t", "b")
        assert mock_post.call_args[1]["timeout"] == 10


# ══════════════════════════════════════════════════════════════
# send_message - emoji mapping
# ══════════════════════════════════════════════════════════════

class TestSendMessageEmoji:
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_trade_emoji(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        assert EMOJI["trade"] in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_analysis_emoji(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("analysis", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        assert EMOJI["analysis"] in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_error_emoji(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("error", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        assert EMOJI["error"] in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_status_emoji(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("status", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        assert EMOJI["status"] in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_unknown_type_default_emoji(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("unknown", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        assert "\U0001f4ac" in text


# ══════════════════════════════════════════════════════════════
# send_message - message formatting
# ══════════════════════════════════════════════════════════════

class TestSendMessageFormat:
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_title_in_bold(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "My Title", "body")
        text = mock_post.call_args[1]["json"]["text"]
        # Title should be wrapped in *...*
        assert "*My Title*" in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_timestamp_in_message(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        assert "KST" in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_special_chars_escaped_in_body(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "title", "price: 100.5 (+2%)")
        text = mock_post.call_args[1]["json"]["text"]
        # Dots, pluses, parentheses in body should be escaped
        assert r"\." in text
        assert r"\+" in text

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_timestamp_in_italic(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_message("trade", "t", "b")
        text = mock_post.call_args[1]["json"]["text"]
        # Timestamp should be wrapped in _..._
        assert text.rstrip().endswith("_")


# ══════════════════════════════════════════════════════════════
# send_message - errors
# ══════════════════════════════════════════════════════════════

class TestSendMessageErrors:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_both_env_vars(self):
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            send_message("trade", "t", "b")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": ""})
    def test_empty_user_id(self):
        with pytest.raises(RuntimeError):
            send_message("trade", "t", "b")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_USER_ID": "123"})
    def test_empty_bot_token(self):
        with pytest.raises(RuntimeError):
            send_message("trade", "t", "b")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_api_400(self, mock_post):
        mock_post.return_value = MagicMock(ok=False, text="Bad Request")
        with pytest.raises(RuntimeError, match="텔레그램 전송 실패"):
            send_message("trade", "t", "b")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_api_500(self, mock_post):
        mock_post.return_value = MagicMock(ok=False, text="Internal Server Error")
        with pytest.raises(RuntimeError, match="텔레그램 전송 실패"):
            send_message("trade", "t", "b")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    def test_network_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.Timeout("timeout")
        with pytest.raises(requests.Timeout):
            send_message("trade", "t", "b")


# ══════════════════════════════════════════════════════════════
# send_photo - happy path
# ══════════════════════════════════════════════════════════════

class TestSendPhotoHappy:
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG"))
    def test_success(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        result = send_photo("/tmp/chart.png", "BTC chart")
        assert result == {"success": True, "type": "photo", "path": "/tmp/chart.png"}

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "my-token", "TELEGRAM_USER_ID": "456"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG"))
    def test_correct_url(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_photo("/tmp/chart.png", "caption")
        url = mock_post.call_args[0][0]
        assert url == "https://api.telegram.org/botmy-token/sendPhoto"

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "789"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG"))
    def test_multipart_fields(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_photo("/tmp/chart.png", "my caption")

        call_kw = mock_post.call_args[1]
        assert call_kw["data"]["chat_id"] == "789"
        assert call_kw["data"]["caption"] == "my caption"
        assert "photo" in call_kw["files"]

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG"))
    def test_timeout_30(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_photo("/tmp/chart.png", "c")
        assert mock_post.call_args[1]["timeout"] == 30

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG"))
    def test_file_opened_as_rb(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)
        send_photo("/tmp/chart.png", "c")
        # The photo tuple should have content type image/png
        files = mock_post.call_args[1]["files"]
        photo_tuple = files["photo"]
        assert photo_tuple[0] == "chart.png"
        assert photo_tuple[2] == "image/png"


# ══════════════════════════════════════════════════════════════
# send_photo - errors
# ══════════════════════════════════════════════════════════════

class TestSendPhotoErrors:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_vars(self):
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            send_photo("/tmp/chart.png", "caption")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    @patch("notify_telegram.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG"))
    def test_api_failure(self, mock_post):
        mock_post.return_value = MagicMock(ok=False, text="Bad Request")
        with pytest.raises(RuntimeError, match="텔레그램 이미지 전송 실패"):
            send_photo("/tmp/chart.png", "caption")

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_USER_ID": "123"})
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            send_photo("/nonexistent/path/chart.png", "caption")
