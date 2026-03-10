"""Unit tests for scripts/save_decision.py"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Patch env vars and load_dotenv before importing the module
with patch.dict("os.environ", {"SUPABASE_URL": "https://fake.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "fake-key"}):
    with patch("dotenv.load_dotenv"):
        import save_decision as sd


# ---------------------------------------------------------------------------
# extract_json_from_response
# ---------------------------------------------------------------------------
class TestExtractJsonFromResponse:
    def test_json_code_fence(self):
        text = '여기 결과입니다:\n```json\n{"decision": "buy"}\n```\n끝'
        result = sd.extract_json_from_response(text)
        assert result == {"decision": "buy"}

    def test_code_fence_without_json_tag(self):
        text = '결과:\n```\n{"decision": "sell"}\n```'
        result = sd.extract_json_from_response(text)
        assert result == {"decision": "sell"}

    def test_raw_json_text(self):
        text = '{"decision": "hold", "confidence": 70}'
        result = sd.extract_json_from_response(text)
        assert result == {"decision": "hold", "confidence": 70}

    def test_json_embedded_in_markdown(self):
        text = (
            "# Analysis\n\nBased on my analysis, here is the decision:\n\n"
            '{"decision": "buy", "confidence": 85}\n\n'
            "The market looks bullish."
        )
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert result["decision"] == "buy"
        assert result["confidence"] == 85

    def test_nested_json_with_inner_braces(self):
        text = '{"decision": "buy", "buy_score": {"fgi": {"value": 25}, "total": 75}}'
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert result["buy_score"]["fgi"]["value"] == 25
        assert result["buy_score"]["total"] == 75

    def test_trailing_commas(self):
        text = '{"decision": "buy", "confidence": 80,}'
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert result["decision"] == "buy"
        assert result["confidence"] == 80

    def test_trailing_comma_in_array(self):
        text = '{"items": [1, 2, 3,]}'
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert result["items"] == [1, 2, 3]

    def test_braces_in_string_values(self):
        text = '{"reason": "price {up} due to {demand}"}'
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert result["reason"] == "price {up} due to {demand}"

    def test_incomplete_json_missing_closing_braces(self):
        text = '{"decision": "buy", "nested": {"value": 1}'
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert result["decision"] == "buy"

    def test_no_json_returns_none(self):
        text = "This is just plain text with no JSON content at all."
        result = sd.extract_json_from_response(text)
        assert result is None

    def test_empty_string_returns_none(self):
        result = sd.extract_json_from_response("")
        assert result is None

    def test_multiple_json_blocks_returns_largest(self):
        text = (
            'Small: {"a": 1}\n\n'
            'Large: {"decision": "buy", "confidence": 85, "reason": "bullish", "current_price": 50000000}\n\n'
            'Medium: {"x": 1, "y": 2}'
        )
        result = sd.extract_json_from_response(text)
        assert result is not None
        assert "decision" in result
        assert result["decision"] == "buy"
        assert result["current_price"] == 50000000


# ---------------------------------------------------------------------------
# map_decision
# ---------------------------------------------------------------------------
class TestMapDecision:
    @pytest.mark.parametrize("raw,expected", [
        ("buy", "매수"),
        ("매수", "매수"),
        ("bid", "매수"),
        ("BUY", "매수"),
        ("Buy", "매수"),
    ])
    def test_buy_variants(self, raw, expected):
        assert sd.map_decision(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("sell", "매도"),
        ("매도", "매도"),
        ("ask", "매도"),
        ("SELL", "매도"),
        ("Sell", "매도"),
    ])
    def test_sell_variants(self, raw, expected):
        assert sd.map_decision(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("hold", "관망"),
        ("관망", "관망"),
        ("HOLD", "관망"),
        ("Hold", "관망"),
    ])
    def test_hold_variants(self, raw, expected):
        assert sd.map_decision(raw) == expected

    @pytest.mark.parametrize("raw", [
        "unknown", "wait", "neutral", "", "something_else",
    ])
    def test_unknown_defaults_to_hold(self, raw):
        assert sd.map_decision(raw) == "관망"

    def test_whitespace_stripped(self):
        assert sd.map_decision("  buy  ") == "매수"


# ---------------------------------------------------------------------------
# _get_nested
# ---------------------------------------------------------------------------
class TestGetNested:
    def test_simple_key_lookup(self):
        data = {"decision": "buy", "confidence": 85}
        assert sd._get_nested(data, "decision") == "buy"

    def test_dotted_path(self):
        data = {"a": {"b": {"c": 42}}}
        assert sd._get_nested(data, "a.b.c") == 42

    def test_dotted_path_intermediate(self):
        data = {"buy_score": {"fgi": {"value": 25}}}
        assert sd._get_nested(data, "buy_score.fgi.value") == 25

    def test_missing_key_returns_default(self):
        data = {"a": 1}
        assert sd._get_nested(data, "b", default="fallback") == "fallback"

    def test_missing_key_returns_none_by_default(self):
        data = {"a": 1}
        assert sd._get_nested(data, "b") is None

    def test_missing_dotted_path_returns_default(self):
        data = {"a": {"b": 1}}
        assert sd._get_nested(data, "a.c.d", default=0) == 0

    def test_multiple_key_candidates_first_match(self):
        data = {"name": "alice", "alias": "bob"}
        assert sd._get_nested(data, "name", "alias") == "alice"

    def test_multiple_key_candidates_fallback(self):
        data = {"alias": "bob"}
        assert sd._get_nested(data, "name", "alias") == "bob"

    def test_none_value_skipped(self):
        data = {"a": None, "b": "value"}
        assert sd._get_nested(data, "a", "b") == "value"

    def test_dotted_path_non_dict_intermediate(self):
        data = {"a": "string_not_dict"}
        assert sd._get_nested(data, "a.b.c", default="nope") == "nope"


# ---------------------------------------------------------------------------
# save_decision — field mapping
# ---------------------------------------------------------------------------
class TestSaveDecision:
    @patch("save_decision.supabase_post")
    def test_confidence_integer_normalized(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "buy",
            "confidence": 85,
            "current_price": 50000000,
            "buy_score": {"fgi": {"value": 25}, "rsi": {"value": 30}, "total": 75},
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["confidence"] == 0.85

    @patch("save_decision.supabase_post")
    def test_confidence_float_not_double_divided(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "hold",
            "confidence": 0.85,
            "buy_score": {},
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["confidence"] == 0.85

    @patch("save_decision.supabase_post")
    def test_confidence_zero(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {"decision": "hold", "confidence": 0, "buy_score": {}}
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["confidence"] == 0.0

    @patch("save_decision.supabase_post")
    def test_buy_score_fgi_value_extraction(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "buy",
            "confidence": 70,
            "buy_score": {
                "fgi": {"value": 22, "score": 30},
                "rsi": {"value": 28, "score": 25},
                "total": 75,
            },
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["fear_greed_value"] == 22
        assert row["rsi_value"] == 28

    @patch("save_decision.supabase_post")
    def test_decision_mapped(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {"decision": "buy", "confidence": 50, "buy_score": {}}
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["decision"] == "매수"

    @patch("save_decision.supabase_post")
    def test_default_market(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {"decision": "hold", "buy_score": {}}
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["market"] == "KRW-BTC"

    @patch("save_decision.supabase_post")
    def test_trade_details_amount(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "buy",
            "confidence": 80,
            "buy_score": {},
            "trade_details": {"amount": 100000},
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["trade_amount"] == 100000

    @patch("save_decision.supabase_post")
    def test_trade_amount_fallback(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "sell",
            "confidence": 60,
            "buy_score": {},
            "trade_amount": 50000,
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["trade_amount"] == 50000

    @patch("save_decision.supabase_post")
    def test_executed_flag(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {"decision": "buy", "confidence": 90, "buy_score": {}, "executed": True}
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["executed"] is True

    @patch("save_decision.supabase_post")
    def test_executed_from_trade_executed(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {"decision": "buy", "confidence": 90, "buy_score": {}, "trade_executed": True}
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["executed"] is True

    @patch("save_decision.supabase_post")
    def test_market_data_snapshot_contains_buy_score(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "hold",
            "buy_score": {"total": 40, "fgi": {"value": 50}},
            "ai_composite_signal": {"score": 0.6},
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        snapshot = json.loads(row["market_data_snapshot"])
        assert snapshot["buy_score"]["total"] == 40
        assert snapshot["ai_composite_signal"]["score"] == 0.6

    @patch("save_decision.supabase_post")
    def test_calls_supabase_with_decisions_table(self, mock_post):
        mock_post.return_value = {"id": "123"}
        data = {"decision": "hold", "buy_score": {}}
        sd.save_decision(data)
        assert mock_post.call_args_list[0][0][0] == "decisions"

    @patch("save_decision.supabase_post")
    def test_returns_none_on_failure(self, mock_post):
        mock_post.return_value = None
        data = {"decision": "hold", "buy_score": {}}
        result = sd.save_decision(data)
        assert result is None

    @patch("save_decision.supabase_post")
    def test_buy_score_non_dict_fgi(self, mock_post):
        """When buy_score.fgi is not a dict (e.g. a number), should not crash."""
        mock_post.return_value = {"id": "123"}
        data = {
            "decision": "buy",
            "confidence": 70,
            "buy_score": {"fgi": 25, "rsi": 30},
        }
        sd.save_decision(data)
        row = mock_post.call_args_list[0][0][1]
        assert row["fear_greed_value"] is None
        assert row["rsi_value"] is None


# ---------------------------------------------------------------------------
# update_past_performance
# ---------------------------------------------------------------------------
class TestUpdatePastPerformance:
    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_buy_decision_profit_loss(self, mock_get, mock_patch):
        """매수 결정: (현재-결정)/결정 * 100 = profit_loss"""
        # First call: supabase GET decisions
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = [
            {"id": "aaa", "decision": "매수", "current_price": 50000000, "profit_loss": None},
        ]
        # Second call: upbit ticker
        ticker_resp = MagicMock()
        ticker_resp.json.return_value = [{"trade_price": 51000000}]

        mock_get.side_effect = [supabase_resp, ticker_resp]
        mock_patch.return_value = MagicMock(ok=True)

        sd.update_past_performance()

        mock_patch.assert_called_once()
        call_kwargs = mock_patch.call_args
        # profit_loss for 매수 = (51000000 - 50000000) / 50000000 * 100 = 2.0
        assert call_kwargs[1]["json"]["profit_loss"] == 2.0

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_sell_decision_profit_loss(self, mock_get, mock_patch):
        """매도 결정: -(현재-결정)/결정 * 100"""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = [
            {"id": "bbb", "decision": "매도", "current_price": 50000000, "profit_loss": None},
        ]
        ticker_resp = MagicMock()
        ticker_resp.json.return_value = [{"trade_price": 51000000}]
        mock_get.side_effect = [supabase_resp, ticker_resp]
        mock_patch.return_value = MagicMock(ok=True)

        sd.update_past_performance()

        call_kwargs = mock_patch.call_args
        # 매도: -price_change_pct = -(51M-50M)/50M*100 = -2.0
        assert call_kwargs[1]["json"]["profit_loss"] == -2.0

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_hold_decision_profit_loss(self, mock_get, mock_patch):
        """관망 결정: -price_change_pct (기회비용)"""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = [
            {"id": "ccc", "decision": "관망", "current_price": 50000000, "profit_loss": None},
        ]
        ticker_resp = MagicMock()
        ticker_resp.json.return_value = [{"trade_price": 52000000}]
        mock_get.side_effect = [supabase_resp, ticker_resp]
        mock_patch.return_value = MagicMock(ok=True)

        sd.update_past_performance()

        call_kwargs = mock_patch.call_args
        # 관망: -price_change_pct = -(52M-50M)/50M*100 = -4.0
        assert call_kwargs[1]["json"]["profit_loss"] == -4.0

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_batch_grouping_by_profit_loss(self, mock_get, mock_patch):
        """같은 profit_loss 값을 가진 결정들이 한 PATCH로 묶인다."""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        # Two buy decisions with same price → same profit_loss
        supabase_resp.json.return_value = [
            {"id": "d1", "decision": "매수", "current_price": 50000000, "profit_loss": None},
            {"id": "d2", "decision": "매수", "current_price": 50000000, "profit_loss": None},
        ]
        ticker_resp = MagicMock()
        ticker_resp.json.return_value = [{"trade_price": 51000000}]
        mock_get.side_effect = [supabase_resp, ticker_resp]
        mock_patch.return_value = MagicMock(ok=True)

        sd.update_past_performance()

        # Both grouped into one PATCH call
        assert mock_patch.call_count == 1
        url = mock_patch.call_args[0][0]
        assert "d1" in url
        assert "d2" in url

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_empty_decisions_no_patch(self, mock_get, mock_patch):
        """결정이 없으면 PATCH 호출하지 않는다."""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = []
        mock_get.return_value = supabase_resp

        sd.update_past_performance()

        mock_patch.assert_not_called()

    @patch("save_decision.requests.get")
    def test_supabase_get_failure_early_return(self, mock_get):
        """Supabase GET 실패 시 조기 리턴."""
        resp = MagicMock()
        resp.ok = False
        mock_get.return_value = resp

        # Should not raise
        sd.update_past_performance()

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_ticker_api_failure_early_return(self, mock_get, mock_patch):
        """Upbit ticker API 실패 시 조기 리턴."""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = [
            {"id": "x", "decision": "매수", "current_price": 50000000},
        ]
        # Ticker call raises
        mock_get.side_effect = [supabase_resp, Exception("network error")]

        sd.update_past_performance()

        mock_patch.assert_not_called()

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_skip_decision_with_no_price(self, mock_get, mock_patch):
        """current_price가 없거나 0인 결정은 건너뛴다."""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = [
            {"id": "z1", "decision": "매수", "current_price": 0, "profit_loss": None},
            {"id": "z2", "decision": "매수", "current_price": None, "profit_loss": None},
        ]
        ticker_resp = MagicMock()
        ticker_resp.json.return_value = [{"trade_price": 51000000}]
        mock_get.side_effect = [supabase_resp, ticker_resp]

        sd.update_past_performance()

        mock_patch.assert_not_called()

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_patch_exception_handled(self, mock_get, mock_patch):
        """PATCH 호출 시 예외가 발생해도 crash하지 않는다."""
        supabase_resp = MagicMock()
        supabase_resp.ok = True
        supabase_resp.json.return_value = [
            {"id": "e1", "decision": "매수", "current_price": 50000000},
        ]
        ticker_resp = MagicMock()
        ticker_resp.json.return_value = [{"trade_price": 51000000}]
        mock_get.side_effect = [supabase_resp, ticker_resp]
        mock_patch.side_effect = Exception("patch failed")

        # Should not raise
        sd.update_past_performance()


# ---------------------------------------------------------------------------
# mark_feedback_applied
# ---------------------------------------------------------------------------
class TestMarkFeedbackApplied:
    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_normal_feedback_applied(self, mock_get, mock_patch):
        """미반영 피드백을 가져와 applied=true로 PATCH한다."""
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = [{"id": "f1"}, {"id": "f2"}]
        mock_get.return_value = resp
        mock_patch.return_value = MagicMock(ok=True)

        sd.mark_feedback_applied()

        mock_patch.assert_called_once()
        call_args = mock_patch.call_args
        assert call_args[1]["json"]["applied"] is True
        assert "applied_at" in call_args[1]["json"]
        # URL should contain both ids
        url = call_args[0][0]
        assert "f1" in url
        assert "f2" in url

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_no_feedback_no_patch(self, mock_get, mock_patch):
        """피드백이 없으면 PATCH 호출하지 않는다."""
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = []
        mock_get.return_value = resp

        sd.mark_feedback_applied()

        mock_patch.assert_not_called()

    @patch("save_decision.requests.patch")
    @patch("save_decision.requests.get")
    def test_get_failure_graceful(self, mock_get, mock_patch):
        """GET 실패 시 graceful 처리."""
        resp = MagicMock()
        resp.ok = False
        mock_get.return_value = resp

        sd.mark_feedback_applied()

        mock_patch.assert_not_called()

    @patch("save_decision.requests.get")
    def test_exception_graceful(self, mock_get):
        """예외 발생 시 crash하지 않는다."""
        mock_get.side_effect = Exception("connection error")

        # Should not raise
        sd.mark_feedback_applied()


# ---------------------------------------------------------------------------
# save_portfolio_snapshot
# ---------------------------------------------------------------------------
class TestSavePortfolioSnapshot:
    @patch("save_decision.supabase_post")
    @patch("save_decision.subprocess.run" if hasattr(sd, "subprocess") else "subprocess.run")
    def test_normal_portfolio_saved(self, mock_run, mock_post):
        """정상적인 포트폴리오 JSON → supabase_post 호출."""
        portfolio = {
            "krw_balance": 500000,
            "holdings": [
                {"currency": "BTC", "eval_amount": 1000000}
            ],
            "total_eval": 1500000,
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(portfolio),
        )
        mock_post.return_value = {"id": "snap1"}

        # Need to import subprocess inside the function, so we patch at module level
        with patch.object(sd, "__builtins__", sd.__builtins__):
            # Actually, save_portfolio_snapshot imports subprocess locally
            with patch("subprocess.run", mock_run):
                sd.save_portfolio_snapshot()

        mock_post.assert_called_once_with("portfolio_snapshots", {
            "total_krw": 500000,
            "total_crypto_value": 1000000,
            "total_value": 1500000,
            "holdings": json.dumps([{"currency": "BTC", "eval_amount": 1000000}], ensure_ascii=False),
        })

    @patch("save_decision.supabase_post")
    def test_subprocess_failure_early_return(self, mock_post):
        """subprocess 실패 시 supabase_post 호출하지 않는다."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            sd.save_portfolio_snapshot()

        mock_post.assert_not_called()

    @patch("save_decision.supabase_post")
    def test_invalid_json_output_caught(self, mock_post):
        """subprocess가 유효하지 않은 JSON 출력 시 예외 처리."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not json at all")
            # Should not raise
            sd.save_portfolio_snapshot()

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
class TestMain:
    @patch("save_decision.SUPABASE_KEY", "")
    @patch("save_decision.SUPABASE_URL", "")
    def test_missing_env_vars_exit_1(self):
        """SUPABASE 환경변수 미설정 시 exit(1)."""
        with pytest.raises(SystemExit) as exc_info:
            sd.main()
        assert exc_info.value.code == 1

    @patch("save_decision.SUPABASE_KEY", "")
    @patch("save_decision.SUPABASE_URL", "https://fake.supabase.co")
    def test_missing_key_exit_1(self):
        """SUPABASE_KEY만 없어도 exit(1)."""
        with pytest.raises(SystemExit) as exc_info:
            sd.main()
        assert exc_info.value.code == 1

    @patch("save_decision.mark_feedback_applied")
    @patch("save_decision.save_portfolio_snapshot")
    @patch("save_decision.update_past_performance")
    @patch("save_decision.save_decision")
    @patch("save_decision.extract_json_from_response")
    @patch("save_decision.SUPABASE_KEY", "fake-key")
    @patch("save_decision.SUPABASE_URL", "https://fake.supabase.co")
    def test_valid_json_from_argv(self, mock_extract, mock_save, mock_perf, mock_snap, mock_fb):
        """argv로 유효한 JSON 전달 시 save_decision 호출."""
        mock_extract.return_value = {"decision": "buy"}
        mock_save.return_value = {"id": "123"}

        with patch.object(sys, "argv", ["save_decision.py", '{"decision": "buy"}']):
            sd.main()

        mock_extract.assert_called_once_with('{"decision": "buy"}')
        mock_save.assert_called_once_with({"decision": "buy"})
        mock_perf.assert_called_once()
        mock_snap.assert_called_once()
        mock_fb.assert_called_once()

    @patch("save_decision.mark_feedback_applied")
    @patch("save_decision.save_portfolio_snapshot")
    @patch("save_decision.update_past_performance")
    @patch("save_decision.save_decision")
    @patch("save_decision.extract_json_from_response")
    @patch("save_decision.SUPABASE_KEY", "fake-key")
    @patch("save_decision.SUPABASE_URL", "https://fake.supabase.co")
    def test_valid_json_from_stdin(self, mock_extract, mock_save, mock_perf, mock_snap, mock_fb):
        """stdin으로 유효한 JSON 전달 시 save_decision 호출."""
        mock_extract.return_value = {"decision": "hold"}
        mock_save.return_value = {"id": "456"}

        from io import StringIO
        with patch.object(sys, "argv", ["save_decision.py"]):
            with patch.object(sys, "stdin", StringIO('{"decision": "hold"}')):
                sd.main()

        mock_extract.assert_called_once_with('{"decision": "hold"}')
        mock_save.assert_called_once_with({"decision": "hold"})

    @patch("save_decision.extract_json_from_response")
    @patch("save_decision.SUPABASE_KEY", "fake-key")
    @patch("save_decision.SUPABASE_URL", "https://fake.supabase.co")
    def test_invalid_input_exit_1(self, mock_extract):
        """JSON 파싱 실패 시 exit(1)."""
        mock_extract.return_value = None

        with patch.object(sys, "argv", ["save_decision.py", "not json"]):
            with pytest.raises(SystemExit) as exc_info:
                sd.main()
            assert exc_info.value.code == 1

    @patch("save_decision.update_past_performance")
    @patch("save_decision.save_decision")
    @patch("save_decision.extract_json_from_response")
    @patch("save_decision.SUPABASE_KEY", "fake-key")
    @patch("save_decision.SUPABASE_URL", "https://fake.supabase.co")
    def test_save_decision_returns_none_exit_1(self, mock_extract, mock_save, mock_perf):
        """save_decision이 None 반환 시 exit(1)."""
        mock_extract.return_value = {"decision": "buy"}
        mock_save.return_value = None

        with patch.object(sys, "argv", ["save_decision.py", '{"decision":"buy"}']):
            with pytest.raises(SystemExit) as exc_info:
                sd.main()
            assert exc_info.value.code == 1

    @patch("save_decision.mark_feedback_applied")
    @patch("save_decision.save_portfolio_snapshot")
    @patch("save_decision.update_past_performance")
    @patch("save_decision.save_decision")
    @patch("save_decision.extract_json_from_response")
    @patch("save_decision.SUPABASE_KEY", "fake-key")
    @patch("save_decision.SUPABASE_URL", "https://fake.supabase.co")
    def test_update_past_performance_exception_does_not_block(self, mock_extract, mock_save, mock_perf, mock_snap, mock_fb):
        """update_past_performance 예외가 main을 중단하지 않는다."""
        mock_extract.return_value = {"decision": "hold"}
        mock_save.return_value = {"id": "789"}
        mock_perf.side_effect = Exception("perf error")

        with patch.object(sys, "argv", ["save_decision.py", '{"decision":"hold"}']):
            sd.main()

        # save_decision still called despite perf error
        mock_save.assert_called_once()
        mock_snap.assert_called_once()
        mock_fb.assert_called_once()
