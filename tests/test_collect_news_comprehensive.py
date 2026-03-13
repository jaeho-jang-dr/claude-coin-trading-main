"""
collect_news.py comprehensive unit tests.

Covers:
  - Tavily API call construction and response parsing
  - Content truncation
  - Usage tracking (monthly + daily counters, auto-reset)
  - Weekend vs weekday query selection
  - Rate limiting / budget (monthly + daily)
  - Error handling (HTTP errors, timeout, missing API key)
  - Output JSON structure and deduplication
  - Daily limit enforcement

All network calls are mocked - no real API calls are made.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

with patch("dotenv.load_dotenv"):
    from scripts.collect_news import (
        TAVILY_API,
        MONTHLY_LIMIT,
        DAILY_LIMIT,
        CRYPTO_QUERIES,
        MACRO_QUERIES,
        _build_queries,
        _load_usage,
        _save_usage,
        _budget_queries,
        fetch_news,
        main,
    )


# ── Helpers ──────────────────────────────────────────────

def _tavily_response(num_results=2, category="crypto_btc"):
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Article {i} - {category}",
            "url": f"https://example.com/{category}-{i}",
            "content": f"Content about {category} article {i}",
            "published_date": "2026-03-13T10:00:00Z",
            "score": 0.95 - i * 0.1,
        })
    return {"results": results}


def _mock_resp(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


# ══════════════════════════════════════════════════════════════
# fetch_news - API call construction
# ══════════════════════════════════════════════════════════════

class TestFetchNews:
    @patch("scripts.collect_news.requests.post")
    def test_api_payload_construction(self, mock_post):
        mock_post.return_value = _mock_resp(_tavily_response(1))

        fetch_news("test-key", "bitcoin price", max_results=5)

        payload = mock_post.call_args[1]["json"]
        assert payload["api_key"] == "test-key"
        assert payload["query"] == "bitcoin price"
        assert payload["max_results"] == 5
        assert payload["search_depth"] == "advanced"
        assert payload["topic"] == "news"
        assert payload["days"] == 1
        assert payload["include_answer"] is False

    @patch("scripts.collect_news.requests.post")
    def test_extracts_correct_fields(self, mock_post):
        mock_post.return_value = _mock_resp(_tavily_response(3))

        articles = fetch_news("key", "query")

        assert len(articles) == 3
        for a in articles:
            assert set(a.keys()) == {"title", "url", "content", "published_date", "score"}

    @patch("scripts.collect_news.requests.post")
    def test_content_truncation_at_500(self, mock_post):
        resp = {"results": [{"title": "T", "url": "http://x", "content": "A" * 1000}]}
        mock_post.return_value = _mock_resp(resp)

        articles = fetch_news("key", "query")
        assert len(articles[0]["content"]) == 500

    @patch("scripts.collect_news.requests.post")
    def test_empty_results(self, mock_post):
        mock_post.return_value = _mock_resp({"results": []})
        assert fetch_news("key", "query") == []

    @patch("scripts.collect_news.requests.post")
    def test_missing_fields_use_defaults(self, mock_post):
        mock_post.return_value = _mock_resp({"results": [{}]})

        articles = fetch_news("key", "query")
        a = articles[0]
        assert a["title"] == ""
        assert a["url"] == ""
        assert a["content"] == ""
        assert a["published_date"] == ""
        assert a["score"] == 0

    @patch("scripts.collect_news.requests.post")
    def test_none_content_handled(self, mock_post):
        """content=None should not crash."""
        resp = {"results": [{"title": "T", "url": "http://x", "content": None}]}
        mock_post.return_value = _mock_resp(resp)

        articles = fetch_news("key", "query")
        assert articles[0]["content"] == ""

    @patch("scripts.collect_news.requests.post")
    def test_no_results_key(self, mock_post):
        mock_post.return_value = _mock_resp({"unexpected": "data"})
        assert fetch_news("key", "query") == []


# ══════════════════════════════════════════════════════════════
# Usage tracking
# ══════════════════════════════════════════════════════════════

class TestUsageTracking:
    @patch("scripts.collect_news.USAGE_FILE")
    def test_no_file_returns_fresh(self, mock_file):
        mock_file.exists.return_value = False
        usage = _load_usage()
        assert usage["count"] == 0
        assert usage["month"] == datetime.now().strftime("%Y-%m")

    @patch("scripts.collect_news.USAGE_FILE")
    def test_current_month_preserved(self, mock_file):
        month = datetime.now().strftime("%Y-%m")
        today = datetime.now().strftime("%Y-%m-%d")
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = json.dumps({
            "month": month, "count": 42, "daily": {today: 5}
        })

        usage = _load_usage()
        assert usage["count"] == 42
        assert usage["daily"][today] == 5

    @patch("scripts.collect_news.USAGE_FILE")
    def test_old_month_resets(self, mock_file):
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = json.dumps({"month": "2025-01", "count": 500})

        usage = _load_usage()
        assert usage["count"] == 0

    @patch("scripts.collect_news.USAGE_FILE")
    def test_new_day_initializes_daily(self, mock_file):
        """When the day changes, the new day's counter should start at 0."""
        month = datetime.now().strftime("%Y-%m")
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = json.dumps({
            "month": month, "count": 10, "daily": {"2026-03-12": 30}
        })

        usage = _load_usage()
        today = datetime.now().strftime("%Y-%m-%d")
        assert usage["daily"].get(today, 0) == 0

    def test_save_creates_directory(self, tmp_path):
        usage_file = tmp_path / "subdir" / "tavily_usage.json"
        with patch("scripts.collect_news.USAGE_FILE", usage_file):
            _save_usage({"month": "2026-03", "count": 10})
        assert usage_file.exists()

    def test_save_writes_valid_json(self, tmp_path):
        usage_file = tmp_path / "tavily_usage.json"
        with patch("scripts.collect_news.USAGE_FILE", usage_file):
            _save_usage({"month": "2026-03", "count": 55, "daily": {"2026-03-13": 10}})
        data = json.loads(usage_file.read_text())
        assert data["count"] == 55


# ══════════════════════════════════════════════════════════════
# Weekend / weekday queries
# ══════════════════════════════════════════════════════════════

class TestBuildQueries:
    @patch("scripts.collect_news.datetime")
    def test_weekday_excludes_onchain(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)  # Monday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()
        cats = [q["category"] for q in queries]
        assert "crypto_onchain" not in cats
        assert len(queries) == 5  # 3 crypto + 2 macro

    @patch("scripts.collect_news.datetime")
    def test_weekend_includes_onchain(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 12, 0)  # Saturday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()
        cats = [q["category"] for q in queries]
        assert "crypto_onchain" in cats
        assert len(queries) == 6

    @patch("scripts.collect_news.datetime")
    def test_weekend_only_key_stripped(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 7, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        for q in _build_queries():
            assert "weekend_only" not in q


# ══════════════════════════════════════════════════════════════
# Budget / rate limiting
# ══════════════════════════════════════════════════════════════

class TestBudgetQueries:
    @patch("scripts.collect_news._build_queries")
    def test_monthly_limit_reached(self, mock_build):
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT, "daily": {}}
        assert _budget_queries(usage) == []

    @patch("scripts.collect_news._build_queries")
    def test_daily_limit_reached(self, mock_build):
        today = datetime.now().strftime("%Y-%m-%d")
        usage = {"month": "2026-03", "count": 0, "daily": {today: DAILY_LIMIT}}
        assert _budget_queries(usage) == []

    @patch("scripts.collect_news._build_queries")
    def test_enough_budget_returns_all(self, mock_build):
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
            {"query": "q2", "category": "macro_geo", "max_results": 3},
        ]
        usage = {"month": "2026-03", "count": 0, "daily": {}}
        assert len(_budget_queries(usage)) == 2

    @patch("scripts.collect_news._build_queries")
    def test_partial_budget_prioritized(self, mock_build):
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
            {"query": "q2", "category": "crypto_alt", "max_results": 3},
            {"query": "q3", "category": "macro_geo", "max_results": 3},
            {"query": "q4", "category": "macro_economy", "max_results": 3},
            {"query": "q5", "category": "crypto_regulation", "max_results": 3},
        ]
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT - 2, "daily": {}}

        result = _budget_queries(usage)
        assert len(result) == 2
        cats = [q["category"] for q in result]
        assert cats[0] == "crypto_btc"
        assert cats[1] == "macro_geo"

    @patch("scripts.collect_news._build_queries")
    def test_daily_remaining_limits_queries(self, mock_build):
        """Daily remaining is the bottleneck, not monthly."""
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
            {"query": "q2", "category": "macro_geo", "max_results": 3},
            {"query": "q3", "category": "crypto_alt", "max_results": 3},
        ]
        today = datetime.now().strftime("%Y-%m-%d")
        # Monthly has plenty, but daily only has 1 left
        usage = {"month": "2026-03", "count": 10, "daily": {today: DAILY_LIMIT - 1}}

        result = _budget_queries(usage)
        assert len(result) == 1
        assert result[0]["category"] == "crypto_btc"


# ══════════════════════════════════════════════════════════════
# Query categorization
# ══════════════════════════════════════════════════════════════

class TestQueryCategorization:
    def test_crypto_categories(self):
        cats = {q["category"] for q in CRYPTO_QUERIES}
        assert {"crypto_btc", "crypto_alt", "crypto_regulation", "crypto_onchain"} == cats

    def test_macro_categories(self):
        cats = {q["category"] for q in MACRO_QUERIES}
        assert {"macro_geo", "macro_economy"} == cats

    def test_onchain_is_weekend_only(self):
        onchain = [q for q in CRYPTO_QUERIES if q["category"] == "crypto_onchain"]
        assert len(onchain) == 1
        assert onchain[0].get("weekend_only") is True

    def test_other_queries_not_weekend_only(self):
        for q in CRYPTO_QUERIES:
            if q["category"] != "crypto_onchain":
                assert not q.get("weekend_only")


# ══════════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════════

class TestErrorHandling:
    @patch("scripts.collect_news.requests.post")
    def test_http_error_raises(self, mock_post):
        mock_post.return_value = _mock_resp({}, status_code=500)
        with pytest.raises(requests.HTTPError):
            fetch_news("key", "query")

    @patch("scripts.collect_news.requests.post")
    def test_timeout_raises(self, mock_post):
        mock_post.side_effect = requests.Timeout("timeout")
        with pytest.raises(requests.Timeout):
            fetch_news("key", "query")

    @patch("scripts.collect_news.requests.post")
    def test_connection_error_raises(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("DNS failure")
        with pytest.raises(requests.ConnectionError):
            fetch_news("key", "query")

    @patch.dict(os.environ, {}, clear=True)
    @patch("scripts.collect_news._load_usage")
    def test_missing_api_key(self, mock_load):
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            main()

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._budget_queries")
    @patch("scripts.collect_news._load_usage")
    def test_quota_exhausted(self, mock_load, mock_budget, mock_save):
        mock_load.return_value = {"month": "2026-03", "count": MONTHLY_LIMIT, "daily": {}}
        mock_budget.return_value = []
        with pytest.raises(RuntimeError, match="한도 소진"):
            main()


# ══════════════════════════════════════════════════════════════
# main() output format
# ══════════════════════════════════════════════════════════════

class TestMainOutput:
    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_output_top_level_keys(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        mock_load.return_value = {"month": "2026-03", "count": 0, "daily": {}}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_post.return_value = _mock_resp(_tavily_response(1))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)
        for key in ["timestamp", "day_type", "queries", "articles_count",
                     "by_category", "articles", "tavily_usage"]:
            assert key in output

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_articles_have_category(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        mock_load.return_value = {"month": "2026-03", "count": 0, "daily": {}}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_post.return_value = _mock_resp(_tavily_response(2))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)
        for article in output["articles"]:
            assert "category" in article

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_duplicate_urls_deduplicated(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        mock_load.return_value = {"month": "2026-03", "count": 0, "daily": {}}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        dup_resp = {"results": [
            {"title": "Same", "url": "https://example.com/same", "content": "c"},
            {"title": "Unique", "url": "https://example.com/unique", "content": "c"},
        ]}
        mock_post.return_value = _mock_resp(dup_resp)

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
                {"query": "q2", "category": "macro_geo", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)
        urls = [a["url"] for a in output["articles"]]
        assert len(urls) == len(set(urls))

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_usage_count_incremented(self, mock_load, mock_save, mock_post, mock_dt):
        mock_load.return_value = {"month": "2026-03", "count": 50, "daily": {}}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_post.return_value = _mock_resp(_tavily_response(1))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
                {"query": "q2", "category": "macro_geo", "max_results": 3},
            ]
            main()

        saved = mock_save.call_args[0][0]
        assert saved["count"] == 52  # 50 + 2 queries

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_daily_count_incremented(self, mock_load, mock_save, mock_post, mock_dt):
        mock_load.return_value = {"month": "2026-03", "count": 0, "daily": {}}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_post.return_value = _mock_resp(_tavily_response(1))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
            ]
            main()

        saved = mock_save.call_args[0][0]
        today = "2026-03-09"
        assert saved["daily"][today] == 1
