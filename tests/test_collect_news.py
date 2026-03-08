"""
collect_news.py unit tests

All network calls are mocked - no real API calls are made.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

# Patch load_dotenv before importing the module
with patch("dotenv.load_dotenv"):
    from scripts.collect_news import (
        TAVILY_API,
        MONTHLY_LIMIT,
        CRYPTO_QUERIES,
        MACRO_QUERIES,
        _build_queries,
        _load_usage,
        _save_usage,
        _budget_queries,
        fetch_news,
        main,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tavily_response(num_results=2, category="crypto_btc"):
    """Build a fake Tavily API response."""
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Test Article {i}",
            "url": f"https://example.com/article-{category}-{i}",
            "content": f"Content for article {i} about {category}",
            "published_date": "2026-03-08T10:00:00Z",
            "score": 0.95 - i * 0.1,
        })
    return {"results": results}


def _make_mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status.return_value = None
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            response=mock_resp
        )
    return mock_resp


# ---------------------------------------------------------------------------
# 1. Tavily API call construction
# ---------------------------------------------------------------------------

class TestFetchNews:
    """Verify Tavily API call construction."""

    @patch("scripts.collect_news.requests.post")
    def test_api_call_params(self, mock_post):
        """fetch_news sends correct payload to Tavily API."""
        mock_post.return_value = _make_mock_response(_make_tavily_response(1))

        fetch_news("test-api-key", "bitcoin BTC price", max_results=5)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == TAVILY_API
        payload = call_args[1]["json"]
        assert payload["api_key"] == "test-api-key"
        assert payload["query"] == "bitcoin BTC price"
        assert payload["max_results"] == 5
        assert payload["search_depth"] == "advanced"
        assert payload["topic"] == "news"
        assert payload["days"] == 1
        assert payload["include_answer"] is False
        assert call_args[1]["timeout"] == 30

    @patch("scripts.collect_news.requests.post")
    def test_article_fields_extraction(self, mock_post):
        """fetch_news extracts the correct fields from results."""
        mock_post.return_value = _make_mock_response(_make_tavily_response(2))

        articles = fetch_news("key", "query")

        assert len(articles) == 2
        for a in articles:
            assert set(a.keys()) == {"title", "url", "content", "published_date", "score"}

    @patch("scripts.collect_news.requests.post")
    def test_content_truncation(self, mock_post):
        """Content longer than 500 chars is truncated."""
        long_content = "A" * 1000
        resp_data = {"results": [{"title": "T", "url": "http://x", "content": long_content}]}
        mock_post.return_value = _make_mock_response(resp_data)

        articles = fetch_news("key", "query")

        assert len(articles[0]["content"]) == 500

    @patch("scripts.collect_news.requests.post")
    def test_empty_results(self, mock_post):
        """fetch_news handles empty results gracefully."""
        mock_post.return_value = _make_mock_response({"results": []})

        articles = fetch_news("key", "query")
        assert articles == []

    @patch("scripts.collect_news.requests.post")
    def test_missing_fields_defaults(self, mock_post):
        """Missing fields in results use sensible defaults."""
        resp_data = {"results": [{}]}
        mock_post.return_value = _make_mock_response(resp_data)

        articles = fetch_news("key", "query")
        assert articles[0]["title"] == ""
        assert articles[0]["url"] == ""
        assert articles[0]["content"] == ""
        assert articles[0]["published_date"] == ""
        assert articles[0]["score"] == 0


# ---------------------------------------------------------------------------
# 2. Usage tracking - monthly counter, auto-reset
# ---------------------------------------------------------------------------

class TestUsageTracking:
    """Test monthly usage counter in tavily_usage.json."""

    @patch("scripts.collect_news.USAGE_FILE")
    def test_load_usage_no_file(self, mock_file):
        """Returns fresh usage when file does not exist."""
        mock_file.exists.return_value = False

        usage = _load_usage()

        assert usage["count"] == 0
        assert usage["month"] == datetime.now().strftime("%Y-%m")

    @patch("scripts.collect_news.USAGE_FILE")
    def test_load_usage_current_month(self, mock_file):
        """Returns existing usage when month matches."""
        current_month = datetime.now().strftime("%Y-%m")
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = json.dumps({"month": current_month, "count": 42})

        usage = _load_usage()

        assert usage["count"] == 42
        assert usage["month"] == current_month

    @patch("scripts.collect_news.USAGE_FILE")
    def test_load_usage_old_month_resets(self, mock_file):
        """Resets counter when month has changed."""
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = json.dumps({"month": "2025-01", "count": 500})

        usage = _load_usage()

        assert usage["count"] == 0
        assert usage["month"] == datetime.now().strftime("%Y-%m")

    def test_save_usage_creates_dir(self, tmp_path):
        """_save_usage creates parent directory if needed."""
        usage_file = tmp_path / "subdir" / "tavily_usage.json"
        with patch("scripts.collect_news.USAGE_FILE", usage_file):
            _save_usage({"month": "2026-03", "count": 10})

        assert usage_file.exists()
        data = json.loads(usage_file.read_text())
        assert data["count"] == 10

    def test_save_usage_writes_json(self, tmp_path):
        """_save_usage writes valid JSON with correct data."""
        usage_file = tmp_path / "tavily_usage.json"
        with patch("scripts.collect_news.USAGE_FILE", usage_file):
            _save_usage({"month": "2026-03", "count": 55})

        data = json.loads(usage_file.read_text())
        assert data == {"month": "2026-03", "count": 55}


# ---------------------------------------------------------------------------
# 3. Weekend vs weekday query differences
# ---------------------------------------------------------------------------

class TestBuildQueries:
    """Weekend adds onchain/whale queries; weekday does not."""

    @patch("scripts.collect_news.datetime")
    def test_weekday_queries(self, mock_dt):
        """Weekday: 3 crypto + 2 macro = 5 queries, no onchain."""
        # Monday
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)  # Monday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()

        categories = [q["category"] for q in queries]
        assert "crypto_onchain" not in categories
        assert len(queries) == 5

    @patch("scripts.collect_news.datetime")
    def test_weekend_queries(self, mock_dt):
        """Weekend: 4 crypto + 2 macro = 6 queries, includes onchain."""
        # Saturday
        mock_dt.now.return_value = datetime(2026, 3, 7, 12, 0)  # Saturday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()

        categories = [q["category"] for q in queries]
        assert "crypto_onchain" in categories
        assert len(queries) == 6

    @patch("scripts.collect_news.datetime")
    def test_weekend_only_flag_stripped(self, mock_dt):
        """weekend_only key is stripped from returned query dicts."""
        mock_dt.now.return_value = datetime(2026, 3, 7, 12, 0)  # Saturday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        queries = _build_queries()

        for q in queries:
            assert "weekend_only" not in q


# ---------------------------------------------------------------------------
# 4. Rate limiting - monthly quota exceeded
# ---------------------------------------------------------------------------

class TestBudgetQueries:
    """Budget queries based on remaining monthly quota."""

    @patch("scripts.collect_news._build_queries")
    def test_zero_remaining(self, mock_build):
        """Returns empty list when monthly limit is reached."""
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT}
        result = _budget_queries(usage)
        assert result == []
        mock_build.assert_not_called()

    @patch("scripts.collect_news._build_queries")
    def test_over_limit(self, mock_build):
        """Returns empty list when count exceeds limit."""
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT + 10}
        result = _budget_queries(usage)
        assert result == []

    @patch("scripts.collect_news._build_queries")
    def test_enough_budget(self, mock_build):
        """Returns all queries when budget is sufficient."""
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
            {"query": "q2", "category": "macro_geo", "max_results": 3},
        ]
        usage = {"month": "2026-03", "count": 0}

        result = _budget_queries(usage)

        assert len(result) == 2

    @patch("scripts.collect_news._build_queries")
    def test_partial_budget_prioritized(self, mock_build):
        """When budget is limited, queries are selected by priority."""
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
            {"query": "q2", "category": "crypto_alt", "max_results": 3},
            {"query": "q3", "category": "crypto_regulation", "max_results": 3},
            {"query": "q4", "category": "macro_geo", "max_results": 3},
            {"query": "q5", "category": "macro_economy", "max_results": 3},
        ]
        # Only 2 remaining
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT - 2}

        result = _budget_queries(usage)

        assert len(result) == 2
        categories = [q["category"] for q in result]
        # crypto_btc is highest priority, macro_geo is second
        assert categories[0] == "crypto_btc"
        assert categories[1] == "macro_geo"

    @patch("scripts.collect_news._build_queries")
    def test_single_remaining(self, mock_build):
        """Only highest priority query when 1 call remains."""
        mock_build.return_value = [
            {"query": "q1", "category": "crypto_btc", "max_results": 5},
            {"query": "q2", "category": "macro_geo", "max_results": 3},
        ]
        usage = {"month": "2026-03", "count": MONTHLY_LIMIT - 1}

        result = _budget_queries(usage)

        assert len(result) == 1
        assert result[0]["category"] == "crypto_btc"


# ---------------------------------------------------------------------------
# 5. News categorization - different query types
# ---------------------------------------------------------------------------

class TestNewsCategorization:
    """Verify query types and categories."""

    def test_crypto_query_categories(self):
        """CRYPTO_QUERIES has expected categories."""
        cats = {q["category"] for q in CRYPTO_QUERIES}
        assert "crypto_btc" in cats
        assert "crypto_alt" in cats
        assert "crypto_regulation" in cats
        assert "crypto_onchain" in cats

    def test_macro_query_categories(self):
        """MACRO_QUERIES has expected categories."""
        cats = {q["category"] for q in MACRO_QUERIES}
        assert "macro_geo" in cats
        assert "macro_economy" in cats

    def test_onchain_is_weekend_only(self):
        """crypto_onchain is marked weekend_only."""
        onchain = [q for q in CRYPTO_QUERIES if q["category"] == "crypto_onchain"]
        assert len(onchain) == 1
        assert onchain[0].get("weekend_only") is True

    def test_non_weekend_queries_not_weekend_only(self):
        """Other crypto queries are not weekend_only."""
        for q in CRYPTO_QUERIES:
            if q["category"] != "crypto_onchain":
                assert not q.get("weekend_only"), f"{q['category']} should not be weekend_only"


# ---------------------------------------------------------------------------
# 6. Error handling - API failure, timeout, invalid response
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Error handling for API failures."""

    @patch("scripts.collect_news.requests.post")
    def test_api_http_error(self, mock_post):
        """fetch_news raises on HTTP error (e.g., 401, 500)."""
        mock_post.return_value = _make_mock_response({}, status_code=500)

        with pytest.raises(requests.HTTPError):
            fetch_news("bad-key", "query")

    @patch("scripts.collect_news.requests.post")
    def test_api_timeout(self, mock_post):
        """fetch_news raises on timeout."""
        mock_post.side_effect = requests.Timeout("Connection timed out")

        with pytest.raises(requests.Timeout):
            fetch_news("key", "query")

    @patch("scripts.collect_news.requests.post")
    def test_api_connection_error(self, mock_post):
        """fetch_news raises on connection error."""
        mock_post.side_effect = requests.ConnectionError("DNS resolution failed")

        with pytest.raises(requests.ConnectionError):
            fetch_news("key", "query")

    @patch("scripts.collect_news.requests.post")
    def test_invalid_json_response(self, mock_post):
        """fetch_news handles response with no 'results' key."""
        mock_post.return_value = _make_mock_response({"unexpected": "data"})

        articles = fetch_news("key", "query")
        assert articles == []

    @patch.dict(os.environ, {}, clear=True)
    @patch("scripts.collect_news._load_usage")
    def test_main_no_api_key(self, mock_load):
        """main() raises when TAVILY_API_KEY is missing."""
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            main()

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._budget_queries")
    @patch("scripts.collect_news._load_usage")
    def test_main_quota_exhausted(self, mock_load, mock_budget, mock_save):
        """main() raises when monthly quota is exhausted."""
        mock_load.return_value = {"month": "2026-03", "count": MONTHLY_LIMIT}
        mock_budget.return_value = []

        with pytest.raises(RuntimeError, match="월간 한도 소진"):
            main()


# ---------------------------------------------------------------------------
# 7. Output format - verify JSON structure
# ---------------------------------------------------------------------------

class TestOutputFormat:
    """Verify main() output JSON structure."""

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_main_output_structure(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        """main() prints valid JSON with expected top-level keys."""
        mock_load.return_value = {"month": "2026-03", "count": 0}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)  # Monday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        mock_post.return_value = _make_mock_response(_make_tavily_response(1))

        # Patch _budget_queries to return a small set
        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "bitcoin BTC", "category": "crypto_btc", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)

        assert "timestamp" in output
        assert "day_type" in output
        assert "queries" in output
        assert "articles_count" in output
        assert "by_category" in output
        assert "articles" in output
        assert "tavily_usage" in output

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_tavily_usage_fields(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        """tavily_usage section has required fields."""
        mock_load.return_value = {"month": "2026-03", "count": 10}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        mock_post.return_value = _make_mock_response(_make_tavily_response(1))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)
        usage = output["tavily_usage"]

        assert "month" in usage
        assert "api_calls_used" in usage
        assert "limit" in usage
        assert usage["limit"] == MONTHLY_LIMIT
        assert "remaining" in usage
        assert "this_run_calls" in usage

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_articles_have_category(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        """Each article in output has a category field."""
        mock_load.return_value = {"month": "2026-03", "count": 0}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        mock_post.return_value = _make_mock_response(_make_tavily_response(2))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)

        for article in output["articles"]:
            assert "category" in article
            assert article["category"] == "crypto_btc"

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_duplicate_urls_deduplicated(self, mock_load, mock_save, mock_post, mock_dt, capsys):
        """Articles with duplicate URLs are deduplicated."""
        mock_load.return_value = {"month": "2026-03", "count": 0}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        # Both queries return articles with same URLs
        dup_response = {
            "results": [
                {"title": "Same", "url": "https://example.com/same", "content": "c1"},
                {"title": "Unique1", "url": "https://example.com/unique1", "content": "c2"},
            ]
        }
        mock_post.return_value = _make_mock_response(dup_response)

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
                {"query": "q2", "category": "macro_geo", "max_results": 3},
            ]
            main()

        output = json.loads(capsys.readouterr().out)
        urls = [a["url"] for a in output["articles"]]
        # Same URL appears in both query results, should be deduplicated
        assert len(urls) == len(set(urls))

    @patch("scripts.collect_news.datetime")
    @patch("scripts.collect_news.requests.post")
    @patch("scripts.collect_news._save_usage")
    @patch("scripts.collect_news._load_usage")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_usage_count_incremented(self, mock_load, mock_save, mock_post, mock_dt):
        """main() increments usage count by number of API calls made."""
        mock_load.return_value = {"month": "2026-03", "count": 50}
        mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        mock_post.return_value = _make_mock_response(_make_tavily_response(1))

        with patch("scripts.collect_news._budget_queries") as mock_budget:
            mock_budget.return_value = [
                {"query": "q1", "category": "crypto_btc", "max_results": 3},
                {"query": "q2", "category": "macro_geo", "max_results": 3},
            ]
            main()

        # _save_usage should be called with count incremented by 2 (number of queries)
        saved = mock_save.call_args[0][0]
        assert saved["count"] == 52
