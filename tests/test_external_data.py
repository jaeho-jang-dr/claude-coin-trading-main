"""
ExternalDataAgent 유닛 테스트

Coverage:
  - analyze_news_sentiment: 키워드 감성 분석, 한국어, 임계값
  - _compress_news: 뉴스 압축
  - _run_script: 성공, 실패, 타임아웃, JSON 파싱
  - _fetch_nvt_signal: NVT 계산, 에러 폴백
  - _inline_fusion: 인라인 Data Fusion
  - _enhance_fusion: 매크로, ETH/BTC, 뉴스, CoinGecko, CMC 보강
  - collect_all: 구조 및 에러 격리
  - load_performance_review: 성과 분석
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.external_data import (
    ExternalDataAgent,
    analyze_news_sentiment,
    _compress_news,
    _fetch_nvt_signal,
    _run_script,
    load_performance_review,
    load_user_feedback,
)


# ============================================================
# News Sentiment
# ============================================================

class TestNewsSentiment:

    def test_no_articles(self):
        result = analyze_news_sentiment({"articles": []})
        assert result["sentiment_score"] == 0
        assert result["overall_sentiment"] == "neutral"

    def test_positive_articles(self):
        articles = [
            {"title": "ETF approved! Major rally expected", "content": "bullish adoption"},
            {"title": "Institutional inflow breaks record", "content": ""},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] == 2
        assert result["sentiment_score"] > 0
        assert result["overall_sentiment"] in ("positive", "slightly_positive")

    def test_negative_articles(self):
        articles = [
            {"title": "Major hack exploit discovered", "content": "crash panic"},
            {"title": "SEC lawsuit against crypto exchange", "content": "bearish sell-off"},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["negative_count"] == 2
        assert result["sentiment_score"] < 0

    def test_mixed_articles(self):
        articles = [
            {"title": "ETF approved bullish rally", "content": ""},
            {"title": "Major hack crash panic", "content": ""},
            {"title": "No relevant crypto news", "content": ""},
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] == 1
        assert result["negative_count"] == 1
        assert result["neutral_count"] == 1

    def test_key_signals_limited_to_5(self):
        articles = [
            {"title": f"ETF approved rally bullish #{i}", "content": "adoption institutional"}
            for i in range(10)
        ]
        result = analyze_news_sentiment({"articles": articles})
        assert len(result["key_signals"]) <= 5

    def test_korean_positive_keywords(self):
        articles = [{"title": "비트코인 상승 반등 돌파", "content": "강세 매수세"}]
        result = analyze_news_sentiment({"articles": articles})
        assert result["positive_count"] == 1

    def test_korean_negative_keywords(self):
        articles = [{"title": "비트코인 폭락 패닉 하락", "content": "약세 매도세"}]
        result = analyze_news_sentiment({"articles": articles})
        assert result["negative_count"] == 1

    def test_sentiment_score_range(self):
        """감성 점수는 -100 ~ +100 범위."""
        all_pos = [{"title": "ETF approved rally", "content": ""} for _ in range(5)]
        result = analyze_news_sentiment({"articles": all_pos})
        assert -100 <= result["sentiment_score"] <= 100

    def test_overall_sentiment_thresholds(self):
        articles = [
            {"title": "ETF approved", "content": ""},
            {"title": "rally expected", "content": ""},
            {"title": "neutral story", "content": ""},
        ]
        result = analyze_news_sentiment({"articles": articles})
        # 2 pos, 0 neg, 1 neutral → 66% → "positive"
        assert result["overall_sentiment"] == "positive"

    def test_slightly_positive_threshold(self):
        """점수 10~29 → slightly_positive."""
        # 1 pos, 0 neg, 4 neutral → score = 1/5 * 100 = 20 → slightly_positive
        articles = [
            {"title": "ETF approved", "content": ""},
        ] + [{"title": "normal", "content": ""} for _ in range(4)]
        result = analyze_news_sentiment({"articles": articles})
        assert result["overall_sentiment"] == "slightly_positive"

    def test_slightly_negative_threshold(self):
        """점수 -29~-10 → slightly_negative."""
        articles = [
            {"title": "hack exploit", "content": ""},
        ] + [{"title": "normal", "content": ""} for _ in range(4)]
        result = analyze_news_sentiment({"articles": articles})
        assert result["overall_sentiment"] == "slightly_negative"

    def test_empty_dict(self):
        result = analyze_news_sentiment({})
        assert result["overall_sentiment"] == "neutral"


# ============================================================
# Compress News
# ============================================================

class TestCompressNews:

    def test_empty_articles(self):
        result = _compress_news({"articles": []})
        assert result == {"articles": []}

    def test_compression(self):
        articles = [
            {"title": "Title1", "content": "X" * 200, "category": "crypto", "score": 0.9},
            {"title": "Title2", "content": "Short", "category": "market"},
        ]
        result = _compress_news({"articles": articles, "timestamp": "2025-01-01"})
        assert result["articles_count"] == 2
        assert "crypto" in result["by_category"]
        assert len(result["categories"]["crypto"][0]["snippet"]) <= 103

    def test_no_articles_key(self):
        result = _compress_news({"timestamp": "2025-01-01"})
        assert result == {"timestamp": "2025-01-01"}


# ============================================================
# _run_script
# ============================================================

class TestRunScript:

    @patch("agents.external_data.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"key": "value"}', stderr="",
        )
        result = _run_script("test_script.py")
        assert result == {"key": "value"}

    @patch("agents.external_data.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error",
        )
        result = _run_script("test_script.py")
        assert "error" in result

    @patch("agents.external_data.subprocess.run")
    def test_timeout(self, mock_run):
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("cmd", 60)
        result = _run_script("test_script.py", timeout=60)
        assert "타임아웃" in result["error"]

    @patch("agents.external_data.subprocess.run")
    def test_json_decode_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not json", stderr="",
        )
        result = _run_script("test_script.py")
        assert "JSON" in result["error"]

    @patch("agents.external_data.subprocess.run")
    def test_empty_stdout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = _run_script("test_script.py")
        assert "error" in result

    @patch("agents.external_data.subprocess.run")
    def test_exception(self, mock_run):
        mock_run.side_effect = OSError("file not found")
        result = _run_script("test_script.py")
        assert "error" in result


# ============================================================
# NVT Signal
# ============================================================

class TestFetchNvt:

    @patch("agents.external_data.requests.get")
    def test_success_normal(self, mock_get):
        mc_resp = MagicMock(ok=True)
        mc_resp.json.return_value = {"values": [{"y": 1000000000000}]}
        tv_resp = MagicMock(ok=True)
        tv_resp.json.return_value = {"values": [{"y": 10000000000}]}
        mock_get.side_effect = [mc_resp, tv_resp]

        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 100.0
        assert result["interpretation"] == "normal"

    @patch("agents.external_data.requests.get")
    def test_overvalued(self, mock_get):
        mc_resp = MagicMock(ok=True)
        mc_resp.json.return_value = {"values": [{"y": 1500000000000}]}
        tv_resp = MagicMock(ok=True)
        tv_resp.json.return_value = {"values": [{"y": 5000000000}]}
        mock_get.side_effect = [mc_resp, tv_resp]

        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 300.0
        assert result["interpretation"] == "overvalued"

    @patch("agents.external_data.requests.get")
    def test_undervalued(self, mock_get):
        mc_resp = MagicMock(ok=True)
        mc_resp.json.return_value = {"values": [{"y": 400000000000}]}
        tv_resp = MagicMock(ok=True)
        tv_resp.json.return_value = {"values": [{"y": 10000000000}]}
        mock_get.side_effect = [mc_resp, tv_resp]

        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 40.0
        assert result["interpretation"] == "undervalued"

    @patch("agents.external_data.requests.get")
    def test_failure_returns_default(self, mock_get):
        mock_get.side_effect = Exception("network error")
        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 100.0
        assert "error" in result

    @patch("agents.external_data.requests.get")
    def test_zero_tx_volume(self, mock_get):
        mc_resp = MagicMock(ok=True)
        mc_resp.json.return_value = {"values": [{"y": 1000000}]}
        tv_resp = MagicMock(ok=True)
        tv_resp.json.return_value = {"values": [{"y": 0}]}
        mock_get.side_effect = [mc_resp, tv_resp]

        result = _fetch_nvt_signal()
        assert result["nvt_signal"] == 100.0  # fallback


# ============================================================
# Inline Fusion
# ============================================================

class TestInlineFusion:

    def test_neutral(self):
        agent = ExternalDataAgent()
        result = agent._inline_fusion({})
        assert result["total_score"] == 0
        assert result["strategy_bonus"] == 0

    def test_binance_negative(self):
        agent = ExternalDataAgent()
        results = {"binance_sentiment": {"sentiment_score": {"score": -10}}}
        result = agent._inline_fusion(results)
        assert result["total_score"] == -10

    def test_bonus_mapping_positive(self):
        agent = ExternalDataAgent()
        results = {"binance_sentiment": {"sentiment_score": {"score": 50}}}
        result = agent._inline_fusion(results)
        assert result["strategy_bonus"] == 20

    def test_bonus_mapping_negative(self):
        agent = ExternalDataAgent()
        results = {"binance_sentiment": {"sentiment_score": {"score": -50}}}
        result = agent._inline_fusion(results)
        assert result["strategy_bonus"] == -20

    def test_bonus_mapping_boundaries(self):
        """보너스 매핑 경계값 테스트."""
        agent = ExternalDataAgent()
        # score 5 → bonus 5
        r = agent._inline_fusion({"binance_sentiment": {"sentiment_score": {"score": 5}}})
        assert r["strategy_bonus"] == 5
        # score 4 → bonus 0 (4 < 5)
        r = agent._inline_fusion({"binance_sentiment": {"sentiment_score": {"score": 4}}})
        assert r["strategy_bonus"] == 0
        # score -5 → bonus -5
        r = agent._inline_fusion({"binance_sentiment": {"sentiment_score": {"score": -5}}})
        assert r["strategy_bonus"] == -5

    def test_whale_score_added(self):
        agent = ExternalDataAgent()
        results = {"whale_tracker": {"whale_score": {"score": 15}}}
        result = agent._inline_fusion(results)
        assert result["total_score"] == 15

    def test_macro_capped_at_15(self):
        agent = ExternalDataAgent()
        results = {"macro": {"analysis": {"macro_score": 50}}}
        result = agent._inline_fusion(results)
        # 50 * 0.5 = 25 → capped at 15
        assert result["total_score"] == 15

    def test_news_sentiment_positive(self):
        agent = ExternalDataAgent()
        results = {"news_sentiment": {"sentiment_score": 35}}
        result = agent._inline_fusion(results)
        assert result["total_score"] == 10

    def test_crypto_signals_high_positive(self):
        agent = ExternalDataAgent()
        results = {
            "crypto_signals": {"btc": {"anomaly_level": "HIGH", "change_24h": 5.0}}
        }
        result = agent._inline_fusion(results)
        assert result["total_score"] == 10

    def test_crypto_signals_high_negative(self):
        agent = ExternalDataAgent()
        results = {
            "crypto_signals": {"btc": {"anomaly_level": "CRITICAL", "change_24h": -5.0}}
        }
        result = agent._inline_fusion(results)
        assert result["total_score"] == -10

    def test_cmc_btc_dominance_high(self):
        agent = ExternalDataAgent()
        results = {"coinmarketcap": {"status": "success", "btc_dominance": 58.0}}
        result = agent._inline_fusion(results)
        assert result["total_score"] == 5

    def test_cmc_btc_dominance_low(self):
        agent = ExternalDataAgent()
        results = {"coinmarketcap": {"status": "success", "btc_dominance": 42.0}}
        result = agent._inline_fusion(results)
        assert result["total_score"] == -5


# ============================================================
# Enhance Fusion
# ============================================================

class TestEnhanceFusion:

    def _base_results(self, **overrides):
        base = {"macro": {"analysis": {}}, "eth_btc": {},
                "news_sentiment": {}, "crypto_signals": {},
                "coinmarketcap": {}}
        base.update(overrides)
        return base

    def test_macro_positive(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 10, "strategy_bonus": 5}
        results = self._base_results(macro={"analysis": {"macro_score": 20, "sentiment": "bullish"}})
        result = agent._enhance_fusion(fusion, results)
        assert result["total_score"] == 20  # 10 + 10
        assert result["extra_components"]["macro"]["score"] == 10

    def test_macro_negative(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 10, "strategy_bonus": 5}
        results = self._base_results(macro={"analysis": {"macro_score": -20, "sentiment": "bearish"}})
        result = agent._enhance_fusion(fusion, results)
        assert result["total_score"] == 0  # 10 + (-10)

    def test_eth_btc_extreme_negative_z(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(eth_btc={"eth_btc_z_score": -2.5})
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["eth_btc"]["score"] == 5

    def test_eth_btc_extreme_positive_z(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(eth_btc={"eth_btc_z_score": 2.5})
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["eth_btc"]["score"] == -5

    def test_eth_btc_normal_z(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(eth_btc={"eth_btc_z_score": 1.0})
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["eth_btc"]["score"] == 0

    def test_news_positive(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(news_sentiment={"sentiment_score": 50, "overall_sentiment": "positive"})
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["news_sentiment"]["score"] == 10

    def test_news_negative(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(news_sentiment={"sentiment_score": -50})
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["news_sentiment"]["score"] == -10

    def test_crypto_signals_high(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(crypto_signals={
            "btc": {"anomaly_level": "HIGH", "change_24h": 5.0},
            "anomaly_alerts": {"count": 10},
        })
        result = agent._enhance_fusion(fusion, results)
        assert result["extra_components"]["crypto_signals"]["score"] == 10

    def test_cmc_dominance_high(self):
        agent = ExternalDataAgent()
        fusion = {"total_score": 0, "strategy_bonus": 0}
        results = self._base_results(coinmarketcap={"status": "success", "btc_dominance": 58.0})
        result = agent._enhance_fusion(fusion, results)
        assert result["total_score"] == 5

    def test_strategy_bonus_recalculation(self):
        """enhance_fusion 후 strategy_bonus가 재계산된다."""
        agent = ExternalDataAgent()
        fusion = {"total_score": 35, "strategy_bonus": 15}
        results = self._base_results(macro={"analysis": {"macro_score": 20, "sentiment": "bullish"}})
        result = agent._enhance_fusion(fusion, results)
        # 35 + 10 = 45 >= 40 → bonus 20
        assert result["strategy_bonus"] == 20


# ============================================================
# collect_all
# ============================================================

class TestCollectAll:

    @patch("agents.external_data._fetch_nvt_signal")
    @patch("agents.external_data._run_script")
    @patch("agents.external_data.load_user_feedback", return_value=[])
    @patch("agents.external_data.load_performance_review", return_value={"available": False})
    def test_returns_structure(self, mock_perf, mock_fb, mock_run, mock_nvt):
        mock_run.return_value = {"test": True}
        mock_nvt.return_value = {"nvt_signal": 100.0}

        agent = ExternalDataAgent()
        with patch.object(agent, '_save_signal_to_db'):
            result = agent.collect_all()

        assert "timestamp" in result
        assert "sources" in result
        assert "external_signal" in result
        assert "errors" in result
        assert "collection_time_sec" in result

    @patch("agents.external_data._fetch_nvt_signal")
    @patch("agents.external_data._run_script")
    @patch("agents.external_data.load_user_feedback", return_value=[])
    @patch("agents.external_data.load_performance_review", return_value={"available": False})
    def test_error_isolation(self, mock_perf, mock_fb, mock_run, mock_nvt):
        """하나가 실패해도 나머지는 정상."""
        def side_effect(script, *args, **kwargs):
            if "fear_greed" in script:
                return {"error": "timeout"}
            return {"status": "ok"}

        mock_run.side_effect = side_effect
        mock_nvt.return_value = {"nvt_signal": 100.0}

        agent = ExternalDataAgent()
        with patch.object(agent, '_save_signal_to_db'):
            result = agent.collect_all()

        assert "fear_greed" in result["errors"]

    def test_get_fgi_value(self):
        agent = ExternalDataAgent()
        results = {"sources": {"fear_greed": {"current": {"value": 25}}}}
        assert agent.get_fgi_value(results) == 25

    def test_get_fgi_value_default(self):
        agent = ExternalDataAgent()
        assert agent.get_fgi_value({}) == 50


# ============================================================
# Performance Review
# ============================================================

class TestPerformanceReview:

    @patch("agents.external_data._load_supabase")
    def test_no_data(self, mock_load):
        mock_load.return_value = []
        result = load_performance_review()
        assert result["available"] is False

    @patch("agents.external_data._load_supabase")
    def test_good_performance(self, mock_load):
        mock_load.return_value = [
            {"decision": "buy", "profit_loss": 5.0, "confidence": 0.8,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "sell", "profit_loss": 3.0, "confidence": 0.7,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "buy", "profit_loss": -2.0, "confidence": 0.6,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
        ]
        result = load_performance_review()
        assert result["available"] is True
        assert result["wins"] == 2
        assert result["losses"] == 1
        assert result["win_rate_pct"] > 60

    @patch("agents.external_data._load_supabase")
    def test_losing_streak(self, mock_load):
        mock_load.return_value = [
            {"decision": "buy", "profit_loss": -3.0, "confidence": 0.5,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "sell", "profit_loss": -2.0, "confidence": 0.5,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
            {"decision": "buy", "profit_loss": 5.0, "confidence": 0.5,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
        ]
        result = load_performance_review()
        assert result["recent_streak_type"] == "loss"
        assert result["recent_streak"] == 2

    @patch("agents.external_data._load_supabase")
    def test_assessment_good(self, mock_load):
        mock_load.return_value = [
            {"decision": "buy", "profit_loss": 5.0, "confidence": 0.8,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
        ] * 4 + [
            {"decision": "sell", "profit_loss": -2.0, "confidence": 0.6,
             "current_price": 50000000, "reason": "test", "created_at": "2025-01-01"},
        ]
        result = load_performance_review()
        assert "양호" in result["assessment"]
