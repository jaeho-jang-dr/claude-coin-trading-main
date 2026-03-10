"""
외부 데이터 수집 에이전트

기존 개별 스크립트를 통합 오케스트레이션한다:
  - collect_fear_greed.py → FGI
  - collect_news.py → 뉴스 + 감성 분석
  - whale_tracker.py → 온체인 고래 (방향 추정 포함)
  - binance_sentiment.py → 롱숏/펀딩비/김치P
  - collect_eth_btc.py → ETH/BTC 비율 + z-score
  - collect_macro.py → 매크로 경제 지표 (S&P500, DXY, 금, 유가, 10Y)
  - collect_crypto_signals.py → CoinGecko 거래량 이상 감지 (MCP 연동)
  - calculate_external_signal.py → Data Fusion 종합
  + 뉴스 감성 분석 (키워드 기반)
  + Supabase: 사용자 피드백, 과거 결정 성과

병렬 수집 + 에러 격리: 하나가 실패해도 나머지는 정상 수집.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
PYTHON = sys.executable


def _run_script(script_name: str, args: list[str] | None = None, timeout: int = 60) -> dict:
    """스크립트를 실행하고 JSON stdout을 파싱한다."""
    cmd = [PYTHON, str(SCRIPTS_DIR / script_name)]
    if args:
        cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_DIR),
            encoding="utf-8",
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return {"error": f"{script_name} 실행 실패 (exit {result.returncode})", "stderr": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"error": f"{script_name} 타임아웃 ({timeout}초)"}
    except json.JSONDecodeError:
        return {"error": f"{script_name} JSON 파싱 실패"}
    except Exception as e:
        return {"error": f"{script_name} 오류: {str(e)[:200]}"}


# ── 뉴스 감성 분석 (키워드 기반) ──────────────────────

POSITIVE_KEYWORDS = {
    # 영어
    "etf approved", "etf approval", "institutional", "bullish", "rally",
    "adoption", "partnership", "upgrade", "halving", "accumulate",
    "inflow", "break out", "all-time high", "ath", "buy signal",
    # 한국어
    "승인", "기관투자", "상승", "반등", "돌파", "강세", "매수세",
    "유입", "채택", "호재", "신고가", "긍정",
}

NEGATIVE_KEYWORDS = {
    # 영어
    "hack", "hacked", "exploit", "ban", "banned", "regulation crackdown",
    "crash", "plunge", "fraud", "scam", "sec lawsuit", "sanctions",
    "war", "attack", "bankruptcy", "liquidation", "sell-off", "bearish",
    "depeg", "panic", "contagion",
    # 한국어
    "해킹", "금지", "규제", "폭락", "사기", "소송", "제재", "전쟁",
    "공격", "파산", "청산", "매도세", "패닉", "약세", "하락",
    "디페그", "경고",
}


def analyze_news_sentiment(news_data: dict) -> dict:
    """뉴스 기사들의 감성을 키워드 기반으로 점수화한다."""
    articles = news_data.get("articles", [])
    if not articles:
        return {
            "sentiment_score": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "overall_sentiment": "neutral",
            "key_signals": [],
        }

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    key_signals: list[str] = []

    for article in articles:
        text = (article.get("title", "") + " " + article.get("content", "")).lower()

        pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
        neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

        if pos_hits > neg_hits:
            positive_count += 1
            if pos_hits >= 2:
                key_signals.append(f"[강세] {article.get('title', '')[:60]}")
        elif neg_hits > pos_hits:
            negative_count += 1
            if neg_hits >= 2:
                key_signals.append(f"[약세] {article.get('title', '')[:60]}")
        else:
            neutral_count += 1

    total = positive_count + negative_count + neutral_count
    if total == 0:
        sentiment_score = 0
    else:
        # -100 ~ +100 범위
        sentiment_score = int((positive_count - negative_count) / total * 100)

    if sentiment_score >= 30:
        overall = "positive"
    elif sentiment_score <= -30:
        overall = "negative"
    elif sentiment_score >= 10:
        overall = "slightly_positive"
    elif sentiment_score <= -10:
        overall = "slightly_negative"
    else:
        overall = "neutral"

    return {
        "sentiment_score": sentiment_score,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_articles": total,
        "overall_sentiment": overall,
        "key_signals": key_signals[:5],
    }


# ── 뉴스 압축 ──────────────────────────────────────

def _compress_news(news_data: dict) -> dict:
    """뉴스 JSON을 압축하여 토큰 비용을 절감한다 (10-15KB → 2-4KB)."""
    articles = news_data.get("articles", [])
    if not articles:
        return news_data

    categories: dict[str, list] = {}
    for a in articles:
        cat = a.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        content = (a.get("content", "") or "")[:100]
        if len(a.get("content", "") or "") > 100:
            content += "..."
        compressed = {"title": a.get("title", ""), "snippet": content}
        score = a.get("score", 0)
        if score:
            compressed["score"] = round(score, 2)
        categories[cat].append(compressed)

    return {
        "timestamp": news_data.get("timestamp", ""),
        "articles_count": len(articles),
        "by_category": {cat: len(arts) for cat, arts in categories.items()},
        "categories": categories,
        "tavily_remaining": news_data.get("tavily_usage", {}).get("remaining", "N/A"),
    }


# ── Supabase 조회 ──────────────────────────────────

def _load_supabase(endpoint: str, params: dict) -> list | dict:
    """Supabase REST API를 조회한다."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return []

    try:
        import requests
        resp = requests.get(
            f"{url}/rest/v1/{endpoint}",
            params=params,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def load_user_feedback() -> list[dict]:
    """미반영 사용자 피드백을 로드한다."""
    return _load_supabase("feedback", {
        "select": "type,content,created_at",
        "applied": "eq.false",
        "order": "created_at.desc",
        "limit": "5",
    })


def load_performance_review() -> dict:
    """과거 결정의 성과를 분석한다."""
    decisions = _load_supabase("decisions", {
        "select": "decision,confidence,current_price,profit_loss,reason,created_at",
        "profit_loss": "not.is.null",
        "order": "created_at.desc",
        "limit": "20",
    })

    if not decisions:
        return {"available": False, "message": "성과 데이터 없음"}

    wins = 0
    losses = 0
    total_pl = 0.0
    recent_streak = 0
    streak_type = None

    for d in decisions:
        pl = float(d.get("profit_loss", 0))
        total_pl += pl
        if pl > 0:
            wins += 1
            if streak_type is None:
                streak_type = "win"
            if streak_type == "win":
                recent_streak += 1
        elif pl < 0:
            losses += 1
            if streak_type is None:
                streak_type = "loss"
            if streak_type == "loss":
                recent_streak += 1

        if streak_type and ((pl > 0 and streak_type == "loss") or (pl < 0 and streak_type == "win")):
            break  # 연속 끊김

    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    avg_pl = round(total_pl / len(decisions), 2) if decisions else 0

    # 성과 평가
    if win_rate >= 60:
        assessment = "양호 — 최근 판단 정확도 높음"
    elif win_rate >= 40:
        assessment = "보통 — 승률 균형"
    else:
        assessment = "주의 — 최근 판단 정확도 낮음, 보수적 접근 권장"

    return {
        "available": True,
        "total_evaluated": len(decisions),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "avg_profit_loss": avg_pl,
        "recent_streak": recent_streak,
        "recent_streak_type": streak_type or "none",
        "assessment": assessment,
    }


# ── 메인 클래스 ──────────────────────────────────────


class ExternalDataAgent:
    """외부 데이터를 병렬 수집하고 통합한다."""

    def __init__(self, snapshot_dir: Path | None = None):
        self.snapshot_dir = snapshot_dir

    def _get_cycle_id(self) -> str:
        """cycle_id를 반환한다. 실패 시 인라인 생성."""
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from scripts.cycle_id import get_or_create_cycle_id
            return get_or_create_cycle_id("agent")
        except Exception:
            from datetime import datetime, timezone, timedelta
            _KST = timezone(timedelta(hours=9))
            return datetime.now(_KST).strftime("%Y%m%d-%H%M") + "-agent"

    def collect_all(self) -> dict:
        """모든 외부 데이터를 병렬 수집한다."""
        tasks = {
            "fear_greed": ("collect_fear_greed.py", None),
            "news": ("collect_news.py", None),
            "whale_tracker": ("whale_tracker.py", ["--blocks", "2"]),
            "binance_sentiment": ("binance_sentiment.py", None),
            "eth_btc": ("collect_eth_btc.py", None),
            "macro": ("collect_macro.py", None),
            "crypto_signals": ("collect_crypto_signals.py", None),
            "coinmarketcap": ("collect_coinmarketcap.py", None),
        }

        results: dict = {}
        start = time.time()

        with ThreadPoolExecutor(max_workers=7) as pool:
            futures = {
                pool.submit(_run_script, script, args): name
                for name, (script, args) in tasks.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"error": str(e)}

        elapsed = round(time.time() - start, 1)

        # 뉴스 감성 분석 (키워드 기반 — 압축 전 원본으로 수행)
        news_sentiment = analyze_news_sentiment(results.get("news", {}))
        results["news_sentiment"] = news_sentiment

        # 뉴스 압축 (토큰 절감: 10-15KB → 2-4KB)
        results["news"] = _compress_news(results.get("news", {}))

        # 사용자 피드백 로드
        feedback = load_user_feedback()
        results["user_feedback"] = feedback

        # 과거 성과 분석
        performance = load_performance_review()
        results["performance_review"] = performance

        # 스냅샷 저장
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            for name, data in results.items():
                path = self.snapshot_dir / f"{name}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        # Data Fusion 종합
        external_signal = self._calculate_fusion(results)

        collected = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "collection_time_sec": elapsed,
            "sources": results,
            "external_signal": external_signal,
            "errors": [name for name, d in results.items()
                       if isinstance(d, dict) and "error" in d],
        }

        # DB에 외부 시그널 기록
        try:
            self._save_signal_to_db(collected, external_signal)
        except Exception as e:
            logging.warning(f"external_signal_log DB 저장 중 예외: {e}")

        return collected

    def _calculate_fusion(self, results: dict) -> dict:
        """수집된 데이터로 Data Fusion 종합 점수를 산출한다."""
        if self.snapshot_dir:
            fusion_result = _run_script(
                "calculate_external_signal.py",
                [str(self.snapshot_dir)],
                timeout=10,
            )
            if "error" not in fusion_result:
                # 매크로/ETH/뉴스 보너스 추가
                fusion_result = self._enhance_fusion(fusion_result, results)
                return fusion_result

        # 인라인 폴백
        return self._inline_fusion(results)

    def _enhance_fusion(self, fusion: dict, results: dict) -> dict:
        """Data Fusion에 매크로/ETH/뉴스 점수를 보강한다."""
        extra_score = 0
        extra_details: list[str] = []

        # 매크로 점수 (±15)
        macro = results.get("macro", {}).get("analysis", {})
        macro_score = macro.get("macro_score", 0) or 0
        if macro_score != 0:
            # ±30 → ±15 리스케일
            macro_adj = max(-15, min(15, int(macro_score * 0.5)))
            extra_score += macro_adj
            extra_details.append(f"매크로 {macro.get('sentiment', 'neutral')}: {macro_adj:+d}점")

        # ETH/BTC 이상 감지 (±5)
        eth = results.get("eth_btc", {})
        z = eth.get("eth_btc_z_score", 0) or 0
        if abs(z) >= 2:
            eth_adj = 5 if z < -2 else -5
            extra_score += eth_adj
            extra_details.append(f"ETH/BTC z={z:.1f}: {eth_adj:+d}점")

        # 뉴스 감성 (±10)
        ns = results.get("news_sentiment", {})
        sent_score = ns.get("sentiment_score", 0) or 0
        if sent_score >= 30:
            news_adj = 10
        elif sent_score >= 10:
            news_adj = 5
        elif sent_score <= -30:
            news_adj = -10
        elif sent_score <= -10:
            news_adj = -5
        else:
            news_adj = 0
        if news_adj != 0:
            extra_score += news_adj
            extra_details.append(f"뉴스 감성 {ns.get('overall_sentiment', 'neutral')}: {news_adj:+d}점")

        # CoinGecko 거래량 이상 감지 (±10)
        cs = results.get("crypto_signals", {}) or {}
        btc_signal = cs.get("btc") or {}
        crypto_adj = 0
        if btc_signal:
            btc_anomaly = btc_signal.get("anomaly_level", "LOW")
            btc_change = btc_signal.get("change_24h", 0) or 0
            if btc_anomaly in ("HIGH", "CRITICAL"):
                # 거래량 급증 + 방향으로 판단
                crypto_adj = 10 if btc_change > 0 else -10
            elif btc_anomaly == "MODERATE" and abs(btc_change) > 3:
                crypto_adj = 5 if btc_change > 0 else -5
        alert_count = cs.get("anomaly_alerts", {}).get("count", 0)
        if alert_count >= 30:
            # 시장 전체 과열 → 변동성 경고
            extra_details.append(f"CoinGecko 이상 {alert_count}건: 시장 변동성 높음")
        if crypto_adj != 0:
            extra_score += crypto_adj
            extra_details.append(f"BTC 거래량 {btc_anomaly}: {crypto_adj:+d}점")

        # CoinMarketCap 매크로 시황 (±5)
        cmc = results.get("coinmarketcap", {}) or {}
        cmc_adj = 0
        if cmc.get("status") == "success":
            btc_dom = cmc.get("btc_dominance")
            if btc_dom:
                if btc_dom > 55:  # BTC 강세장 리드
                    cmc_adj = 5
                elif btc_dom < 45: # 알트 시즌 (BTC 모멘텀 약화)
                    cmc_adj = -5
            if cmc_adj != 0:
                extra_score += cmc_adj
                extra_details.append(f"CMC BTC 도미넌스({btc_dom:.1f}%): {cmc_adj:+d}점")

        # 기존 total_score에 추가
        old_total = fusion.get("total_score", 0)
        new_total = old_total + extra_score

        # strategy_bonus 재계산 (새 범위: -100~+100)
        fusion["total_score"] = new_total
        fusion["extra_components"] = {
            "macro": {"score": macro_adj if macro_score != 0 else 0, "max": 15},
            "eth_btc": {"score": eth_adj if abs(z) >= 2 else 0, "max": 5},
            "news_sentiment": {"score": news_adj, "max": 10},
            "crypto_signals": {"score": crypto_adj, "max": 10},
        }
        fusion["extra_details"] = extra_details

        # ±100 → ±20 보너스 재계산
        if new_total >= 40:
            fusion["strategy_bonus"] = 20
        elif new_total >= 25:
            fusion["strategy_bonus"] = 15
        elif new_total >= 15:
            fusion["strategy_bonus"] = 10
        elif new_total >= 5:
            fusion["strategy_bonus"] = 5
        elif new_total <= -40:
            fusion["strategy_bonus"] = -20
        elif new_total <= -25:
            fusion["strategy_bonus"] = -15
        elif new_total <= -15:
            fusion["strategy_bonus"] = -10
        elif new_total <= -5:
            fusion["strategy_bonus"] = -5
        else:
            fusion["strategy_bonus"] = 0

        return fusion

    def _inline_fusion(self, results: dict) -> dict:
        """인라인 Data Fusion 간이 계산 (calculate_external_signal.py 실패 시 폴백)."""
        score = 0

        # 바이낸스 심리
        bs = results.get("binance_sentiment", {})
        sentiment = bs.get("sentiment_score", {})
        bs_score = sentiment.get("score", 0) if isinstance(sentiment, dict) else 0
        score -= bs_score

        # 고래 활동
        wt = results.get("whale_tracker", {})
        ws_obj = wt.get("whale_score", {})
        ws = ws_obj.get("score", 0) if isinstance(ws_obj, dict) else 0
        score += ws

        # 매크로
        macro = results.get("macro", {}).get("analysis", {})
        macro_score = macro.get("macro_score", 0) or 0
        score += max(-15, min(15, int(macro_score * 0.5)))

        # 뉴스 감성
        ns = results.get("news_sentiment", {})
        sent = ns.get("sentiment_score", 0) or 0
        if sent >= 30:
            score += 10
        elif sent >= 10:
            score += 5
        elif sent <= -30:
            score -= 10
        elif sent <= -10:
            score -= 5

        # CoinGecko 거래량 이상
        cs = results.get("crypto_signals", {}) or {}
        btc_sig = cs.get("btc") or {}
        if btc_sig:
            btc_anom = btc_sig.get("anomaly_level", "LOW")
            btc_chg = btc_sig.get("change_24h", 0) or 0
            if btc_anom in ("HIGH", "CRITICAL"):
                score += 10 if btc_chg > 0 else -10
            elif btc_anom == "MODERATE" and abs(btc_chg) > 3:
                score += 5 if btc_chg > 0 else -5

        # CoinMarketCap
        cmc = results.get("coinmarketcap", {}) or {}
        if cmc.get("status") == "success":
            b_dom = cmc.get("btc_dominance")
            if b_dom and b_dom > 55:
                score += 5
            elif b_dom and b_dom < 45:
                score -= 5

        # strategy_bonus 매핑
        if score >= 40:
            bonus = 20
        elif score >= 25:
            bonus = 15
        elif score >= 15:
            bonus = 10
        elif score >= 5:
            bonus = 5
        elif score <= -40:
            bonus = -20
        elif score <= -25:
            bonus = -15
        elif score <= -15:
            bonus = -10
        elif score <= -5:
            bonus = -5
        else:
            bonus = 0

        return {
            "total_score": score,
            "strategy_bonus": bonus,
            "fusion": {"signal": "neutral", "note": "인라인 간이 계산"},
        }

    def _save_signal_to_db(self, results: dict, external_signal: dict) -> None:
        """외부 시그널 데이터를 Supabase external_signal_log 테이블에 저장한다."""
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            logging.warning("Supabase 환경변수 미설정 — external_signal_log 저장 건너뜀")
            return

        sources = results.get("sources", {})
        ext_sig = external_signal

        # -- Binance Futures --
        bs = sources.get("binance_sentiment", {})
        bs_sent = bs.get("sentiment_score", {})
        binance_score = bs_sent.get("score", None) if isinstance(bs_sent, dict) else None
        # top_trader_long_short.current_ratio (실제 출력 키)
        tls = bs.get("top_trader_long_short", {})
        ls_ratio = tls.get("current_ratio", None) if isinstance(tls, dict) else None
        # funding_rate.current_rate (실제 출력 키)
        fr = bs.get("funding_rate", {})
        funding_rate = fr.get("current_rate", None) if isinstance(fr, dict) else None
        # open_interest.oi_change_24h_pct (실제 출력 키)
        oi = bs.get("open_interest", {})
        oi_change = oi.get("oi_change_24h_pct", None) if isinstance(oi, dict) else None
        kimchi_pct = bs.get("kimchi_premium", {}).get("premium_pct", None) if isinstance(bs.get("kimchi_premium"), dict) else None

        # -- Whale / On-chain --
        wt = sources.get("whale_tracker", {})
        ws_obj = wt.get("whale_score", {})
        whale_score = ws_obj.get("score", None) if isinstance(ws_obj, dict) else None
        whale_flow = ws_obj.get("direction", None) if isinstance(ws_obj, dict) else None
        ba = wt.get("block_analysis", {})
        large_tx_count = ba.get("whale_txs_count", None) if isinstance(ba, dict) else None
        dir_btc = ba.get("direction_btc", {}) if isinstance(ba, dict) else {}
        exchange_inflow = dir_btc.get("exchange_deposit", None) if isinstance(dir_btc, dict) else None
        exchange_outflow = dir_btc.get("exchange_withdrawal", None) if isinstance(dir_btc, dict) else None

        # -- Macro --
        macro_analysis = sources.get("macro", {}).get("analysis", {})
        macro_score = macro_analysis.get("macro_score", None)
        macro_quotes = sources.get("macro", {}).get("quotes", {})
        sp500 = macro_quotes.get("sp500", {}).get("change_pct", None)
        dxy = macro_quotes.get("dxy", {}).get("change_pct", None)
        gold = macro_quotes.get("gold", {}).get("change_pct", None)
        us10y = macro_quotes.get("us10y", {}).get("change_pct", None)

        # -- FGI --
        fg = sources.get("fear_greed", {})
        fg_current = fg.get("current", {})
        fgi_value = int(fg_current.get("value", 0)) if fg_current.get("value") is not None else None
        fgi_class = fg_current.get("classification", None)

        # -- News --
        ns = sources.get("news_sentiment", {})
        news_sentiment = ns.get("sentiment_score", None)
        news_count = ns.get("total_articles", None)
        news_score_val = ext_sig.get("extra_components", {}).get("news_sentiment", {}).get("score", None)

        # -- ETH/BTC --
        eth = sources.get("eth_btc", {})
        eth_btc_ratio = eth.get("eth_btc_ratio", None)
        eth_btc_z = eth.get("eth_btc_z_score", None)
        eth_btc_score = ext_sig.get("extra_components", {}).get("eth_btc", {}).get("score", None)
        eth_btc_trend = eth.get("signal", None)

        # -- CoinGecko --
        cs = sources.get("crypto_signals", {}) or {}
        btc_sig = cs.get("btc") or {}
        coingecko_score_val = ext_sig.get("extra_components", {}).get("crypto_signals", {}).get("score", None)
        volume_anomaly = btc_sig.get("anomaly_level", "LOW") in ("HIGH", "CRITICAL", "MODERATE") if btc_sig else None
        volume_change_pct = btc_sig.get("change_24h", None) if btc_sig else None

        # -- Fusion --
        fusion_score = ext_sig.get("total_score", None)
        fusion_obj = ext_sig.get("fusion", {})
        fusion_signal = fusion_obj.get("signal", None) if isinstance(fusion_obj, dict) else None
        fusion_details = ext_sig  # store entire fusion result as JSONB

        row = {
            "binance_score": binance_score,
            "long_short_ratio": ls_ratio,
            "funding_rate": funding_rate,
            "open_interest_change": oi_change,
            "whale_score": whale_score,
            "whale_flow": whale_flow,
            "large_tx_count": large_tx_count,
            "exchange_inflow": exchange_inflow,
            "exchange_outflow": exchange_outflow,
            "macro_score": macro_score,
            "sp500_change": sp500,
            "dxy_change": dxy,
            "gold_change": gold,
            "us10y_change": us10y,
            "fgi_value": fgi_value,
            "fgi_classification": fgi_class,
            "news_sentiment": news_sentiment,
            "news_score": news_score_val,
            "news_count": news_count,
            "eth_btc_ratio": eth_btc_ratio,
            "eth_btc_score": eth_btc_score,
            "eth_btc_trend": eth_btc_trend,
            "coingecko_score": coingecko_score_val,
            "volume_anomaly": volume_anomaly,
            "volume_change_pct": volume_change_pct,
            "kimchi_premium_pct": kimchi_pct,
            "fusion_score": fusion_score,
            "fusion_signal": fusion_signal,
            "fusion_details": json.dumps(fusion_details, ensure_ascii=False) if fusion_details else None,
            "source": "agent",
            "cycle_id": self._get_cycle_id(),
        }

        # None 값 제거 (Supabase가 DEFAULT 처리하도록)
        row = {k: v for k, v in row.items() if v is not None}

        try:
            resp = requests.post(
                f"{url}/rest/v1/external_signal_log",
                json=row,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                timeout=10,
            )
            if resp.status_code in (200, 201):
                logging.info("external_signal_log 저장 완료")
            else:
                logging.warning(f"external_signal_log 저장 실패: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            logging.warning(f"external_signal_log 저장 오류: {e}")

    def get_fgi_value(self, results: dict) -> int:
        fg = results.get("sources", {}).get("fear_greed", {})
        current = fg.get("current", {})
        return int(current.get("value", 50))
