#!/usr/bin/env python3
"""
매크로 경제 지표 수집 (무료 API 전용)

수집 항목:
  - S&P 500 (^GSPC) — 미국 주식시장 대표 지수
  - 나스닥 (^IXIC) — 기술주 중심
  - 달러 인덱스 (DX-Y.NYB) — 달러 강세/약세
  - 금 (GC=F) — 안전자산 수요
  - WTI 원유 (CL=F) — 지정학 리스크
  - 미국 10년 국채 (^TNX) — 금리 방향

데이터 소스: Yahoo Finance (무료, API 키 불필요)
출력: JSON (stdout)
"""

from __future__ import annotations

import io
import json
import sys
import time

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TIMEOUT = 15

# Yahoo Finance chart API (무료, 키 불필요)
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

SYMBOLS = {
    "sp500": {"symbol": "^GSPC", "name": "S&P 500", "category": "stock"},
    "nasdaq": {"symbol": "^IXIC", "name": "NASDAQ", "category": "stock"},
    "dxy": {"symbol": "DX-Y.NYB", "name": "Dollar Index", "category": "currency"},
    "gold": {"symbol": "GC=F", "name": "Gold Futures", "category": "commodity"},
    "oil": {"symbol": "CL=F", "name": "WTI Crude Oil", "category": "commodity"},
    "us10y": {"symbol": "^TNX", "name": "US 10Y Yield", "category": "bond"},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 커넥션 재사용을 위한 세션
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """모듈 레벨 requests.Session을 반환한다 (커넥션 풀 재사용)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
        _session.headers.update({"Accept": "application/json"})
    return _session


def fetch_yahoo(symbol: str, range_: str = "5d", interval: str = "1d") -> dict | None:
    """Yahoo Finance에서 시세를 조회한다 (Session 재사용)."""
    session = _get_session()
    try:
        resp = session.get(
            YAHOO_CHART.format(symbol=symbol),
            params={"range": range_, "interval": interval, "includePrePost": "false"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"[macro] {symbol} HTTP {resp.status_code}", file=sys.stderr)
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        return result[0]
    except Exception as e:
        print(f"[macro] {symbol} 오류: {e}", file=sys.stderr)
        return None


def parse_quote(chart_data: dict) -> dict:
    """Yahoo chart 데이터를 파싱한다."""
    meta = chart_data.get("meta", {})
    indicators = chart_data.get("indicators", {}).get("quote", [{}])[0]

    closes = [c for c in (indicators.get("close") or []) if c is not None]
    if not closes:
        return {"error": "데이터 없음"}

    current = meta.get("regularMarketPrice", closes[-1])
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose") or (closes[-2] if len(closes) >= 2 else current)

    change_pct = ((current - prev_close) / prev_close * 100) if prev_close else 0

    # 5일 추세
    if len(closes) >= 5:
        five_day_change = ((closes[-1] - closes[0]) / closes[0] * 100)
    else:
        five_day_change = change_pct

    return {
        "price": round(current, 2),
        "prev_close": round(prev_close, 2),
        "change_pct": round(change_pct, 2),
        "five_day_change_pct": round(five_day_change, 2),
        "currency": meta.get("currency", "USD"),
    }


def analyze_macro(quotes: dict) -> dict:
    """매크로 지표를 종합 분석한다."""
    score = 0
    signals: list[str] = []

    # S&P 500 분석
    sp = quotes.get("sp500", {})
    if "price" in sp:
        if sp["change_pct"] < -2:
            score -= 15
            signals.append(f"S&P500 급락 {sp['change_pct']:+.1f}% → 위험자산 회피")
        elif sp["change_pct"] < -1:
            score -= 8
            signals.append(f"S&P500 하락 {sp['change_pct']:+.1f}%")
        elif sp["change_pct"] > 1:
            score += 8
            signals.append(f"S&P500 상승 {sp['change_pct']:+.1f}% → 위험자산 선호")
        elif sp["five_day_change_pct"] > 3:
            score += 5
            signals.append(f"S&P500 5일 강세 {sp['five_day_change_pct']:+.1f}%")

    # 달러 인덱스 분석 (역 상관: 달러 강세 = 크립토 약세)
    dxy = quotes.get("dxy", {})
    if "price" in dxy:
        if dxy["change_pct"] > 0.5:
            score -= 8
            signals.append(f"달러 강세 {dxy['change_pct']:+.1f}% → 크립토 압박")
        elif dxy["change_pct"] < -0.5:
            score += 8
            signals.append(f"달러 약세 {dxy['change_pct']:+.1f}% → 크립토 우호")
        if dxy["five_day_change_pct"] > 2:
            score -= 5
            signals.append(f"달러 5일 강세 {dxy['five_day_change_pct']:+.1f}%")
        elif dxy["five_day_change_pct"] < -2:
            score += 5
            signals.append(f"달러 5일 약세 {dxy['five_day_change_pct']:+.1f}%")

    # 금 분석 (동행: 금 상승 = 불확실성 증가, BTC와 동반 가능)
    gold = quotes.get("gold", {})
    if "price" in gold:
        if gold["change_pct"] > 1.5:
            score += 3  # 안전자산 수요 = 불확실성이지만 BTC도 디지털 금
            signals.append(f"금 급등 {gold['change_pct']:+.1f}% → 안전자산 수요↑")
        elif gold["change_pct"] < -1.5:
            score -= 3
            signals.append(f"금 급락 {gold['change_pct']:+.1f}%")

    # 원유 분석 (급등 = 지정학 리스크)
    oil = quotes.get("oil", {})
    if "price" in oil:
        if oil["change_pct"] > 3:
            score -= 10
            signals.append(f"유가 급등 {oil['change_pct']:+.1f}% → 지정학 리스크 경고")
        elif oil["change_pct"] > 1.5:
            score -= 5
            signals.append(f"유가 상승 {oil['change_pct']:+.1f}%")
        elif oil["change_pct"] < -3:
            score += 3
            signals.append(f"유가 급락 {oil['change_pct']:+.1f}% → 리스크 완화")

    # 미국 10년 국채 (급등 = 금리 상승 = 크립토 약세)
    bond = quotes.get("us10y", {})
    if "price" in bond:
        if bond["change_pct"] > 3:  # yield 기준이라 3%가 큰 변동
            score -= 8
            signals.append(f"10Y 수익률 급등 {bond['change_pct']:+.1f}% → 금리 압박")
        elif bond["change_pct"] < -3:
            score += 8
            signals.append(f"10Y 수익률 급락 {bond['change_pct']:+.1f}% → 금리 완화 기대")

    score = max(-30, min(30, score))

    # 종합 판정
    if score >= 15:
        sentiment = "risk_on"
        summary = "매크로 환경 강세 — 위험자산 선호"
    elif score >= 5:
        sentiment = "slightly_bullish"
        summary = "매크로 환경 소폭 우호적"
    elif score <= -15:
        sentiment = "risk_off"
        summary = "매크로 환경 약세 — 위험자산 회피"
    elif score <= -5:
        sentiment = "slightly_bearish"
        summary = "매크로 환경 소폭 약세"
    else:
        sentiment = "neutral"
        summary = "매크로 환경 중립"

    return {
        "macro_score": score,
        "max_score": 30,
        "sentiment": sentiment,
        "summary": summary,
        "signals": signals,
    }


def _fetch_and_parse(key: str, info: dict) -> tuple[str, dict]:
    """개별 심볼 데이터를 수집·파싱한다 (병렬 실행용)."""
    chart_data = fetch_yahoo(info["symbol"])
    if chart_data:
        parsed = parse_quote(chart_data)
        parsed["name"] = info["name"]
        parsed["category"] = info["category"]
        return key, parsed
    return key, {"error": f"{info['name']} 조회 실패"}


def main():
    quotes: dict = {}

    # 6개 Yahoo Finance 호출을 병렬로 실행
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_fetch_and_parse, key, info): key
            for key, info in SYMBOLS.items()
        }
        for future in as_completed(futures):
            try:
                key, parsed = future.result()
                quotes[key] = parsed
            except Exception as e:
                key = futures[future]
                quotes[key] = {"error": f"조회 실패: {e}"}

    analysis = analyze_macro(quotes)

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "source": "Yahoo Finance (free, no API key)",
        "quotes": quotes,
        "analysis": analysis,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stdout, ensure_ascii=False)
        sys.exit(1)
