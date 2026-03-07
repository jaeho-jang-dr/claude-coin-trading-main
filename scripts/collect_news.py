#!/usr/bin/env python3
"""
Tavily API를 사용한 암호화폐 + 매크로 뉴스 수집 스크립트

조회 구성 (1회 실행당 5 API 호출):
  - 크립토 3종: BTC 시장, 알트코인/ETH, 크립토 규제/기관
  - 매크로 2종: 지정학/전쟁/유가, 경제/금리/고용

월간 한도: 999 API 호출 (Tavily 무료 티어)
  - 6회/일 x 5호출 = 30/일 x 31일 = 930/월 (여유 69호출)

출력: JSON (stdout)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

TAVILY_API = "https://api.tavily.com/search"
MONTHLY_LIMIT = 999
USAGE_FILE = Path(__file__).resolve().parent.parent / "data" / "tavily_usage.json"

# 크립토/매크로 쿼리 풀
# 주중(월~금): 크립토 3 + 매크로 2 = 5 호출
# 주말(토~일): 크립토 4 + 매크로 2 = 6 호출
# 월간 예산: 주중 30/일 x 22일 + 주말 36/일 x 9일 = 984 < 999

CRYPTO_QUERIES = [
    {
        "query": "비트코인 Bitcoin BTC 가격 시장 전망",
        "category": "crypto_btc",
        "max_results": 5,
    },
    {
        "query": "이더리움 Ethereum ETH 알트코인 altcoin crypto market",
        "category": "crypto_alt",
        "max_results": 3,
    },
    {
        "query": "crypto regulation institutional ETF 암호화폐 규제 기관투자",
        "category": "crypto_regulation",
        "max_results": 3,
    },
    # 주말 전용 (4번째 크립토)
    {
        "query": "Bitcoin whale on-chain mining 비트코인 고래 온체인 채굴 해시레이트",
        "category": "crypto_onchain",
        "max_results": 3,
        "weekend_only": True,
    },
]

MACRO_QUERIES = [
    {
        "query": "war geopolitics oil sanctions 전쟁 지정학 유가 제재",
        "category": "macro_geo",
        "max_results": 3,
    },
    {
        "query": "Federal Reserve interest rate economy employment CPI 금리 경제 고용 인플레이션",
        "category": "macro_economy",
        "max_results": 3,
    },
]


def _build_queries() -> list:
    """요일에 따라 쿼리 목록을 구성한다."""
    is_weekend = datetime.now().weekday() >= 5  # 5=토, 6=일
    queries = []
    for q in CRYPTO_QUERIES:
        if q.get("weekend_only") and not is_weekend:
            continue
        queries.append({k: v for k, v in q.items() if k != "weekend_only"})
    queries.extend(MACRO_QUERIES)
    return queries


def _load_usage() -> dict:
    """월별 사용량 파일 로드. 월이 바뀌면 자동 리셋."""
    current_month = datetime.now().strftime("%Y-%m")
    if USAGE_FILE.exists():
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        if data.get("month") == current_month:
            return data
    return {"month": current_month, "count": 0}


def _save_usage(data: dict):
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _budget_queries(usage: dict) -> list:
    """잔여 한도에 맞춰 쿼리를 조정한다."""
    remaining = MONTHLY_LIMIT - usage["count"]
    if remaining <= 0:
        return []

    queries = _build_queries()
    calls_needed = len(queries)

    if remaining >= calls_needed:
        return queries

    # 잔여 부족 -> 우선순위대로 배분
    priority = ["crypto_btc", "macro_geo", "crypto_alt", "macro_economy", "crypto_regulation", "crypto_onchain"]
    adjusted = []
    for cat in priority:
        if remaining <= 0:
            break
        q = next((q for q in queries if q["category"] == cat), None)
        if q:
            adjusted.append(dict(q))
            remaining -= 1
    return adjusted


def fetch_news(api_key: str, query: str, max_results: int = 5):
    r = requests.post(
        TAVILY_API,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "max_results": max_results,
            "topic": "news",
            "days": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    return [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "content": (a.get("content", "") or "")[:500],
            "published_date": a.get("published_date", ""),
            "score": a.get("score", 0),
        }
        for a in r.json().get("results", [])
    ]


def main():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 환경변수가 설정되지 않았습니다.")

    usage = _load_usage()
    queries = _budget_queries(usage)

    if not queries:
        remaining = MONTHLY_LIMIT - usage["count"]
        raise RuntimeError(
            f"Tavily 월간 한도 소진: {usage['count']}/{MONTHLY_LIMIT} (잔여 {remaining}건)"
        )

    api_calls = len(queries)
    is_weekend = datetime.now().weekday() >= 5

    all_articles = {}
    for q in queries:
        articles = fetch_news(api_key, q["query"], q["max_results"])
        for a in articles:
            a["category"] = q["category"]
        for a in articles:
            if a["url"] not in all_articles:
                all_articles[a["url"]] = a

    # API 호출 수 기준으로 사용량 기록
    usage["count"] += api_calls
    _save_usage(usage)

    articles_list = list(all_articles.values())

    # 카테고리별 집계
    categories = {}
    for a in articles_list:
        cat = a["category"]
        categories[cat] = categories.get(cat, 0) + 1

    day_type = "weekend" if is_weekend else "weekday"
    calls_per_day = api_calls * 6

    result = {
        "timestamp": datetime.now().isoformat(),
        "day_type": day_type,
        "queries": [q["query"][:50] for q in queries],
        "articles_count": len(articles_list),
        "by_category": categories,
        "articles": articles_list,
        "tavily_usage": {
            "month": usage["month"],
            "api_calls_used": usage["count"],
            "limit": MONTHLY_LIMIT,
            "remaining": MONTHLY_LIMIT - usage["count"],
            "this_run_calls": api_calls,
            "day_type": f"{day_type} ({api_calls} calls/run)",
            "daily_budget": f"{calls_per_day}/day ({day_type})",
            "monthly_projection": "weekday 30 x 22 + weekend 36 x 9 = 984/month",
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
