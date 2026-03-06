#!/usr/bin/env python3
"""
Tavily API를 사용한 암호화폐 뉴스 수집 스크립트

수집: 최근 24시간 BTC 관련 뉴스 최대 10건
감성 분석은 LLM이 수행 (이 스크립트는 수집만 담당)

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

# Windows cp949 인코딩 이슈 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

TAVILY_API = "https://api.tavily.com/search"
MONTHLY_LIMIT = 999
USAGE_FILE = Path(__file__).resolve().parent.parent / "data" / "tavily_usage.json"

# 기본 조회 건수: crypto 7 + macro 4 = 11건/회, 하루 3회 = 33건/일
QUERIES = [
    {"query": "비트코인 Bitcoin BTC 시장", "category": "crypto", "max_results": 7},
    {"query": "war geopolitics oil sanctions economy crisis 전쟁 금리 경제위기", "category": "macro", "max_results": 4},
]
NORMAL_PER_RUN = sum(q["max_results"] for q in QUERIES)  # 11


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
    """잔여 한도에 맞춰 조회 건수를 조정한다.

    - 잔여 >= 11: 정상 (crypto 7 + macro 4)
    - 잔여 1~10: 잔여만큼만 (crypto 우선 배분)
    - 잔여 0: 조회 불가
    """
    remaining = MONTHLY_LIMIT - usage["count"]
    if remaining <= 0:
        return []
    if remaining >= NORMAL_PER_RUN:
        return [dict(q) for q in QUERIES]
    # 잔여 부족 → crypto 우선, 나머지 macro
    crypto_count = min(remaining, 7)
    macro_count = min(remaining - crypto_count, 4)
    adjusted = []
    if crypto_count > 0:
        adjusted.append({"query": QUERIES[0]["query"], "category": "crypto", "max_results": crypto_count})
    if macro_count > 0:
        adjusted.append({"query": QUERIES[1]["query"], "category": "macro", "max_results": macro_count})
    return adjusted


def fetch_news(api_key: str, query: str, max_results: int = 10):
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

    requested = sum(q["max_results"] for q in queries)

    all_articles = {}
    for q in queries:
        articles = fetch_news(api_key, q["query"], q["max_results"])
        for a in articles:
            a["category"] = q["category"]
        for a in articles:
            if a["url"] not in all_articles:
                all_articles[a["url"]] = a

    # 성공 후 사용량 기록
    usage["count"] += requested
    _save_usage(usage)

    articles_list = list(all_articles.values())
    result = {
        "timestamp": datetime.now().isoformat(),
        "queries": [q["query"] for q in queries],
        "articles_count": len(articles_list),
        "crypto_count": sum(1 for a in articles_list if a["category"] == "crypto"),
        "macro_count": sum(1 for a in articles_list if a["category"] == "macro"),
        "articles": articles_list,
        "tavily_usage": {
            "month": usage["month"],
            "used": usage["count"],
            "limit": MONTHLY_LIMIT,
            "remaining": MONTHLY_LIMIT - usage["count"],
            "this_run": requested,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
