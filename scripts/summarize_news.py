#!/usr/bin/env python3
"""
뉴스 JSON을 압축하여 토큰 비용을 절감한다.

입력: collect_news.py의 전체 JSON (stdin 또는 파일 경로)
출력: 압축된 JSON (stdout)

압축 전략:
  - URL 제거 (분석에 불필요)
  - content를 100자로 절단
  - published_date 제거 (timestamp로 충분)
  - tavily_usage 메타데이터 간소화
  - 카테고리별 그룹핑으로 구조화

목표: 10-15KB → 2-4KB
"""

import json
import sys
from pathlib import Path


def summarize_news(data: dict) -> dict:
    """뉴스 JSON을 압축한다."""
    articles = data.get("articles", [])

    if not articles:
        return {
            "articles_count": 0,
            "by_category": {},
            "categories": {},
            "tavily_remaining": data.get("tavily_usage", {}).get("remaining", "N/A"),
        }

    # 카테고리별 그룹핑 + 압축
    categories: dict[str, list] = {}
    for a in articles:
        cat = a.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []

        # content를 100자로 절단
        content = (a.get("content", "") or "")[:100]
        if len(a.get("content", "") or "") > 100:
            content += "..."

        compressed = {
            "title": a.get("title", ""),
            "snippet": content,
        }
        # score가 있으면 보존 (relevance 판단용)
        score = a.get("score", 0)
        if score:
            compressed["score"] = round(score, 2)

        categories[cat].append(compressed)

    # 카테고리별 기사 수
    by_category = {cat: len(arts) for cat, arts in categories.items()}

    # tavily 사용량은 remaining만
    tavily = data.get("tavily_usage", {})

    return {
        "timestamp": data.get("timestamp", ""),
        "articles_count": len(articles),
        "by_category": by_category,
        "categories": categories,
        "tavily_remaining": tavily.get("remaining", "N/A"),
    }


def main():
    # stdin 또는 파일 경로에서 입력
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if input_path.exists():
            data = json.loads(input_path.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"error": f"파일 없음: {sys.argv[1]}"}), file=sys.stderr)
            sys.exit(1)
    else:
        data = json.load(sys.stdin)

    result = summarize_news(data)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
