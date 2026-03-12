#!/usr/bin/env python3
"""
RAG 기반 과거 거래 리콜 -- 현재 시장과 가장 유사한 과거 경험 조회

사용법:
  python3 scripts/recall_rag.py                    # 현재 시장 데이터 자동 수집
  python3 scripts/recall_rag.py data/snapshots/market.json  # JSON 파일에서 로드
  python3 scripts/recall_rag.py --top 5            # 상위 5건 조회 (기본 3건)
  python3 scripts/recall_rag.py --json             # JSON 구조화 출력

출력: 토큰 절약형 한 줄 요약 (파이프라인 프롬프트 주입용)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
KST = timezone(timedelta(hours=9))


def _get_venv_python() -> str:
    venv = PROJECT_ROOT / ".venv" / "bin" / "python3"
    return str(venv) if venv.exists() else "python3"


def collect_current_market_data() -> dict:
    """collect_market_data.py와 collect_fear_greed.py를 실행하여 현재 시장 데이터를 수집."""
    python = _get_venv_python()
    result = {}

    # 시장 데이터 수집
    try:
        proc = subprocess.run(
            [python, "scripts/collect_market_data.py"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode == 0:
            result = json.loads(proc.stdout)
    except Exception as e:
        print(f"[recall_rag] 시장 데이터 수집 실패: {e}", file=sys.stderr)

    # 공포탐욕지수 수집
    try:
        proc = subprocess.run(
            [python, "scripts/collect_fear_greed.py"],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode == 0:
            fgi_data = json.loads(proc.stdout)
            result["fear_greed"] = fgi_data
            # fear_greed_value를 최상위에도 배치 (build_embedding_text 호환)
            current = fgi_data.get("current", {})
            if current.get("value"):
                result["fear_greed_value"] = current["value"]
    except Exception as e:
        print(f"[recall_rag] FGI 수집 실패: {e}", file=sys.stderr)

    return result


def get_embedding(text: str) -> list | None:
    """OpenAI API로 텍스트 임베딩을 생성."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"[recall_rag] 임베딩 생성 실패: {e}", file=sys.stderr)
        return None


def query_similar_decisions(embedding: list, limit: int = 3) -> list[dict]:
    """Supabase RPC match_similar_decisions를 호출하여 유사 결정을 조회.

    psycopg2 직접 연결을 우선 사용하고, 실패 시 REST API fallback.
    """
    # 방법 1: psycopg2 직접 SQL (더 안정적)
    db_url = os.getenv("SUPABASE_DB_URL")
    if db_url:
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM match_similar_decisions(%s::vector, %s)",
                (str(embedding), limit),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[recall_rag] psycopg2 RPC 호출 실패: {e}", file=sys.stderr)

    # 방법 2: REST API fallback
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not supabase_key:
        print("[recall_rag] SUPABASE 환경변수 미설정", file=sys.stderr)
        return []

    try:
        import requests
        r = requests.post(
            f"{supabase_url}/rest/v1/rpc/match_similar_decisions",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
            },
            json={
                "query_embedding": str(embedding),
                "match_limit": limit,
            },
            timeout=15,
        )
        if r.ok:
            return r.json()
        else:
            print(f"[recall_rag] RPC REST 호출 실패 ({r.status_code}): {r.text[:300]}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"[recall_rag] REST RPC 예외: {e}", file=sys.stderr)
        return []


def format_result_line(row: dict) -> str:
    """한 건의 유사 결정을 한 줄 요약으로 포맷."""
    similarity = row.get("similarity", 0)
    pct = int(similarity * 100) if similarity <= 1 else int(similarity)

    created = row.get("created_at", "")
    if isinstance(created, datetime):
        date_str = created.strftime("%Y-%m-%d")
    elif isinstance(created, str):
        date_str = created[:10]
    else:
        date_str = "unknown"

    decision = row.get("decision", "?")
    rsi = row.get("rsi_value", row.get("rsi_14"))
    fgi = row.get("fear_greed_value", row.get("fgi_value"))
    profit = row.get("profit_loss")

    # 지표 요약
    indicators = []
    if rsi is not None:
        indicators.append(f"RSI {int(float(rsi))}")
    if fgi is not None:
        indicators.append(f"FGI {int(float(fgi))}")
    ind_str = ", ".join(indicators) if indicators else "지표 없음"

    # 성과 요약
    if profit is not None:
        profit = float(profit)
        if profit > 0:
            perf = f"수익 +{profit:.1f}%"
        elif profit < 0:
            perf = f"손실 {profit:.1f}%"
        else:
            perf = "변동 없음"
    else:
        perf = "성과 미측정"

    # 사유 요약 (최대 30자)
    reason = row.get("reason", "")
    if reason and len(reason) > 30:
        reason = reason[:27] + "..."
    reason_str = f" ({reason})" if reason else ""

    return f"[유사도 {pct}%] {date_str} {decision}: {ind_str} → {perf}{reason_str}"


def main():
    parser = argparse.ArgumentParser(description="RAG 기반 과거 거래 리콜")
    parser.add_argument("market_json", nargs="?", help="시장 데이터 JSON 파일 경로")
    parser.add_argument("--top", type=int, default=3, help="조회 건수 (기본 3)")
    parser.add_argument("--json", action="store_true", help="JSON 구조화 출력")
    args = parser.parse_args()

    # 1. 시장 데이터 로드
    if args.market_json:
        try:
            with open(args.market_json) as f:
                market_data = json.load(f)
        except Exception as e:
            print(f"[recall_rag] JSON 파일 로드 실패: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        market_data = collect_current_market_data()
        if not market_data:
            print("[recall_rag] 시장 데이터 수집 실패", file=sys.stderr)
            sys.exit(1)

    # 2. 임베딩 텍스트 생성 (save_decision.py와 동일한 함수 사용)
    from scripts.save_decision import build_embedding_text
    text = build_embedding_text(market_data)
    if not text:
        print("[recall_rag] 임베딩 텍스트 생성 불가 (데이터 부족)", file=sys.stderr)
        sys.exit(1)

    # 3. 임베딩 벡터 생성
    embedding = get_embedding(text)
    if not embedding:
        print("[recall_rag] 임베딩 벡터 생성 실패", file=sys.stderr)
        sys.exit(1)

    # 4. 유사 결정 조회
    results = query_similar_decisions(embedding, args.top)

    if not results:
        msg = "과거 임베딩 데이터 없음 -- 거래 이력이 쌓이면 유사 경험을 조회할 수 있습니다."
        if args.json:
            print(json.dumps({"status": "no_data", "message": msg, "query_text": text}, ensure_ascii=False))
        else:
            print(msg)
        return

    # 5. 출력
    if args.json:
        output = {
            "status": "ok",
            "query_text": text,
            "results": [],
        }
        for row in results:
            output["results"].append({
                "similarity": round(float(row.get("similarity", 0)), 4),
                "date": str(row.get("created_at", ""))[:10],
                "decision": row.get("decision"),
                "reason": row.get("reason", ""),
                "profit_loss": float(row["profit_loss"]) if row.get("profit_loss") is not None else None,
                "rsi": float(row["rsi_value"]) if row.get("rsi_value") is not None else None,
                "fgi": int(row["fear_greed_value"]) if row.get("fear_greed_value") is not None else None,
                "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
            })
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for row in results:
            print(format_result_line(row))


if __name__ == "__main__":
    main()
