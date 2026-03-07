#!/usr/bin/env python3
"""
매매 결정을 Supabase에 저장하는 스크립트

cron_run.sh에서 claude -p 응답을 파싱하여 decisions 테이블에 기록한다.
이전 결정의 사후 성과도 함께 업데이트한다.

사용법:
  echo '{"decision":"hold","confidence":0.85,...}' | python3 scripts/save_decision.py
  python3 scripts/save_decision.py '{"decision":"hold",...}'
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

KST = timezone(timedelta(hours=9))

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def extract_json_from_response(text: str) -> dict | None:
    """claude -p 응답에서 JSON 블록을 추출한다."""
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(\{.*?\})\n\s*```",
        r"(\{[^{}]*\"decision\"[^{}]*\})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def map_decision(raw: str) -> str:
    """다양한 결정 표현을 DB enum으로 매핑."""
    raw_lower = raw.lower().strip()
    if raw_lower in ("hold", "관망"):
        return "관망"
    if raw_lower in ("buy", "bid", "매수"):
        return "매수"
    if raw_lower in ("sell", "ask", "매도"):
        return "매도"
    return "관망"


def save_decision(data: dict) -> dict:
    """decisions 테이블에 저장."""
    analysis = data.get("analysis", data.get("market_summary", {}))
    portfolio = data.get("portfolio", {})

    row = {
        "market": data.get("market", "KRW-BTC"),
        "decision": map_decision(data.get("decision", "hold")),
        "confidence": float(data.get("confidence", 0)),
        "reason": data.get("reason", data.get("note", "")),
        "fear_greed_value": analysis.get("fgi", analysis.get("fgi_value")),
        "rsi_value": analysis.get("rsi_14", analysis.get("rsi14")),
        "current_price": analysis.get("price", analysis.get("current_price")),
        "sma20_price": analysis.get("sma20", analysis.get("sma_20")),
        "executed": data.get("trade_executed", False),
        "execution_result": json.dumps(data, ensure_ascii=False),
        "market_data_snapshot": json.dumps(analysis, ensure_ascii=False),
    }

    if data.get("trade_executed") and data.get("action_taken") != "none":
        row["trade_amount"] = data.get("trade_amount", 0)

    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/decisions",
        headers=supabase_headers(),
        json=row,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def update_past_performance():
    """이전 결정의 사후 성과를 업데이트한다.

    profit_loss가 NULL인 과거 결정을 찾아,
    결정 시점 가격 vs 현재 가격으로 성과를 기록한다.
    """
    # profit_loss가 NULL인 최근 결정 조회
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/decisions"
        "?profit_loss=is.null&order=created_at.desc&limit=20",
        headers=supabase_headers(),
        timeout=10,
    )
    if not r.ok:
        return

    decisions = r.json()
    if not decisions:
        return

    # 현재 BTC 가격
    try:
        ticker = requests.get(
            "https://api.upbit.com/v1/ticker?markets=KRW-BTC", timeout=5
        ).json()[0]
        current_price = ticker["trade_price"]
    except Exception:
        return

    for d in decisions:
        decision_price = d.get("current_price")
        if not decision_price or decision_price == 0:
            continue

        decision_type = d.get("decision", "관망")
        price_change_pct = (current_price - decision_price) / decision_price * 100

        # 관망: 가격이 올랐으면 기회 손실(-), 내렸으면 올바른 판단(+)
        # 매수: 가격이 올랐으면 수익(+), 내렸으면 손실(-)
        # 매도: 가격이 올랐으면 기회 손실(-), 내렸으면 올바른 판단(+)
        if decision_type == "관망":
            profit_loss = -price_change_pct  # 안 샀는데 올랐으면 기회비용
        elif decision_type == "매수":
            profit_loss = price_change_pct
        elif decision_type == "매도":
            profit_loss = -price_change_pct
        else:
            profit_loss = 0

        requests.patch(
            f"{SUPABASE_URL}/rest/v1/decisions?id=eq.{d['id']}",
            headers=supabase_headers(),
            json={"profit_loss": round(profit_loss, 2)},
            timeout=10,
        )


def save_portfolio_snapshot():
    """현재 포트폴리오 스냅샷을 저장한다."""
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "scripts/get_portfolio.py"],
            capture_output=True, text=True, timeout=30,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode != 0:
            return
        portfolio = json.loads(result.stdout)

        row = {
            "total_krw": int(portfolio.get("krw_balance", 0)),
            "total_crypto_value": int(
                sum(h.get("eval_amount", 0) for h in portfolio.get("holdings", []))
            ),
            "total_value": int(portfolio.get("total_eval", 0)),
            "holdings": json.dumps(portfolio.get("holdings", []), ensure_ascii=False),
        }

        requests.post(
            f"{SUPABASE_URL}/rest/v1/portfolio_snapshots",
            headers=supabase_headers(),
            json=row,
            timeout=10,
        )
    except Exception:
        pass


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print(json.dumps({"error": "SUPABASE 환경변수 미설정"}))
        sys.exit(1)

    # 입력: 파이프 또는 인자
    if len(sys.argv) > 1:
        raw_text = sys.argv[1]
    else:
        raw_text = sys.stdin.read()

    data = extract_json_from_response(raw_text)
    if not data:
        print(json.dumps({"error": "JSON 파싱 실패", "raw": raw_text[:200]}))
        sys.exit(1)

    # 1. 이전 결정 성과 업데이트
    try:
        update_past_performance()
    except Exception as e:
        print(json.dumps({"warning": f"성과 업데이트 실패: {e}"}), file=sys.stderr)

    # 2. 현재 결정 저장
    saved = save_decision(data)
    print(json.dumps({"success": True, "saved": saved}, ensure_ascii=False, default=str))

    # 3. 포트폴리오 스냅샷 저장
    try:
        save_portfolio_snapshot()
    except Exception:
        pass


if __name__ == "__main__":
    main()
