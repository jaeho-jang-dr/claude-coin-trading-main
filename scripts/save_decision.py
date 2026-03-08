#!/usr/bin/env python3
"""
매매 결정을 Supabase에 저장하는 스크립트

cron_run.sh에서 claude -p 응답을 파싱하여 decisions 테이블에 기록한다.
이전 결정의 사후 성과도 함께 업데이트한다.

사용법:
  echo "$CLAUDE_RESPONSE" | python3 scripts/save_decision.py
  python3 scripts/save_decision.py "$CLAUDE_RESPONSE"
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


def supabase_post(table: str, row: dict) -> dict | None:
    """Supabase 테이블에 INSERT. 실패 시 stderr에 로그."""
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=supabase_headers(),
            json=row,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[save_decision] {table} INSERT 실패: {e}", file=sys.stderr)
        return None


def extract_json_from_response(text: str) -> dict | None:
    """claude -p 응답에서 JSON 블록을 추출한다.

    Claude는 보통 ```json ... ``` 으로 감싸서 출력한다.
    중첩 JSON 객체가 있으므로 re.DOTALL로 전체를 캡처한다.
    """
    # 1) ```json 코드펜스 (가장 일반적)
    m = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) ``` 코드펜스 (json 태그 없이)
    m = re.search(r"```\s*\n(\{.*\})\n\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) 전체 텍스트가 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4) 텍스트 내에서 가장 큰 { ... } 블록 찾기
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1

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


def _get_nested(data: dict, *keys, default=None):
    """여러 키 후보 중 첫 번째로 존재하는 값을 반환."""
    for key in keys:
        if "." in key:
            parts = key.split(".")
            val = data
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p)
                else:
                    val = None
                    break
            if val is not None:
                return val
        elif key in data and data[key] is not None:
            return data[key]
    return default


def save_decision(data: dict) -> dict | None:
    """decisions 테이블에 저장.

    Claude의 실제 출력 구조에 맞춘 필드 매핑:
    - current_price: 최상위 current_price
    - buy_score: buy_score.total / buy_score.fgi.value 등
    - fear_greed_value: buy_score.fgi.value
    - rsi_value: buy_score.rsi.value
    - sma20_price: 시장 데이터에서 추출 (별도 저장 안 하면 NULL 허용)
    """
    buy_score = data.get("buy_score", {})
    fgi_obj = buy_score.get("fgi", {}) if isinstance(buy_score.get("fgi"), dict) else {}
    rsi_obj = buy_score.get("rsi", {}) if isinstance(buy_score.get("rsi"), dict) else {}

    # confidence: Claude가 85 같은 정수로 출력 → 0.85로 변환 (DB는 DECIMAL(3,2))
    raw_conf = float(data.get("confidence", 0))
    confidence = raw_conf / 100.0 if raw_conf > 1 else raw_conf

    row = {
        "market": data.get("market", "KRW-BTC"),
        "decision": map_decision(data.get("decision", "hold")),
        "confidence": round(confidence, 2),
        "reason": data.get("reason", ""),
        "current_price": data.get("current_price"),
        "fear_greed_value": fgi_obj.get("value"),
        "rsi_value": rsi_obj.get("value"),
        "sma20_price": data.get("sma20_price"),
        "executed": data.get("executed", data.get("trade_executed", False)),
        # 전체 JSON을 execution_result에 저장 (감사 추적용)
        "execution_result": json.dumps(data, ensure_ascii=False),
        # buy_score + ai_signal + portfolio를 market_data_snapshot에 저장
        "market_data_snapshot": json.dumps({
            "buy_score": buy_score,
            "ai_composite_signal": data.get("ai_composite_signal"),
            "portfolio_status": data.get("portfolio_status", data.get("portfolio")),
            "risk_alerts": data.get("risk_alerts", []),
            "eth_btc_signal": data.get("eth_btc_signal"),
            "strategy_switch": data.get("strategy_switch_recommendation"),
        }, ensure_ascii=False),
    }

    # 매매 실행된 경우 금액 기록
    trade_details = data.get("trade_details", {})
    if trade_details.get("amount"):
        row["trade_amount"] = trade_details["amount"]
    elif data.get("trade_amount"):
        row["trade_amount"] = data["trade_amount"]

    return supabase_post("decisions", row)


def update_past_performance():
    """이전 결정의 사후 성과를 업데이트한다.

    profit_loss가 NULL인 과거 결정을 찾아,
    결정 시점 가격 vs 현재 가격으로 성과를 기록한다.
    """
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

        if decision_type == "관망":
            profit_loss = -price_change_pct
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


def mark_feedback_applied():
    """프롬프트에 주입된 미반영 피드백을 applied=true로 갱신."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/feedback?applied=eq.false&select=id",
            headers=supabase_headers(),
            timeout=10,
        )
        if not r.ok:
            return
        feedbacks = r.json()
        if not feedbacks:
            return

        ids = ",".join(f"'{f['id']}'" for f in feedbacks)
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/feedback?id=in.({ids})",
            headers=supabase_headers(),
            json={
                "applied": True,
                "applied_at": datetime.now(KST).isoformat(),
            },
            timeout=10,
        )
        print(f"[save_decision] {len(feedbacks)}건 피드백 applied 처리", file=sys.stderr)
    except Exception as e:
        print(f"[save_decision] 피드백 applied 갱신 실패: {e}", file=sys.stderr)


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

        supabase_post("portfolio_snapshots", row)
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
        print(json.dumps({"error": "JSON 파싱 실패", "raw": raw_text[:500]}))
        sys.exit(1)

    # 1. 이전 결정 성과 업데이트
    try:
        update_past_performance()
    except Exception as e:
        print(f"[save_decision] 성과 업데이트 실패: {e}", file=sys.stderr)

    # 2. 현재 결정 저장
    saved = save_decision(data)
    if saved:
        print(json.dumps({"success": True, "saved": saved}, ensure_ascii=False, default=str))
    else:
        print(json.dumps({"success": False, "error": "decisions INSERT 실패"}))
        sys.exit(1)

    # 3. 포트폴리오 스냅샷 저장
    try:
        save_portfolio_snapshot()
    except Exception:
        pass

    # 4. 사용된 피드백 applied 처리
    try:
        mark_feedback_applied()
    except Exception:
        pass


if __name__ == "__main__":
    main()
