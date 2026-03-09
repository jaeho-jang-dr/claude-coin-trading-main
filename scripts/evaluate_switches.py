"""
전환 성과 평가 스크립트

agent_switches 테이블에서 미평가된 전환을 찾아,
전환 후 4h/24h 가격 변화를 기록하여 학습 데이터를 축적한다.

cron으로 1시간마다 실행하거나, run_agents.sh 끝에 호출한다.
"""

import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import requests

KST = timezone(timedelta(hours=9))


def get_current_price() -> int:
    """Upbit에서 현재 BTC 가격을 조회한다."""
    try:
        resp = requests.get(
            "https://api.upbit.com/v1/ticker",
            params={"markets": "KRW-BTC"},
            timeout=5,
        )
        return int(resp.json()[0]["trade_price"])
    except Exception:
        return 0


def evaluate_pending_switches():
    """미평가 전환을 찾아 성과를 기록한다."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("SUPABASE 환경변수 미설정", file=sys.stderr)
        return

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    # 미평가 + 4시간 이상 경과한 전환 조회
    cutoff_4h = (datetime.now(KST) - timedelta(hours=4)).isoformat()
    resp = requests.get(
        f"{url}/rest/v1/agent_switches",
        params={
            "select": "id,price_at_switch,created_at,price_after_4h",
            "evaluated_at": "is.null",
            "created_at": f"lt.{cutoff_4h}",
            "order": "created_at.asc",
            "limit": "20",
        },
        headers=headers,
        timeout=10,
    )

    if resp.status_code != 200:
        print(f"조회 실패: {resp.status_code}", file=sys.stderr)
        return

    switches = resp.json()
    if not switches:
        print("평가할 전환 없음")
        return

    current_price = get_current_price()
    if current_price == 0:
        print("현재가 조회 실패", file=sys.stderr)
        return

    now = datetime.now(KST)
    evaluated = 0

    for sw in switches:
        sw_id = sw["id"]
        price_at = sw.get("price_at_switch")
        created_at = sw.get("created_at", "")

        if not price_at:
            # 전환 시 가격이 없으면 스킵
            continue

        price_at = int(price_at)

        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        hours_elapsed = (now - created_dt).total_seconds() / 3600
        update = {}

        # 4시간 평가
        if sw.get("price_after_4h") is None and hours_elapsed >= 4:
            update["price_after_4h"] = current_price
            update["profit_after_4h"] = round((current_price - price_at) / price_at * 100, 2)

        # 24시간 평가 (최종)
        if hours_elapsed >= 24:
            update["price_after_24h"] = current_price
            profit_24h = round((current_price - price_at) / price_at * 100, 2)
            update["profit_after_24h"] = profit_24h

            # 성과 판정
            if profit_24h > 1:
                update["outcome"] = "good"
                update["outcome_reason"] = f"전환 후 24h 수익 {profit_24h:+.2f}%"
            elif profit_24h < -1:
                update["outcome"] = "bad"
                update["outcome_reason"] = f"전환 후 24h 손실 {profit_24h:+.2f}%"
            else:
                update["outcome"] = "neutral"
                update["outcome_reason"] = f"전환 후 24h 변동 {profit_24h:+.2f}% (중립)"

            update["evaluated_at"] = now.isoformat()

        if update:
            patch_resp = requests.patch(
                f"{url}/rest/v1/agent_switches",
                params={"id": f"eq.{sw_id}"},
                json=update,
                headers={**headers, "Prefer": "return=minimal"},
                timeout=5,
            )
            if patch_resp.status_code in (200, 204):
                evaluated += 1
                print(f"  평가 완료: {sw_id[:8]}... {update.get('outcome', '4h만')}")

    print(f"총 {evaluated}/{len(switches)}건 평가 완료")


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    evaluate_pending_switches()
