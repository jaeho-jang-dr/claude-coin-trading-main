#!/usr/bin/env python3
"""
결정 사후 추적 스크립트 -- 매 시간 cron으로 실행

1h/4h/24h 전 결정들의 가격 변동을 기록하여 결정 정확도를 측정한다.

사용법:
  python3 scripts/retrospective.py           # 모든 미평가 결정 업데이트
  python3 scripts/retrospective.py report    # 정확도 리포트 출력

cron 예시 (매시 15분):
  15 * * * * cd ~/workspace/blockchain && .venv/bin/python3 scripts/retrospective.py
"""

import json, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
UPBIT_API = "https://api.upbit.com/v1"
KST = timezone(timedelta(hours=9))


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def get_btc_price() -> int:
    """현재 BTC/KRW 가격"""
    r = requests.get(f"{UPBIT_API}/ticker", params={"markets": "KRW-BTC"}, timeout=10)
    if r.ok:
        return int(r.json()[0]["trade_price"])
    return 0


def get_historical_price(target_time: datetime) -> int:
    """특정 시점의 BTC 가격 (분봉으로 근사)"""
    # Use minute candle closest to target time
    params = {
        "market": "KRW-BTC",
        "to": target_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": 1,
    }
    r = requests.get(f"{UPBIT_API}/candles/minutes/60", params=params, timeout=10)
    if r.ok and r.json():
        return int(r.json()[0]["trade_price"])
    return 0


def _evaluate_correctness(decision_type: str, outcome_pct: float, window: str) -> bool:
    """결정이 올바랐는지 판단"""
    if decision_type == "매수":
        return outcome_pct > 0
    elif decision_type == "매도":
        return outcome_pct < 0
    else:  # 관망
        threshold = 1.0 if window == "24h" else 0.5
        return abs(outcome_pct) < threshold


def _update_window(window: str, hours: int, extra_filter: dict = None):
    """특정 시간 윈도우의 aftermath 업데이트"""
    now = datetime.now(KST)
    cutoff = (now - timedelta(hours=hours)).isoformat()

    price_col = f"price_{window}_after"
    outcome_col = f"outcome_{window}_pct"
    correct_col = f"was_correct_{window}"

    params = {
        price_col: "is.null",
        "created_at": f"lt.{cutoff}",
        "select": "id,decision,current_price,created_at",
        "limit": "100",
    }
    if extra_filter:
        params.update(extra_filter)

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/decisions",
        headers=supabase_headers(),
        params=params,
        timeout=15,
    )
    if not r.ok:
        print(f"[retrospective] {window} 조회 실패: {r.status_code}", file=sys.stderr)
        return 0

    updated = 0
    for row in r.json():
        decision_price = row.get("current_price", 0)
        if not decision_price:
            continue

        decision_time = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        target_time = decision_time + timedelta(hours=hours)

        if target_time > now:
            continue

        price_after = get_historical_price(target_time)
        if not price_after:
            continue

        outcome_pct = round((price_after - decision_price) / decision_price * 100, 3)
        decision_type = row.get("decision", "")
        was_correct = _evaluate_correctness(decision_type, outcome_pct, window)

        patch = {
            price_col: price_after,
            outcome_col: outcome_pct,
            correct_col: was_correct,
            "aftermath_updated_at": now.isoformat(),
        }
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/decisions?id=eq.{row['id']}",
            headers={**supabase_headers(), "Prefer": "return=minimal"},
            json=patch,
            timeout=10,
        )
        updated += 1
        time.sleep(0.2)  # Rate limit for Upbit API

    return updated


def update_decisions():
    """미평가 결정들의 aftermath 업데이트"""
    updated_1h = _update_window("1h", 1)
    updated_4h = _update_window("4h", 4, extra_filter={"price_1h_after": "not.is.null"})
    updated_24h = _update_window("24h", 24, extra_filter={"price_4h_after": "not.is.null"})

    print(f"[retrospective] 업데이트 완료: 1h={updated_1h}, 4h={updated_4h}, 24h={updated_24h}")
    return {"1h": updated_1h, "4h": updated_4h, "24h": updated_24h}


def update_scalp_aftermath():
    """초단타 거래 aftermath 업데이트"""
    now = datetime.now(KST)
    cutoff = (now - timedelta(hours=1)).isoformat()

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/scalp_trade_log",
        headers=supabase_headers(),
        params={
            "price_1h_after": "is.null",
            "entry_time": f"lt.{cutoff}",
            "select": "id,entry_price,exit_price,pnl_pct,entry_time",
            "limit": "100",
        },
        timeout=15,
    )
    if not r.ok:
        return

    updated = 0
    for row in r.json():
        entry_price = row.get("entry_price", 0)
        if not entry_price:
            continue

        entry_time = datetime.fromisoformat(row["entry_time"].replace("Z", "+00:00"))
        target_time = entry_time + timedelta(hours=1)

        if target_time > now:
            continue

        price_after = get_historical_price(target_time)
        if not price_after:
            continue

        outcome_pct = round((price_after - entry_price) / entry_price * 100, 3)

        # 매수 진입이었으므로: 1시간 후 가격이 진입가보다 높으면 좋은 진입
        was_good_entry = outcome_pct > 0

        patch = {
            "price_1h_after": price_after,
            "outcome_1h_pct": outcome_pct,
            "was_good_entry": was_good_entry,
            "aftermath_updated_at": now.isoformat(),
        }
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/scalp_trade_log?id=eq.{row['id']}",
            headers={**supabase_headers(), "Prefer": "return=minimal"},
            json=patch,
            timeout=10,
        )
        updated += 1
        time.sleep(0.2)

    if updated:
        print(f"[retrospective] 초단타 {updated}건 업데이트")


def update_signal_attempts():
    """차단된/생성된 시그널의 1시간 후 가격 업데이트"""
    now = datetime.now(KST)
    cutoff = (now - timedelta(hours=1)).isoformat()

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/signal_attempt_log",
        headers=supabase_headers(),
        params={
            "price_1h_after": "is.null",
            "signal_type": "neq.no_signal",
            "recorded_at": f"lt.{cutoff}",
            "select": "id,btc_price,recorded_at,signal_type,action",
            "limit": "100",
        },
        timeout=15,
    )
    if not r.ok:
        print(f"[retrospective] signal_attempt 조회 실패: {r.status_code}", file=sys.stderr)
        return

    updated = 0
    for row in r.json():
        btc_price = row.get("btc_price", 0)
        if not btc_price:
            continue

        recorded_at = datetime.fromisoformat(row["recorded_at"].replace("Z", "+00:00"))
        target_time = recorded_at + timedelta(hours=1)

        if target_time > now:
            continue

        price_after = get_historical_price(target_time)
        if not price_after:
            continue

        outcome_pct = round((price_after - btc_price) / btc_price * 100, 3)

        # 매수 시그널이었다면: 1시간 후 가격 상승 = 거래했으면 수익
        action = row.get("action", "buy")
        if action == "buy":
            would_have_won = outcome_pct > 0
        elif action == "sell":
            would_have_won = outcome_pct < 0
        else:
            would_have_won = None

        patch = {
            "price_1h_after": price_after,
            "outcome_1h_pct": outcome_pct,
            "would_have_won": would_have_won,
            "aftermath_updated_at": now.isoformat(),
        }
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/signal_attempt_log?id=eq.{row['id']}",
            headers={**supabase_headers(), "Prefer": "return=minimal"},
            json=patch,
            timeout=10,
        )
        updated += 1
        time.sleep(0.2)

    if updated:
        print(f"[retrospective] signal_attempt {updated}건 업데이트")


def update_buy_score_aftermath():
    """니어미스/AI거부 사후 추적 -- 1h/4h 후 가격을 기록하여 판단 정확도를 측정한다."""
    now = datetime.now(KST)
    current_price = get_btc_price()
    if not current_price:
        print("[retrospective] BTC 가격 조회 실패, buy_score aftermath 스킵", file=sys.stderr)
        return

    headers = {**supabase_headers(), "Prefer": "return=minimal"}

    # ── 1시간 후 가격 업데이트 ──
    cutoff_1h = (now - timedelta(hours=1)).isoformat()
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/buy_score_detail",
        headers=supabase_headers(),
        params={
            "price_1h_after": "is.null",
            "recorded_at": f"lt.{cutoff_1h}",
            "or": "(is_near_miss.eq.true,was_ai_vetoed.eq.true)",
            "select": "id,price_at_decision,recorded_at,action",
            "limit": "100",
        },
        timeout=15,
    )
    updated_1h = 0
    if r.ok:
        for row in r.json():
            decision_price = row.get("price_at_decision", 0)
            if not decision_price:
                continue

            recorded_at = datetime.fromisoformat(row["recorded_at"].replace("Z", "+00:00"))
            target_time = recorded_at + timedelta(hours=1)
            if target_time > now:
                continue

            price_after = get_historical_price(target_time)
            if not price_after:
                continue

            outcome_pct = round((price_after - decision_price) / decision_price * 100, 3)

            patch = {
                "price_1h_after": price_after,
                "outcome_1h_pct": outcome_pct,
                "aftermath_updated_at": now.isoformat(),
            }
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/buy_score_detail?id=eq.{row['id']}",
                headers=headers,
                json=patch,
                timeout=10,
            )
            updated_1h += 1
            time.sleep(0.2)

    # ── 4시간 후 가격 업데이트 ──
    cutoff_4h = (now - timedelta(hours=4)).isoformat()
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/buy_score_detail",
        headers=supabase_headers(),
        params={
            "price_4h_after": "is.null",
            "price_1h_after": "not.is.null",
            "recorded_at": f"lt.{cutoff_4h}",
            "or": "(is_near_miss.eq.true,was_ai_vetoed.eq.true)",
            "select": "id,price_at_decision,recorded_at,action",
            "limit": "100",
        },
        timeout=15,
    )
    updated_4h = 0
    if r.ok:
        for row in r.json():
            decision_price = row.get("price_at_decision", 0)
            if not decision_price:
                continue

            recorded_at = datetime.fromisoformat(row["recorded_at"].replace("Z", "+00:00"))
            target_time = recorded_at + timedelta(hours=4)
            if target_time > now:
                continue

            price_after = get_historical_price(target_time)
            if not price_after:
                continue

            outcome_pct = round((price_after - decision_price) / decision_price * 100, 3)
            action = row.get("action", "hold")
            # would_have_profited: 관망/거부했는데 올랐으면 True
            if action in ("hold",):
                would_have_profited = outcome_pct > 0.5  # 0.5% 이상 상승
            elif action in ("buy", "매수"):
                would_have_profited = outcome_pct > 0
            else:
                would_have_profited = None

            patch = {
                "price_4h_after": price_after,
                "outcome_4h_pct": outcome_pct,
                "would_have_profited": would_have_profited,
                "aftermath_updated_at": now.isoformat(),
            }
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/buy_score_detail?id=eq.{row['id']}",
                headers=headers,
                json=patch,
                timeout=10,
            )
            updated_4h += 1
            time.sleep(0.2)

    if updated_1h or updated_4h:
        print(f"[retrospective] buy_score aftermath: 1h={updated_1h}, 4h={updated_4h}")


def report():
    """정확도 리포트"""
    # Decision accuracy
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/v_decision_accuracy",
        headers=supabase_headers(),
        timeout=15,
    )
    if r.ok and r.json():
        print("=== 결정 정확도 ===")
        print(
            f"  {'결정':<6} {'소스':<8} {'총':<5} {'1h정확':>7} {'1h평균':>7} "
            f"{'4h정확':>7} {'4h평균':>7} {'24h정확':>8} {'24h평균':>7}"
        )
        for row in r.json():
            d = row.get('decision') or '?'
            s = row.get('source') or '?'
            t = row.get('total') or 0
            a1 = row.get('accuracy_1h')
            m1 = row.get('avg_1h_pct')
            a4 = row.get('accuracy_4h')
            m4 = row.get('avg_4h_pct')
            a24 = row.get('accuracy_24h')
            m24 = row.get('avg_24h_pct')
            print(
                f"  {d:<6} {s:<8} {t:<5} "
                f"{f'{a1}%' if a1 is not None else '-':>7} "
                f"{f'{m1}%' if m1 is not None else '-':>7} "
                f"{f'{a4}%' if a4 is not None else '-':>7} "
                f"{f'{m4}%' if m4 is not None else '-':>7} "
                f"{f'{a24}%' if a24 is not None else '-':>8} "
                f"{f'{m24}%' if m24 is not None else '-':>7}"
            )
    else:
        print("=== 결정 정확도 === (데이터 없음)")

    # Missed opportunities
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/v_missed_opportunities",
        headers=supabase_headers(),
        params={"limit": "10"},
        timeout=15,
    )
    if r.ok and r.json():
        print(f"\n=== 놓친 기회 TOP {len(r.json())} ===")
        for row in r.json():
            print(
                f"  {row.get('created_at', '')[:16]} | "
                f"가격: {row.get('current_price', 0):,} | "
                f"24h후: +{row.get('outcome_24h_pct', 0):.1f}% | "
                f"FGI:{row.get('fear_greed_value', '?')} RSI:{row.get('rsi_value', '?')}"
            )
            print(f"    사유: {(row.get('reason') or '')[:80]}")

    # Bad trades
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/v_bad_trades",
        headers=supabase_headers(),
        params={"limit": "10"},
        timeout=15,
    )
    if r.ok and r.json():
        print(f"\n=== 잘못된 매수 TOP {len(r.json())} ===")
        for row in r.json():
            print(
                f"  {row.get('created_at', '')[:16]} | "
                f"가격: {row.get('current_price', 0):,} | "
                f"4h후: {row.get('outcome_4h_pct', 0):+.1f}% | "
                f"금액: {row.get('trade_amount', 0):,}"
            )
            print(f"    사유: {(row.get('reason') or '')[:80]}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        update_decisions()
        update_scalp_aftermath()
        update_signal_attempts()
        update_buy_score_aftermath()
        if "--report" in sys.argv:
            report()


if __name__ == "__main__":
    main()
