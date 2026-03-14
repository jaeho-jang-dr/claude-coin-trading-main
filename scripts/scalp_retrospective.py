#!/usr/bin/env python3
"""
초단타 시그널 사후 추적 스크립트

signal_attempt_log에 기록된 시그널의 1m/5m/15m/30m 후 가격을 추적하여
시그널 품질을 평가한다.

실행: cron으로 1분마다 실행 (가벼운 스크립트)
  * * * * * cd /Users/drj00/workspace/blockchain && .venv/bin/python3 scripts/scalp_retrospective.py

또는 봇 내부에서 asyncio 태스크로 실행 가능.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── 환경 설정 ──
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

KST = timezone(timedelta(hours=9))
MARKET = "KRW-BTC"
UPBIT_API = "https://api.upbit.com/v1"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("scalp_retro")

# ── 시간대별 설정 ──
TIMEFRAMES = [
    {"suffix": "1m", "minutes": 1, "col_price": "price_1m_after", "col_pct": "outcome_1m_pct"},
    {"suffix": "5m", "minutes": 5, "col_price": "price_5m_after", "col_pct": "outcome_5m_pct", "col_won": "would_have_won_5m"},
    {"suffix": "15m", "minutes": 15, "col_price": "price_15m_after", "col_pct": "outcome_15m_pct", "col_won": "would_have_won_15m"},
    {"suffix": "30m", "minutes": 30, "col_price": "price_30m_after", "col_pct": "outcome_30m_pct", "col_won": "would_have_won_30m"},
]

# 승리 판정 기준 (수수료 왕복 0.10% + 최소 마진)
WIN_THRESHOLD_PCT = 0.15


def supabase_get(table: str, params: dict) -> list[dict]:
    """Supabase REST API GET"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            params=params,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            timeout=10,
        )
        if resp.ok:
            return resp.json()
        log.warning(f"GET {table} 실패 ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        log.warning(f"GET {table} 예외: {e}")
    return []


def supabase_patch(table: str, filters: dict, data: dict) -> bool:
    """Supabase REST API PATCH (조건부 업데이트)"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        params = {f"{k}": f"eq.{v}" for k, v in filters.items()}
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}",
            params=params,
            json=data,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=10,
        )
        return resp.status_code < 300
    except Exception as e:
        log.warning(f"PATCH {table} 예외: {e}")
        return False


def get_current_price() -> float:
    """현재 BTC 가격"""
    try:
        r = requests.get(f"{UPBIT_API}/ticker", params={"markets": MARKET}, timeout=5)
        return r.json()[0]["trade_price"]
    except Exception as e:
        log.error(f"가격 조회 실패: {e}")
        return 0


def get_historical_price(minutes_ago: int) -> float | None:
    """N분 전 가격 (1분봉에서 추출)"""
    try:
        r = requests.get(
            f"{UPBIT_API}/candles/minutes/1",
            params={"market": MARKET, "count": minutes_ago + 2},
            timeout=5,
        )
        if r.ok:
            candles = r.json()
            if len(candles) > minutes_ago:
                return candles[minutes_ago]["trade_price"]
    except Exception:
        pass
    return None


def process_timeframe(tf: dict, now: datetime, current_price: float) -> int:
    """특정 시간대의 미추적 시그널을 업데이트. 처리 건수 반환."""
    col_price = tf["col_price"]
    col_pct = tf["col_pct"]
    col_won = tf.get("col_won")
    minutes = tf["minutes"]

    # 미추적 시그널 조회 (시간 경과 + 미기록)
    cutoff = (now - timedelta(minutes=minutes + 1)).isoformat()

    signals = supabase_get("signal_attempt_log", {
        "select": "id,recorded_at,btc_price,action,signal_type",
        f"{col_price}": "is.null",
        "signal_type": "neq.no_signal",
        "recorded_at": f"lt.{cutoff}",
        "order": "recorded_at.desc",
        "limit": "50",
    })

    if not signals:
        return 0

    updated = 0
    for sig in signals:
        entry_price = sig.get("btc_price")
        if not entry_price:
            continue

        # 현재 가격을 사후 가격으로 사용 (정확한 N분 후가 아닐 수 있지만 근사값)
        after_price = current_price
        outcome_pct = round((after_price - entry_price) / entry_price * 100, 3)

        update_data = {
            col_price: int(after_price),
            col_pct: outcome_pct,
        }

        # 승리 판정 (매수 시그널 기준)
        if col_won:
            action = sig.get("action", "buy")
            if action == "buy":
                update_data[col_won] = outcome_pct >= WIN_THRESHOLD_PCT
            elif action == "sell":
                update_data[col_won] = outcome_pct <= -WIN_THRESHOLD_PCT
            else:
                update_data[col_won] = abs(outcome_pct) >= WIN_THRESHOLD_PCT

        if supabase_patch("signal_attempt_log", {"id": sig["id"]}, update_data):
            updated += 1

    return updated


def update_best_worst_30m(now: datetime, current_price: float) -> int:
    """30분 내 최고/최저 가격 추적 (best_price_30m, worst_price_30m)"""
    cutoff_30m = (now - timedelta(minutes=30)).isoformat()
    cutoff_start = (now - timedelta(minutes=35)).isoformat()

    # 30분 전후의 시그널 중 best/worst 미기록건
    signals = supabase_get("signal_attempt_log", {
        "select": "id,btc_price,action",
        "signal_type": "neq.no_signal",
        "recorded_at": f"gt.{cutoff_start}",
        "best_price_30m": "is.null",
        "order": "recorded_at.desc",
        "limit": "30",
    })

    if not signals:
        return 0

    # 최근 30분 가격 범위 추출 (1분봉)
    try:
        r = requests.get(
            f"{UPBIT_API}/candles/minutes/1",
            params={"market": MARKET, "count": 35},
            timeout=5,
        )
        if not r.ok:
            return 0
        candles = r.json()
        highs = [c["high_price"] for c in candles]
        lows = [c["low_price"] for c in candles]
        best_30m = max(highs) if highs else current_price
        worst_30m = min(lows) if lows else current_price
    except Exception:
        return 0

    updated = 0
    for sig in signals:
        entry_price = sig.get("btc_price")
        if not entry_price:
            continue

        best_exit_pct = round((best_30m - entry_price) / entry_price * 100, 3)
        worst_dd_pct = round((worst_30m - entry_price) / entry_price * 100, 3)

        update_data = {
            "best_price_30m": int(best_30m),
            "worst_price_30m": int(worst_30m),
            "best_exit_pct": best_exit_pct,
            "worst_drawdown_pct": worst_dd_pct,
        }

        if supabase_patch("signal_attempt_log", {"id": sig["id"]}, update_data):
            updated += 1

    return updated


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 미설정")
        sys.exit(1)

    now = datetime.now(KST)
    current_price = get_current_price()
    if not current_price:
        log.error("현재가 조회 실패 — 종료")
        sys.exit(1)

    total_updated = 0

    # 각 시간대별 사후 추적
    for tf in TIMEFRAMES:
        count = process_timeframe(tf, now, current_price)
        if count > 0:
            log.info(f"[{tf['suffix']}] {count}건 사후 추적 완료")
        total_updated += count

    # 30분 best/worst 추적
    bw_count = update_best_worst_30m(now, current_price)
    if bw_count > 0:
        log.info(f"[best/worst] {bw_count}건 추적 완료")
    total_updated += bw_count

    if total_updated > 0:
        log.info(f"총 {total_updated}건 사후 추적 완료 (BTC {current_price:,.0f}원)")
    else:
        log.debug("추적 대상 없음")


if __name__ == "__main__":
    main()
