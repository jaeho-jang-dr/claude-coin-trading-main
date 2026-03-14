#!/usr/bin/env python3
"""
DB 자동 정리 스크립트

Supabase 무료 플랜 500MB 한도 내에서 운영하기 위한 데이터 정리.
매일 1회 cron으로 실행 권장.

실행:
  python3 scripts/db_cleanup.py              # 정리 실행
  python3 scripts/db_cleanup.py --dry-run    # 미리보기만
  python3 scripts/db_cleanup.py --status     # DB 용량 현황
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

KST = timezone(timedelta(hours=9))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("db_cleanup")

# ── 정리 규칙 ──
CLEANUP_RULES = [
    {
        "name": "하트비트 7일+",
        "table": "worker_heartbeats",
        "condition": f"recorded_at=lt.{(datetime.now(KST) - timedelta(days=7)).isoformat()}",
    },
    {
        "name": "no_signal 30일+",
        "table": "signal_attempt_log",
        "condition": f"signal_type=eq.no_signal&recorded_at=lt.{(datetime.now(KST) - timedelta(days=30)).isoformat()}",
    },
    {
        "name": "blocked 사후추적완료 90일+",
        "table": "signal_attempt_log",
        "condition": f"signal_type=eq.blocked&outcome_5m_pct=not.is.null&recorded_at=lt.{(datetime.now(KST) - timedelta(days=90)).isoformat()}",
    },
    {
        "name": "시스템 헬스 로그 30일+",
        "table": "system_health_logs",
        "condition": f"recorded_at=lt.{(datetime.now(KST) - timedelta(days=30)).isoformat()}",
    },
    {
        "name": "전략 알림 30일+",
        "table": "strategy_alerts",
        "condition": f"created_at=lt.{(datetime.now(KST) - timedelta(days=30)).isoformat()}",
    },
]


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def count_rows(table: str, condition: str) -> int:
    """조건에 맞는 행 수 조회"""
    params = {"select": "id", "limit": "0"}
    # condition 파싱
    for part in condition.split("&"):
        key, val = part.split("=", 1)
        params[key] = val
    params["Prefer"] = "count=exact"

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            params=params,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Prefer": "count=exact",
            },
            timeout=10,
        )
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range:
            total = content_range.split("/")[1]
            return int(total) if total != "*" else 0
    except Exception:
        pass
    return 0


def delete_rows(table: str, condition: str) -> int:
    """조건에 맞는 행 삭제"""
    params = {}
    for part in condition.split("&"):
        key, val = part.split("=", 1)
        params[key] = val

    try:
        resp = requests.delete(
            f"{SUPABASE_URL}/rest/v1/{table}",
            params=params,
            headers=supabase_headers(),
            timeout=30,
        )
        if resp.status_code < 300:
            return -1  # 성공 (정확한 삭제 수 모름)
        log.warning(f"삭제 실패 ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        log.warning(f"삭제 예외: {e}")
    return 0


def show_status():
    """DB 용량 현황"""
    try:
        # Management API로 DB 크기 조회
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from supabase_query import find_token, execute_sql
        token = find_token()

        # 전체 크기
        result = execute_sql(token, "SELECT pg_size_pretty(pg_database_size('postgres')) AS total")
        total = result[0]["total"] if result else "?"

        # 테이블별 크기
        tables = execute_sql(token, """
            SELECT relname AS name,
                   pg_size_pretty(pg_total_relation_size(relid)) AS size,
                   n_live_tup AS rows
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(relid) DESC
            LIMIT 10
        """)

        print(f"\n{'='*55}")
        print(f"  DB 총 용량: {total} / 500 MB (무료 한도)")
        print(f"{'='*55}")
        print(f"  {'테이블':<25} {'크기':>8}  {'행 수':>8}")
        print(f"  {'-'*25} {'-'*8}  {'-'*8}")
        for t in tables:
            print(f"  {t['name']:<25} {t['size']:>8}  {t['rows']:>8}")
        print()

    except Exception as e:
        log.error(f"상태 조회 실패: {e}")


def run_cleanup(dry_run: bool = False):
    """정리 실행"""
    total_deleted = 0

    print(f"\n{'='*55}")
    print(f"  DB 정리 {'(DRY RUN - 미리보기)' if dry_run else '실행'}")
    print(f"  {datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}")
    print(f"{'='*55}\n")

    for rule in CLEANUP_RULES:
        count = count_rows(rule["table"], rule["condition"])

        if count == 0:
            log.info(f"  [{rule['name']}] 정리 대상 없음")
            continue

        if dry_run:
            log.info(f"  [{rule['name']}] {count}건 삭제 예정 ({rule['table']})")
        else:
            log.info(f"  [{rule['name']}] {count}건 삭제 중...")
            result = delete_rows(rule["table"], rule["condition"])
            if result != 0:
                log.info(f"  [{rule['name']}] 삭제 완료")
                total_deleted += count
            else:
                log.warning(f"  [{rule['name']}] 삭제 실패")

    if not dry_run and total_deleted > 0:
        log.info(f"\n  총 {total_deleted}건 정리 완료")

        # 텔레그램 알림
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_USER_ID", "")
        if token and chat_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": f"🗑️ DB 정리 완료: {total_deleted}건 삭제",
                    },
                    timeout=5,
                )
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="DB 자동 정리")
    parser.add_argument("--dry-run", action="store_true", help="미리보기만")
    parser.add_argument("--status", action="store_true", help="DB 용량 현황")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 미설정")
        sys.exit(1)

    if args.status:
        show_status()
    else:
        run_cleanup(dry_run=args.dry_run)
        if not args.dry_run:
            show_status()


if __name__ == "__main__":
    main()
