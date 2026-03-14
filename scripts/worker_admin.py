#!/usr/bin/env python3
"""
워커 관리 CLI (관리자 전용)

등록 절차:
  1. 관리자 → 초대 이메일 발송
  2. 참여자 → 동의 회신
  3. 관리자 → 승인 + worker_admin.py approve 실행
  4. 참여자에게 worker-id + .env 파일 이메일 전달
  5. 참여자 → 워커 실행

사용법:
  python scripts/worker_admin.py list                          # 전체 워커 목록
  python scripts/worker_admin.py approve --id "jsh-pc" --tier coworker --name "JSH PC"
  python scripts/worker_admin.py approve --id "community-01" --tier collaborator --name "홍길동"
  python scripts/worker_admin.py suspend --id "community-01"   # 일시 정지
  python scripts/worker_admin.py remove --id "community-01"    # 완전 삭제
  python scripts/worker_admin.py promote --id "jsh-pc" --tier owner  # 티어 승격
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

KST = timezone(timedelta(hours=9))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def db_query(sql: str):
    """supabase_query.py의 execute_sql 호출"""
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    from supabase_query import find_token, execute_sql
    token = find_token()
    if not token:
        print("❌ Supabase 토큰을 찾을 수 없습니다")
        sys.exit(1)
    return execute_sql(token, sql)


def cmd_list(args):
    """전체 워커 목록"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id,worker_name,tier,status,is_main_brain,ram_gb,last_heartbeat",
                "order": "tier.asc,worker_id.asc"},
        headers=HEADERS,
        timeout=15,
    )
    workers = resp.json() if resp.ok else []

    if not workers:
        print("등록된 워커가 없습니다")
        return

    print(f"\n{'워커 ID':<16} {'이름':<24} {'티어':<14} {'상태':<10} {'RAM':<6} {'Main'}")
    print("-" * 90)
    for w in workers:
        main = "✅" if w.get("is_main_brain") else ""
        ram = f"{w.get('ram_gb', '-')}GB" if w.get('ram_gb') else "-"
        print(f"{w['worker_id']:<16} {(w.get('worker_name') or '-'):<24} "
              f"{w['tier']:<14} {w['status']:<10} {ram:<6} {main}")
    print(f"\n총 {len(workers)}대")


def cmd_approve(args):
    """워커 승인 등록"""
    if not args.id or not args.tier:
        print("❌ --id와 --tier는 필수입니다")
        return

    if args.tier not in ("owner", "coworker", "collaborator", "viewer"):
        print(f"❌ 유효하지 않은 티어: {args.tier}")
        return

    is_main = args.tier == "owner"
    name = args.name or args.id

    # 이미 존재하는지 확인
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id", "worker_id": f"eq.{args.id}"},
        headers=HEADERS,
        timeout=15,
    )
    if resp.ok and resp.json():
        print(f"⚠️  '{args.id}'는 이미 등록되어 있습니다. promote 명령으로 티어를 변경하세요.")
        return

    data = {
        "worker_id": args.id,
        "worker_name": name,
        "tier": args.tier,
        "status": "offline",
        "is_main_brain": is_main,
        "notes": f"승인: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}",
    }

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        json=data,
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )

    if resp.status_code < 300:
        print(f"✅ 워커 승인 완료: {args.id} (tier: {args.tier})")
        print(f"   이름: {name}")
        print()
        print(f"   📧 참여자에게 전달할 정보:")
        print(f"   - 워커 ID: {args.id}")
        print(f"   - 티어: {args.tier}")
        if args.tier == "coworker":
            print(f"   - 환경변수: .env.coworker 파일 첨부")
        elif args.tier == "collaborator":
            print(f"   - 환경변수: .env.collaborator 파일 첨부")
        print(f"   - 실행 명령: python -m scalp_ml.worker --worker-id \"{args.id}\" --tier {args.tier}")
    else:
        print(f"❌ 등록 실패: {resp.text}")


def cmd_suspend(args):
    """워커 일시 정지"""
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        json={"status": "suspended",
              "notes": f"정지: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"⏸️  워커 정지: {args.id}")
    else:
        print(f"❌ 실패: {resp.text}")


def cmd_remove(args):
    """워커 완전 삭제"""
    # heartbeat 로그 먼저 삭제
    requests.delete(
        f"{SUPABASE_URL}/rest/v1/worker_heartbeats",
        params={"worker_id": f"eq.{args.id}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    resp = requests.delete(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"🗑️  워커 삭제: {args.id}")
    else:
        print(f"❌ 실패: {resp.text}")


def cmd_promote(args):
    """워커 티어 변경"""
    if args.tier not in ("owner", "coworker", "collaborator", "viewer"):
        print(f"❌ 유효하지 않은 티어: {args.tier}")
        return

    is_main = args.tier == "owner"
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        json={"tier": args.tier, "is_main_brain": is_main,
              "notes": f"티어 변경→{args.tier}: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"⬆️  티어 변경: {args.id} → {args.tier}")
    else:
        print(f"❌ 실패: {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="워커 관리 CLI (관리자 전용)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="전체 워커 목록")

    p_approve = sub.add_parser("approve", help="워커 승인 등록")
    p_approve.add_argument("--id", required=True, help="워커 ID")
    p_approve.add_argument("--tier", required=True, help="티어")
    p_approve.add_argument("--name", help="표시 이름")

    p_suspend = sub.add_parser("suspend", help="워커 일시 정지")
    p_suspend.add_argument("--id", required=True)

    p_remove = sub.add_parser("remove", help="워커 완전 삭제")
    p_remove.add_argument("--id", required=True)

    p_promote = sub.add_parser("promote", help="티어 변경")
    p_promote.add_argument("--id", required=True)
    p_promote.add_argument("--tier", required=True)

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 미설정")
        sys.exit(1)

    cmds = {
        "list": cmd_list,
        "approve": cmd_approve,
        "suspend": cmd_suspend,
        "remove": cmd_remove,
        "promote": cmd_promote,
    }

    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
