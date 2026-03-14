#!/usr/bin/env python3
"""
워커 관리 CLI (관리자 전용)

=== 방식 A: 초대코드 (자가등록, 제로 지식) ===
  1. 관리자 → invite 생성 (초대코드 발급)
  2. 관리자 → 초대코드 + .env.{tier} 셋업 파일을 이메일로 전달
  3. 참여자 → 초대코드로 자가등록 (토큰 자동 발급, 본인만 확인)
  4. 참여자 → .env에 토큰 입력 후 워커 실행
  ※ 관리자는 토큰을 알 수 없음 (제로 지식)

=== 방식 B: 직접 발송 (관리자가 토큰 생성 → 즉시 전달) ===
  1. 관리자 → direct-send 실행 (워커 등록 + 토큰 생성 + 이메일/텔레그램 발송)
  2. 참여자 → 수신한 토큰을 .env에 입력 후 워커 실행
  ※ 관리자가 토큰을 알 수 있음 (편의 우선)

사용법:
  python scripts/worker_admin.py list                                      # 워커 목록
  python scripts/worker_admin.py invite --tier coworker --name "홍길동" --email "hong@email.com"
  python scripts/worker_admin.py invite --tier collaborator --name "김철수" --hours 72
  python scripts/worker_admin.py direct-send --tier collaborator --name "김철수" --email "kim@email.com"
  python scripts/worker_admin.py direct-send --tier coworker --name "홍길동" --telegram "123456789"
  python scripts/worker_admin.py invites                                   # 초대 현황
  python scripts/worker_admin.py revoke-invite --code "ABC12345"           # 초대 취소
  python scripts/worker_admin.py msg --id "worker-id" -m "안녕하세요"      # 워커에게 텔레그램 메시지
  python scripts/worker_admin.py msg-all -m "서버 점검 예정"               # 전체 공지
  python scripts/worker_admin.py listen                                    # 수신 대기 (폴링)
  python scripts/worker_admin.py inbox                                     # 수신함 확인
  python scripts/worker_admin.py reply --to "worker-id" -m "확인했습니다"  # 답장
  python scripts/worker_admin.py chat --id "worker-id"                     # 실시간 대화
  python scripts/worker_admin.py suspend --id "worker-id"                  # 일시 정지
  python scripts/worker_admin.py unsuspend --id "worker-id"                # 정지 해제
  python scripts/worker_admin.py remove --id "worker-id"                   # 완전 삭제
  python scripts/worker_admin.py promote --id "worker-id" --tier owner     # 티어 변경
  python scripts/worker_admin.py invalidate --id "worker-id"               # 토큰 무효화 (재등록 필요)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import smtplib
import string
import sys
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

# Windows cp949 대응
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

# 이메일 발송 설정 (선택사항)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "") or SMTP_USER

# 텔레그램 발송
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


def send_email(to_email: str, subject: str, body: str) -> bool:
    """SMTP 이메일 발송 (설정된 경우)"""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS]):
        print("  ⚠ SMTP 미설정 — 이메일 발송 건너뜀")
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print(f"  ✅ 이메일 발송 완료: {to_email}")
        return True
    except Exception as e:
        print(f"  ❌ 이메일 발송 실패: {e}")
        return False


def send_telegram(chat_id: str, text: str) -> bool:
    """텔레그램 메시지 발송"""
    if not TG_TOKEN:
        print("  ⚠ TELEGRAM_BOT_TOKEN 미설정 — 텔레그램 발송 건너뜀")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.ok:
            print(f"  ✅ 텔레그램 발송 완료: chat_id={chat_id}")
            return True
        else:
            print(f"  ❌ 텔레그램 발송 실패: {resp.text}")
            return False
    except Exception as e:
        print(f"  ❌ 텔레그램 발송 실패: {e}")
        return False


def generate_invite_code(length: int = 8) -> str:
    """읽기 쉬운 초대코드 생성 (대문자+숫자, 혼동 문자 제외)"""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # O/0/I/1 제외
    return "".join(secrets.choice(chars) for _ in range(length))


# ── 관리자 명령 ──────────────────────────────────────


def cmd_list(args):
    """전체 워커 목록"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id,worker_name,tier,status,is_main_brain,ram_gb,gpu_info,telegram_chat_id,email,last_heartbeat",
                "order": "tier.asc,worker_id.asc"},
        headers=HEADERS,
        timeout=15,
    )
    workers = resp.json() if resp.ok else []

    if not workers:
        print("등록된 워커가 없습니다")
        return

    print(f"\n{'워커 ID':<16} {'이름':<14} {'티어':<12} {'상태':<10} {'텔레그램':<14} {'이메일':<24} {'Main'}")
    print("-" * 110)
    for w in workers:
        main = "Y" if w.get("is_main_brain") else ""
        tg = w.get('telegram_chat_id') or '-'
        email = (w.get('email') or '-')[:22]
        print(f"{w['worker_id']:<16} {(w.get('worker_name') or '-'):<14} "
              f"{w['tier']:<12} {w['status']:<10} {tg:<14} {email:<24} {main}")
    print(f"\n총 {len(workers)}대")


def cmd_invite(args):
    """초대코드 생성 → 관리자가 이메일로 전달"""
    if args.tier not in ("owner", "coworker", "collaborator", "viewer"):
        print(f"유효하지 않은 티어: {args.tier}")
        return

    code = generate_invite_code()
    hours = args.hours or 48
    expires = datetime.now(KST) + timedelta(hours=hours)
    name = args.name or ""
    email = args.email or ""

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/worker_invites",
        json={
            "invite_code": code,
            "tier": args.tier,
            "worker_name": name,
            "email": email,
            "expires_at": expires.isoformat(),
        },
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )

    if resp.status_code < 300:
        print(f"초대코드 생성 완료")
        print(f"  코드: {code}")
        print(f"  티어: {args.tier}")
        print(f"  이름: {name}")
        print(f"  이메일: {email}")
        print(f"  만료: {expires.strftime('%Y-%m-%d %H:%M')} KST ({hours}시간)")
        print()
        print(f"=== 이메일로 전달할 내용 ===")
        print()
        print(f"  안녕하세요, {name}님.")
        print(f"  암호화폐 자동매매 분산 컴퓨팅 시스템에 {args.tier}로 초대합니다.")
        print()
        print(f"  1. 프로젝트를 클론하세요:")
        print(f"     git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git")
        print(f"     cd claude-coin-trading-main")
        print(f"     pip install -r requirements.txt")
        print()
        print(f"  2. .env.{args.tier} 파일을 .env로 복사하세요:")
        print(f"     cp .env.{args.tier} .env")
        print()
        print(f"  3. .env 파일에 SUPABASE_URL과 SUPABASE_ANON_KEY를 입력하세요.")
        print(f"     (이 값은 별도로 전달드립니다)")
        print()
        print(f"  4. 자가등록 명령을 실행하세요:")
        print(f"     python -m scalp_ml.worker --register --invite-code {code}")
        print()
        print(f"  5. 화면에 표시된 WORKER_TOKEN을 .env 파일에 입력하세요.")
        print(f"     (이 토큰은 본인만 확인할 수 있습니다)")
        print()
        print(f"  6. 워커를 실행하세요:")
        print(f"     python -m scalp_ml.worker --worker-id \"your-pc-name\" --tier {args.tier}")
        print()
        print(f"  초대코드 만료: {expires.strftime('%Y-%m-%d %H:%M')} KST")
        print(f"=== 끝 ===")
    else:
        print(f"초대 생성 실패: {resp.text}")


def cmd_direct_send(args):
    """워커 직접 등록 + 토큰 발송 (이메일/텔레그램/콘솔)"""
    if args.tier not in ("owner", "coworker", "collaborator", "viewer"):
        print(f"유효하지 않은 티어: {args.tier}")
        return

    if not args.email and not args.telegram:
        print("--email 또는 --telegram 중 하나는 필수입니다")
        return

    name = args.name or ""
    worker_id = args.worker_id or f"{name.lower().replace(' ', '-')}-{secrets.token_hex(3)}"
    token = str(uuid.uuid4())
    is_main = args.tier == "owner"

    # DB에 워커 직접 등록
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        json={
            "worker_id": worker_id,
            "worker_name": name,
            "tier": args.tier,
            "status": "offline",
            "api_token": token,
            "is_main_brain": is_main,
            "telegram_chat_id": args.telegram or None,
            "email": args.email or None,
            "notes": f"직접 등록: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}",
        },
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )

    if resp.status_code >= 300:
        print(f"워커 등록 실패: {resp.text}")
        return

    print(f"워커 등록 완료")
    print(f"  워커 ID: {worker_id}")
    print(f"  이름: {name}")
    print(f"  티어: {args.tier}")
    print(f"  토큰: {token}")

    # 발송할 메시지 구성
    setup_msg = (
        f"안녕하세요, {name}님.\n"
        f"암호화폐 자동매매 분산 컴퓨팅 시스템에 {args.tier}로 등록되었습니다.\n\n"
        f"=== 셋업 안내 ===\n\n"
        f"1. 프로젝트를 클론하세요:\n"
        f"   git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git\n"
        f"   cd claude-coin-trading-main\n"
        f"   pip install -r requirements.txt\n\n"
        f"2. .env.{args.tier} 파일을 .env로 복사하세요:\n"
        f"   cp .env.{args.tier} .env\n\n"
        f"3. .env 파일에 아래 값을 입력하세요:\n"
        f"   SUPABASE_URL=(별도 전달)\n"
        f"   SUPABASE_ANON_KEY=(별도 전달)\n"
        f"   WORKER_TOKEN={token}\n\n"
        f"4. 워커를 실행하세요:\n"
        f"   python -m scalp_ml.worker --worker-id \"{worker_id}\" --tier {args.tier}\n\n"
        f"⚠ 토큰은 이 메시지에서만 확인 가능합니다. 안전하게 보관하세요.\n"
    )

    sent = False

    # 이메일 발송
    if args.email:
        sent = send_email(
            args.email,
            f"[CoinTrading] {args.tier} 워커 등록 안내",
            setup_msg,
        ) or sent

    # 텔레그램 발송
    if args.telegram:
        tg_msg = (
            f"<b>🔑 워커 등록 안내</b>\n\n"
            f"<b>이름:</b> {name}\n"
            f"<b>티어:</b> {args.tier}\n"
            f"<b>워커 ID:</b> <code>{worker_id}</code>\n"
            f"<b>토큰:</b> <code>{token}</code>\n\n"
            f"<b>셋업:</b>\n"
            f"1. git clone → pip install\n"
            f"2. cp .env.{args.tier} .env\n"
            f"3. .env에 WORKER_TOKEN 입력\n"
            f"4. python -m scalp_ml.worker --worker-id \"{worker_id}\" --tier {args.tier}\n\n"
            f"⚠ 토큰을 안전하게 보관하세요."
        )
        sent = send_telegram(args.telegram, tg_msg) or sent

    # 콘솔 출력 (발송 실패 시 또는 항상)
    if not sent:
        print()
        print("=== 아래 내용을 직접 전달하세요 ===")
        print()
        print(setup_msg)
        print("=== 끝 ===")
    else:
        print()
        print("발송 완료. 콘솔에서 토큰을 다시 확인하려면:")
        print(f"  WORKER_TOKEN={token}")


def cmd_invites(args):
    """초대 현황 조회"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/worker_invites",
        params={"select": "*", "order": "created_at.desc"},
        headers=HEADERS,
        timeout=15,
    )
    invites = resp.json() if resp.ok else []

    if not invites:
        print("발급된 초대가 없습니다")
        return

    now = datetime.now(KST)
    print(f"\n{'코드':<10} {'티어':<14} {'이름':<16} {'이메일':<24} {'상태':<10} {'만료'}")
    print("-" * 90)
    for inv in invites:
        exp = datetime.fromisoformat(inv["expires_at"])
        if inv.get("used_at"):
            status = f"사용됨({inv.get('used_by_worker_id', '?')})"
        elif exp.astimezone(KST) < now:
            status = "만료"
        else:
            status = "유효"
        print(f"{inv['invite_code']:<10} {inv['tier']:<14} {(inv.get('worker_name') or '-'):<16} "
              f"{(inv.get('email') or '-'):<24} {status:<10} "
              f"{exp.strftime('%m-%d %H:%M')}")
    print(f"\n총 {len(invites)}건")


def cmd_revoke_invite(args):
    """초대코드 취소 (만료 처리)"""
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/worker_invites",
        params={"invite_code": f"eq.{args.code}"},
        json={"expires_at": datetime.now(KST).isoformat()},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"초대코드 취소됨: {args.code}")
    else:
        print(f"실패: {resp.text}")


def cmd_suspend(args):
    """워커 일시 정지 (RLS 토큰도 차단됨 — status=suspended 체크)"""
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        json={"status": "suspended",
              "notes": f"정지: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"워커 정지: {args.id} (RLS 접근도 차단됨)")
    else:
        print(f"실패: {resp.text}")


def cmd_unsuspend(args):
    """워커 정지 해제"""
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        json={"status": "offline",
              "notes": f"정지 해제: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"정지 해제: {args.id}")
    else:
        print(f"실패: {resp.text}")


def cmd_remove(args):
    """워커 완전 삭제"""
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
        print(f"워커 삭제: {args.id}")
    else:
        print(f"실패: {resp.text}")


def cmd_invalidate(args):
    """워커 토큰 무효화 (재등록 필요 — 관리자는 새 토큰을 모름)"""
    # 랜덤 토큰으로 교체하여 기존 토큰 무효화
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        json={"api_token": str(uuid.uuid4()), "status": "suspended",
              "notes": f"토큰 무효화: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"토큰 무효화: {args.id}")
        print(f"  워커는 새 초대코드로 재등록해야 합니다.")
        print(f"  관리자는 새 토큰을 알 수 없습니다.")
    else:
        print(f"실패: {resp.text}")


def cmd_msg(args):
    """연락처에게 텔레그램 메시지 발송 (워커ID/이름/chat_id)"""
    contact = _lookup_contact_by_name(args.id)
    if contact and contact.get("chat_id"):
        chat_id = contact["chat_id"]
        name = contact.get("name") or args.id
    else:
        # chat_id 직접 입력 시도
        chat_id = args.id
        name = args.id

    if not chat_id:
        print(f"'{args.id}'의 텔레그램 ID를 찾을 수 없습니다.")
        return

    ok = send_telegram(chat_id, args.message)
    if ok:
        _save_message(chat_id=chat_id, direction="outgoing", message=args.message,
                      worker_id=contact.get("worker_id") if contact else None,
                      worker_name=name)


def cmd_msg_all(args):
    """전체 연락처에게 텔레그램 공지 발송 (워커 + 친구)"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id",
                "is_active": "eq.true"},
        headers=HEADERS,
        timeout=15,
    )
    contacts = resp.json() if resp.ok else []

    if not contacts:
        print("등록된 연락처가 없습니다.")
        return

    print(f"발송 대상: {len(contacts)}명")
    for c in contacts:
        print(f"  → {c['name']} [{c['role']}] ({c['chat_id']})")
        send_telegram(c['chat_id'], f"[공지] {args.message}")
        _save_message(chat_id=c['chat_id'], direction="outgoing",
                      message=f"[공지] {args.message}",
                      worker_id=c.get("worker_id"), worker_name=c["name"])
    print(f"\n발송 완료")


def _lookup_contact_by_chat_id(chat_id: str) -> dict | None:
    """chat_id로 연락처 조회 (워커 + 친구 통합)"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,email",
                "chat_id": f"eq.{chat_id}", "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp.ok and resp.json():
        c = resp.json()[0]
        return {"worker_id": c.get("worker_id") or c["chat_id"],
                "worker_name": c["name"], "tier": c["role"]}
    # fallback: compute_workers
    resp2 = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id,worker_name,tier",
                "telegram_chat_id": f"eq.{chat_id}"},
        headers=HEADERS, timeout=10,
    )
    if resp2.ok and resp2.json():
        return resp2.json()[0]
    return None


def _lookup_contact_by_name(name_or_id: str) -> dict | None:
    """이름/별명/워커ID로 연락처 조회 → chat_id 반환"""
    # 이름 또는 워커 ID로 시도
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,aliases",
                "or": f"(worker_id.eq.{name_or_id},name.eq.{name_or_id})",
                "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp.ok and resp.json():
        return resp.json()[0]

    # 별명 검색 (aliases array contains)
    resp2 = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,aliases",
                "aliases": f"cs.{{{name_or_id}}}",
                "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp2.ok and resp2.json():
        return resp2.json()[0]

    # compute_workers fallback
    resp3 = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id,worker_name,telegram_chat_id",
                "worker_id": f"eq.{name_or_id}"},
        headers=HEADERS, timeout=10,
    )
    if resp3.ok and resp3.json():
        w = resp3.json()[0]
        return {"chat_id": w.get("telegram_chat_id"), "name": w.get("worker_name"),
                "role": "worker", "worker_id": w["worker_id"]}
    return None


def cmd_contacts(args):
    """전체 연락처 목록 (워커 + 친구)"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,email,aliases,is_active",
                "order": "role.asc,name.asc"},
        headers=HEADERS, timeout=15,
    )
    contacts = resp.json() if resp.ok else []

    if not contacts:
        print("등록된 연락처가 없습니다")
        return

    print(f"\n{'이름':<14} {'별명':<14} {'역할':<12} {'chat_id':<14} {'이메일':<22} {'상태'}")
    print("-" * 90)
    for c in contacts:
        status = "활성" if c.get("is_active") else "비활성"
        aliases = ", ".join(c.get("aliases") or []) or "-"
        print(f"{c['name']:<14} {aliases:<14} {c['role']:<12} {c['chat_id']:<14} "
              f"{(c.get('email') or '-'):<22} {status}")
    print(f"\n총 {len(contacts)}명")


def cmd_contact_add(args):
    """연락처 추가 (친구/외부인)"""
    role = args.role or "friend"
    if role not in ("owner", "coworker", "collaborator", "viewer", "friend"):
        print(f"유효하지 않은 역할: {role}")
        return

    aliases = [a.strip() for a in args.aliases.split(",")] if args.aliases else []

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        json={
            "chat_id": args.chat_id,
            "name": args.name,
            "role": role,
            "aliases": aliases,
            "email": args.email or None,
            "notes": args.notes or None,
        },
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"연락처 추가: {args.name} ({role}) chat_id={args.chat_id}")
        # 등록 인사 메시지
        send_telegram(args.chat_id,
            f"안녕하세요 {args.name}님! CoinTrading 봇에 연락처로 등록되었습니다.\n"
            f"이 채팅으로 관리자와 메시지를 주고받을 수 있습니다.")
    else:
        print(f"실패: {resp.text}")


def cmd_contact_remove(args):
    """연락처 비활성화"""
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"or": f"(chat_id.eq.{args.target},name.eq.{args.target})"},
        json={"is_active": False},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"연락처 비활성화: {args.target}")
    else:
        print(f"실패: {resp.text}")


def cmd_sync_contacts(args):
    """compute_workers의 텔레그램 ID를 연락처에 동기화"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id,worker_name,tier,telegram_chat_id,email",
                "telegram_chat_id": "not.is.null"},
        headers=HEADERS, timeout=15,
    )
    workers = resp.json() if resp.ok else []

    synced = 0
    for w in workers:
        # upsert (chat_id unique)
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/telegram_contacts",
            json={
                "chat_id": w["telegram_chat_id"],
                "name": w.get("worker_name") or w["worker_id"],
                "role": w["tier"],
                "worker_id": w["worker_id"],
                "email": w.get("email"),
            },
            headers={**HEADERS, "Prefer": "return=minimal,resolution=merge-duplicates"},
            timeout=15,
        )
        if r.status_code < 300:
            synced += 1
    print(f"워커 → 연락처 동기화: {synced}명")


def _save_message(chat_id: str, direction: str, message: str,
                  worker_id: str = None, worker_name: str = None,
                  tg_msg_id: int = None, tg_username: str = None,
                  tg_first_name: str = None):
    """메시지를 DB에 저장"""
    requests.post(
        f"{SUPABASE_URL}/rest/v1/telegram_messages",
        json={
            "chat_id": chat_id,
            "direction": direction,
            "message": message,
            "worker_id": worker_id,
            "worker_name": worker_name,
            "telegram_message_id": tg_msg_id,
            "telegram_username": tg_username,
            "telegram_first_name": tg_first_name,
        },
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=10,
    )


def cmd_listen(args):
    """텔레그램 메시지 수신 대기 (폴링) — Ctrl+C로 종료"""
    import time

    if not TG_TOKEN:
        print("TELEGRAM_BOT_TOKEN 미설정")
        return

    interval = args.interval or 5
    offset = 0

    print(f"텔레그램 수신 대기 중... (폴링 {interval}초 간격, Ctrl+C 종료)")
    print("-" * 60)

    try:
        while True:
            try:
                resp = requests.get(
                    f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                    params={"offset": offset, "timeout": interval,
                            "allowed_updates": '["message"]'},
                    timeout=interval + 10,
                )
                if not resp.ok:
                    time.sleep(interval)
                    continue

                updates = resp.json().get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if not msg or not msg.get("text"):
                        continue

                    chat_id = str(msg["chat"]["id"])
                    text = msg["text"]
                    tg_username = msg["chat"].get("username", "")
                    tg_first = msg["chat"].get("first_name", "")
                    tg_msg_id = msg.get("message_id")

                    # 연락처 매핑
                    contact = _lookup_contact_by_chat_id(chat_id)
                    w_id = contact["worker_id"] if contact else None
                    w_name = contact["worker_name"] if contact else tg_first

                    # DB 저장
                    _save_message(
                        chat_id=chat_id, direction="incoming", message=text,
                        worker_id=w_id, worker_name=w_name,
                        tg_msg_id=tg_msg_id, tg_username=tg_username,
                        tg_first_name=tg_first,
                    )

                    # 콘솔 출력
                    sender = f"{w_name or tg_first}" + (f" (@{tg_username})" if tg_username else "")
                    role = f" [{contact['tier']}]" if contact else " [미등록]"
                    ts = datetime.now(KST).strftime("%H:%M:%S")
                    print(f"[{ts}] {sender}{role}: {text}")

                    # 미등록 유저에게 안내 + chat_id 알려주기
                    if not contact:
                        send_telegram(chat_id,
                            f"안녕하세요! 아직 등록되지 않은 사용자입니다.\n"
                            f"관리자에게 아래 chat_id를 알려주세요:\n"
                            f"  chat_id: {chat_id}")

            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.ConnectionError:
                print("연결 오류, 재시도...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n수신 대기 종료")


def cmd_inbox(args):
    """수신 메시지 목록 (읽지 않은 것 먼저)"""
    limit = args.limit or 20

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_messages",
        params={
            "select": "id,chat_id,worker_id,worker_name,telegram_username,message,is_read,created_at",
            "direction": "eq.incoming",
            "order": "is_read.asc,created_at.desc",
            "limit": str(limit),
        },
        headers=HEADERS, timeout=15,
    )
    messages = resp.json() if resp.ok else []

    if not messages:
        print("수신된 메시지가 없습니다")
        return

    unread = sum(1 for m in messages if not m.get("is_read"))
    print(f"\n수신함 (읽지 않은 메시지: {unread}건)\n")
    print(f"{'상태':<6} {'시간':<18} {'발신자':<20} {'메시지'}")
    print("-" * 80)

    for m in messages:
        status = "● NEW" if not m.get("is_read") else "  "
        ts = datetime.fromisoformat(m["created_at"]).astimezone(KST).strftime("%m-%d %H:%M")
        sender = m.get("worker_name") or m.get("telegram_username") or m["chat_id"]
        text = m["message"][:50] + ("..." if len(m["message"]) > 50 else "")
        print(f"{status:<6} {ts:<18} {sender:<20} {text}")

    # 읽음 처리
    unread_ids = [m["id"] for m in messages if not m.get("is_read")]
    if unread_ids:
        for uid in unread_ids:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/telegram_messages",
                params={"id": f"eq.{uid}"},
                json={"is_read": True},
                headers={**HEADERS, "Prefer": "return=minimal"},
                timeout=10,
            )
        print(f"\n{len(unread_ids)}건 읽음 처리 완료")


def cmd_reply(args):
    """답장 (이름/워커ID/chat_id 모두 가능)"""
    contact = _lookup_contact_by_name(args.to)
    if contact and contact.get("chat_id"):
        chat_id = contact["chat_id"]
        name = contact.get("name") or args.to
        w_id = contact.get("worker_id")
    else:
        chat_id = args.to  # chat_id 직접 입력
        name = args.to
        w_id = None

    ok = send_telegram(chat_id, args.message)
    if ok:
        _save_message(chat_id=chat_id, direction="outgoing", message=args.message,
                      worker_id=w_id, worker_name=name)


def cmd_chat(args):
    """실시간 대화 (이름/워커ID/chat_id로 대상 지정)"""
    import time
    import threading

    if not TG_TOKEN:
        print("TELEGRAM_BOT_TOKEN 미설정")
        return

    target_chat_id = getattr(args, 'chat_id', None)
    contact = None
    name = None

    if args.id:
        contact = _lookup_contact_by_name(args.id)
        if contact and contact.get("chat_id"):
            target_chat_id = contact["chat_id"]
            name = contact.get("name") or args.id

    if not target_chat_id:
        print(f"'{args.id or ''}'의 텔레그램 ID를 찾을 수 없습니다. --chat-id로 직접 지정하세요.")
        return

    if not name:
        name = target_chat_id
    print(f"💬 {name}과(와) 실시간 대화 (빈 줄 입력=종료)")
    print("-" * 50)

    # 수신 스레드
    offset = 0
    running = True

    # 초기 offset 설정 (기존 메시지 건너뛰기)
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params={"offset": -1, "timeout": 0},
            timeout=10,
        )
        if r.ok and r.json().get("result"):
            offset = r.json()["result"][-1]["update_id"] + 1
    except Exception:
        pass

    def receiver():
        nonlocal offset, running
        while running:
            try:
                r = requests.get(
                    f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                    params={"offset": offset, "timeout": 3,
                            "allowed_updates": '["message"]'},
                    timeout=8,
                )
                if not r.ok:
                    continue
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if not msg or not msg.get("text"):
                        continue
                    cid = str(msg["chat"]["id"])
                    if cid != target_chat_id:
                        continue
                    ts = datetime.now(KST).strftime("%H:%M:%S")
                    print(f"\n  [{ts}] {name}: {msg['text']}")
                    print("  나> ", end="", flush=True)
                    _save_message(chat_id=cid, direction="incoming", message=msg["text"],
                                  worker_id=contact.get("worker_id") if contact else None,
                                  worker_name=name, tg_msg_id=msg.get("message_id"))
            except Exception:
                time.sleep(2)

    t = threading.Thread(target=receiver, daemon=True)
    t.start()

    # 송신 루프
    try:
        while True:
            text = input("  나> ").strip()
            if not text:
                break
            send_telegram(target_chat_id, text)
            _save_message(chat_id=target_chat_id, direction="outgoing", message=text,
                          worker_id=contact.get("worker_id") if contact else None,
                          worker_name=name)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        running = False
        print("\n대화 종료")


def cmd_promote(args):
    """워커 티어 변경"""
    if args.tier not in ("owner", "coworker", "collaborator", "viewer"):
        print(f"유효하지 않은 티어: {args.tier}")
        return

    is_main = args.tier == "owner"
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"worker_id": f"eq.{args.id}"},
        json={"tier": args.tier, "is_main_brain": is_main,
              "notes": f"티어 변경 {args.tier}: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')}"},
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=15,
    )
    if resp.status_code < 300:
        print(f"티어 변경: {args.id} -> {args.tier}")
    else:
        print(f"실패: {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="워커 관리 CLI (관리자 전용)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="전체 워커 목록")

    p_invite = sub.add_parser("invite", help="초대코드 생성")
    p_invite.add_argument("--tier", required=True, help="티어 (coworker/collaborator)")
    p_invite.add_argument("--name", help="참여자 이름")
    p_invite.add_argument("--email", help="참여자 이메일")
    p_invite.add_argument("--hours", type=int, default=48, help="만료 시간 (기본 48시간)")

    p_direct = sub.add_parser("direct-send", help="직접 등록 + 토큰 발송")
    p_direct.add_argument("--tier", required=True, help="티어 (coworker/collaborator)")
    p_direct.add_argument("--name", help="참여자 이름")
    p_direct.add_argument("--email", help="이메일 주소 (SMTP 설정 필요)")
    p_direct.add_argument("--telegram", help="텔레그램 chat_id")
    p_direct.add_argument("--worker-id", help="워커 ID (미지정 시 자동 생성)")

    sub.add_parser("invites", help="초대 현황 조회")

    p_revoke = sub.add_parser("revoke-invite", help="초대코드 취소")
    p_revoke.add_argument("--code", required=True, help="초대코드")

    p_suspend = sub.add_parser("suspend", help="워커 정지")
    p_suspend.add_argument("--id", required=True)

    p_unsuspend = sub.add_parser("unsuspend", help="정지 해제")
    p_unsuspend.add_argument("--id", required=True)

    p_remove = sub.add_parser("remove", help="워커 삭제")
    p_remove.add_argument("--id", required=True)

    p_invalidate = sub.add_parser("invalidate", help="토큰 무효화 (재등록 필요)")
    p_invalidate.add_argument("--id", required=True)

    p_msg = sub.add_parser("msg", help="워커에게 텔레그램 메시지")
    p_msg.add_argument("--id", required=True, help="워커 ID")
    p_msg.add_argument("--message", "-m", required=True, help="메시지 내용")

    p_msg_all = sub.add_parser("msg-all", help="전체 워커에게 공지")
    p_msg_all.add_argument("--message", "-m", required=True, help="공지 내용")

    sub.add_parser("contacts", help="전체 연락처 목록")

    p_cadd = sub.add_parser("contact-add", help="연락처 추가 (친구 등)")
    p_cadd.add_argument("--chat-id", required=True, help="텔레그램 chat_id")
    p_cadd.add_argument("--name", required=True, help="이름")
    p_cadd.add_argument("--role", default="friend", help="역할 (기본 friend)")
    p_cadd.add_argument("--aliases", help="별명 (쉼표 구분, 예: '목수,나무꾼')")
    p_cadd.add_argument("--email", help="이메일")
    p_cadd.add_argument("--notes", help="메모")

    p_crm = sub.add_parser("contact-remove", help="연락처 비활성화")
    p_crm.add_argument("--target", required=True, help="이름 또는 chat_id")

    sub.add_parser("sync-contacts", help="워커 → 연락처 동기화")

    p_listen = sub.add_parser("listen", help="텔레그램 수신 대기 (폴링)")
    p_listen.add_argument("--interval", type=int, default=5, help="폴링 간격 초 (기본 5)")

    p_inbox = sub.add_parser("inbox", help="수신 메시지 목록")
    p_inbox.add_argument("--limit", type=int, default=20, help="표시 개수 (기본 20)")

    p_reply = sub.add_parser("reply", help="메시지 답장")
    p_reply.add_argument("--to", required=True, help="워커 ID 또는 chat_id")
    p_reply.add_argument("--message", "-m", required=True, help="답장 내용")

    p_chat = sub.add_parser("chat", help="워커와 실시간 대화")
    p_chat.add_argument("--id", help="워커 ID")
    p_chat.add_argument("--chat-id", help="텔레그램 chat_id (직접 지정)")

    p_promote = sub.add_parser("promote", help="티어 변경")
    p_promote.add_argument("--id", required=True)
    p_promote.add_argument("--tier", required=True)

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 미설정")
        sys.exit(1)

    cmds = {
        "list": cmd_list,
        "invite": cmd_invite,
        "direct-send": cmd_direct_send,
        "invites": cmd_invites,
        "revoke-invite": cmd_revoke_invite,
        "suspend": cmd_suspend,
        "unsuspend": cmd_unsuspend,
        "remove": cmd_remove,
        "invalidate": cmd_invalidate,
        "contacts": cmd_contacts,
        "contact-add": cmd_contact_add,
        "contact-remove": cmd_contact_remove,
        "sync-contacts": cmd_sync_contacts,
        "msg": cmd_msg,
        "msg-all": cmd_msg_all,
        "listen": cmd_listen,
        "inbox": cmd_inbox,
        "reply": cmd_reply,
        "chat": cmd_chat,
        "promote": cmd_promote,
    }

    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
