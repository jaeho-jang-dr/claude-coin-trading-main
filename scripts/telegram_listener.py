#!/usr/bin/env python3
"""
텔레그램 수신 데몬 — 백그라운드 상시 실행
봇으로 들어오는 메시지를 DB에 저장하고, 등록된 사용자 간 메시지 중계

봇 명령어:
  /start              — 봇 소개 + chat_id 표시
  /chat_id            — 내 chat_id 확인
  /list               — 연락처 목록
  /msg 이름 메시지     — 특정 사용자에게 메시지 전송
  /all 메시지          — 전체 공지

실행: python scripts/telegram_listener.py
중지: Ctrl+C 또는 taskkill
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Windows cp949 대응 + unbuffered
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

KST = timezone(timedelta(hours=9))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

POLL_INTERVAL = 3
CLEANUP_INTERVAL = 3600
_last_cleanup = 0
_active_chats: dict[str, "subprocess.Popen"] = {}  # chat_id → 터미널 프로세스


# ── 연락처 조회 ──────────────────────────────────

def lookup_by_chat_id(chat_id: str) -> dict | None:
    """chat_id로 연락처 조회"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,aliases",
                "chat_id": f"eq.{chat_id}", "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp.ok and resp.json():
        return resp.json()[0]
    return None


def lookup_by_name(name: str) -> dict | None:
    """이름 또는 별명으로 연락처 조회"""
    # 이름 exact match
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,aliases",
                "name": f"eq.{name}", "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp.ok and resp.json():
        return resp.json()[0]

    # worker_id match
    resp2 = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,aliases",
                "worker_id": f"eq.{name}", "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp2.ok and resp2.json():
        return resp2.json()[0]

    # 별명 검색 (aliases array contains)
    resp3 = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,worker_id,aliases",
                "aliases": f"cs.{{{name}}}",
                "is_active": "eq.true"},
        headers=HEADERS, timeout=10,
    )
    if resp3.ok and resp3.json():
        return resp3.json()[0]

    return None


def get_all_contacts() -> list:
    """활성 연락처 전체 조회"""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "chat_id,name,role,aliases",
                "is_active": "eq.true",
                "order": "role.asc,name.asc"},
        headers=HEADERS, timeout=10,
    )
    return resp.json() if resp.ok else []


# ── 메시지 저장/발송 ──────────────────────────────

def save_message(chat_id, direction, message, **kwargs):
    """메시지 DB 저장"""
    data = {"chat_id": chat_id, "direction": direction, "message": message}
    data.update({k: v for k, v in kwargs.items() if v is not None})
    requests.post(
        f"{SUPABASE_URL}/rest/v1/telegram_messages",
        json=data,
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=10,
    )


def send_telegram(chat_id: str, text: str) -> bool:
    """텔레그램 메시지 발송"""
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=15,
    )
    return r.ok


# ── 봇 명령어 처리 ──────────────────────────────

def handle_command(chat_id: str, text: str, sender: dict | None):
    """봇 명령어 처리 — 등록된 사용자만 사용 가능"""
    cmd = text.strip()
    cmd_lower = cmd.lower()

    # /start
    if cmd_lower == "/start":
        if sender:
            send_telegram(chat_id,
                f"안녕하세요 {sender['name']}님! [{sender['role']}]\n\n"
                f"사용 가능한 명령어:\n"
                f"  /list — 연락처 목록\n"
                f"  /msg 이름 메시지 — 메시지 보내기\n"
                f"  /all 메시지 — 전체 공지\n"
                f"  /chat_id — 내 chat_id 확인")
        else:
            send_telegram(chat_id,
                f"CoinTrading 봇입니다.\n"
                f"당신의 chat_id: {chat_id}\n"
                f"관리자에게 이 번호를 알려주면 등록해드립니다.")
        return True

    # /chat_id
    if cmd_lower == "/chat_id":
        send_telegram(chat_id, f"chat_id: {chat_id}")
        return True

    # 미등록 유저는 여기서 차단
    if not sender:
        return False

    sender_name = sender["name"]

    # /list — 연락처 목록
    if cmd_lower == "/list":
        contacts = get_all_contacts()
        if not contacts:
            send_telegram(chat_id, "등록된 연락처가 없습니다.")
            return True
        lines = ["[연락처 목록]"]
        for c in contacts:
            aliases = c.get("aliases") or []
            alias_str = f" ({', '.join(aliases)})" if aliases else ""
            lines.append(f"  {c['name']}{alias_str} [{c['role']}]")
        send_telegram(chat_id, "\n".join(lines))
        return True

    # /msg 이름 메시지 — 특정 사용자에게 전송
    if cmd_lower.startswith("/msg "):
        parts = cmd[5:].strip().split(None, 1)
        if len(parts) < 2:
            send_telegram(chat_id, "사용법: /msg 이름 메시지내용")
            return True

        target_name, msg_text = parts
        target = lookup_by_name(target_name)
        if not target:
            send_telegram(chat_id, f"'{target_name}'을(를) 찾을 수 없습니다.\n/list 로 목록을 확인하세요.")
            return True

        if target["chat_id"] == chat_id:
            send_telegram(chat_id, "자신에게는 메시지를 보낼 수 없습니다.")
            return True

        # 수신자에게 전달
        forward_text = f"[{sender_name}] {msg_text}"
        ok = send_telegram(target["chat_id"], forward_text)

        if ok:
            send_telegram(chat_id, f"✓ {target['name']}에게 전송 완료")
            # 발신 기록
            save_message(chat_id=target["chat_id"], direction="incoming",
                         message=forward_text, worker_name=sender_name)
            save_message(chat_id=chat_id, direction="outgoing",
                         message=f"→ {target['name']}: {msg_text}",
                         worker_name=sender_name)
        else:
            send_telegram(chat_id, f"전송 실패. {target['name']}의 텔레그램 연결을 확인하세요.")
        return True

    # /all 메시지 — 전체 공지
    if cmd_lower.startswith("/all "):
        msg_text = cmd[5:].strip()
        if not msg_text:
            send_telegram(chat_id, "사용법: /all 공지내용")
            return True

        contacts = get_all_contacts()
        sent = 0
        for c in contacts:
            if c["chat_id"] == chat_id:
                continue  # 자신 제외
            send_telegram(c["chat_id"], f"[공지 from {sender_name}] {msg_text}")
            save_message(chat_id=c["chat_id"], direction="incoming",
                         message=f"[공지 from {sender_name}] {msg_text}",
                         worker_name=sender_name)
            sent += 1

        send_telegram(chat_id, f"✓ {sent}명에게 공지 발송 완료")
        return True

    return False


# ── 일반 메시지 처리 (명령어가 아닌 경우) ──────────

def handle_plain_message(chat_id: str, text: str, sender: dict | None):
    """일반 메시지 — 이름>메시지 형식이면 전달, 아니면 그냥 기록"""
    if not sender:
        send_telegram(chat_id,
            f"아직 등록되지 않은 사용자입니다.\n"
            f"관리자에게 chat_id를 알려주세요: {chat_id}")
        return

    # "이름>메시지" 또는 "이름: 메시지" 형식 감지
    for sep in [">", ":"]:
        if sep in text:
            target_name, msg_text = text.split(sep, 1)
            target_name = target_name.strip()
            msg_text = msg_text.strip()
            if target_name and msg_text:
                target = lookup_by_name(target_name)
                if target and target["chat_id"] != chat_id:
                    forward_text = f"[{sender['name']}] {msg_text}"
                    ok = send_telegram(target["chat_id"], forward_text)
                    if ok:
                        send_telegram(chat_id, f"✓ {target['name']}에게 전송")
                        save_message(chat_id=target["chat_id"], direction="incoming",
                                     message=forward_text, worker_name=sender["name"])
                        save_message(chat_id=chat_id, direction="outgoing",
                                     message=f"→ {target['name']}: {msg_text}",
                                     worker_name=sender["name"])
                    return

    # 매칭 안 되면 기록만 (관리자가 inbox에서 확인)


# ── 채팅 터미널 자동 실행 ──────────────────────

def open_chat_terminal(chat_id: str, name: str, contact: dict):
    """등록된 사용자가 메시지를 보내면 전용 채팅 터미널을 자동 실행"""
    # 이미 열린 터미널이 살아있는지 확인
    if chat_id in _active_chats:
        proc = _active_chats[chat_id]
        if proc.poll() is None:
            return  # 아직 실행 중
        del _active_chats[chat_id]

    worker_id = contact.get("worker_id") or name
    title = f"Chat - {name} [{contact.get('role', '?')}]"

    try:
        if sys.platform == "win32":
            # Windows: 새 cmd 창으로 chat 실행
            cmd = (
                f'start "{title}" cmd /k '
                f'"cd /d {PROJECT_DIR} && python scripts/worker_admin.py chat '
                f'--chat-id {chat_id}"'
            )
            proc = _sp.Popen(cmd, shell=True, cwd=str(PROJECT_DIR))
        else:
            # Linux/Mac: 새 터미널
            proc = _sp.Popen(
                [sys.executable, "scripts/worker_admin.py", "chat",
                 "--chat-id", chat_id],
                cwd=str(PROJECT_DIR),
            )

        _active_chats[chat_id] = proc
        print(f"  💬 채팅 터미널 열림: {name} (chat_id={chat_id})", flush=True)
    except Exception as e:
        print(f"  채팅 터미널 실행 실패: {e}", flush=True)


# ── 메인 루프 ──────────────────────────────────

import subprocess as _sp

LOCK_FILE = PROJECT_DIR / "data" / "telegram_listener.lock"


def acquire_lock():
    """중복 실행 방지 — PID 기반 락 파일"""
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            # 프로세스가 살아있는지 확인
            import subprocess
            r = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}"],
                               capture_output=True, text=True, timeout=5)
            if str(old_pid) in r.stdout:
                print(f"이미 실행 중 (PID {old_pid}). 종료합니다.")
                sys.exit(0)
        except Exception:
            pass
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_lock():
    """락 파일 삭제"""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    if not TG_TOKEN:
        print("TELEGRAM_BOT_TOKEN 미설정")
        sys.exit(1)

    acquire_lock()

    print(f"[{datetime.now(KST).strftime('%H:%M:%S')}] 텔레그램 수신 데몬 시작 (PID {os.getpid()}, 폴링 {POLL_INTERVAL}초)")
    print(f"  봇 명령어: /start /list /msg /all /chat_id")
    print(f"  단축 전송: 이름>메시지  또는  이름: 메시지")
    print("-" * 50)

    # 연결 알림: 등록된 모든 연락처에게 온라인 알림
    try:
        contacts = get_all_contacts()
        ts = datetime.now(KST).strftime("%H:%M")
        for c in contacts:
            send_telegram(c["chat_id"],
                f"🟢 CoinTrading 봇 온라인 ({ts} KST)\n"
                f"명령어: /list /msg /all /chat_id")
        print(f"  연결 알림 발송: {len(contacts)}명", flush=True)
    except Exception:
        pass

    # 기존 메시지 건너뛰기
    offset = 0
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params={"offset": -1, "timeout": 0},
            timeout=10,
        )
        if r.ok and r.json().get("result"):
            offset = r.json()["result"][-1]["update_id"] + 1
            print(f"  기존 메시지 건너뜀 (offset={offset})")
    except Exception:
        pass

    def cleanup_old_messages():
        """7일 이전 메시지 삭제"""
        global _last_cleanup
        now = time.time()
        if now - _last_cleanup < CLEANUP_INTERVAL:
            return
        _last_cleanup = now
        cutoff = (datetime.now(KST) - timedelta(days=7)).isoformat()
        resp = requests.delete(
            f"{SUPABASE_URL}/rest/v1/telegram_messages",
            params={"created_at": f"lt.{cutoff}"},
            headers={**HEADERS, "Prefer": "return=minimal"},
            timeout=15,
        )
        if resp.status_code < 300:
            print(f"[{datetime.now(KST).strftime('%H:%M:%S')}] 7일 이전 메시지 정리 완료", flush=True)

    try:
        while True:
            cleanup_old_messages()
            try:
                resp = requests.get(
                    f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                    params={"offset": offset, "timeout": POLL_INTERVAL,
                            "allowed_updates": '["message"]'},
                    timeout=POLL_INTERVAL + 10,
                )
                if not resp.ok:
                    time.sleep(POLL_INTERVAL)
                    continue

                for upd in resp.json().get("result", []):
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
                    contact = lookup_by_chat_id(chat_id)

                    # DB 기록 (모든 메시지)
                    save_message(
                        chat_id=chat_id, direction="incoming", message=text,
                        worker_id=contact.get("worker_id") if contact else None,
                        worker_name=contact["name"] if contact else tg_first,
                        telegram_message_id=tg_msg_id,
                        telegram_username=tg_username,
                        telegram_first_name=tg_first,
                    )

                    # 콘솔 출력
                    ts = datetime.now(KST).strftime("%H:%M:%S")
                    name = contact["name"] if contact else tg_first
                    role = contact["role"] if contact else "미등록"
                    sender_str = f"{name}" + (f" (@{tg_username})" if tg_username else "")
                    print(f"[{ts}] {sender_str} [{role}]: {text}", flush=True)

                    # 명령어 처리
                    if text.startswith("/"):
                        handle_command(chat_id, text, contact)
                    else:
                        handle_plain_message(chat_id, text, contact)

                    # 등록된 사용자의 첫 메시지 → 전용 채팅 터미널 자동 실행
                    if contact and chat_id not in _active_chats:
                        open_chat_terminal(chat_id, name, contact)

            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.ConnectionError:
                print(f"[{datetime.now(KST).strftime('%H:%M:%S')}] 연결 오류, 5초 후 재시도...", flush=True)
                time.sleep(5)
            except Exception as e:
                print(f"[{datetime.now(KST).strftime('%H:%M:%S')}] 오류: {e}", flush=True)
                time.sleep(5)

    except KeyboardInterrupt:
        pass
    finally:
        # 오프라인 알림
        try:
            contacts = get_all_contacts()
            for c in contacts:
                send_telegram(c["chat_id"], "🔴 CoinTrading 봇 오프라인")
        except Exception:
            pass
        release_lock()
        print(f"\n[{datetime.now(KST).strftime('%H:%M:%S')}] 수신 데몬 종료")


if __name__ == "__main__":
    main()
