#!/usr/bin/env python3
"""
Upbit 매매 실행 스크립트

안전장치:
  - EMERGENCY_STOP=true → 모든 매매 차단
  - DRY_RUN=true → 분석만 수행, 실제 주문 미실행
  - MAX_TRADE_AMOUNT → 1회 매매 금액 상한

사용법:
  python3 scripts/execute_trade.py bid KRW-BTC 100000   # 시장가 매수 (10만원)
  python3 scripts/execute_trade.py ask KRW-BTC 0.001    # 시장가 매도 (0.001 BTC)

출력: JSON (stdout)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
import jwt
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

UPBIT_API = "https://api.upbit.com/v1"
PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCK_FILE = PROJECT_DIR / "data" / "trading.lock"
KST = timezone(timedelta(hours=9))


LOCK_TIMEOUT_SECONDS = 120  # 2분 이상 된 락은 stale로 판단


def acquire_lock():
    """정규 매매 실행 중 락파일 생성. stale lock 자동 해제."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    # stale lock 검사: 타임아웃 초과 또는 프로세스 사망 시 강제 해제
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
            lock_time = datetime.fromisoformat(data["timestamp"])
            age = (datetime.now(KST) - lock_time).total_seconds()
            lock_pid = data.get("pid", 0)
            # 프로세스 생존 확인 (pid=0이면 검사 생략)
            pid_alive = False
            if lock_pid > 0:
                try:
                    os.kill(lock_pid, 0)
                    pid_alive = True
                except (OSError, ProcessLookupError):
                    pid_alive = False
            if age > LOCK_TIMEOUT_SECONDS or not pid_alive:
                print(f"[lock] stale lock 제거 (age={age:.0f}s, pid={lock_pid}, alive={pid_alive})", file=sys.stderr)
                LOCK_FILE.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"다른 매매 프로세스 실행 중 (pid={lock_pid}, age={age:.0f}s)")
        except (json.JSONDecodeError, KeyError):
            LOCK_FILE.unlink(missing_ok=True)

    LOCK_FILE.write_text(json.dumps({
        "process": "execute_trade",
        "pid": os.getpid(),
        "timestamp": datetime.now(KST).isoformat(),
    }))


def release_lock():
    """락파일 해제"""
    try:
        if LOCK_FILE.exists():
            data = json.loads(LOCK_FILE.read_text())
            if data.get("pid") == os.getpid():
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        LOCK_FILE.unlink(missing_ok=True)


def make_auth_header(query_string: str) -> dict:
    payload = {
        "access_key": os.environ["UPBIT_ACCESS_KEY"],
        "nonce": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "query_hash": hashlib.sha512(query_string.encode()).hexdigest(),
        "query_hash_alg": "SHA512",
    }
    token = jwt.encode(payload, os.environ["UPBIT_SECRET_KEY"], algorithm="HS256")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def execute(side: str, market: str, amount: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")

    # 1) 긴급 정지 확인
    if os.environ.get("EMERGENCY_STOP", "false").lower() == "true":
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": "EMERGENCY_STOP 활성화 - 매매 차단",
            "timestamp": ts,
        }

    # 2) DRY_RUN 확인
    if os.environ.get("DRY_RUN", "true").lower() == "true":
        return {
            "success": True,
            "dry_run": True,
            "side": side,
            "market": market,
            "amount": amount,
            "timestamp": ts,
        }

    # 3) 매수 금액 상한 확인
    max_amount = int(os.environ.get("MAX_TRADE_AMOUNT", "100000"))
    if side == "bid" and int(float(amount)) > max_amount:
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": f"매매 금액 상한 초과: {amount} > {max_amount}",
            "timestamp": ts,
        }

    # 4) 주문 실행 (락파일로 단타 봇과 동시 실행 방지)
    try:
        acquire_lock()
    except RuntimeError as e:
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": str(e),
            "timestamp": ts,
        }

    try:
        body = {"market": market, "side": side}
        if side == "bid":
            body["ord_type"] = "price"  # 시장가 매수
            body["price"] = amount
        else:
            body["ord_type"] = "market"  # 시장가 매도
            body["volume"] = amount

        qs = urlencode(body)
        headers = make_auth_header(qs)

        r = requests.post(f"{UPBIT_API}/orders", json=body, headers=headers, timeout=10)
        try:
            response = r.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            response = {"raw_response": r.text[:500]}

        # 잔고 부족, 최소 주문 금액 등 API 에러 상세 처리
        error_msg = None
        if not r.ok:
            err_name = response.get("error", {}).get("name", "")
            err_msg = response.get("error", {}).get("message", "")
            error_msg = f"{err_name}: {err_msg}" if err_name else json.dumps(response, ensure_ascii=False)

        return {
            "success": r.ok,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "response": response,
            "error": error_msg,
            "timestamp": ts,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": f"주문 요청 실패: {e}",
            "timestamp": ts,
        }
    finally:
        release_lock()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "사용법: python3 execute_trade.py [bid|ask] [market] [amount]",
            file=sys.stderr,
        )
        sys.exit(1)

    result = execute(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result["success"]:
        sys.exit(1)
