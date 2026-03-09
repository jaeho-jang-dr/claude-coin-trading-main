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


def acquire_lock(timeout=15):
    """정규 매매 실행 중 원자적 락파일 생성"""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    while True:
        try:
            with LOCK_FILE.open('x', encoding='utf-8') as f:
                f.write(json.dumps({
                    "process": "execute_trade",
                    "pid": os.getpid(),
                    "timestamp": datetime.now(KST).isoformat(),
                }))
            break
        except FileExistsError:
            # 락이 너무 오래된 경우 강제 해제 (좀비 프로세스 방지)
            try:
                if LOCK_FILE.exists():
                    data = json.loads(LOCK_FILE.read_text())
                    lock_time = datetime.fromisoformat(data['timestamp'])
                    if datetime.now(KST) - lock_time > timedelta(minutes=5):
                        LOCK_FILE.unlink(missing_ok=True)
                        continue
            except Exception:
                pass
            
            if time.time() - start_time > timeout:
                raise TimeoutError("락을 획득하지 못했습니다. 다른 매매가 진행 중일 수 있습니다.")
            time.sleep(0.5)


def release_lock():
    """락파일 해제"""
    try:
        # 안전한 해제: 현재 pid가 일치할 때만 해제 (필요에 따라 강제 해제 허용)
        if LOCK_FILE.exists():
            data = json.loads(LOCK_FILE.read_text())
            if data.get("pid") == os.getpid():
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def check_open_orders_and_cancel(market: str, side: str):
    """현재 진행 중인 미체결 주문(수정주문 lock)을 확인하고, 동일 방향이면 취소한다."""
    try:
        qs = urlencode({"market": market, "state": "wait"})
        headers = make_auth_header(qs)
        r = requests.get(f"{UPBIT_API}/orders", params={"market": market, "state": "wait"}, headers=headers, timeout=10)
        if not r.ok:
            return
        
        open_orders = r.json()
        for order in open_orders:
            # 같은 방향이거나 양방향 안전 확보를 위해 기존 주문 정리
            if order.get("side") == side:
                uuid_to_cancel = order.get("uuid")
                cancel_qs = urlencode({"uuid": uuid_to_cancel})
                cancel_headers = make_auth_header(cancel_qs)
                requests.delete(f"{UPBIT_API}/order", params={"uuid": uuid_to_cancel}, headers=cancel_headers, timeout=10)
                time.sleep(0.2)
    except Exception as e:
        print(f"[warning] 미체결 주문 정리 중 오류: {e}", file=sys.stderr)



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

    # 1a) 사용자 긴급 정지 확인
    if os.environ.get("EMERGENCY_STOP", "false").lower() == "true":
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": "사용자 EMERGENCY_STOP 활성화 - 매매 차단",
            "timestamp": ts,
        }

    # 1b) 감독 자동 긴급정지 확인 (긴급정지 중 매도는 허용)
    auto_em_file = PROJECT_DIR / "data" / "auto_emergency.json"
    if auto_em_file.exists() and side == "bid":  # 매수만 차단, 매도(청산)는 허용
        try:
            import json as _json
            auto_em = _json.loads(auto_em_file.read_text(encoding="utf-8"))
            if auto_em.get("active"):
                return {
                    "success": False,
                    "dry_run": False,
                    "side": side,
                    "market": market,
                    "amount": amount,
                    "error": f"감독 자동긴급정지 활성 - 매수 차단 (사유: {auto_em.get('reason', '?')})",
                    "timestamp": ts,
                }
        except Exception:
            pass

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

    # 4) 주문 실행 (원자적 락파일로 동시 실행 차단)
    try:
        acquire_lock()
    except TimeoutError as e:
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
        # 5) 수정주문 / 미체결 주문 처리 (수정주문 lock 관리)
        check_open_orders_and_cancel(market, side)
        
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
        response = r.json()

        return {
            "success": r.ok,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "response": response,
            "error": None if r.ok else json.dumps(response, ensure_ascii=False),
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
