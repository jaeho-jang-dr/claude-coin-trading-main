#!/usr/bin/env python3
"""
Upbit 매매 실행 스크립트

안전장치:
  - EMERGENCY_STOP=true → 모든 매매 차단
  - DRY_RUN=true → 분석만 수행, 실제 주문 미실행
  - MAX_TRADE_AMOUNT → 1회 매매 금액 상한
  - MAX_DAILY_TRADES → 일일 매매 횟수 상한
  - MIN_TRADE_INTERVAL_HOURS → 최소 매매 간격
  - MAX_POSITION_RATIO → 총 자산 대비 최대 투자 비율

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


def acquire_lock(timeout=15):
    """정규 매매 실행 중 락파일 생성. stale lock 자동 해제 + 타임아웃."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    while True:
        # stale lock 검사: 타임아웃 초과 또는 프로세스 사망 시 강제 해제
        if LOCK_FILE.exists():
            try:
                data = json.loads(LOCK_FILE.read_text())
                lock_time = datetime.fromisoformat(data.get("timestamp") or data.get("time", ""))
                age = (datetime.now(KST) - lock_time).total_seconds()
                lock_pid = data.get("pid", 0)
                # 프로세스 생존 확인 (pid=0이면 검사 생략)
                pid_alive = False
                if lock_pid > 0:
                    try:
                        if os.name == "nt":
                            import ctypes
                            kernel32 = ctypes.windll.kernel32
                            handle = kernel32.OpenProcess(0x100000, False, lock_pid)  # SYNCHRONIZE
                            if handle:
                                kernel32.CloseHandle(handle)
                                pid_alive = True
                            else:
                                pid_alive = False
                        else:
                            os.kill(lock_pid, 0)
                            pid_alive = True
                    except (OSError, ProcessLookupError):
                        pid_alive = False
                if age > LOCK_TIMEOUT_SECONDS or not pid_alive:
                    print(f"[lock] stale lock 제거 (age={age:.0f}s, pid={lock_pid}, alive={pid_alive})", file=sys.stderr)
                    LOCK_FILE.unlink(missing_ok=True)
                else:
                    # Lock is valid and held by a live process -- wait or timeout
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"락을 획득하지 못했습니다. 다른 매매 프로세스 실행 중 (pid={lock_pid}, age={age:.0f}s)")
                    time.sleep(0.5)
                    continue
            except (json.JSONDecodeError, KeyError):
                LOCK_FILE.unlink(missing_ok=True)

        # 원자적 락 생성 시도
        try:
            with LOCK_FILE.open('x', encoding='utf-8') as f:
                f.write(json.dumps({
                    "process": "execute_trade",
                    "pid": os.getpid(),
                    "timestamp": datetime.now(KST).isoformat(),
                }))
            break
        except FileExistsError:
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
        LOCK_FILE.unlink(missing_ok=True)


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
    access_key = os.environ.get("UPBIT_ACCESS_KEY", "")
    secret_key = os.environ.get("UPBIT_SECRET_KEY", "")
    if not access_key or not secret_key:
        raise ValueError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY 환경변수가 설정되지 않았습니다")
    payload = {
        "access_key": access_key,
        "nonce": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "query_hash": hashlib.sha512(query_string.encode()).hexdigest(),
        "query_hash_alg": "SHA512",
    }
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


DAILY_TRADES_FILE = PROJECT_DIR / "data" / "daily_trades.json"
LAST_TRADE_TIME_FILE = PROJECT_DIR / "data" / "last_trade_time.txt"


def _get_daily_trades() -> dict:
    """오늘의 매매 횟수를 조회한다."""
    today = datetime.now(KST).strftime("%Y-%m-%d")
    try:
        if DAILY_TRADES_FILE.exists():
            data = json.loads(DAILY_TRADES_FILE.read_text(encoding="utf-8"))
            if data.get("date") == today:
                return data
    except (json.JSONDecodeError, KeyError):
        pass
    return {"date": today, "count": 0}


def _increment_daily_trades():
    """오늘의 매매 횟수를 1 증가시킨다."""
    data = _get_daily_trades()
    data["count"] = data.get("count", 0) + 1
    DAILY_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAILY_TRADES_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _update_last_trade_time():
    """마지막 매매 시각을 기록한다."""
    LAST_TRADE_TIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_TRADE_TIME_FILE.write_text(datetime.now(KST).isoformat(), encoding="utf-8")


def _get_last_trade_time():
    """마지막 매매 시각을 조회한다. 없으면 None."""
    try:
        if LAST_TRADE_TIME_FILE.exists():
            text = LAST_TRADE_TIME_FILE.read_text(encoding="utf-8").strip()
            return datetime.fromisoformat(text)
    except (ValueError, OSError):
        pass
    return None


def _get_btc_position_ratio() -> float | None:
    """현재 BTC 포지션 비율(0~1)을 조회한다. 실패 시 None."""
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from scripts.get_portfolio import main as _portfolio_main
        # get_portfolio.main()은 stdout에 JSON을 출력하므로, 직접 API 호출로 대체
        import io
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            _portfolio_main()
        finally:
            sys.stdout = old_stdout
        portfolio = json.loads(buf.getvalue())
        total_eval = portfolio.get("total_eval", 0)
        if total_eval <= 0:
            return 0.0
        btc_eval = 0.0
        for h in portfolio.get("holdings", []):
            if h.get("currency") == "BTC":
                btc_eval = h.get("eval_amount", 0)
                break
        return btc_eval / total_eval
    except Exception as e:
        print(f"[warning] BTC 포지션 비율 조회 실패: {e}", file=sys.stderr)
        return None


def execute(side: str, market: str, amount: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")

    # 0) side 유효성 검사 (ET-07)
    if side not in ("bid", "ask"):
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": f"유효하지 않은 side: '{side}'. 'bid' 또는 'ask'만 허용됩니다.",
            "timestamp": ts,
        }

    # 1a) 사용자 긴급 정지 확인 (매도는 청산을 위해 허용)
    if os.environ.get("EMERGENCY_STOP", "false").lower() == "true":
        if side == "bid":
            return {
                "success": False,
                "dry_run": False,
                "side": side,
                "market": market,
                "amount": amount,
                "error": "EMERGENCY_STOP active - buys blocked",
                "timestamp": ts,
            }
        else:
            print("⚠️ EMERGENCY_STOP active but allowing sell for position liquidation", file=sys.stderr)

    # 1b) 감독 자동 긴급정지 확인 (긴급정지 중 매도는 허용)
    auto_em_file = PROJECT_DIR / "data" / "auto_emergency.json"
    if auto_em_file.exists() and side == "bid":  # 매수만 차단, 매도(청산)는 허용
        try:
            auto_em = json.loads(auto_em_file.read_text(encoding="utf-8"))
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

    # 2a) 일일 매매 횟수 상한 확인 (ET-01)
    max_daily = int(os.environ.get("MAX_DAILY_TRADES", "6"))
    daily_data = _get_daily_trades()
    if daily_data["count"] >= max_daily:
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": f"일일 매매 횟수 상한 도달: {daily_data['count']}/{max_daily}",
            "timestamp": ts,
        }

    # 2b) 최소 매매 간격 확인 (ET-02)
    min_interval = float(os.environ.get("MIN_TRADE_INTERVAL_HOURS", "4"))
    last_trade = _get_last_trade_time()
    if last_trade is not None:
        elapsed_hours = (datetime.now(KST) - last_trade).total_seconds() / 3600
        if elapsed_hours < min_interval:
            remaining = min_interval - elapsed_hours
            return {
                "success": False,
                "dry_run": False,
                "side": side,
                "market": market,
                "amount": amount,
                "error": f"최소 매매 간격 미충족: {elapsed_hours:.1f}h / {min_interval}h (잔여 {remaining:.1f}h)",
                "timestamp": ts,
            }

    # 2c) 매수 시 최대 포지션 비율 확인 (ET-03)
    if side == "bid":
        max_pos_ratio = float(os.environ.get("MAX_POSITION_RATIO", "0.5"))
        current_ratio = _get_btc_position_ratio()
        if current_ratio is not None and current_ratio >= max_pos_ratio:
            return {
                "success": False,
                "dry_run": False,
                "side": side,
                "market": market,
                "amount": amount,
                "error": f"최대 포지션 비율 초과: {current_ratio:.1%} >= {max_pos_ratio:.0%}",
                "timestamp": ts,
            }

    # 2d) 매도 시 보유량 초과 확인 (ET-04)
    if side == "ask":
        try:
            sell_volume = float(amount)
            ratio = _get_btc_position_ratio()
            if ratio is not None:
                # 보유량 직접 확인
                sys.path.insert(0, str(PROJECT_DIR))
                import io as _io
                from scripts.get_portfolio import main as _pf_main
                old_stdout = sys.stdout
                sys.stdout = buf = _io.StringIO()
                try:
                    _pf_main()
                finally:
                    sys.stdout = old_stdout
                pf = json.loads(buf.getvalue())
                for h in pf.get("holdings", []):
                    if h.get("currency") == "BTC":
                        held = h.get("balance", 0)
                        if sell_volume > held:
                            return {
                                "success": False,
                                "dry_run": False,
                                "side": side,
                                "market": market,
                                "amount": amount,
                                "error": f"매도 수량이 보유량 초과: {sell_volume} > {held}",
                                "timestamp": ts,
                            }
                        break
        except Exception as e:
            print(f"[warning] 매도 보유량 검증 실패 (계속 진행): {e}", file=sys.stderr)

    # 3) 매수 금액 상한 확인 (ET-08: float 비교로 수정)
    max_amount = int(os.environ.get("MAX_TRADE_AMOUNT", "100000"))
    if side == "bid" and float(amount) > max_amount:
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

        exec_started = datetime.now(KST)
        r = requests.post(f"{UPBIT_API}/orders", json=body, headers=headers, timeout=10)
        exec_completed = datetime.now(KST)
        latency_ms = int((exec_completed - exec_started).total_seconds() * 1000)

        try:
            response = r.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            response = {"raw_response": r.text[:500]}

        # 성공 시 일일 매매 횟수 증가 + 마지막 매매 시각 기록
        if r.ok:
            try:
                _increment_daily_trades()
                _update_last_trade_time()
            except Exception as e:
                print(f"[warning] 매매 추적 기록 실패: {e}", file=sys.stderr)

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
            "_exec_started": exec_started.isoformat(),
            "_exec_completed": exec_completed.isoformat(),
            "_latency_ms": latency_ms,
        }
    except requests.exceptions.RequestException as e:
        exec_completed = datetime.now(KST)
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": f"주문 요청 실패: {e}",
            "timestamp": ts,
            "_exec_started": exec_started.isoformat() if 'exec_started' in locals() else None,
            "_exec_completed": exec_completed.isoformat(),
            "_latency_ms": None,
        }
    except ValueError as e:
        # API key 미설정 등
        return {
            "success": False,
            "dry_run": False,
            "side": side,
            "market": market,
            "amount": amount,
            "error": str(e),
            "timestamp": ts,
        }
    finally:
        release_lock()


def _record_trade_to_db(result: dict, source: str = "manual"):
    """매매 결과를 DB에 기록 (수동 매매 포함 모든 경로)"""
    try:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        # decisions 테이블에 기록
        side = result.get("side", "")
        action_kr = "매수" if side == "bid" else "매도" if side == "ask" else "관망"
        dry_run = result.get("dry_run", True)

        # cycle_id 생성
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from scripts.cycle_id import make_cycle_id
            _cycle_id = make_cycle_id(source)
        except Exception:
            _cycle_id = datetime.now(KST).strftime("%Y%m%d-%H%M") + f"-{source}"

        decision_row = {
            "decision": action_kr,
            "reason": f"[{source}] {result.get('market', 'KRW-BTC')} {result.get('amount', '')}",
            "confidence": 1.0 if source == "manual" else 0.5,
            "market": result.get("market", "KRW-BTC"),
            "execution_status": "success" if result.get("success") else "failed",
            "execution_error": result.get("error"),
            "dry_run": dry_run,
            "cycle_id": _cycle_id,
            "source": source,
        }

        # 실행 추적 필드 추가
        if result.get("_exec_started"):
            decision_row["execution_attempted"] = True
            decision_row["execution_started_at"] = result.get("_exec_started")
            decision_row["execution_completed_at"] = result.get("_exec_completed")
            decision_row["execution_latency_ms"] = result.get("_latency_ms")
        elif not result.get("dry_run"):
            decision_row["execution_attempted"] = False

        # 체결 정보 추가
        resp = result.get("response", {})
        if isinstance(resp, dict) and resp.get("uuid"):
            decision_row["order_uuid"] = resp["uuid"]
            decision_row["executed_price"] = int(float(resp.get("price", 0) or 0)) or None
            decision_row["executed_volume"] = float(resp.get("volume", 0) or 0) or None

        r = requests.post(
            f"{url}/rest/v1/decisions",
            json=decision_row,
            headers=headers,
            timeout=10,
        )
        if r.ok:
            print(f"[DB] 매매 기록 저장 완료 (source={source})", file=sys.stderr)
        elif "dry_run" in r.text:
            # dry_run 컬럼 없으면 제거 후 재시도
            decision_row.pop("dry_run", None)
            r2 = requests.post(
                f"{url}/rest/v1/decisions",
                json=decision_row,
                headers=headers,
                timeout=10,
            )
            if r2.ok:
                print(f"[DB] 매매 기록 저장 완료 (dry_run 컬럼 없음, source={source})", file=sys.stderr)
            else:
                print(f"[DB] 매매 기록 저장 실패: {r2.status_code} {r2.text[:200]}", file=sys.stderr)
        else:
            print(f"[DB] 매매 기록 저장 실패: {r.status_code} {r.text[:200]}", file=sys.stderr)

    except Exception as e:
        print(f"[DB] 매매 기록 저장 오류: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "사용법: python3 execute_trade.py [bid|ask] [market] [amount]",
            file=sys.stderr,
        )
        sys.exit(1)

    result = execute(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 수동 실행 시에도 DB에 기록
    _record_trade_to_db(result, source="manual")

    if not result["success"]:
        sys.exit(1)
