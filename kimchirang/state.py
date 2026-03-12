"""Kimchirang State Persistence -- 포지션 상태 파일 저장/복원

봇 재시작 시 포지션 유실 방지를 위해 JSON 파일로 영속화.
파일 잠금으로 동시 접근 방지.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("kimchirang.state")

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_DIR / "data" / "kimchirang_state.json"
LOCK_FILE = STATE_FILE.with_suffix(".lock")

# 기본 상태 (포지션 없음)
DEFAULT_STATE = {
    "side": "none",
    "entry_kp": 0.0,
    "entry_time": 0.0,
    "upbit_qty": 0.0,
    "binance_qty": 0.0,
    "upbit_entry_price": 0.0,
    "binance_entry_price": 0.0,
    "trade_count_today": 0,
    "last_trade_time": 0.0,
    "last_trade_date": "",
}


def _acquire_lock(timeout: float = 5.0) -> bool:
    """파일 잠금 획득 (Windows 호환)"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # 오래된 잠금 파일 제거 (60초 이상)
            try:
                if LOCK_FILE.exists():
                    age = time.time() - LOCK_FILE.stat().st_mtime
                    if age > 60:
                        LOCK_FILE.unlink(missing_ok=True)
                        continue
            except OSError:
                pass
            time.sleep(0.1)
    logger.warning("상태 파일 잠금 획득 실패 (타임아웃)")
    return False


def _release_lock():
    """파일 잠금 해제"""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def load_position() -> dict:
    """저장된 포지션 상태 로드"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not STATE_FILE.exists():
        logger.info("상태 파일 없음 -- 기본 상태 사용")
        return dict(DEFAULT_STATE)

    _acquire_lock()
    try:
        raw = STATE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)

        # 일자 변경 시 trade_count_today 리셋
        today = datetime.now().strftime("%Y-%m-%d")
        if data.get("last_trade_date", "") != today:
            data["trade_count_today"] = 0
            data["last_trade_date"] = today

        # 누락 필드 보충
        for k, v in DEFAULT_STATE.items():
            if k not in data:
                data[k] = v

        logger.info(
            f"상태 복원: side={data['side']}, entry_kp={data['entry_kp']:.2f}%, "
            f"trades_today={data['trade_count_today']}"
        )
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"상태 파일 읽기 실패: {e} -- 기본 상태 사용")
        return dict(DEFAULT_STATE)
    finally:
        _release_lock()


def save_position(state: dict):
    """포지션 상태 저장"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 일자 갱신
    state["last_trade_date"] = datetime.now().strftime("%Y-%m-%d")

    _acquire_lock()
    try:
        STATE_FILE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        logger.error(f"상태 파일 저장 실패: {e}")
    finally:
        _release_lock()
