"""Kimchirang State Persistence -- 포지션 상태 파일 저장/복원

봇 재시작 시 포지션 유실 방지를 위해 JSON 파일로 영속화.
파일 잠금으로 동시 접근 방지.

Graceful degradation:
  - 잠금 실패 시에도 읽기는 진행 (경합 주의 로그)
  - 잠금 실패 시 쓰기는 in-memory 백업에 저장
  - 파일 손상 시 백업 파일에서 복원 시도
  - 원자적 쓰기 (tmp -> rename)로 크래시 안전성 확보
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
BACKUP_FILE = STATE_FILE.with_suffix(".bak")
LOCK_FILE = STATE_FILE.with_suffix(".lock")

# 디렉토리 존재 여부 캐시 (프로세스 생존 동안 한 번만 생성)
_dir_ensured = False

# 메모리 백업 (파일 저장 실패 시 사용)
_memory_backup: dict = {}

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


def _ensure_dir():
    """상태 파일 디렉토리를 한 번만 생성"""
    global _dir_ensured
    if not _dir_ensured:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _dir_ensured = True
        except OSError as e:
            logger.error(f"상태 디렉토리 생성 실패: {e}")


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
                        logger.warning(f"오래된 잠금 파일 제거 (수명 {age:.0f}초)")
                        LOCK_FILE.unlink(missing_ok=True)
                        continue
            except OSError:
                pass
            time.sleep(0.1)
        except OSError as e:
            logger.error(f"잠금 파일 생성 오류: {e}")
            return False
    logger.warning("상태 파일 잠금 획득 실패 (타임아웃)")
    return False


def _release_lock():
    """파일 잠금 해제"""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _read_json_safe(filepath: Path) -> dict:
    """JSON 파일을 안전하게 읽기 (실패 시 None 반환)"""
    try:
        if filepath.exists():
            raw = filepath.read_text(encoding="utf-8")
            if raw.strip():
                return json.loads(raw)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"JSON 읽기 실패 ({filepath.name}): {e}")
    return None


def load_position() -> dict:
    """저장된 포지션 상태 로드

    복원 우선순위:
      1. 메인 상태 파일
      2. 백업 파일 (.bak)
      3. 메모리 백업
      4. 기본 상태
    """
    global _memory_backup
    _ensure_dir()

    if not STATE_FILE.exists() and not BACKUP_FILE.exists():
        if _memory_backup:
            logger.info("상태 파일 없음 -- 메모리 백업 사용")
            return dict(_memory_backup)
        logger.info("상태 파일 없음 -- 기본 상태 사용")
        return dict(DEFAULT_STATE)

    locked = _acquire_lock()
    if not locked:
        logger.warning("잠금 실패 -- 파일 읽기를 진행하되 경합 주의")

    try:
        # 1. 메인 파일 시도
        data = _read_json_safe(STATE_FILE)

        # 2. 메인 실패 시 백업 시도
        if data is None:
            logger.warning("메인 상태 파일 손상 -- 백업에서 복원 시도")
            data = _read_json_safe(BACKUP_FILE)
            if data:
                logger.info("백업 파일에서 상태 복원 성공")
            else:
                # 3. 메모리 백업 시도
                if _memory_backup:
                    logger.warning("백업 파일도 실패 -- 메모리 백업 사용")
                    data = dict(_memory_backup)
                else:
                    logger.error("모든 복원 소스 실패 -- 기본 상태 사용")
                    return dict(DEFAULT_STATE)

        # 일자 변경 시 trade_count_today 리셋
        today = datetime.now().strftime("%Y-%m-%d")
        if data.get("last_trade_date", "") != today:
            data["trade_count_today"] = 0
            data["last_trade_date"] = today

        # 누락 필드 보충
        for k, v in DEFAULT_STATE.items():
            if k not in data:
                data[k] = v

        # 메모리 백업 갱신
        _memory_backup = dict(data)

        logger.info(
            f"상태 복원: side={data['side']}, entry_kp={data['entry_kp']:.2f}%, "
            f"trades_today={data['trade_count_today']}"
        )
        return data
    except Exception as e:
        logger.error(f"상태 로드 예상치 못한 오류: {e} -- 기본 상태 사용")
        return dict(DEFAULT_STATE)
    finally:
        if locked:
            _release_lock()


def save_position(state: dict):
    """포지션 상태 저장

    실패 시에도 메모리 백업에는 항상 저장한다.
    """
    global _memory_backup
    _ensure_dir()

    # 일자 갱신
    state["last_trade_date"] = datetime.now().strftime("%Y-%m-%d")

    # 메모리 백업은 항상 갱신 (파일 저장 실패해도 봇 재시작 전까지 유지)
    _memory_backup = dict(state)

    locked = _acquire_lock()
    if not locked:
        logger.error(
            "잠금 실패 -- 상태를 메모리 백업에만 저장 "
            "(다음 저장 시 파일 기록 재시도)"
        )
        return

    try:
        content = json.dumps(state, indent=2, ensure_ascii=False)

        # 기존 파일을 백업으로 복사 (크래시 시 복원용)
        try:
            if STATE_FILE.exists():
                import shutil
                shutil.copy2(str(STATE_FILE), str(BACKUP_FILE))
        except OSError as e:
            logger.debug(f"백업 파일 생성 실패 (치명적 아님): {e}")

        # 원자적 쓰기: 임시 파일에 먼저 쓰고 rename (크래시 시 데이터 보호)
        tmp_file = STATE_FILE.with_suffix(".tmp")
        tmp_file.write_text(content, encoding="utf-8")
        tmp_file.replace(STATE_FILE)
    except OSError as e:
        logger.error(f"상태 파일 저장 실패: {e} (메모리 백업에는 저장됨)")
    finally:
        if locked:
            _release_lock()
