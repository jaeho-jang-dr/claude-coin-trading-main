"""통합 cycle_id 생성 — 모든 DB 기록의 연결 키"""
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

def make_cycle_id(source: str = "unknown") -> str:
    """cycle_id 생성: YYYYMMDD-HHmm-{source}"""
    now = datetime.now(KST)
    return f"{now.strftime('%Y%m%d-%H%M')}-{source}"

# 현재 세션의 cycle_id (같은 프로세스 내 동일 ID 유지)
_current_cycle_id = None

def get_or_create_cycle_id(source: str = "unknown") -> str:
    """프로세스 내 동일 cycle_id 반환 (최초 호출 시 생성)"""
    global _current_cycle_id
    if _current_cycle_id is None:
        _current_cycle_id = make_cycle_id(source)
    return _current_cycle_id

def set_cycle_id(cycle_id: str):
    """외부에서 cycle_id 지정 (파이프라인에서 전달받을 때)"""
    global _current_cycle_id
    _current_cycle_id = cycle_id
