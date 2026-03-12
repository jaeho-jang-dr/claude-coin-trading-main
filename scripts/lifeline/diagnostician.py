#!/usr/bin/env python3
"""
Lifeline Diagnostician - 헬스체크 실패 원인 분석 + 복구 액션 추천

Sentinel의 체크 결과를 받아 규칙 기반으로 근본 원인을 진단하고
Healer가 실행할 복구 액션을 추천한다.

출력: JSON (stdout)
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── KST 타임존 ────────────────────────────────────────
KST = timezone(timedelta(hours=9))

# ── 상수 ──────────────────────────────────────────────

# status → severity 매핑
STATUS_SEVERITY = {
    "OK": "none",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "critical",
}

# 복구 액션 종류
ACTION_RESTART = "restart_process"
ACTION_CLEAR_CACHE = "clear_cache"
ACTION_ADJUST_CONFIG = "adjust_config"
ACTION_RETRY = "retry_connection"
ACTION_ALERT = "alert_only"
ACTION_EMERGENCY = "emergency_stop"
ACTION_NONE = "none"

# ── 알려진 에러 패턴 (v1 규칙 기반) ──────────────────────

# 키: (component, status) → dict(cause, action, confidence, auto_healable, detail_check)
# detail_check: message 내 특정 키워드가 있으면 오버라이드
_KNOWN_PATTERNS: dict[tuple[str, str], dict] = {
    # Upbit API
    ("upbit_api", "ERROR"): {
        "cause": "Upbit API 연결 실패 또는 일시적 장애",
        "action": ACTION_RETRY,
        "confidence": 0.8,
        "auto_healable": True,
        "overrides": {
            "429": {
                "cause": "Upbit API 요청 한도 초과 (429 Too Many Requests)",
                "action": ACTION_ADJUST_CONFIG,
                "confidence": 0.95,
                "auto_healable": True,
            },
        },
    },
    ("upbit_api", "CRITICAL"): {
        "cause": "Upbit API 완전 불통 — 장기 장애 가능",
        "action": ACTION_RETRY,
        "confidence": 0.7,
        "auto_healable": True,
        "overrides": {
            "429": {
                "cause": "Upbit API 요청 한도 초과 (429 Too Many Requests)",
                "action": ACTION_ADJUST_CONFIG,
                "confidence": 0.95,
                "auto_healable": True,
            },
        },
    },
    # Supabase
    ("supabase", "ERROR"): {
        "cause": "Supabase DB 연결 실패",
        "action": ACTION_RETRY,
        "confidence": 0.8,
        "auto_healable": True,
    },
    ("supabase", "CRITICAL"): {
        "cause": "Supabase DB 완전 불통 — 서비스 장애 가능",
        "action": ACTION_RETRY,
        "confidence": 0.7,
        "auto_healable": True,
    },
    # 디스크
    ("disk", "CRITICAL"): {
        "cause": "디스크 공간 부족 — 오래된 로그/스냅샷 정리 필요",
        "action": ACTION_CLEAR_CACHE,
        "confidence": 0.9,
        "auto_healable": True,
    },
    ("disk", "WARNING"): {
        "cause": "디스크 사용량 높음 — 모니터링 필요",
        "action": ACTION_ALERT,
        "confidence": 0.85,
        "auto_healable": False,
    },
    # 메모리
    ("memory", "CRITICAL"): {
        "cause": "메모리 부족 — 프로세스 재시작 필요",
        "action": ACTION_RESTART,
        "confidence": 0.85,
        "auto_healable": True,
    },
    ("memory", "WARNING"): {
        "cause": "메모리 사용량 높음 — 모니터링 필요",
        "action": ACTION_ALERT,
        "confidence": 0.8,
        "auto_healable": False,
    },
    # 프로세스
    ("process", "ERROR"): {
        "cause": "핵심 프로세스 비정상 종료 또는 미실행",
        "action": ACTION_RESTART,
        "confidence": 0.85,
        "auto_healable": True,
    },
    ("process", "CRITICAL"): {
        "cause": "핵심 프로세스 장시간 미실행",
        "action": ACTION_RESTART,
        "confidence": 0.8,
        "auto_healable": True,
    },
    # RL 모델
    ("rl_model", "WARNING"): {
        "cause": "RL 모델 성능 저하 — 재훈련 권장",
        "action": ACTION_ALERT,
        "confidence": 0.7,
        "auto_healable": False,
    },
    ("rl_model", "ERROR"): {
        "cause": "RL 모델 로드 실패 또는 파일 손상",
        "action": ACTION_ALERT,
        "confidence": 0.75,
        "auto_healable": False,
    },
    # 긴급정지 플래그
    ("emergency_flags", "WARNING"): {
        "cause": "자동 긴급정지 플래그 활성 상태",
        "action": ACTION_ALERT,
        "confidence": 0.95,
        "auto_healable": False,
    },
    ("emergency_flags", "ERROR"): {
        "cause": "수동 긴급정지(EMERGENCY_STOP) 활성 상태",
        "action": ACTION_ALERT,
        "confidence": 0.95,
        "auto_healable": False,
    },
    # 오래된 락 파일
    ("stale_locks", "WARNING"): {
        "cause": "오래된 락 파일 잔존 — 이전 프로세스 비정상 종료 흔적",
        "action": ACTION_CLEAR_CACHE,
        "confidence": 0.9,
        "auto_healable": True,
    },
    ("stale_locks", "ERROR"): {
        "cause": "다수의 오래된 락 파일 — 반복 비정상 종료 의심",
        "action": ACTION_CLEAR_CACHE,
        "confidence": 0.85,
        "auto_healable": True,
    },
}


class Diagnostician:
    """헬스체크 실패 원인을 규칙 기반으로 진단하고 복구 액션을 추천한다."""

    def __init__(self) -> None:
        self.patterns = _KNOWN_PATTERNS

    def diagnose(self, check_result: dict) -> dict:
        """단일 체크 결과를 진단한다.

        Args:
            check_result: Sentinel이 반환하는 체크 결과 dict.
                필수 키: component, status
                선택 키: message, details

        Returns:
            진단 결과 dict (component, severity, issue_summary,
            ai_diagnosis, recommended_action, confidence, auto_healable)
        """
        component = check_result.get("component", "unknown")
        status = check_result.get("status", "OK")
        message = check_result.get("message", "")
        details = check_result.get("details", {})

        severity = STATUS_SEVERITY.get(status, "warning")

        # OK 상태면 진단 불필요
        if status == "OK":
            return {
                "component": component,
                "severity": "none",
                "issue_summary": "정상",
                "ai_diagnosis": "이상 없음",
                "recommended_action": ACTION_NONE,
                "confidence": 1.0,
                "auto_healable": False,
                "diagnosed_at": datetime.now(KST).isoformat(),
            }

        # 알려진 패턴 매칭
        key = (component, status)
        pattern = self.patterns.get(key)

        if pattern is not None:
            cause = pattern["cause"]
            action = pattern["action"]
            confidence = pattern["confidence"]
            auto_healable = pattern["auto_healable"]

            # 메시지 내 키워드로 오버라이드 체크
            overrides = pattern.get("overrides", {})
            for keyword, override in overrides.items():
                if keyword in message or keyword in str(details):
                    cause = override["cause"]
                    action = override["action"]
                    confidence = override["confidence"]
                    auto_healable = override["auto_healable"]
                    break

            return {
                "component": component,
                "severity": severity,
                "issue_summary": f"[{component}] {status}: {message}" if message else f"[{component}] {status}",
                "ai_diagnosis": cause,
                "recommended_action": action,
                "confidence": confidence,
                "auto_healable": auto_healable,
                "diagnosed_at": datetime.now(KST).isoformat(),
            }

        # 알려진 패턴 없음 — 기본 진단
        return {
            "component": component,
            "severity": severity,
            "issue_summary": f"[{component}] {status}: {message}" if message else f"[{component}] {status}",
            "ai_diagnosis": f"알 수 없는 에러 패턴 ({component}/{status}). 수동 확인 필요.",
            "recommended_action": ACTION_ALERT,
            "confidence": 0.3,
            "auto_healable": False,
            "diagnosed_at": datetime.now(KST).isoformat(),
        }

    def diagnose_all(self, checks: list[dict]) -> list[dict]:
        """모든 체크 결과 중 실패(status != 'OK')인 것만 진단한다.

        Args:
            checks: Sentinel 체크 결과 리스트

        Returns:
            진단 결과 리스트 (OK가 아닌 항목만)
        """
        diagnoses = []
        for check in checks:
            if check.get("status", "OK") != "OK":
                diagnoses.append(self.diagnose(check))
        return diagnoses


# ── CLI ───────────────────────────────────────────────


def main() -> None:
    """stdin에서 JSON 체크 결과를 읽어 진단 결과를 stdout에 출력한다."""
    diagnostician = Diagnostician()

    try:
        raw = sys.stdin.read()
        checks = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(json.dumps({
            "error": f"입력 파싱 실패: {e}",
            "timestamp": datetime.now(KST).isoformat(),
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    # 단일 dict이면 리스트로 감싸기
    if isinstance(checks, dict):
        checks = [checks]

    diagnoses = diagnostician.diagnose_all(checks)

    result = {
        "timestamp": datetime.now(KST).isoformat(),
        "total_checks": len(checks),
        "failed_checks": len(diagnoses),
        "diagnoses": diagnoses,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
