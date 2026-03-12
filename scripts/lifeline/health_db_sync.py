#!/usr/bin/env python3
"""
HealthDBSync — Lifeline 헬스체크 결과를 Supabase에 동기화

system_health_logs 테이블에 점검 결과, AI 진단, 자가치유 이력을 기록한다.
DB 동기화 실패가 모니터링 시스템을 중단시키지 않도록 모든 메서드는 예외를 삼킨다.

출력: JSON (stdout)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

KST = timezone(timedelta(hours=9))

# 점검 status → DB severity 매핑
_SEVERITY_MAP = {
    "OK": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class HealthDBSync:
    """Supabase system_health_logs 테이블에 헬스체크 결과를 기록한다."""

    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL", "")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        })

    # ── 단건 기록 ────────────────────────────────────────

    def log_check(
        self,
        check_result: dict,
        diagnosis: dict | None = None,
        healing_result: dict | None = None,
    ) -> bool:
        """점검 결과 1건을 system_health_logs에 INSERT한다.

        Args:
            check_result: sentinel._result() 형식 (component, status, message, details)
            diagnosis: AI 진단 결과 (ai_diagnosis, healing_action 등)
            healing_result: 치유 실행 결과 (resolution_status, healing_duration_ms 등)

        Returns:
            True: 저장 성공, False: 실패 (예외를 삼킨다)
        """
        try:
            severity = _SEVERITY_MAP.get(check_result.get("status", "OK"), "INFO")
            payload: dict = {
                "component": check_result.get("component", "unknown"),
                "severity": severity,
                "issue_summary": check_result.get("message", ""),
                "raw_traceback": json.dumps(
                    check_result.get("details", {}), ensure_ascii=False
                ) if check_result.get("details") else None,
            }

            # AI 진단 정보
            if diagnosis:
                payload["ai_diagnosis"] = diagnosis.get("ai_diagnosis", "")
                payload["healing_action"] = diagnosis.get("healing_action", "")
                payload["resolution_status"] = "DIAGNOSING"

            # 치유 결과 정보
            if healing_result:
                payload["resolution_status"] = healing_result.get(
                    "resolution_status", "DETECTED"
                )
                payload["healing_duration_ms"] = healing_result.get(
                    "healing_duration_ms"
                )
                # 진단 정보가 healing_result에도 있을 수 있음
                if not diagnosis:
                    if healing_result.get("ai_diagnosis"):
                        payload["ai_diagnosis"] = healing_result["ai_diagnosis"]
                    if healing_result.get("healing_action"):
                        payload["healing_action"] = healing_result["healing_action"]

            url = f"{self.supabase_url}/rest/v1/system_health_logs"
            resp = self.session.post(url, json=payload, timeout=10)
            return resp.ok

        except Exception as e:
            print(f"[health_db_sync] log_check 실패: {e}", file=sys.stderr)
            return False

    # ── 배치 기록 ────────────────────────────────────────

    def log_batch(self, results: list[dict]) -> int:
        """여러 점검 결과를 기록하고 성공 건수를 반환한다.

        Args:
            results: 각 항목은 {"check": dict, "diagnosis": dict|None, "healing": dict|None}
                     또는 단순 check_result dict

        Returns:
            성공적으로 INSERT된 건수
        """
        count = 0
        for item in results:
            if isinstance(item, dict) and "check" in item:
                ok = self.log_check(
                    item["check"],
                    diagnosis=item.get("diagnosis"),
                    healing_result=item.get("healing"),
                )
            else:
                ok = self.log_check(item)
            if ok:
                count += 1
        return count

    # ── 최근 인시던트 조회 ────────────────────────────────

    def get_recent_incidents(self, hours: int = 24, limit: int = 50) -> list[dict]:
        """최근 N시간 동안의 인시던트(INFO 제외)를 조회한다.

        Args:
            hours: 조회 범위 (기본 24시간)
            limit: 최대 반환 건수 (기본 50)

        Returns:
            인시던트 목록 (최신순)
        """
        try:
            since = (
                datetime.now(timezone.utc) - timedelta(hours=hours)
            ).isoformat()
            url = f"{self.supabase_url}/rest/v1/system_health_logs"
            params = {
                "severity": "neq.INFO",
                "timestamp": f"gte.{since}",
                "order": "timestamp.desc",
                "limit": str(limit),
            }
            resp = self.session.get(url, params=params, timeout=10)
            if resp.ok:
                return resp.json()
            print(
                f"[health_db_sync] get_recent_incidents 실패: {resp.status_code}",
                file=sys.stderr,
            )
            return []
        except Exception as e:
            print(f"[health_db_sync] get_recent_incidents 예외: {e}", file=sys.stderr)
            return []

    # ── 컴포넌트별 통계 ──────────────────────────────────

    def get_component_stats(self, hours: int = 24) -> dict:
        """v_system_health_summary 뷰에서 컴포넌트별 건강 통계를 조회한다.

        Args:
            hours: 뷰는 최근 24시간 기준 (DB 뷰 자체가 필터링)

        Returns:
            컴포넌트별 통계 딕셔너리
        """
        try:
            url = f"{self.supabase_url}/rest/v1/v_system_health_summary"
            resp = self.session.get(url, timeout=10)
            if resp.ok:
                rows = resp.json()
                return {
                    "components": rows,
                    "total_components": len(rows),
                    "unhealthy": [
                        r for r in rows
                        if (r.get("critical_count", 0) or 0) > 0
                        or (r.get("error_count", 0) or 0) > 0
                    ],
                }
            print(
                f"[health_db_sync] get_component_stats 실패: {resp.status_code}",
                file=sys.stderr,
            )
            return {}
        except Exception as e:
            print(f"[health_db_sync] get_component_stats 예외: {e}", file=sys.stderr)
            return {}


if __name__ == "__main__":
    # Windows cp949 인코딩 문제 방지
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    sync = HealthDBSync()

    # 테스트: 최근 인시던트 + 컴포넌트 통계 출력
    result = {
        "recent_incidents": sync.get_recent_incidents(),
        "component_stats": sync.get_component_stats(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
