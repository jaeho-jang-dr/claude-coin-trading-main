#!/usr/bin/env python3
"""
Lifeline Main Orchestrator — 감시 → 진단 → 치유 → DB 기록 → 알림

자가치유 시스템의 전체 사이클을 실행하고 결과를 JSON으로 출력한다.

사용법:
  python scripts/lifeline/main.py                # 1회 실행 (기본)
  python scripts/lifeline/main.py --once          # 1회 실행
  python scripts/lifeline/main.py --interval 120  # 120초 간격 연속 실행
  python scripts/lifeline/main.py --verbose        # 상세 로그

출력: JSON (stdout)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

KST = timezone(timedelta(hours=9))

# ── 내부 모듈 임포트 ─────────────────────────────────────
# sentinel, diagnostician, healer, health_db_sync는 같은 패키지
_LIFELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_LIFELINE_DIR.parent))

from lifeline.sentinel import run_all_checks
from lifeline.health_db_sync import HealthDBSync

# diagnostician / healer는 같은 패키지 — 안전하게 임포트
try:
    from lifeline.diagnostician import Diagnostician  # type: ignore[import-not-found]
    _HAS_DIAGNOSTICIAN = True
except ImportError:
    _HAS_DIAGNOSTICIAN = False

try:
    from lifeline.healer import Healer  # type: ignore[import-not-found]
    _HAS_HEALER = True
except ImportError:
    _HAS_HEALER = False


# ── 텔레그램 알림 ─────────────────────────────────────────

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def _escape_md(text: str) -> str:
    """MarkdownV2 특수문자 이스케이프"""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", str(text))


def send_health_alert(critical_issues: list[dict]) -> bool:
    """CRITICAL 이슈를 텔레그램으로 알린다.

    Args:
        critical_issues: 각 항목은 check_result + 선택적 diagnosis/healing 정보

    Returns:
        True: 전송 성공, False: 실패
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    user_id = os.environ.get("TELEGRAM_USER_ID")
    if not bot_token or not user_id:
        print("[lifeline] TELEGRAM_BOT_TOKEN/USER_ID 미설정", file=sys.stderr)
        return False

    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = [f"\U0001f3e5 *Lifeline Health Alert*\n"]

    for issue in critical_issues:
        check = issue.get("check", issue)
        component = _escape_md(check.get("component", "unknown"))
        severity = _escape_md(check.get("status", "CRITICAL"))
        message = _escape_md(check.get("message", ""))

        lines.append(f"\u26a0\ufe0f *{component}* \\[{severity}\\]")
        lines.append(f"  {message}")

        diagnosis = issue.get("diagnosis")
        if diagnosis:
            action = diagnosis.get("healing_action", "")
            if action:
                lines.append(f"  \U0001f527 {_escape_md(action)}")

        healing = issue.get("healing")
        if healing:
            status = healing.get("resolution_status", "")
            if status:
                lines.append(f"  \U0001f4cb {_escape_md(status)}")

        lines.append("")

    lines.append(f"_{_escape_md(ts)}_")
    text = "\n".join(lines)

    try:
        resp = requests.post(
            f"{TELEGRAM_API.format(token=bot_token)}/sendMessage",
            json={
                "chat_id": user_id,
                "text": text,
                "parse_mode": "MarkdownV2",
            },
            timeout=10,
        )
        if not resp.ok:
            print(f"[lifeline] 텔레그램 전송 실패: {resp.text[:200]}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[lifeline] 텔레그램 전송 예외: {e}", file=sys.stderr)
        return False


# ── 헬스 사이클 ──────────────────────────────────────────

def run_health_cycle(verbose: bool = False) -> dict:
    """감시 → 진단 → 치유 → DB 기록 → 알림 전체 사이클 실행.

    Returns:
        사이클 결과 요약 딕셔너리
    """
    cycle_start = time.time()
    now_kst = datetime.now(KST).isoformat()

    # 1. Sentinel: 전체 점검
    if verbose:
        print("[lifeline] 1/5 Sentinel 점검 시작...", file=sys.stderr)
    health = run_all_checks()
    checks = health.get("checks", [])

    # 2. 비정상 항목 필터
    non_ok = [c for c in checks if c.get("status") != "OK"]
    diagnoses: dict[str, dict] = {}  # component → diagnosis
    healings: dict[str, dict] = {}   # component → healing_result

    # 3. Diagnostician: 비정상 항목 진단
    if non_ok and _HAS_DIAGNOSTICIAN:
        if verbose:
            print(
                f"[lifeline] 2/5 Diagnostician 진단 ({len(non_ok)}건)...",
                file=sys.stderr,
            )
        try:
            dx = Diagnostician()
            diag_results = dx.diagnose_all(non_ok)
            for d in diag_results:
                comp = d.get("component", "")
                diagnoses[comp] = d
        except Exception as e:
            print(f"[lifeline] 진단 실패: {e}", file=sys.stderr)
    elif verbose:
        if non_ok:
            print("[lifeline] 2/5 Diagnostician 미설치 — 건너뜀", file=sys.stderr)
        else:
            print("[lifeline] 2/5 모든 점검 OK — 진단 불필요", file=sys.stderr)

    # 4. Healer: 자동 치유 가능한 항목 치유
    healable = [
        d for d in diagnoses.values()
        if d.get("auto_healable", False)
    ]
    if healable and _HAS_HEALER:
        if verbose:
            print(
                f"[lifeline] 3/5 Healer 치유 ({len(healable)}건)...",
                file=sys.stderr,
            )
        try:
            healer = Healer()
            heal_results = healer.heal_all(healable)
            for h in heal_results:
                comp = h.get("component", "")
                healings[comp] = h
        except Exception as e:
            print(f"[lifeline] 치유 실패: {e}", file=sys.stderr)
    elif verbose:
        if healable:
            print("[lifeline] 3/5 Healer 미설치 — 건너뜀", file=sys.stderr)
        else:
            print("[lifeline] 3/5 치유 대상 없음", file=sys.stderr)

    # 5. DB 동기화
    if verbose:
        print("[lifeline] 4/5 DB 동기화...", file=sys.stderr)
    db_sync = HealthDBSync()
    batch = []
    for c in checks:
        comp = c.get("component", "")
        batch.append({
            "check": c,
            "diagnosis": diagnoses.get(comp),
            "healing": healings.get(comp),
        })
    synced = db_sync.log_batch(batch)

    # 6. CRITICAL 이슈 텔레그램 알림
    critical_issues = [
        b for b in batch if b["check"].get("status") == "CRITICAL"
    ]
    alert_sent = False
    if critical_issues:
        if verbose:
            print(
                f"[lifeline] 5/5 CRITICAL 알림 ({len(critical_issues)}건)...",
                file=sys.stderr,
            )
        alert_sent = send_health_alert(critical_issues)
    elif verbose:
        print("[lifeline] 5/5 CRITICAL 없음 — 알림 불필요", file=sys.stderr)

    cycle_ms = int((time.time() - cycle_start) * 1000)

    summary = {
        "timestamp": now_kst,
        "overall_status": health.get("overall_status", "UNKNOWN"),
        "checks_summary": health.get("summary", {}),
        "non_ok_count": len(non_ok),
        "diagnosed_count": len(diagnoses),
        "healed_count": len(healings),
        "db_synced_count": synced,
        "critical_count": len(critical_issues),
        "alert_sent": alert_sent,
        "cycle_duration_ms": cycle_ms,
    }
    return summary


# ── 연속 실행 ────────────────────────────────────────────

def run_continuous(interval_seconds: int = 60, verbose: bool = False):
    """지정 간격으로 헬스 사이클을 반복 실행한다.

    Args:
        interval_seconds: 실행 간격 (초)
        verbose: 상세 로그 출력
    """
    print(
        f"[lifeline] 연속 모드 시작 (간격: {interval_seconds}초, Ctrl+C로 종료)",
        file=sys.stderr,
    )
    try:
        while True:
            result = run_health_cycle(verbose=verbose)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.stdout.flush()

            status = result.get("overall_status", "")
            ts = result.get("timestamp", "")
            cycle_ms = result.get("cycle_duration_ms", 0)
            if verbose:
                print(
                    f"[lifeline] {ts} | {status} | {cycle_ms}ms | "
                    f"다음 실행: {interval_seconds}초 후",
                    file=sys.stderr,
                )

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n[lifeline] 연속 모드 종료 (KeyboardInterrupt)", file=sys.stderr)


# ── 엔트리포인트 ─────────────────────────────────────────

if __name__ == "__main__":
    # Windows cp949 인코딩 문제 방지
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Lifeline 자가치유 시스템 — 감시/진단/치유 오케스트레이터"
    )
    parser.add_argument(
        "--once", action="store_true", default=False,
        help="1회 실행 후 종료 (기본 동작)",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="연속 실행 간격 (초). 미지정 시 1회 실행",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="stderr에 상세 진행 로그 출력",
    )
    args = parser.parse_args()

    if args.interval is not None:
        run_continuous(interval_seconds=args.interval, verbose=args.verbose)
    else:
        result = run_health_cycle(verbose=args.verbose)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("overall_status") in ("ERROR", "CRITICAL"):
            sys.exit(1)
        sys.exit(0)
