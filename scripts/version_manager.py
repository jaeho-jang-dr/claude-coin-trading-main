"""
앱 버전 관리 + 변경 이력 DB 기록

사용법:
  # 변경사항 기록 (DB + VERSION 파일 자동 업데이트)
  python scripts/version_manager.py log \
    --severity major \
    --category bugfix \
    --summary "FGI 기본값 캐싱 + DB학습 활성화" \
    --files "agents/orchestrator.py,agents/base_agent.py" \
    --details '{"fixes": ["FGI fallback", "price_at_switch"]}' \
    --verified

  # 버전 조회
  python scripts/version_manager.py version

  # 최근 변경이력 조회
  python scripts/version_manager.py history [--limit 10]

  # 버전 범프 (patch/minor/major)
  python scripts/version_manager.py bump patch

severity 등급:
  critical  → 긴급 버그 수정, 데이터 손실 방지
  major     → 주요 기능 추가/변경, 중요 버그 수정
  minor     → 소규모 개선, 부분 수정
  patch     → 문서, 설정, 코드 정리
"""

import io
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
VERSION_FILE = PROJECT_DIR / "VERSION"
KST = timezone(timedelta(hours=9))


def get_version() -> str:
    """현재 버전을 읽는다."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return "0.0.0"


def set_version(version: str) -> None:
    """VERSION 파일에 버전을 기록한다."""
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")


def bump_version(part: str) -> str:
    """시맨틱 버전을 올린다. part: major/minor/patch"""
    current = get_version()
    parts = current.split(".")
    if len(parts) != 3:
        parts = ["0", "0", "0"]

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")

    new_version = f"{major}.{minor}.{patch}"
    set_version(new_version)
    return new_version


def _severity_to_bump(severity: str) -> str:
    """severity에 따라 자동 버전 범프 결정."""
    return {
        "critical": "minor",
        "major": "minor",
        "minor": "patch",
        "patch": "patch",
    }.get(severity, "patch")


def log_change(
    severity: str,
    category: str,
    summary: str,
    files: list[str] | None = None,
    details: dict | None = None,
    verified: bool = False,
    verification_result: str | None = None,
    auto_bump: bool = True,
    changed_by: str = "claude_session",
) -> dict:
    """변경사항을 DB에 기록하고, VERSION을 자동 범프한다."""
    import requests
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("[ERROR] SUPABASE 환경변수 미설정", file=sys.stderr)
        return {"error": "no_supabase"}

    # 버전 범프
    if auto_bump:
        bump_part = _severity_to_bump(severity)
        version = bump_version(bump_part)
    else:
        version = get_version()

    row = {
        "version": version,
        "severity": severity,
        "category": category,
        "summary": summary,
        "files_modified": files or [],
        "details": details or {},
        "verified": verified,
        "verification_result": verification_result,
        "changed_by": changed_by,
    }

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        resp = requests.post(
            f"{url}/rest/v1/app_changelog",
            json=row,
            headers=headers,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            result = resp.json()
            if isinstance(result, list):
                result = result[0]
            print(f"[OK] v{version} ({severity}/{category}): {summary}")
            return result
        else:
            print(f"[ERROR] DB 기록 실패: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
            return {"error": resp.text}
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return {"error": str(e)}


def get_history(limit: int = 10) -> list[dict]:
    """최근 변경이력을 조회한다."""
    import requests
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return []

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    try:
        resp = requests.get(
            f"{url}/rest/v1/app_changelog",
            params={
                "select": "version,severity,category,summary,files_modified,verified,created_at",
                "order": "created_at.desc",
                "limit": str(limit),
            },
            headers=headers,
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="앱 버전 관리")
    sub = parser.add_subparsers(dest="command")

    # version
    sub.add_parser("version", help="현재 버전 조회")

    # bump
    p_bump = sub.add_parser("bump", help="버전 범프")
    p_bump.add_argument("part", choices=["major", "minor", "patch"])

    # log
    p_log = sub.add_parser("log", help="변경사항 기록")
    p_log.add_argument("--severity", required=True, choices=["critical", "major", "minor", "patch"])
    p_log.add_argument("--category", required=True, choices=["bugfix", "feature", "improvement", "refactor", "config", "strategy"])
    p_log.add_argument("--summary", required=True, help="변경 요약 (1줄)")
    p_log.add_argument("--files", default="", help="수정 파일 (콤마 구분)")
    p_log.add_argument("--details", default="{}", help="상세 JSON")
    p_log.add_argument("--verified", action="store_true", help="검증 완료")
    p_log.add_argument("--verification-result", default=None, help="검증 결과")
    p_log.add_argument("--no-bump", action="store_true", help="자동 버전 범프 안 함")
    p_log.add_argument("--changed-by", default="claude_session")

    # history
    p_hist = sub.add_parser("history", help="변경이력 조회")
    p_hist.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.command == "version":
        print(f"v{get_version()}")

    elif args.command == "bump":
        new = bump_version(args.part)
        print(f"v{new}")

    elif args.command == "log":
        files = [f.strip() for f in args.files.split(",") if f.strip()]
        details = json.loads(args.details)
        log_change(
            severity=args.severity,
            category=args.category,
            summary=args.summary,
            files=files,
            details=details,
            verified=args.verified,
            verification_result=args.verification_result,
            auto_bump=not args.no_bump,
            changed_by=args.changed_by,
        )

    elif args.command == "history":
        history = get_history(args.limit)
        for h in history:
            sev_icon = {"critical": "🔴", "major": "🟡", "minor": "🟢", "patch": "⚪"}.get(h["severity"], "?")
            verified = "✅" if h.get("verified") else "  "
            ts = h.get("created_at", "")[:16]
            print(f"  {sev_icon} v{h['version']} [{h['category']}] {h['summary']} {verified} ({ts})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
