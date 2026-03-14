#!/usr/bin/env python3
"""
Supabase Management API로 SQL 직접 실행

libpq dotted username 문제를 우회하여 SQL을 실행한다.
모든 컴퓨터(Mac/Windows)에서 사용 가능.

사용법:
  # 마이그레이션 파일 실행
  python3 scripts/supabase_query.py --file supabase/migrations/024_scalp_ml_system.sql

  # 직접 쿼리
  python3 scripts/supabase_query.py --sql "SELECT count(*) FROM scalp_market_snapshot"

  # 테이블 목록 조회
  python3 scripts/supabase_query.py --tables

  # 토큰 직접 지정
  python3 scripts/supabase_query.py --token "sbp_xxx" --sql "SELECT 1"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests 패키지 필요: pip install requests", file=sys.stderr)
    sys.exit(1)

PROJECT_REF = "tvxgibvbpndorpcqluis"
API_URL = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"


def get_token_from_keychain() -> str | None:
    """macOS Keychain에서 Supabase CLI 토큰 추출"""
    if platform.system() != "Darwin":
        return None
    try:
        raw = subprocess.check_output(
            ["security", "find-generic-password", "-s", "Supabase CLI", "-w"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if raw.startswith("go-keyring-base64:"):
            import base64
            return base64.b64decode(raw.replace("go-keyring-base64:", "")).decode()
        return raw
    except Exception:
        return None


def get_token_from_env() -> str | None:
    """환경변수에서 토큰 추출"""
    return os.getenv("SUPABASE_ACCESS_TOKEN")


def get_token_from_file() -> str | None:
    """프로젝트 로컬 파일에서 토큰 추출 (.supabase_token)"""
    token_file = Path(__file__).resolve().parent.parent / ".supabase_token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    return None


def find_token() -> str:
    """토큰을 여러 소스에서 찾기"""
    # 1. 환경변수
    token = get_token_from_env()
    if token:
        return token

    # 2. 로컬 파일
    token = get_token_from_file()
    if token:
        return token

    # 3. macOS Keychain
    token = get_token_from_keychain()
    if token:
        return token

    print(
        "Supabase access token을 찾을 수 없습니다.\n"
        "다음 중 하나를 설정하세요:\n"
        "  1. 환경변수: export SUPABASE_ACCESS_TOKEN=sbp_xxx\n"
        "  2. 파일: .supabase_token 에 토큰 저장\n"
        "  3. macOS: supabase login (Keychain에 자동 저장)\n"
        "  4. 직접 지정: --token sbp_xxx",
        file=sys.stderr,
    )
    sys.exit(1)


def execute_sql(token: str, sql: str) -> list | dict:
    """SQL 실행"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(API_URL, headers=headers, json={"query": sql}, timeout=30)

    if resp.status_code in (200, 201):
        return resp.json()
    else:
        print(f"에러 (HTTP {resp.status_code}):", file=sys.stderr)
        print(resp.text[:500], file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Supabase Management API로 SQL 실행 (libpq 우회)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sql", help="실행할 SQL 쿼리")
    group.add_argument("--file", help="실행할 SQL 파일 경로")
    group.add_argument("--tables", action="store_true", help="테이블 목록 조회")
    parser.add_argument("--token", help="Supabase access token (미지정 시 자동 탐색)")
    parser.add_argument("--json", action="store_true", help="JSON 형식 출력")
    args = parser.parse_args()

    token = args.token or find_token()

    if args.tables:
        sql = "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            # 프로젝트 루트 기준으로 재시도
            path = Path(__file__).resolve().parent.parent / args.file
        if not path.exists():
            print(f"파일 없음: {args.file}", file=sys.stderr)
            sys.exit(1)
        sql = path.read_text(encoding="utf-8")
        print(f"파일 로드: {path} ({len(sql):,} bytes)")
    else:
        sql = args.sql

    print(f"실행 중...")
    result = execute_sql(token, sql)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif isinstance(result, list) and result:
        # 테이블 형식 출력
        if isinstance(result[0], dict):
            keys = list(result[0].keys())
            # 컬럼 폭 계산
            widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in result)) for k in keys}
            # 헤더
            header = " | ".join(k.ljust(widths[k]) for k in keys)
            print(header)
            print("-+-".join("-" * widths[k] for k in keys))
            # 데이터
            for row in result:
                print(" | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))
        else:
            for row in result:
                print(row)
        print(f"\n({len(result)}건)")
    else:
        print("완료 (결과 없음 또는 DDL 성공)")


if __name__ == "__main__":
    main()
