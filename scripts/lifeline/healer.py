#!/usr/bin/env python3
"""
Lifeline Healer - 진단 결과에 따른 자동 복구 실행

Diagnostician의 진단 결과를 받아 auto_healable인 항목을 자동 복구한다.
모든 액션은 DRY_RUN 모드를 존중한다.

출력: JSON (stdout)
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from scripts.hide_console import subprocess_kwargs
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import requests

# ── KST 타임존 ────────────────────────────────────────
KST = timezone(timedelta(hours=9))

# ── 상수 ──────────────────────────────────────────────

# 캐시 정리 기준 (일)
LOG_RETENTION_DAYS = 7
SNAPSHOT_RETENTION_DAYS = 30
# 오래된 락 파일 기준 (초)
STALE_LOCK_SECONDS = 300  # 5분

# 연결 재시도
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2  # 2^attempt 초

# 테스트 엔드포인트
UPBIT_TEST_URL = "https://api.upbit.com/v1/market/all"
SUPABASE_HEALTH_PATH = "/rest/v1/"

# requests.Session 재사용
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """모듈 레벨 requests.Session을 반환한다 (커넥션 풀 재사용)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Accept": "application/json"})
    return _session


class Healer:
    """진단 결과에 따라 자동 복구를 실행한다."""

    def __init__(self) -> None:
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.project_root = Path(__file__).resolve().parent.parent.parent

    def heal(self, diagnosis: dict) -> dict:
        """단일 진단 결과에 대해 복구 액션을 실행한다.

        Args:
            diagnosis: Diagnostician이 반환하는 진단 결과 dict.
                필수 키: component, recommended_action, auto_healable

        Returns:
            복구 결과 dict (component, action_taken, success, message,
            duration_ms, dry_run)
        """
        component = diagnosis.get("component", "unknown")
        action = diagnosis.get("recommended_action", "none")
        auto_healable = diagnosis.get("auto_healable", False)

        start = time.monotonic()

        # auto_healable이 아니면 스킵
        if not auto_healable or action in ("none", "alert_only"):
            return self._result(
                component=component,
                action_taken=action,
                success=True,
                message="자동 복구 대상 아님 — 알림만 전송",
                start=start,
            )

        # 액션별 디스패치
        dispatch = {
            "retry_connection": self._retry_connection,
            "clear_cache": self._clear_cache,
            "restart_process": self._restart_process,
            "adjust_config": self._adjust_config_wrapper,
            "emergency_stop": self._emergency_stop_wrapper,
            "clean_junk": self._clean_junk_files,
            "restart_bot": self._restart_bot,
            "clean_logs": self._clean_large_logs,
        }

        handler = dispatch.get(action)
        if handler is None:
            return self._result(
                component=component,
                action_taken=action,
                success=False,
                message=f"알 수 없는 복구 액션: {action}",
                start=start,
            )

        try:
            if action == "adjust_config":
                success = handler(component, diagnosis.get("details", {}))
            else:
                success = handler(component)

            msg = "복구 성공" if success else "복구 실패"
            if self.dry_run:
                msg = f"[DRY_RUN] {msg} (실제 실행 안 함)"

            return self._result(
                component=component,
                action_taken=action,
                success=success,
                message=msg,
                start=start,
            )
        except Exception as e:
            return self._result(
                component=component,
                action_taken=action,
                success=False,
                message=f"복구 중 예외 발생: {e}",
                start=start,
            )

    def heal_all(self, diagnoses: list[dict]) -> list[dict]:
        """auto_healable인 모든 진단에 대해 복구를 실행한다.

        Args:
            diagnoses: Diagnostician.diagnose_all()이 반환하는 진단 리스트

        Returns:
            복구 결과 리스트
        """
        results = []
        for diagnosis in diagnoses:
            results.append(self.heal(diagnosis))
        return results

    # ── 복구 액션 구현 ────────────────────────────────────

    def _retry_connection(self, component: str) -> bool:
        """외부 서비스 연결을 재시도한다 (exponential backoff, 3회).

        지원: upbit_api, supabase
        """
        if self.dry_run:
            print(
                f"[healer][DRY_RUN] {component} 연결 재시도 (3회, backoff) 수행 예정",
                file=sys.stderr,
            )
            return True

        session = _get_session()

        if component == "upbit_api":
            url = UPBIT_TEST_URL
        elif component == "supabase":
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")
            if not supabase_url:
                print("[healer] SUPABASE_URL 미설정", file=sys.stderr)
                return False
            url = supabase_url.rstrip("/") + SUPABASE_HEALTH_PATH
            session.headers.update({
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
            })
        else:
            print(f"[healer] {component}에 대한 연결 테스트 미지원", file=sys.stderr)
            return False

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code < 500:
                    print(
                        f"[healer] {component} 연결 성공 (시도 {attempt + 1})",
                        file=sys.stderr,
                    )
                    return True
            except requests.RequestException as e:
                print(
                    f"[healer] {component} 연결 실패 (시도 {attempt + 1}): {e}",
                    file=sys.stderr,
                )

            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            print(f"[healer] {wait}초 후 재시도...", file=sys.stderr)
            time.sleep(wait)

        print(f"[healer] {component} 연결 {RETRY_MAX_ATTEMPTS}회 모두 실패", file=sys.stderr)
        return False

    def _clear_cache(self, component: str) -> bool:
        """오래된 캐시/로그/락 파일을 정리한다.

        - disk: logs/ 7일 이상, data/snapshots/ 30일 이상 파일 삭제
        - stale_locks: data/ 내 5분 이상 된 .lock 파일 삭제
        """
        if self.dry_run:
            print(
                f"[healer][DRY_RUN] {component} 캐시 정리 수행 예정",
                file=sys.stderr,
            )
            if component == "disk":
                self._list_old_files("disk")
            elif component in ("stale_locks",):
                self._list_old_files("stale_locks")
            return True

        if component == "disk":
            return self._clean_disk()
        elif component in ("stale_locks",):
            return self._clean_stale_locks()
        else:
            print(f"[healer] {component}에 대한 캐시 정리 미지원", file=sys.stderr)
            return False

    def _clean_disk(self) -> bool:
        """오래된 로그 및 스냅샷 파일을 삭제한다."""
        now = time.time()
        removed = 0

        # logs/ 내 7일 이상 파일
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            cutoff = now - (LOG_RETENTION_DAYS * 86400)
            for f in logs_dir.rglob("*"):
                if f.is_file() and f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        removed += 1
                        print(f"[healer] 삭제: {f}", file=sys.stderr)
                    except OSError as e:
                        print(f"[healer] 삭제 실패: {f} — {e}", file=sys.stderr)

        # data/snapshots/ 내 30일 이상 파일
        snapshots_dir = self.project_root / "data" / "snapshots"
        if snapshots_dir.exists():
            cutoff = now - (SNAPSHOT_RETENTION_DAYS * 86400)
            for f in snapshots_dir.rglob("*"):
                if f.is_file() and f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        removed += 1
                        print(f"[healer] 삭제: {f}", file=sys.stderr)
                    except OSError as e:
                        print(f"[healer] 삭제 실패: {f} — {e}", file=sys.stderr)

        print(f"[healer] 디스크 정리 완료: {removed}개 파일 삭제", file=sys.stderr)
        return True

    def _clean_stale_locks(self) -> bool:
        """data/ 내 5분 이상 된 .lock 파일을 삭제한다."""
        now = time.time()
        cutoff = now - STALE_LOCK_SECONDS
        data_dir = self.project_root / "data"
        removed = 0

        if not data_dir.exists():
            return True

        for f in data_dir.rglob("*.lock"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    removed += 1
                    print(f"[healer] 락 파일 삭제: {f}", file=sys.stderr)
                except OSError as e:
                    print(f"[healer] 락 파일 삭제 실패: {f} — {e}", file=sys.stderr)

        print(f"[healer] 오래된 락 정리 완료: {removed}개 삭제", file=sys.stderr)
        return True

    def _list_old_files(self, target: str) -> None:
        """DRY_RUN 모드에서 삭제 대상 파일을 stderr에 나열한다."""
        now = time.time()

        if target == "disk":
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                cutoff = now - (LOG_RETENTION_DAYS * 86400)
                old = [f for f in logs_dir.rglob("*") if f.is_file() and f.stat().st_mtime < cutoff]
                print(f"[healer][DRY_RUN] logs/ 내 {LOG_RETENTION_DAYS}일 이상 파일: {len(old)}개", file=sys.stderr)

            snap_dir = self.project_root / "data" / "snapshots"
            if snap_dir.exists():
                cutoff = now - (SNAPSHOT_RETENTION_DAYS * 86400)
                old = [f for f in snap_dir.rglob("*") if f.is_file() and f.stat().st_mtime < cutoff]
                print(f"[healer][DRY_RUN] data/snapshots/ 내 {SNAPSHOT_RETENTION_DAYS}일 이상 파일: {len(old)}개", file=sys.stderr)

        elif target == "stale_locks":
            data_dir = self.project_root / "data"
            if data_dir.exists():
                cutoff = now - STALE_LOCK_SECONDS
                old = [f for f in data_dir.rglob("*.lock") if f.is_file() and f.stat().st_mtime < cutoff]
                print(f"[healer][DRY_RUN] data/ 내 {STALE_LOCK_SECONDS}초 이상 .lock 파일: {len(old)}개", file=sys.stderr)

    def _restart_process(self, component: str) -> bool:
        """프로세스를 재시작한다.

        DRY_RUN 모드에서는 로그만 출력한다.
        """
        # 컴포넌트별 재시작 커맨드 매핑
        restart_commands = {
            "process": [
                sys.executable,
                str(self.project_root / "scripts" / "run_agents.py"),
            ],
            "memory": [
                sys.executable,
                str(self.project_root / "scripts" / "run_agents.py"),
            ],
        }

        cmd = restart_commands.get(component)
        if cmd is None:
            print(f"[healer] {component}에 대한 재시작 커맨드 미정의", file=sys.stderr)
            return False

        if self.dry_run:
            print(
                f"[healer][DRY_RUN] 재시작 예정: {' '.join(cmd)}",
                file=sys.stderr,
            )
            return True

        try:
            print(f"[healer] 프로세스 재시작: {' '.join(cmd)}", file=sys.stderr)
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(self.project_root),
                **subprocess_kwargs(),
            )
            return True
        except OSError as e:
            print(f"[healer] 재시작 실패: {e}", file=sys.stderr)
            return False

    def _adjust_config_wrapper(self, component: str, details: dict) -> bool:
        """설정 조정 래퍼."""
        return self._adjust_config(component, details)

    def _adjust_config(self, component: str, details: dict) -> bool:
        """임시 설정 파일에 조정값을 기록한다.

        .env 파일은 절대 직접 수정하지 않는다.
        대신 data/temp_config.json에 임시 오버라이드를 기록한다.
        """
        temp_config_path = self.project_root / "data" / "temp_config.json"

        # 기존 임시 설정 로드
        temp_config: dict = {}
        if temp_config_path.exists():
            try:
                temp_config = json.loads(temp_config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                temp_config = {}

        if component == "upbit_api":
            # 429 대응: MIN_TRADE_INTERVAL_HOURS 증가
            current = float(os.getenv("MIN_TRADE_INTERVAL_HOURS", "4"))
            new_interval = min(current + 1, 12)  # 최대 12시간
            adjustment = {
                "MIN_TRADE_INTERVAL_HOURS": new_interval,
                "reason": "Upbit API 429 대응 — 매매 간격 임시 증가",
                "adjusted_at": datetime.now(KST).isoformat(),
                "original_value": current,
            }
            temp_config["upbit_api_throttle"] = adjustment

            if self.dry_run:
                print(
                    f"[healer][DRY_RUN] MIN_TRADE_INTERVAL_HOURS: {current} → {new_interval} "
                    f"(temp_config.json에 기록 예정)",
                    file=sys.stderr,
                )
                return True

            try:
                temp_config_path.parent.mkdir(parents=True, exist_ok=True)
                temp_config_path.write_text(
                    json.dumps(temp_config, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    f"[healer] MIN_TRADE_INTERVAL_HOURS: {current} → {new_interval} "
                    f"(temp_config.json에 기록)",
                    file=sys.stderr,
                )
                return True
            except OSError as e:
                print(f"[healer] 설정 조정 실패: {e}", file=sys.stderr)
                return False
        else:
            print(f"[healer] {component}에 대한 설정 조정 미지원", file=sys.stderr)
            return False

    def _emergency_stop_wrapper(self, component: str) -> bool:
        """긴급정지 래퍼."""
        return self._emergency_stop()

    def _emergency_stop(self) -> bool:
        """auto_emergency.json 플래그 파일을 생성하여 긴급정지를 발동한다."""
        emergency_file = self.project_root / "data" / "auto_emergency.json"

        payload = {
            "activated": True,
            "reason": "Lifeline Healer 자동 긴급정지 발동",
            "activated_at": datetime.now(KST).isoformat(),
            "source": "lifeline_healer",
        }

        if self.dry_run:
            print(
                f"[healer][DRY_RUN] 긴급정지 발동 예정: {emergency_file}",
                file=sys.stderr,
            )
            return True

        try:
            emergency_file.parent.mkdir(parents=True, exist_ok=True)
            emergency_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[healer] 긴급정지 발동: {emergency_file}", file=sys.stderr)
            return True
        except OSError as e:
            print(f"[healer] 긴급정지 파일 생성 실패: {e}", file=sys.stderr)
            return False

    # ── 잔여 파일 정리 ──────────────────────────────────────

    def _clean_junk_files(self, component: str) -> bool:
        """프로젝트 루트의 잔여/임시 파일을 삭제한다."""
        junk_patterns = ["*.png", "download_*.py"]
        junk_dirs = ["claude-coin-trading-main", ".claude/worktrees"]

        removed = 0

        for pattern in junk_patterns:
            for f in self.project_root.glob(pattern):
                if f.is_file():
                    if self.dry_run:
                        print(f"[healer][DRY_RUN] 삭제 예정: {f.name}", file=sys.stderr)
                    else:
                        try:
                            f.unlink()
                            removed += 1
                            print(f"[healer] 삭제: {f.name}", file=sys.stderr)
                        except OSError as e:
                            print(f"[healer] 삭제 실패: {f.name} — {e}", file=sys.stderr)

        for d in junk_dirs:
            p = self.project_root / d
            if p.exists() and p.is_dir():
                if self.dry_run:
                    print(f"[healer][DRY_RUN] 디렉토리 삭제 예정: {d}", file=sys.stderr)
                else:
                    try:
                        import shutil
                        shutil.rmtree(p)
                        removed += 1
                        print(f"[healer] 디렉토리 삭제: {d}", file=sys.stderr)
                    except OSError as e:
                        print(f"[healer] 디렉토리 삭제 실패: {d} — {e}", file=sys.stderr)

        print(f"[healer] 잔여 파일 정리: {removed}개 삭제", file=sys.stderr)
        return True

    def _restart_bot(self, component: str) -> bool:
        """중단된 봇 프로세스를 재시작한다."""
        bot_commands = {
            "kimchirang": [sys.executable, "-m", "kimchirang.main"],
            "short_term": [sys.executable, "scripts/short_term_trader.py"],
            "dashboard": [sys.executable, "scripts/dashboard.py"],
        }

        # 어떤 봇이 죽었는지 확인
        dead_bots = []
        bot_keywords = {
            "kimchirang": "kimchirang.main",
            "short_term": "short_term_trader.py",
            "dashboard": "dashboard.py",
        }

        for name, keyword in bot_keywords.items():
            try:
                result = subprocess.run(
                    ["pgrep", "-f", keyword],
                    capture_output=True, text=True, timeout=5,
                )
                if not result.stdout.strip():
                    dead_bots.append(name)
            except Exception:
                dead_bots.append(name)

        if not dead_bots:
            print("[healer] 모든 봇이 실행 중 — 재시작 불필요", file=sys.stderr)
            return True

        for bot_name in dead_bots:
            cmd = bot_commands.get(bot_name)
            if not cmd:
                continue

            if self.dry_run:
                print(f"[healer][DRY_RUN] {bot_name} 재시작 예정", file=sys.stderr)
                continue

            try:
                env = os.environ.copy()
                if bot_name == "kimchirang":
                    env["KR_DRY_RUN"] = env.get("KR_DRY_RUN", "true")

                log_file = self.project_root / "logs" / f"{bot_name}.log"
                with open(log_file, "a") as lf:
                    subprocess.Popen(
                        cmd,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        cwd=str(self.project_root),
                        env=env,
                    )
                print(f"[healer] {bot_name} 재시작 완료", file=sys.stderr)
            except OSError as e:
                print(f"[healer] {bot_name} 재시작 실패: {e}", file=sys.stderr)
                return False

        return True

    def _clean_large_logs(self, component: str) -> bool:
        """50MB 이상 로그 파일을 truncate한다."""
        logs_dir = self.project_root / "logs"
        if not logs_dir.exists():
            return True

        truncated = 0
        size_limit = 50 * 1024 * 1024  # 50MB

        for f in logs_dir.rglob("*"):
            if f.is_file():
                try:
                    if f.stat().st_size > size_limit:
                        if self.dry_run:
                            size_mb = f.stat().st_size / (1024 * 1024)
                            print(f"[healer][DRY_RUN] truncate 예정: {f.name} ({size_mb:.0f}MB)", file=sys.stderr)
                        else:
                            # 마지막 1000줄만 유지
                            try:
                                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
                                keep = lines[-1000:] if len(lines) > 1000 else lines
                                f.write_text("\n".join(keep) + "\n", encoding="utf-8")
                                truncated += 1
                                print(f"[healer] truncate: {f.name} (마지막 1000줄 유지)", file=sys.stderr)
                            except Exception as e:
                                print(f"[healer] truncate 실패: {f.name} — {e}", file=sys.stderr)
                except OSError:
                    pass

        print(f"[healer] 로그 정리: {truncated}개 파일 truncate", file=sys.stderr)
        return True

    # ── 헬퍼 ──────────────────────────────────────────────

    def _result(
        self,
        component: str,
        action_taken: str,
        success: bool,
        message: str,
        start: float,
    ) -> dict:
        """복구 결과 dict를 생성한다."""
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "component": component,
            "action_taken": action_taken,
            "success": success,
            "message": message,
            "duration_ms": duration_ms,
            "dry_run": self.dry_run,
            "healed_at": datetime.now(KST).isoformat(),
        }


# ── CLI ───────────────────────────────────────────────


def main() -> None:
    """stdin에서 JSON 진단 결과를 읽어 복구를 실행하고 결과를 stdout에 출력한다."""
    healer = Healer()

    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(json.dumps({
            "error": f"입력 파싱 실패: {e}",
            "timestamp": datetime.now(KST).isoformat(),
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    # diagnose_all 출력 형식 (diagnoses 키) 또는 진단 리스트 직접
    if isinstance(data, dict) and "diagnoses" in data:
        diagnoses = data["diagnoses"]
    elif isinstance(data, list):
        diagnoses = data
    elif isinstance(data, dict):
        diagnoses = [data]
    else:
        print(json.dumps({"error": "지원하지 않는 입력 형식"}, ensure_ascii=False, indent=2))
        sys.exit(1)

    results = healer.heal_all(diagnoses)

    output = {
        "timestamp": datetime.now(KST).isoformat(),
        "dry_run": healer.dry_run,
        "total_diagnoses": len(diagnoses),
        "healed": sum(1 for r in results if r["success"] and r["action_taken"] not in ("none", "alert_only")),
        "failed": sum(1 for r in results if not r["success"]),
        "skipped": sum(1 for r in results if r["action_taken"] in ("none", "alert_only")),
        "results": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
