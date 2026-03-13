#!/usr/bin/env python3
"""
Sentinel — Lifeline 자기치유 시스템의 감시 스크립트

모든 시스템 컴포넌트를 점검하고 상태를 JSON으로 출력한다.
종료 코드: OK/WARNING → 0, ERROR/CRITICAL → 1

점검 항목:
  - Upbit API 연결
  - Supabase DB 연결
  - 디스크 여유 공간
  - 메모리 사용량
  - 트레이딩 프로세스 생존
  - RL 모델 파일 최신성
  - 긴급정지 플래그
  - 오래된 락 파일

출력: JSON (stdout)
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import requests

# ── 상수 ────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

STATUS_ORDER = {"OK": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}

# ── requests 세션 (커넥션 풀 재사용) ────────────────────
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """모듈 레벨 requests.Session을 반환한다 (커넥션 풀 재사용)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Accept": "application/json"})
    return _session


def _result(component: str, status: str, message: str, details: dict | None = None) -> dict:
    """표준 점검 결과 딕셔너리를 반환한다."""
    return {
        "component": component,
        "status": status,
        "message": message,
        "details": details or {},
    }


# ── 개별 점검 ──────────────────────────────────────────

def check_upbit_api() -> dict:
    """Upbit API 연결 상태를 점검한다."""
    component = "upbit_api"
    try:
        session = _get_session()
        resp = session.get("https://api.upbit.com/v1/market/all", timeout=10)
        if resp.status_code == 200:
            market_count = len(resp.json()) if isinstance(resp.json(), list) else 0
            return _result(component, "OK", "Upbit API 정상", {"status_code": 200, "markets": market_count})
        return _result(component, "ERROR", f"Upbit API 비정상 응답: {resp.status_code}", {"status_code": resp.status_code})
    except requests.exceptions.Timeout:
        return _result(component, "ERROR", "Upbit API 타임아웃 (10s)", {"error": "timeout"})
    except Exception as e:
        return _result(component, "ERROR", f"Upbit API 연결 실패: {e}", {"error": str(e)})


def check_supabase() -> dict:
    """Supabase DB 연결 상태를 점검한다."""
    component = "supabase"
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_KEY", "")

    if not supabase_url:
        return _result(component, "WARNING", "SUPABASE_URL 미설정", {"configured": False})
    if not supabase_key:
        return _result(component, "WARNING", "SUPABASE_KEY 미설정", {"configured": False})

    try:
        session = _get_session()
        resp = session.get(
            f"{supabase_url.rstrip('/')}/rest/v1/",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return _result(component, "OK", "Supabase 연결 정상", {"status_code": 200})
        return _result(component, "ERROR", f"Supabase 비정상 응답: {resp.status_code}", {"status_code": resp.status_code})
    except requests.exceptions.Timeout:
        return _result(component, "ERROR", "Supabase 타임아웃 (10s)", {"error": "timeout"})
    except Exception as e:
        return _result(component, "ERROR", f"Supabase 연결 실패: {e}", {"error": str(e)})


def check_disk_space() -> dict:
    """프로젝트 루트 디스크 여유 공간을 점검한다."""
    component = "disk_space"
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = (usage.used / usage.total) * 100

        details = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_percent": round(used_pct, 1),
        }

        if free_gb < 0.5:
            return _result(component, "CRITICAL", f"디스크 여유 공간 부족: {free_gb:.2f}GB", details)
        if free_gb < 1.0:
            return _result(component, "WARNING", f"디스크 여유 공간 주의: {free_gb:.2f}GB", details)
        return _result(component, "OK", f"디스크 여유 공간 충분: {free_gb:.2f}GB", details)
    except Exception as e:
        return _result(component, "ERROR", f"디스크 점검 실패: {e}", {"error": str(e)})


def check_memory() -> dict:
    """시스템 메모리 사용량을 점검한다."""
    component = "memory"
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_pct = mem.percent
        available_gb = mem.available / (1024 ** 3)

        details = {
            "used_percent": round(used_pct, 1),
            "available_gb": round(available_gb, 2),
            "total_gb": round(mem.total / (1024 ** 3), 2),
        }

        if used_pct > 95:
            return _result(component, "CRITICAL", f"메모리 사용량 위험: {used_pct:.1f}%", details)
        if used_pct > 85:
            return _result(component, "WARNING", f"메모리 사용량 주의: {used_pct:.1f}%", details)
        return _result(component, "OK", f"메모리 사용량 정상: {used_pct:.1f}%", details)
    except ImportError:
        return _result(component, "WARNING", "psutil 미설치 — 메모리 점검 건너뜀", {"psutil_available": False})
    except Exception as e:
        return _result(component, "ERROR", f"메모리 점검 실패: {e}", {"error": str(e)})


def check_process(pid_file: str | Path | None = None) -> dict:
    """트레이딩 프로세스 생존 여부를 PID 파일로 점검한다."""
    component = "process"
    pid_dir = DATA_DIR
    pid_files: list[Path] = []

    if pid_file is not None:
        p = Path(pid_file)
        if p.exists():
            pid_files = [p]
        else:
            return _result(component, "WARNING", f"PID 파일 없음: {p.name}", {"pid_file": str(p), "exists": False})
    else:
        # data/ 디렉토리에서 모든 .pid 파일 탐색
        if pid_dir.exists():
            pid_files = list(pid_dir.glob("*.pid"))

    if not pid_files:
        return _result(component, "OK", "활성 PID 파일 없음 (프로세스 미실행 상태)", {"pid_files": []})

    results = []
    alive_count = 0
    dead_count = 0

    for pf in pid_files:
        try:
            pid = int(pf.read_text().strip())
            try:
                # os.kill(pid, 0)은 프로세스 존재 여부만 확인 (신호를 보내지 않음)
                os.kill(pid, 0)
                results.append({"file": pf.name, "pid": pid, "alive": True})
                alive_count += 1
            except OSError:
                results.append({"file": pf.name, "pid": pid, "alive": False})
                dead_count += 1
        except (ValueError, OSError) as e:
            results.append({"file": pf.name, "error": str(e)})
            dead_count += 1

    details = {"processes": results, "alive": alive_count, "dead": dead_count}

    if dead_count > 0:
        return _result(component, "WARNING", f"종료된 프로세스 발견: {dead_count}개", details)
    return _result(component, "OK", f"프로세스 {alive_count}개 정상 실행 중", details)


def check_rl_models() -> dict:
    """RL 모델 파일 존재 및 최신성을 점검한다."""
    component = "rl_models"
    model_dir = DATA_DIR / "rl_models"

    if not model_dir.exists():
        return _result(component, "WARNING", "rl_models 디렉토리 없음", {"directory_exists": False})

    # 모델 파일 확장자
    model_extensions = {".zip", ".pt", ".pth", ".pkl", ".onnx", ".h5"}
    model_files: list[Path] = []
    for ext in model_extensions:
        model_files.extend(model_dir.rglob(f"*{ext}"))

    if not model_files:
        return _result(component, "WARNING", "모델 파일 없음", {"file_count": 0})

    now = time.time()
    seven_days_sec = 7 * 24 * 3600
    newest_age_sec = float("inf")
    oldest_age_sec = 0
    stale_files = []

    for mf in model_files:
        try:
            mtime = mf.stat().st_mtime
            age_sec = now - mtime
            newest_age_sec = min(newest_age_sec, age_sec)
            oldest_age_sec = max(oldest_age_sec, age_sec)
            if age_sec > seven_days_sec:
                stale_files.append(mf.name)
        except OSError:
            pass

    newest_age_days = newest_age_sec / 86400 if newest_age_sec != float("inf") else -1
    oldest_age_days = oldest_age_sec / 86400

    details = {
        "file_count": len(model_files),
        "newest_age_days": round(newest_age_days, 1),
        "oldest_age_days": round(oldest_age_days, 1),
        "stale_files": stale_files[:10],  # 최대 10개만 표시
    }

    if newest_age_days > 7:
        return _result(component, "WARNING", f"모든 모델이 7일 이상 오래됨 (최신: {newest_age_days:.1f}일 전)", details)
    return _result(component, "OK", f"RL 모델 {len(model_files)}개 (최신: {newest_age_days:.1f}일 전)", details)


def check_emergency_flags() -> dict:
    """긴급정지 플래그 상태를 점검한다."""
    component = "emergency_flags"
    env_stop = os.getenv("EMERGENCY_STOP", "false").lower() == "true"

    auto_stop = False
    auto_details: dict = {}
    auto_file = DATA_DIR / "auto_emergency.json"
    if auto_file.exists():
        try:
            data = json.loads(auto_file.read_text(encoding="utf-8"))
            auto_stop = data.get("active", False)
            auto_details = {
                "active": auto_stop,
                "reason": data.get("reason", ""),
                "activated_at": data.get("activated_at", ""),
            }
        except (json.JSONDecodeError, OSError):
            auto_details = {"parse_error": True}

    details = {
        "env_emergency_stop": env_stop,
        "auto_emergency": auto_details,
    }

    if env_stop:
        return _result(component, "CRITICAL", "EMERGENCY_STOP 활성화 (.env 수동 설정)", details)
    if auto_stop:
        reason = auto_details.get("reason", "알 수 없음")
        return _result(component, "WARNING", f"자동 긴급정지 활성: {reason}", details)
    return _result(component, "OK", "긴급정지 플래그 없음", details)


def check_stale_locks() -> dict:
    """data/ 디렉토리의 오래된 .lock 파일을 점검한다."""
    component = "stale_locks"

    if not DATA_DIR.exists():
        return _result(component, "OK", "data 디렉토리 없음", {"directory_exists": False})

    lock_files = list(DATA_DIR.glob("*.lock"))
    if not lock_files:
        return _result(component, "OK", "락 파일 없음", {"lock_count": 0})

    now = time.time()
    stale_threshold_sec = 5 * 60  # 5분
    stale = []

    for lf in lock_files:
        try:
            mtime = lf.stat().st_mtime
            age_sec = now - mtime
            if age_sec > stale_threshold_sec:
                stale.append({
                    "file": lf.name,
                    "age_minutes": round(age_sec / 60, 1),
                })
        except OSError:
            pass

    details = {"lock_count": len(lock_files), "stale_locks": stale}

    if stale:
        return _result(component, "WARNING", f"오래된 락 파일 {len(stale)}개 발견 (5분 초과)", details)
    return _result(component, "OK", f"락 파일 {len(lock_files)}개 (모두 정상)", details)


def check_junk_files() -> dict:
    """프로젝트 루트의 잔여/임시 파일을 점검한다."""
    component = "junk_files"
    junk_patterns = [
        # (glob 패턴, 설명)
        ("*.png", "스크린샷"),
        ("download_*.py", "임시 다운로드 스크립트"),
        ("claude-coin-trading-main", "clone 잔여물"),
        (".claude/worktrees", "워크트리 잔여물"),
    ]

    found = []
    for pattern, desc in junk_patterns:
        matches = list(PROJECT_ROOT.glob(pattern))
        # 디렉토리도 체크
        p = PROJECT_ROOT / pattern
        if p.exists() and p not in matches:
            matches.append(p)
        for m in matches:
            # .gitignore에 있는 파일은 제외하지 않음 (정리 대상)
            found.append({"path": str(m.relative_to(PROJECT_ROOT)), "type": desc})

    details = {"junk_count": len(found), "files": found[:20]}

    if len(found) > 5:
        return _result(component, "WARNING", f"잔여 파일 {len(found)}개 발견 — 정리 필요", details)
    if found:
        return _result(component, "OK", f"잔여 파일 {len(found)}개 (허용 범위)", details)
    return _result(component, "OK", "잔여 파일 없음", details)


def check_bot_processes() -> dict:
    """김치랑/초단타/대시보드 봇 프로세스 생존 여부를 점검한다."""
    component = "bot_processes"

    expected_bots = [
        {"name": "kimchirang", "keyword": "kimchirang.main"},
        {"name": "short_term", "keyword": "short_term_trader.py"},
        {"name": "dashboard", "keyword": "dashboard.py"},
    ]

    alive = []
    dead = []

    for bot in expected_bots:
        try:
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", bot["keyword"]],
                capture_output=True, text=True, timeout=5,
            )
            pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
            if pids:
                alive.append({"name": bot["name"], "pids": pids})
            else:
                dead.append(bot["name"])
        except Exception:
            dead.append(bot["name"])

    details = {"alive": alive, "dead": dead}

    if dead:
        return _result(component, "WARNING", f"중단된 봇: {', '.join(dead)}", details)
    return _result(component, "OK", f"봇 {len(alive)}개 정상 실행 중", details)


def check_git_status() -> dict:
    """git 미추적/충돌 파일을 점검한다."""
    component = "git_status"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        untracked = [l[3:] for l in lines if l.startswith("??")]
        conflicts = [l[3:] for l in lines if l.startswith("UU") or l.startswith("AA")]
        modified = [l[3:] for l in lines if l.startswith(" M") or l.startswith("M ")]

        details = {
            "untracked": len(untracked),
            "conflicts": len(conflicts),
            "modified": len(modified),
            "untracked_files": untracked[:10],
            "conflict_files": conflicts[:5],
        }

        if conflicts:
            return _result(component, "ERROR", f"git 충돌 {len(conflicts)}개 — 즉시 해결 필요", details)
        if len(untracked) > 10:
            return _result(component, "WARNING", f"미추적 파일 {len(untracked)}개 — 정리 또는 커밋 필요", details)
        return _result(component, "OK", f"git 상태 정상 (미추적 {len(untracked)}, 수정 {len(modified)})", details)
    except Exception as e:
        return _result(component, "ERROR", f"git 상태 확인 실패: {e}", {"error": str(e)})


def check_log_size() -> dict:
    """로그 디렉토리 크기를 점검한다."""
    component = "log_size"
    logs_dir = PROJECT_ROOT / "logs"

    if not logs_dir.exists():
        return _result(component, "OK", "logs 디렉토리 없음", {"exists": False})

    total_bytes = 0
    file_count = 0
    large_files = []

    for f in logs_dir.rglob("*"):
        if f.is_file():
            try:
                size = f.stat().st_size
                total_bytes += size
                file_count += 1
                if size > 50 * 1024 * 1024:  # 50MB 이상
                    large_files.append({
                        "file": str(f.relative_to(PROJECT_ROOT)),
                        "size_mb": round(size / (1024 * 1024), 1),
                    })
            except OSError:
                pass

    total_mb = total_bytes / (1024 * 1024)
    details = {
        "total_mb": round(total_mb, 1),
        "file_count": file_count,
        "large_files": large_files[:5],
    }

    if total_mb > 500:
        return _result(component, "WARNING", f"로그 크기 과다: {total_mb:.0f}MB — 정리 필요", details)
    return _result(component, "OK", f"로그 크기 정상: {total_mb:.0f}MB ({file_count}개 파일)", details)


# ── 전체 점검 실행 ─────────────────────────────────────

def run_all_checks() -> dict:
    """모든 점검을 실행하고 종합 결과를 반환한다."""
    checks = [
        check_upbit_api(),
        check_supabase(),
        check_disk_space(),
        check_memory(),
        check_process(),
        check_rl_models(),
        check_emergency_flags(),
        check_stale_locks(),
        check_junk_files(),
        check_bot_processes(),
        check_git_status(),
        check_log_size(),
    ]

    # 최악 상태를 전체 상태로 결정
    worst = "OK"
    for c in checks:
        if STATUS_ORDER.get(c["status"], 0) > STATUS_ORDER.get(worst, 0):
            worst = c["status"]

    now_kst = datetime.now(KST).isoformat()

    return {
        "timestamp": now_kst,
        "overall_status": worst,
        "checks": checks,
        "summary": {
            "total": len(checks),
            "ok": sum(1 for c in checks if c["status"] == "OK"),
            "warning": sum(1 for c in checks if c["status"] == "WARNING"),
            "error": sum(1 for c in checks if c["status"] == "ERROR"),
            "critical": sum(1 for c in checks if c["status"] == "CRITICAL"),
        },
    }


if __name__ == "__main__":
    # Windows cp949 인코딩 문제 방지
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    result = run_all_checks()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["overall_status"] in ("ERROR", "CRITICAL"):
        sys.exit(1)
    sys.exit(0)
