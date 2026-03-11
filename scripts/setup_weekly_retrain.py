"""Windows 작업 스케줄러에 주간 재학습 등록/해제

사용법:
    python scripts/setup_weekly_retrain.py           # 등록 (일요일 03:00)
    python scripts/setup_weekly_retrain.py --remove   # 해제
    python scripts/setup_weekly_retrain.py --status   # 상태 확인
"""

import argparse
import os
import subprocess
import sys

TASK_NAME = "CoinTrading_Weekly_Retrain"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_python_path():
    """가상환경 Python 경로"""
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    return venv_python if os.path.exists(venv_python) else sys.executable


def register_task():
    """Windows 작업 스케줄러에 주간 재학습 등록"""
    python = get_python_path()
    script = os.path.join(PROJECT_ROOT, "scripts", "weekly_retrain.py")

    # schtasks 명령으로 등록
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", f'"{python}" "{script}"',
        "/SC", "WEEKLY",
        "/D", "SUN",          # 일요일
        "/ST", "03:00",       # 새벽 3시
        "/RL", "HIGHEST",     # 관리자 권한
        "/F",                 # 기존 덮어쓰기
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] 작업 등록 완료: {TASK_NAME}")
        print(f"  실행: 매주 일요일 03:00")
        print(f"  Python: {python}")
        print(f"  스크립트: {script}")
    else:
        print(f"[FAIL] 등록 실패: {result.stderr}")
        print("  관리자 권한으로 실행해주세요.")


def remove_task():
    """작업 해제"""
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"[OK] 작업 해제 완료: {TASK_NAME}")
    else:
        print(f"[FAIL] 해제 실패: {result.stderr}")


def check_status():
    """작업 상태 확인"""
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", TASK_NAME, "/V", "/FO", "LIST"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"작업이 등록되어 있지 않습니다: {TASK_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="주간 재학습 스케줄러 설정")
    parser.add_argument("--remove", action="store_true", help="작업 해제")
    parser.add_argument("--status", action="store_true", help="상태 확인")
    args = parser.parse_args()

    if args.remove:
        remove_task()
    elif args.status:
        check_status()
    else:
        register_task()
