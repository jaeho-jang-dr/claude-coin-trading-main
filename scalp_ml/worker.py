#!/usr/bin/env python3
"""
ML 학습 워커 데몬

Supabase의 scalp_training_tasks 테이블을 폴링하여 학습 작업을 실행한다.
Mac Mini, Windows PC 모두에서 실행 가능.

실행:
  python3 -m scalp_ml.worker                                    # 기본
  python3 -m scalp_ml.worker --worker-id win-pc1 --tier coworker # Windows PC
  python3 -m scalp_ml.worker --worker-id community-01 --tier collaborator
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# 프로젝트 루트 찾기
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

KST = timezone(timedelta(hours=9))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ml_worker")

# ── 티어별 허용 작업 ──
TIER_PERMISSIONS = {
    "owner": {"train_lgbm", "train_xgboost", "train_pytorch", "backtest",
              "evaluate", "parameter_sweep", "analyze"},
    "coworker": {"train_lgbm", "train_xgboost", "train_pytorch", "backtest",
                 "evaluate", "parameter_sweep", "analyze"},
    "collaborator": {"train_lgbm", "train_xgboost", "train_pytorch", "backtest",
                     "parameter_sweep"},
    "viewer": set(),  # 작업 실행 불가
}

# 티어별 금지 행위 (코드 레벨 강제)
TIER_RESTRICTIONS = {
    "collaborator": {
        "no_model_deploy": True,     # is_active=true 설정 불가
        "no_live_trading": True,     # 매매 실행 불가
        "no_strategy_edit": True,    # 전략 파일 수정 불가
        "read_only_tables": {"decisions", "portfolio_snapshots", "feedback",
                             "strategy_history", "agent_switches"},
    },
    "viewer": {
        "no_model_deploy": True,
        "no_live_trading": True,
        "no_strategy_edit": True,
        "no_training": True,
    },
}


class SupabaseClient:
    """Supabase REST API 클라이언트"""

    def __init__(self):
        self.url = SUPABASE_URL
        self.headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

    def get(self, table: str, params: dict) -> list[dict]:
        try:
            resp = requests.get(
                f"{self.url}/rest/v1/{table}",
                params=params,
                headers=self.headers,
                timeout=15,
            )
            return resp.json() if resp.ok else []
        except Exception as e:
            log.warning(f"DB GET 실패: {e}")
            return []

    def patch(self, table: str, filters: dict, data: dict) -> bool:
        try:
            params = {k: f"eq.{v}" for k, v in filters.items()}
            resp = requests.patch(
                f"{self.url}/rest/v1/{table}",
                params=params,
                json=data,
                headers={**self.headers, "Prefer": "return=minimal"},
                timeout=15,
            )
            return resp.status_code < 300
        except Exception as e:
            log.warning(f"DB PATCH 실패: {e}")
            return False

    def insert(self, table: str, data: dict) -> bool:
        try:
            resp = requests.post(
                f"{self.url}/rest/v1/{table}",
                json=data,
                headers={**self.headers, "Prefer": "return=minimal"},
                timeout=15,
            )
            return resp.status_code < 300
        except Exception as e:
            log.warning(f"DB INSERT 실패: {e}")
            return False

    def upsert(self, table: str, data: dict, on_conflict: str = "worker_id") -> bool:
        try:
            resp = requests.post(
                f"{self.url}/rest/v1/{table}",
                json=data,
                headers={
                    **self.headers,
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
                timeout=15,
            )
            return resp.status_code < 300
        except Exception as e:
            log.warning(f"DB UPSERT 실패: {e}")
            return False


def get_system_info() -> dict:
    """현재 시스템 정보 수집"""
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }

    # RAM
    try:
        if platform.system() == "Darwin":
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            info["ram_gb"] = int(int(raw) / (1024**3))
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong), ('dwMemoryLoad', c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram_gb"] = int(stat.ullTotalPhys / (1024**3))
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info["ram_gb"] = int(kb / (1024**2))
                        break
    except Exception:
        pass

    # CPU
    try:
        if platform.system() == "Darwin":
            info["cpu_info"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif platform.system() == "Windows":
            info["cpu_info"] = platform.processor()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_info"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["gpu_info"] = result.stdout.strip().split("\n")[0]
            info["can_gpu"] = True
    except Exception:
        info["can_gpu"] = False

    return info


def get_resource_usage() -> dict:
    """현재 CPU/RAM/디스크 사용률"""
    usage = {}
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        usage["disk_free_gb"] = round(free / (1024**3), 2)
    except Exception:
        pass

    try:
        if platform.system() == "Darwin":
            # macOS: vm_stat으로 간이 계산
            raw = subprocess.check_output(["vm_stat"], text=True)
            lines = raw.strip().split("\n")
            page_size = 16384  # Apple Silicon
            free_pages = 0
            for line in lines:
                if "Pages free" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
            total_mem = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True).strip())
            free_mem = free_pages * page_size
            usage["ram_usage_pct"] = round((1 - free_mem / total_mem) * 100, 1)

            # CPU (loadavg)
            load = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            usage["cpu_usage_pct"] = round(min(load / cpu_count * 100, 100), 1)
        elif platform.system() == "Windows":
            # Windows: wmic
            result = subprocess.run(
                ["wmic", "cpu", "get", "loadpercentage", "/value"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                if "LoadPercentage" in line:
                    usage["cpu_usage_pct"] = float(line.split("=")[1].strip())
    except Exception:
        pass

    return usage


class MLWorker:
    """ML 학습 워커"""

    def __init__(self, worker_id: str, tier: str = "collaborator",
                 poll_interval: int = 30, worker_name: str | None = None):
        self.worker_id = worker_id
        self.tier = tier
        self.worker_name = worker_name
        self.poll_interval = poll_interval
        self.running = True
        self.db = SupabaseClient()
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.allowed_tasks = TIER_PERMISSIONS.get(tier, set())
        self.current_task_id = None
        self._last_heartbeat = 0

    def register(self):
        """워커를 DB에 등록 (이미 있으면 업데이트)"""
        sys_info = get_system_info()

        worker_data = {
            "worker_id": self.worker_id,
            "worker_name": self.worker_name or self.worker_id,
            "tier": self.tier,
            "status": "online",
            "is_main_brain": self.tier == "owner",
            "last_heartbeat": datetime.now(KST).isoformat(),
            "last_online_at": datetime.now(KST).isoformat(),
            "allowed_task_types": list(self.allowed_tasks),
            **sys_info,
        }

        if self.db.upsert("compute_workers", worker_data):
            log.info(f"워커 등록 완료: {self.worker_id} (tier: {self.tier})")
        else:
            log.warning("워커 등록 실패 — 계속 실행")

    def heartbeat(self):
        """하트비트 전송 (60초 간격)"""
        now = time.time()
        if now - self._last_heartbeat < 60:
            return
        self._last_heartbeat = now

        usage = get_resource_usage()

        # compute_workers 상태 업데이트
        status = "busy" if self.current_task_id else "online"
        self.db.patch("compute_workers", {"worker_id": self.worker_id}, {
            "status": status,
            "last_heartbeat": datetime.now(KST).isoformat(),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        })

        # 하트비트 로그
        self.db.insert("worker_heartbeats", {
            "worker_id": self.worker_id,
            "status": status,
            "current_task_id": self.current_task_id,
            **usage,
        })

    def go_offline(self):
        """종료 시 오프라인 표시"""
        self.db.patch("compute_workers", {"worker_id": self.worker_id}, {
            "status": "offline",
            "last_online_at": datetime.now(KST).isoformat(),
        })

    def claim_task(self) -> dict | None:
        """대기 중인 작업 1건을 가져와서 running으로 변경 (티어 권한 확인)"""
        tasks = self.db.get("scalp_training_tasks", {
            "select": "*",
            "status": "eq.pending",
            "order": "priority.asc,created_at.asc",
            "limit": "5",  # 여러 건 가져와서 권한 필터링
        })

        if not tasks:
            return None

        # 티어 권한에 맞는 작업만 선택
        for task in tasks:
            if task["task_type"] not in self.allowed_tasks:
                continue

            success = self.db.patch("scalp_training_tasks", {"id": task["id"]}, {
                "status": "running",
                "assigned_worker": self.worker_id,
                "started_at": datetime.now(KST).isoformat(),
            })

            if success:
                self.current_task_id = task["id"]
                log.info(f"작업 수임: {task['task_type']} (ID: {task['id'][:8]})")
                return task

        return None

    def complete_task(self, task_id: str, result: dict):
        """작업 완료 처리"""
        self.db.patch("scalp_training_tasks", {"id": task_id}, {
            "status": "completed",
            "result": json.dumps(result, default=str),
            "completed_at": datetime.now(KST).isoformat(),
        })
        self.current_task_id = None
        self.tasks_completed += 1
        log.info(f"작업 완료: {task_id[:8]} (총 {self.tasks_completed}건)")

    def fail_task(self, task_id: str, error: str):
        """작업 실패 처리"""
        self.db.patch("scalp_training_tasks", {"id": task_id}, {
            "status": "failed",
            "error_message": error[:500],
            "completed_at": datetime.now(KST).isoformat(),
        })
        self.current_task_id = None
        self.tasks_failed += 1
        log.error(f"작업 실패: {task_id[:8]} — {error[:200]}")

    def execute_task(self, task: dict):
        """작업 실행 (타입별 분기)"""
        task_type = task["task_type"]
        params = task.get("params", {})
        if isinstance(params, str):
            params = json.loads(params)

        try:
            if task_type == "analyze":
                result = self.run_analyze(params)
            elif task_type == "backtest":
                result = self.run_backtest(params)
            elif task_type in ("train_lgbm", "train_xgboost"):
                result = self.run_train_tree(task_type, params)
            elif task_type == "train_pytorch":
                result = self.run_train_pytorch(params)
            elif task_type == "evaluate":
                result = self.run_evaluate(params)
            elif task_type == "parameter_sweep":
                result = self.run_parameter_sweep(params)
            else:
                raise ValueError(f"알 수 없는 작업 타입: {task_type}")

            self.complete_task(task["id"], result)

        except Exception as e:
            self.fail_task(task["id"], str(e))

    # ── 작업 실행기 ──

    def run_analyze(self, params: dict) -> dict:
        """시그널 분석"""
        from scalp_ml.analyzer import SignalAnalyzer
        analyzer = SignalAnalyzer(self.db)
        return analyzer.run(params)

    def run_backtest(self, params: dict) -> dict:
        """백테스트"""
        log.info(f"백테스트 실행: {params}")
        return {"status": "not_implemented", "message": "Phase 2에서 구현 예정"}

    def run_train_tree(self, task_type: str, params: dict) -> dict:
        """LightGBM / XGBoost 학습"""
        log.info(f"{task_type} 학습 실행: {params}")
        return {"status": "not_implemented", "message": "Phase 3에서 구현 예정"}

    def run_train_pytorch(self, params: dict) -> dict:
        """PyTorch 학습"""
        log.info(f"PyTorch 학습 실행: {params}")
        return {"status": "not_implemented", "message": "Phase 4에서 구현 예정"}

    def run_evaluate(self, params: dict) -> dict:
        """모델 평가"""
        log.info(f"모델 평가 실행: {params}")
        return {"status": "not_implemented", "message": "Phase 3에서 구현 예정"}

    def run_parameter_sweep(self, params: dict) -> dict:
        """하이퍼파라미터 탐색"""
        log.info(f"파라미터 스윕 실행: {params}")
        return {"status": "not_implemented", "message": "Phase 3에서 구현 예정"}

    # ── 메인 루프 ──

    def run(self):
        """워커 메인 루프"""
        sys_info = get_system_info()
        log.info(f"{'='*50}")
        log.info(f"ML 워커 시작")
        log.info(f"  워커 ID: {self.worker_id}")
        log.info(f"  티어: {self.tier}")
        log.info(f"  플랫폼: {sys_info.get('platform')} {sys_info.get('architecture')}")
        log.info(f"  RAM: {sys_info.get('ram_gb', '?')}GB")
        log.info(f"  CPU: {sys_info.get('cpu_info', '?')}")
        log.info(f"  GPU: {sys_info.get('gpu_info', '없음')}")
        log.info(f"  허용 작업: {', '.join(sorted(self.allowed_tasks))}")
        log.info(f"  폴링 간격: {self.poll_interval}초")
        log.info(f"{'='*50}")

        # DB에 워커 등록
        self.register()

        # graceful shutdown
        def handle_signal(signum, frame):
            log.info("종료 신호 수신")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        idle_count = 0
        while self.running:
            try:
                # 하트비트
                self.heartbeat()

                task = self.claim_task()
                if task:
                    idle_count = 0
                    self.execute_task(task)
                else:
                    idle_count += 1
                    if idle_count % 10 == 0:
                        log.debug(f"대기 중... (완료: {self.tasks_completed}, 실패: {self.tasks_failed})")
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"워커 루프 에러: {e}")
                time.sleep(self.poll_interval * 2)

        # 종료 처리
        self.go_offline()
        log.info(f"워커 종료 (완료: {self.tasks_completed}, 실패: {self.tasks_failed})")


def main():
    parser = argparse.ArgumentParser(description="ML 학습 워커 데몬")
    parser.add_argument("--worker-id", default=platform.node(),
                        help="워커 식별자 (기본: hostname)")
    parser.add_argument("--worker-name", default=None,
                        help="표시 이름 (예: 'JSH PC', '커뮤니티 홍길동')")
    parser.add_argument("--tier", default="collaborator",
                        choices=["owner", "coworker", "collaborator", "viewer"],
                        help="워커 티어 (기본: collaborator)")
    parser.add_argument("--interval", type=int, default=30,
                        help="폴링 간격 (초, 기본: 30)")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 미설정")
        sys.exit(1)

    if args.tier == "viewer":
        log.error("viewer 티어는 작업 실행 불가 — 대시보드만 사용하세요")
        sys.exit(1)

    worker = MLWorker(
        worker_id=args.worker_id,
        tier=args.tier,
        poll_interval=args.interval,
        worker_name=args.worker_name,
    )
    worker.run()


if __name__ == "__main__":
    main()
