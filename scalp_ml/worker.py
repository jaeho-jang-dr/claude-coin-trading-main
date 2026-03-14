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
# Owner: SERVICE_ROLE_KEY (bypasses RLS), Non-owner: ANON_KEY (RLS enforced)
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")  # Non-owner workers: per-worker UUID token

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
        # Non-owner workers: RLS token for tier-based access control
        if WORKER_TOKEN:
            self.headers["x-worker-token"] = WORKER_TOKEN

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
        """워커를 DB에 등록 (티어 권한 검증 포함)

        보안 규칙:
        - DB에 이미 등록된 워커: DB의 tier를 강제 적용 (CLI --tier 무시)
        - 신규 워커: collaborator로만 자동 등록 가능
        - owner/coworker 승격: 관리자가 DB에서 직접 변경해야 함
        """
        sys_info = get_system_info()

        # 1) DB에 이미 등록된 워커인지 확인
        existing = self.db.get("compute_workers", {
            "select": "worker_id,tier,is_main_brain,status",
            "worker_id": f"eq.{self.worker_id}",
        })

        if existing:
            # DB에 등록된 티어를 강제 적용
            db_tier = existing[0]["tier"]
            if db_tier != self.tier:
                log.warning(
                    f"⚠️  CLI 티어({self.tier})와 DB 티어({db_tier}) 불일치 "
                    f"— DB 티어({db_tier})를 적용합니다"
                )
                log.warning(
                    f"   티어 변경은 관리자가 DB에서 직접 수행해야 합니다"
                )
                self.tier = db_tier
                self.allowed_tasks = TIER_PERMISSIONS.get(db_tier, set())

            # 기존 워커 상태 업데이트 (tier는 변경하지 않음)
            update_data = {
                "status": "online",
                "last_heartbeat": datetime.now(KST).isoformat(),
                "last_online_at": datetime.now(KST).isoformat(),
                "allowed_task_types": list(self.allowed_tasks),
                **sys_info,
            }
            if self.worker_name:
                update_data["worker_name"] = self.worker_name

            if self.db.patch("compute_workers", {"worker_id": self.worker_id}, update_data):
                log.info(f"워커 재접속: {self.worker_id} (tier: {self.tier})")
            else:
                log.warning("워커 상태 업데이트 실패 — 계속 실행")

        else:
            # 미등록 워커: 모든 티어 자동 등록 차단
            # 등록 절차: 관리자 초대 이메일 → 동의 → 승인 → DB 등록
            log.error(
                f"❌ 미등록 워커입니다: '{self.worker_id}'"
            )
            log.error(
                f"   등록 절차:"
            )
            log.error(
                f"   1. 관리자에게 참여 요청 이메일 전송"
            )
            log.error(
                f"   2. 관리자 승인 후 워커 ID와 .env 파일을 이메일로 수령"
            )
            log.error(
                f"   3. 수령한 워커 ID로 재실행"
            )
            self.running = False
            return

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

    # ── 결과 동기화 (수동 전용) ──

    def preview_sync(self) -> list[dict]:
        """동기화 대상 파일을 미리보기 (업로드하지 않음)"""
        results_dir = PROJECT_DIR / "data" / "training_results"
        if not results_dir.exists():
            return []

        sync_marker = results_dir / ".last_sync"
        last_sync_time = 0
        if sync_marker.exists():
            last_sync_time = sync_marker.stat().st_mtime

        previews = []
        for f in sorted(results_dir.glob("*.json")):
            if f.name.startswith("."):
                continue
            if f.stat().st_mtime > last_sync_time:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    previews.append({
                        "file": f.name,
                        "size_kb": round(f.stat().st_size / 1024, 1),
                        "task_type": data.get("task_type", "unknown"),
                        "created": datetime.fromtimestamp(
                            f.stat().st_mtime, tz=KST
                        ).strftime("%Y-%m-%d %H:%M"),
                        "keys": list(data.keys()),
                        "_path": f,
                        "_data": data,
                    })
                except Exception:
                    pass

        return previews

    def do_sync(self, previews: list[dict]) -> tuple[int, int]:
        """확인된 파일만 실제 업로드"""
        results_dir = PROJECT_DIR / "data" / "training_results"
        sync_marker = results_dir / ".last_sync"

        uploaded = 0
        for p in previews:
            try:
                record = {
                    "worker_id": self.worker_id,
                    "task_type": p["_data"].get("task_type", "unknown"),
                    "status": "completed",
                    "result": json.dumps(p["_data"], default=str),
                    "completed_at": p["created"],
                    "params": json.dumps(p["_data"].get("params", {})),
                }

                if self.db.insert("scalp_training_tasks", record):
                    uploaded += 1
            except Exception as e:
                log.warning(f"업로드 실패 ({p['file']}): {e}")

        if uploaded > 0:
            sync_marker.touch()

        return uploaded, len(previews)

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


def self_register(args):
    """초대코드로 자가등록 — 토큰은 본인만 확인 가능"""
    import uuid as _uuid

    invite_code = args.invite_code
    if not invite_code:
        invite_code = input("초대코드를 입력하세요: ").strip()

    if not invite_code:
        print("초대코드가 필요합니다.")
        sys.exit(1)

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    # 1. 초대코드 유효성 확인
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/worker_invites",
        params={"select": "*", "invite_code": f"eq.{invite_code}"},
        headers=headers,
        timeout=15,
    )
    if not resp.ok or not resp.json():
        print("유효하지 않은 초대코드입니다.")
        sys.exit(1)

    invite = resp.json()[0]

    # 만료 체크
    from datetime import datetime, timezone
    expires = datetime.fromisoformat(invite["expires_at"])
    if expires.astimezone(timezone.utc) < datetime.now(timezone.utc):
        print("초대코드가 만료되었습니다. 관리자에게 새 코드를 요청하세요.")
        sys.exit(1)

    # 이미 사용된 코드
    if invite.get("used_at"):
        print("이미 사용된 초대코드입니다.")
        sys.exit(1)

    tier = invite["tier"]
    invite_name = invite.get("worker_name", "")

    # 2. 워커 정보 입력
    print(f"\n초대코드 확인 완료: {tier} 티어")
    if invite_name:
        print(f"등록 이름: {invite_name}")

    worker_id = args.worker_id or input("워커 ID를 입력하세요 (예: my-pc): ").strip()
    if not worker_id:
        worker_id = platform.node()
    worker_name = args.worker_name or invite_name

    # 필수 정보 수집 (오너가 관리에 필요)
    if not worker_name:
        worker_name = input("닉네임을 입력하세요 (필수): ").strip()
    if not worker_name:
        print("닉네임은 필수입니다.")
        sys.exit(1)

    tg_chat_id = getattr(args, 'telegram_id', None)
    if not tg_chat_id:
        tg_chat_id = input("텔레그램 chat_id를 입력하세요 (필수, 숫자): ").strip()
    if not tg_chat_id:
        print("텔레그램 chat_id는 필수입니다. @userinfobot 에서 확인하세요.")
        sys.exit(1)

    reg_email = getattr(args, 'email', None) or invite.get("email", "")
    if not reg_email:
        reg_email = input("이메일을 입력하세요 (필수): ").strip()
    if not reg_email:
        print("이메일은 필수입니다.")
        sys.exit(1)

    # 이미 등록된 worker_id 체크
    resp2 = requests.get(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        params={"select": "worker_id", "worker_id": f"eq.{worker_id}"},
        headers=headers,
        timeout=15,
    )
    if resp2.ok and resp2.json():
        print(f"'{worker_id}'는 이미 등록되어 있습니다. 다른 ID를 사용하세요.")
        sys.exit(1)

    # 3. 토큰 생성 + 워커 등록 (토큰은 여기서만 표시)
    new_token = str(_uuid.uuid4())

    # 시스템 정보 수집
    ram_gb = 0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        pass

    gpu_info = None
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_info = r.stdout.strip().split("\n")[0]
    except Exception:
        pass

    worker_data = {
        "worker_id": worker_id,
        "worker_name": worker_name,
        "tier": tier,
        "status": "offline",
        "is_main_brain": tier == "owner",
        "api_token": new_token,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "ram_gb": ram_gb or None,
        "cpu_info": platform.processor() or None,
        "gpu_info": gpu_info,
        "python_version": platform.python_version(),
        "telegram_chat_id": tg_chat_id,
        "email": reg_email,
        "notes": f"자가등록: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')} (초대 {invite_code})",
    }

    resp3 = requests.post(
        f"{SUPABASE_URL}/rest/v1/compute_workers",
        json=worker_data,
        headers={**headers, "Prefer": "return=minimal"},
        timeout=15,
    )

    if resp3.status_code >= 300:
        print(f"등록 실패: {resp3.text}")
        sys.exit(1)

    # 4. 초대코드 사용 처리
    requests.patch(
        f"{SUPABASE_URL}/rest/v1/worker_invites",
        params={"invite_code": f"eq.{invite_code}"},
        json={
            "used_at": datetime.now(KST).isoformat(),
            "used_by_worker_id": worker_id,
        },
        headers={**headers, "Prefer": "return=minimal"},
        timeout=15,
    )

    # 5. 결과 출력 — 토큰은 이 순간에만 표시
    print()
    print("=" * 60)
    print("  등록 완료!")
    print("=" * 60)
    print()
    print(f"  워커 ID:    {worker_id}")
    print(f"  이름:       {worker_name}")
    print(f"  티어:       {tier}")
    print(f"  플랫폼:     {platform.system()} {platform.machine()}")
    if ram_gb:
        print(f"  RAM:        {ram_gb}GB")
    if gpu_info:
        print(f"  GPU:        {gpu_info}")
    print()
    print("  *** 아래 토큰을 .env 파일에 저장하세요 ***")
    print(f"  WORKER_TOKEN={new_token}")
    print()
    print("  이 토큰은 다시 표시되지 않습니다.")
    print("  분실 시 관리자에게 재등록을 요청하세요.")
    print()
    print(f"  워커 실행: python -m scalp_ml.worker --worker-id \"{worker_id}\" --tier {tier}")
    print("=" * 60)


def cmd_contacts(args):
    """등록된 연락처 목록 조회 (모든 티어 가능)"""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if WORKER_TOKEN:
        headers["x-worker-token"] = WORKER_TOKEN

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "name,role,aliases",
                "is_active": "eq.true",
                "order": "role.asc,name.asc"},
        headers=headers, timeout=15,
    )

    if not resp.ok:
        print(f"조회 실패: {resp.status_code} {resp.text}")
        return

    contacts = resp.json()
    if not contacts:
        print("등록된 연락처가 없습니다")
        return

    print(f"\n{'이름':<16} {'별명':<16} {'역할'}")
    print("-" * 50)
    for c in contacts:
        aliases = ", ".join(c.get("aliases") or []) or "-"
        print(f"{c['name']:<16} {aliases:<16} {c['role']}")
    print(f"\n총 {len(contacts)}명")


def cmd_send_msg(args):
    """등록된 사용자에게 텔레그램 메시지 발송"""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if WORKER_TOKEN:
        headers["x-worker-token"] = WORKER_TOKEN

    # 수신자 조회 (이름 또는 별명)
    target = None
    for param_key, param_val in [
        ("name", f"eq.{args.to}"),
        ("aliases", f"cs.{{{args.to}}}"),
    ]:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/telegram_contacts",
            params={"select": "chat_id,name,role",
                    param_key: param_val, "is_active": "eq.true"},
            headers=headers, timeout=10,
        )
        if resp.ok and resp.json():
            target = resp.json()[0]
            break

    if not target:
        print(f"'{args.to}'을(를) 찾을 수 없습니다. --contacts 로 목록을 확인하세요.")
        return

    # 발신자 이름 조회
    sender_name = args.worker_id or platform.node()
    my_resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_contacts",
        params={"select": "name",
                "or": f"(worker_id.eq.{sender_name},name.eq.{sender_name})",
                "is_active": "eq.true"},
        headers=headers, timeout=10,
    )
    if my_resp.ok and my_resp.json():
        sender_name = my_resp.json()[0]["name"]

    # 텔레그램 발송 (봇 토큰은 .env에 있는 경우만)
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not tg_token:
        # 봇 토큰 없으면 DB에 메시지만 기록 (listener가 전달)
        requests.post(
            f"{SUPABASE_URL}/rest/v1/telegram_messages",
            json={"chat_id": target["chat_id"], "direction": "incoming",
                  "message": f"[{sender_name}] {args.message}",
                  "worker_name": sender_name},
            headers={**headers, "Prefer": "return=minimal"},
            timeout=10,
        )
        print(f"메시지 저장 완료 (봇 토큰 없음 — listener가 전달 예정)")
        return

    r = requests.post(
        f"https://api.telegram.org/bot{tg_token}/sendMessage",
        json={"chat_id": target["chat_id"],
              "text": f"[{sender_name}] {args.message}"},
        timeout=15,
    )
    if r.ok:
        print(f"전송 완료: {target['name']} [{target['role']}]")
        # 기록
        requests.post(
            f"{SUPABASE_URL}/rest/v1/telegram_messages",
            json={"chat_id": target["chat_id"], "direction": "incoming",
                  "message": f"[{sender_name}] {args.message}",
                  "worker_name": sender_name},
            headers={**headers, "Prefer": "return=minimal"},
            timeout=10,
        )
    else:
        print(f"전송 실패: {r.text}")


def cmd_my_inbox(args):
    """내 수신 메시지 확인"""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if WORKER_TOKEN:
        headers["x-worker-token"] = WORKER_TOKEN

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/telegram_messages",
        params={"select": "worker_name,message,is_read,created_at",
                "direction": "eq.incoming",
                "order": "created_at.desc",
                "limit": str(args.limit or 20)},
        headers=headers, timeout=15,
    )

    if not resp.ok:
        print(f"조회 실패: {resp.status_code}")
        return

    messages = resp.json()
    if not messages:
        print("수신된 메시지가 없습니다")
        return

    print(f"\n{'시간':<18} {'발신자':<16} {'메시지'}")
    print("-" * 60)
    for m in messages:
        ts = datetime.fromisoformat(m["created_at"]).astimezone(KST).strftime("%m-%d %H:%M")
        sender = m.get("worker_name") or "?"
        text = m["message"][:40] + ("..." if len(m["message"]) > 40 else "")
        new = " *" if not m.get("is_read") else ""
        print(f"{ts:<18} {sender:<16} {text}{new}")


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
    parser.add_argument("--sync", action="store_true",
                        help="결과를 즉시 동기화하고 종료 (수동 전송)")
    parser.add_argument("--register", action="store_true",
                        help="초대코드로 자가등록 (토큰 자동 발급)")
    parser.add_argument("--invite-code", default=None,
                        help="관리자로부터 받은 초대코드")
    parser.add_argument("--telegram-id", default=None,
                        help="텔레그램 chat_id (숫자, @userinfobot에서 확인)")
    parser.add_argument("--email", default=None,
                        help="이메일 주소")
    parser.add_argument("--contacts", action="store_true",
                        help="등록된 연락처 목록 조회")
    parser.add_argument("--msg-to", default=None,
                        help="메시지 수신자 (이름 또는 별명)")
    parser.add_argument("--message", "-m", default=None,
                        help="전송할 메시지 내용")
    parser.add_argument("--inbox", action="store_true",
                        help="내 수신 메시지 확인")
    parser.add_argument("--limit", type=int, default=20,
                        help="inbox 표시 개수 (기본 20)")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL / SUPABASE_ANON_KEY (또는 SERVICE_ROLE_KEY) 미설정")
        sys.exit(1)

    # ── 자가등록 모드 ──
    if args.register:
        self_register(args)
        return

    # ── 연락처 조회 ──
    if args.contacts:
        cmd_contacts(args)
        return

    # ── 메시지 발송 ──
    if args.msg_to:
        if not args.message:
            print("메시지 내용이 필요합니다: -m \"내용\"")
            sys.exit(1)
        args.to = args.msg_to
        cmd_send_msg(args)
        return

    # ── 수신함 ──
    if args.inbox:
        cmd_my_inbox(args)
        return

    # Non-owner: ANON_KEY + WORKER_TOKEN 필수
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY") and not WORKER_TOKEN:
        log.error("WORKER_TOKEN 미설정 — 초대코드로 등록하세요: --register --invite-code <코드>")
        sys.exit(1)

    if args.sync:
        # 수동 동기화 모드: 프리뷰 → 안내 → 확인 → 업로드
        worker = MLWorker(
            worker_id=args.worker_id,
            tier=args.tier,
            worker_name=args.worker_name,
        )
        worker.register()
        if not worker.running:
            sys.exit(1)

        previews = worker.preview_sync()

        if not previews:
            print("\n  동기화할 새 결과가 없습니다.\n")
            worker.go_offline()
            return

        # ── 데이터 프리뷰 ──
        print()
        print("=" * 60)
        print("  결과 동기화 미리보기")
        print("=" * 60)
        print()
        print(f"  워커 ID: {worker.worker_id}")
        print(f"  티어: {worker.tier}")
        print(f"  대상 파일: {len(previews)}건")
        print()

        total_kb = 0
        for i, p in enumerate(previews, 1):
            total_kb += p["size_kb"]
            print(f"  [{i}] {p['file']}")
            print(f"      유형: {p['task_type']}  |  크기: {p['size_kb']}KB  |  생성: {p['created']}")
            # 데이터 키 미리보기 (어떤 정보가 포함되는지)
            safe_keys = [k for k in p["keys"] if k not in ("_path", "_data")]
            print(f"      포함 항목: {', '.join(safe_keys[:10])}")
            print()

        print("-" * 60)
        print(f"  총 {len(previews)}건, {total_kb:.1f}KB")
        print("-" * 60)

        # ── 안전 안내 ──
        print()
        print("  [데이터 안전 안내]")
        print()
        print("  - 전송되는 데이터: 백테스트/학습 결과 (수치 데이터만)")
        print("  - 전송되지 않는 데이터:")
        print("      API 키, 비밀번호, 개인정보, .env 파일,")
        print("      거래소 계정 정보, 잔고/포트폴리오 정보")
        print("  - 전송 후에도 로컬 파일은 그대로 유지됩니다 (삭제 안 됨)")
        print("  - 전송 대상: 프로젝트 공유 DB (Supabase)")
        print("  - 전송된 데이터는 관리자가 조회할 수 있습니다")
        print("  - 동기화를 취소해도 워커 실행에 영향 없습니다")
        print()
        print("-" * 60)

        # ── 확인 ──
        try:
            answer = input("  동기화를 진행하시겠습니까? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer not in ("y", "yes"):
            print("\n  동기화를 취소했습니다. 로컬 데이터에 변경 없음.\n")
            worker.go_offline()
            return

        uploaded, total = worker.do_sync(previews)
        print()
        if uploaded > 0:
            print(f"  📤 동기화 완료: {uploaded}/{total}건 업로드")
        else:
            print(f"  ⚠️  업로드 실패: 0/{total}건")
        print(f"  로컬 파일은 그대로 유지됩니다.\n")
        worker.go_offline()
        return

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
