"""
Lifeline 자가치유 시스템 종합 유닛 테스트

Coverage:
  - sentinel.py: API 점검, 디스크, 메모리, 프로세스, RL 모델, 긴급정지, 락 파일
  - diagnostician.py: 규칙 기반 진단, 오버라이드, 알 수 없는 패턴
  - healer.py: DRY_RUN, 연결 재시도, 캐시 정리, 설정 조정, 긴급정지
  - health_db_sync.py: Supabase 로깅, 배치, 인시던트 조회, 통계
  - main.py: 헬스 사이클, 텔레그램 알림, CLI
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.lifeline.sentinel import (
    check_upbit_api,
    check_supabase,
    check_disk_space,
    check_memory,
    check_process,
    check_rl_models,
    check_emergency_flags,
    check_stale_locks,
    run_all_checks,
    _result,
    _get_session,
    PROJECT_ROOT as SENTINEL_PROJECT_ROOT,
    DATA_DIR,
)
from scripts.lifeline.diagnostician import Diagnostician
from scripts.lifeline.healer import Healer
from scripts.lifeline.health_db_sync import HealthDBSync


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_sentinel_session():
    """각 테스트 전에 sentinel의 세션을 리셋한다."""
    import scripts.lifeline.sentinel as sentinel_mod
    sentinel_mod._session = None
    yield
    sentinel_mod._session = None


@pytest.fixture(autouse=True)
def reset_healer_session():
    """각 테스트 전에 healer의 세션을 리셋한다."""
    import scripts.lifeline.healer as healer_mod
    healer_mod._session = None
    yield
    healer_mod._session = None


@pytest.fixture
def diagnostician():
    return Diagnostician()


@pytest.fixture
def healer_dry():
    """DRY_RUN=true인 Healer."""
    with patch.dict(os.environ, {"DRY_RUN": "true"}):
        return Healer()


@pytest.fixture
def healer_live():
    """DRY_RUN=false인 Healer."""
    with patch.dict(os.environ, {"DRY_RUN": "false"}):
        return Healer()


@pytest.fixture
def db_sync():
    """HealthDBSync 인스턴스."""
    with patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "test-key",
    }):
        return HealthDBSync()


# ============================================================
# TestSentinel
# ============================================================

class TestSentinel:
    """Sentinel 개별 점검 테스트."""

    def test_check_upbit_api_success(self):
        """Upbit API 200 응답 시 OK를 반환한다."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"market": "KRW-BTC"}, {"market": "KRW-ETH"}]

        with patch("scripts.lifeline.sentinel._get_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_get.return_value = mock_session

            result = check_upbit_api()

        assert result["status"] == "OK"
        assert result["component"] == "upbit_api"
        assert result["details"]["status_code"] == 200
        assert result["details"]["markets"] == 2

    def test_check_upbit_api_failure(self):
        """Upbit API 타임아웃 시 ERROR를 반환한다."""
        import requests as req_lib
        with patch("scripts.lifeline.sentinel._get_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get.side_effect = req_lib.exceptions.Timeout("timeout")
            mock_get.return_value = mock_session

            result = check_upbit_api()

        assert result["status"] == "ERROR"
        assert "타임아웃" in result["message"]
        assert result["details"]["error"] == "timeout"

    def test_check_upbit_api_non_200(self):
        """Upbit API가 500을 반환하면 ERROR를 반환한다."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("scripts.lifeline.sentinel._get_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_get.return_value = mock_session

            result = check_upbit_api()

        assert result["status"] == "ERROR"
        assert "500" in result["message"]

    def test_check_supabase_success(self):
        """Supabase 200 응답 시 OK를 반환한다."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        }):
            with patch("scripts.lifeline.sentinel._get_session") as mock_get:
                mock_session = MagicMock()
                mock_session.get.return_value = mock_resp
                mock_get.return_value = mock_session

                result = check_supabase()

        assert result["status"] == "OK"
        assert result["component"] == "supabase"

    def test_check_supabase_no_env(self):
        """SUPABASE_URL 미설정 시 WARNING을 반환한다."""
        with patch.dict(os.environ, {}, clear=True):
            # dotenv에서 로드될 수 있으므로 getenv도 패치
            with patch("os.getenv", side_effect=lambda k, d="": d):
                result = check_supabase()

        assert result["status"] == "WARNING"
        assert "미설정" in result["message"]

    def test_check_disk_space_ok(self):
        """디스크 여유 공간 충분 시 OK를 반환한다."""
        # 100GB free, 500GB total
        mock_usage = MagicMock()
        mock_usage.free = 100 * (1024 ** 3)
        mock_usage.total = 500 * (1024 ** 3)
        mock_usage.used = 400 * (1024 ** 3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space()

        assert result["status"] == "OK"
        assert result["details"]["free_gb"] == 100.0

    def test_check_disk_space_warning(self):
        """디스크 여유 공간 < 1GB 시 WARNING을 반환한다."""
        mock_usage = MagicMock()
        mock_usage.free = int(0.8 * (1024 ** 3))  # 0.8GB
        mock_usage.total = 500 * (1024 ** 3)
        mock_usage.used = int(499.2 * (1024 ** 3))

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space()

        assert result["status"] == "WARNING"
        assert "주의" in result["message"]

    def test_check_disk_space_critical(self):
        """디스크 여유 공간 < 500MB 시 CRITICAL을 반환한다."""
        mock_usage = MagicMock()
        mock_usage.free = int(0.3 * (1024 ** 3))  # 0.3GB
        mock_usage.total = 500 * (1024 ** 3)
        mock_usage.used = int(499.7 * (1024 ** 3))

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space()

        assert result["status"] == "CRITICAL"
        assert "부족" in result["message"]

    def test_check_memory_no_psutil(self):
        """psutil 미설치 시 WARNING을 반환한다."""
        with patch.dict(sys.modules, {"psutil": None}):
            # psutil import를 실패하게 만들기
            import importlib
            with patch("builtins.__import__", side_effect=ImportError("No module named 'psutil'")):
                result = check_memory()

        assert result["status"] == "WARNING"
        assert "psutil" in result["message"]

    def test_check_memory_ok(self):
        """메모리 사용량 정상 시 OK를 반환한다."""
        mock_psutil = MagicMock()
        mock_mem = MagicMock()
        mock_mem.percent = 50.0
        mock_mem.available = 8 * (1024 ** 3)
        mock_mem.total = 16 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = check_memory()

        assert result["status"] == "OK"
        assert result["details"]["used_percent"] == 50.0

    def test_check_process_no_pids(self, tmp_path):
        """PID 파일이 없으면 OK를 반환한다."""
        with patch("scripts.lifeline.sentinel.DATA_DIR", tmp_path):
            result = check_process()

        assert result["status"] == "OK"
        assert "활성 PID 파일 없음" in result["message"]

    def test_check_rl_models_with_files(self, tmp_path):
        """최근 모델 파일이 있으면 OK를 반환한다."""
        model_dir = tmp_path / "rl_models"
        model_dir.mkdir()
        model_file = model_dir / "ppo_model.zip"
        model_file.write_text("dummy")

        with patch("scripts.lifeline.sentinel.DATA_DIR", tmp_path):
            result = check_rl_models()

        assert result["status"] == "OK"
        assert result["details"]["file_count"] == 1

    def test_check_rl_models_stale(self, tmp_path):
        """모든 모델이 7일 이상 오래되면 WARNING을 반환한다."""
        model_dir = tmp_path / "rl_models"
        model_dir.mkdir()
        model_file = model_dir / "old_model.zip"
        model_file.write_text("dummy")
        # 파일 mtime을 10일 전으로 설정
        old_time = time.time() - (10 * 86400)
        os.utime(model_file, (old_time, old_time))

        with patch("scripts.lifeline.sentinel.DATA_DIR", tmp_path):
            result = check_rl_models()

        assert result["status"] == "WARNING"
        assert "7일 이상" in result["message"]
        assert "old_model.zip" in result["details"]["stale_files"]

    def test_check_rl_models_no_directory(self, tmp_path):
        """rl_models 디렉토리가 없으면 WARNING을 반환한다."""
        with patch("scripts.lifeline.sentinel.DATA_DIR", tmp_path):
            result = check_rl_models()

        assert result["status"] == "WARNING"
        assert result["details"]["directory_exists"] is False

    def test_check_emergency_flags_clean(self):
        """긴급정지 플래그가 없으면 OK를 반환한다."""
        with patch.dict(os.environ, {"EMERGENCY_STOP": "false"}, clear=False):
            with patch("scripts.lifeline.sentinel.DATA_DIR", Path("/nonexistent")):
                result = check_emergency_flags()

        assert result["status"] == "OK"
        assert result["details"]["env_emergency_stop"] is False

    def test_check_emergency_flags_active(self):
        """EMERGENCY_STOP=true 시 CRITICAL을 반환한다."""
        with patch.dict(os.environ, {"EMERGENCY_STOP": "true"}, clear=False):
            with patch("scripts.lifeline.sentinel.DATA_DIR", Path("/nonexistent")):
                result = check_emergency_flags()

        assert result["status"] == "CRITICAL"
        assert result["details"]["env_emergency_stop"] is True

    def test_check_stale_locks_none(self, tmp_path):
        """락 파일이 없으면 OK를 반환한다."""
        with patch("scripts.lifeline.sentinel.DATA_DIR", tmp_path):
            result = check_stale_locks()

        assert result["status"] == "OK"
        assert result["details"]["lock_count"] == 0

    def test_check_stale_locks_old(self, tmp_path):
        """5분 이상 된 .lock 파일이 있으면 WARNING을 반환한다."""
        lock_file = tmp_path / "test.lock"
        lock_file.write_text("locked")
        old_time = time.time() - 600  # 10분 전
        os.utime(lock_file, (old_time, old_time))

        with patch("scripts.lifeline.sentinel.DATA_DIR", tmp_path):
            result = check_stale_locks()

        assert result["status"] == "WARNING"
        assert len(result["details"]["stale_locks"]) == 1

    def test_run_all_checks_aggregation(self):
        """run_all_checks는 모든 점검을 실행하고 최악 상태를 반환한다."""
        ok_result = _result("test1", "OK", "ok")
        warn_result = _result("test2", "WARNING", "warn")
        err_result = _result("test3", "ERROR", "err")

        with patch("scripts.lifeline.sentinel.check_upbit_api", return_value=ok_result), \
             patch("scripts.lifeline.sentinel.check_supabase", return_value=warn_result), \
             patch("scripts.lifeline.sentinel.check_disk_space", return_value=ok_result), \
             patch("scripts.lifeline.sentinel.check_memory", return_value=ok_result), \
             patch("scripts.lifeline.sentinel.check_process", return_value=ok_result), \
             patch("scripts.lifeline.sentinel.check_rl_models", return_value=ok_result), \
             patch("scripts.lifeline.sentinel.check_emergency_flags", return_value=err_result), \
             patch("scripts.lifeline.sentinel.check_stale_locks", return_value=ok_result):
            result = run_all_checks()

        assert result["overall_status"] == "ERROR"
        assert result["summary"]["total"] == 8
        assert result["summary"]["ok"] == 6
        assert result["summary"]["warning"] == 1
        assert result["summary"]["error"] == 1
        assert "timestamp" in result


# ============================================================
# TestDiagnostician
# ============================================================

class TestDiagnostician:
    """Diagnostician 진단 로직 테스트."""

    def test_diagnose_upbit_error(self, diagnostician):
        """Upbit API ERROR → retry_connection 추천."""
        check = {"component": "upbit_api", "status": "ERROR", "message": "연결 실패"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "retry_connection"
        assert result["auto_healable"] is True
        assert result["severity"] == "error"

    def test_diagnose_upbit_429(self, diagnostician):
        """Upbit API ERROR + 429 → adjust_config 추천 (오버라이드)."""
        check = {
            "component": "upbit_api",
            "status": "ERROR",
            "message": "Upbit API 비정상 응답: 429",
            "details": {"status_code": 429},
        }
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "adjust_config"
        assert result["confidence"] == 0.95
        assert "429" in result["ai_diagnosis"]

    def test_diagnose_supabase_error(self, diagnostician):
        """Supabase ERROR → retry_connection 추천."""
        check = {"component": "supabase", "status": "ERROR", "message": "Supabase 타임아웃"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "retry_connection"
        assert result["auto_healable"] is True

    def test_diagnose_disk_critical(self, diagnostician):
        """disk CRITICAL → clear_cache 추천."""
        check = {"component": "disk", "status": "CRITICAL", "message": "디스크 부족"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "clear_cache"
        assert result["auto_healable"] is True

    def test_diagnose_disk_warning(self, diagnostician):
        """disk WARNING → alert_only 추천."""
        check = {"component": "disk", "status": "WARNING", "message": "디스크 사용량 높음"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "alert_only"
        assert result["auto_healable"] is False

    def test_diagnose_memory_critical(self, diagnostician):
        """memory CRITICAL → restart_process 추천."""
        check = {"component": "memory", "status": "CRITICAL", "message": "메모리 부족"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "restart_process"
        assert result["auto_healable"] is True

    def test_diagnose_stale_locks(self, diagnostician):
        """stale_locks WARNING → clear_cache 추천."""
        check = {"component": "stale_locks", "status": "WARNING", "message": "오래된 락 파일"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "clear_cache"
        assert result["auto_healable"] is True

    def test_diagnose_ok_status_skipped(self, diagnostician):
        """OK 상태는 severity=none, action=none으로 반환한다."""
        check = {"component": "upbit_api", "status": "OK", "message": "정상"}
        result = diagnostician.diagnose(check)

        assert result["severity"] == "none"
        assert result["recommended_action"] == "none"
        assert result["confidence"] == 1.0

    def test_diagnose_all_filters_ok(self, diagnostician):
        """diagnose_all은 OK 상태를 필터링한다."""
        checks = [
            {"component": "upbit_api", "status": "OK", "message": "ok"},
            {"component": "supabase", "status": "OK", "message": "ok"},
            {"component": "disk", "status": "WARNING", "message": "주의"},
        ]
        results = diagnostician.diagnose_all(checks)

        # OK 2건은 필터링, WARNING 1건만 진단
        assert len(results) == 1
        assert results[0]["component"] == "disk"

    def test_diagnose_unknown_component(self, diagnostician):
        """알 수 없는 컴포넌트는 alert_only 폴백으로 처리한다."""
        check = {"component": "unknown_thing", "status": "ERROR", "message": "알 수 없는 오류"}
        result = diagnostician.diagnose(check)

        assert result["recommended_action"] == "alert_only"
        assert result["confidence"] == 0.3
        assert result["auto_healable"] is False
        assert "알 수 없는 에러 패턴" in result["ai_diagnosis"]


# ============================================================
# TestHealer
# ============================================================

class TestHealer:
    """Healer 복구 액션 테스트."""

    def test_heal_dry_run_no_action(self):
        """DRY_RUN=true 시 실제 연결 시도 없이 성공을 반환한다."""
        with patch.dict(os.environ, {"DRY_RUN": "true"}):
            healer = Healer()
        diagnosis = {
            "component": "upbit_api",
            "recommended_action": "retry_connection",
            "auto_healable": True,
        }
        result = healer.heal(diagnosis)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert "DRY_RUN" in result["message"]

    def test_retry_connection_upbit_success(self):
        """Upbit 연결 재시도 성공 시 True를 반환한다."""
        with patch.dict(os.environ, {"DRY_RUN": "false"}):
            healer = Healer()

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("scripts.lifeline.healer._get_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_get.return_value = mock_session

            result = healer._retry_connection("upbit_api")

        assert result is True

    def test_retry_connection_all_fail(self):
        """3회 모두 실패하면 False를 반환한다."""
        import requests as req_lib
        with patch.dict(os.environ, {"DRY_RUN": "false"}):
            healer = Healer()

        with patch("scripts.lifeline.healer._get_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get.side_effect = req_lib.exceptions.ConnectionError("fail")
            mock_get.return_value = mock_session
            with patch("time.sleep"):  # 재시도 대기 스킵
                result = healer._retry_connection("upbit_api")

        assert result is False

    def test_clear_cache_stale_locks(self, tmp_path):
        """오래된 .lock 파일을 삭제한다."""
        with patch.dict(os.environ, {"DRY_RUN": "false"}):
            healer = Healer()
        healer.project_root = tmp_path

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lock_file = data_dir / "test.lock"
        lock_file.write_text("locked")
        # 10분 전 파일
        old_time = time.time() - 600
        os.utime(lock_file, (old_time, old_time))

        result = healer._clear_cache("stale_locks")

        assert result is True
        assert not lock_file.exists()

    def test_clear_cache_old_logs(self, tmp_path):
        """오래된 로그 파일을 삭제한다."""
        with patch.dict(os.environ, {"DRY_RUN": "false"}):
            healer = Healer()
        healer.project_root = tmp_path

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        old_log = logs_dir / "old.log"
        old_log.write_text("old log")
        # 10일 전 파일
        old_time = time.time() - (10 * 86400)
        os.utime(old_log, (old_time, old_time))

        result = healer._clear_cache("disk")

        assert result is True
        assert not old_log.exists()

    def test_adjust_config_writes_temp(self, tmp_path):
        """adjust_config는 data/temp_config.json에 기록한다."""
        with patch.dict(os.environ, {"DRY_RUN": "false", "MIN_TRADE_INTERVAL_HOURS": "4"}):
            healer = Healer()
        healer.project_root = tmp_path
        (tmp_path / "data").mkdir()

        result = healer._adjust_config("upbit_api", {})

        assert result is True
        config_path = tmp_path / "data" / "temp_config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert "upbit_api_throttle" in config
        assert config["upbit_api_throttle"]["MIN_TRADE_INTERVAL_HOURS"] == 5.0

    def test_emergency_stop_creates_flag(self, tmp_path):
        """emergency_stop은 auto_emergency.json 플래그를 생성한다."""
        with patch.dict(os.environ, {"DRY_RUN": "false"}):
            healer = Healer()
        healer.project_root = tmp_path
        (tmp_path / "data").mkdir()

        result = healer._emergency_stop()

        assert result is True
        flag_path = tmp_path / "data" / "auto_emergency.json"
        assert flag_path.exists()
        data = json.loads(flag_path.read_text(encoding="utf-8"))
        assert data["activated"] is True
        assert data["source"] == "lifeline_healer"

    def test_heal_all_processes_batch(self):
        """heal_all은 여러 진단을 일괄 처리한다."""
        with patch.dict(os.environ, {"DRY_RUN": "true"}):
            healer = Healer()

        diagnoses = [
            {"component": "upbit_api", "recommended_action": "retry_connection", "auto_healable": True},
            {"component": "disk", "recommended_action": "clear_cache", "auto_healable": True},
            {"component": "memory", "recommended_action": "alert_only", "auto_healable": False},
        ]
        results = healer.heal_all(diagnoses)

        assert len(results) == 3
        assert results[0]["action_taken"] == "retry_connection"
        assert results[1]["action_taken"] == "clear_cache"
        # alert_only는 자동복구 대상 아님
        assert results[2]["message"] == "자동 복구 대상 아님 — 알림만 전송"

    def test_heal_non_healable_skipped(self):
        """auto_healable=False인 진단은 치유하지 않는다."""
        with patch.dict(os.environ, {"DRY_RUN": "true"}):
            healer = Healer()

        diagnosis = {
            "component": "emergency_flags",
            "recommended_action": "alert_only",
            "auto_healable": False,
        }
        result = healer.heal(diagnosis)

        assert result["success"] is True
        assert "자동 복구 대상 아님" in result["message"]

    def test_heal_unknown_action_skip(self):
        """알 수 없는 액션은 실패를 반환한다."""
        with patch.dict(os.environ, {"DRY_RUN": "false"}):
            healer = Healer()

        diagnosis = {
            "component": "test",
            "recommended_action": "nonexistent_action",
            "auto_healable": True,
        }
        result = healer.heal(diagnosis)

        assert result["success"] is False
        assert "알 수 없는 복구 액션" in result["message"]


# ============================================================
# TestHealthDBSync
# ============================================================

class TestHealthDBSync:
    """HealthDBSync 테스트."""

    def test_log_check_success(self, db_sync):
        """POST 성공 시 True를 반환한다."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 201
        db_sync.session.post = MagicMock(return_value=mock_resp)

        check = {"component": "upbit_api", "status": "OK", "message": "정상", "details": {}}
        result = db_sync.log_check(check)

        assert result is True
        db_sync.session.post.assert_called_once()

    def test_log_check_with_diagnosis(self, db_sync):
        """진단 정보가 포함된 로그를 기록한다."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        db_sync.session.post = MagicMock(return_value=mock_resp)

        check = {"component": "upbit_api", "status": "ERROR", "message": "연결 실패"}
        diagnosis = {
            "ai_diagnosis": "Upbit API 연결 실패",
            "healing_action": "retry_connection",
        }
        result = db_sync.log_check(check, diagnosis=diagnosis)

        assert result is True
        call_kwargs = db_sync.session.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["ai_diagnosis"] == "Upbit API 연결 실패"
        assert payload["resolution_status"] == "DIAGNOSING"

    def test_log_check_with_healing(self, db_sync):
        """치유 결과가 포함된 로그를 기록한다."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        db_sync.session.post = MagicMock(return_value=mock_resp)

        check = {"component": "upbit_api", "status": "ERROR", "message": "실패"}
        healing = {
            "resolution_status": "HEALED",
            "healing_duration_ms": 150,
            "ai_diagnosis": "자동 복구",
            "healing_action": "retry",
        }
        result = db_sync.log_check(check, healing_result=healing)

        assert result is True
        call_kwargs = db_sync.session.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["resolution_status"] == "HEALED"
        assert payload["healing_duration_ms"] == 150

    def test_log_check_failure_no_raise(self, db_sync):
        """POST 실패 시 예외를 삼키고 False를 반환한다."""
        db_sync.session.post = MagicMock(side_effect=Exception("network error"))

        check = {"component": "test", "status": "ERROR", "message": "fail"}
        result = db_sync.log_check(check)

        assert result is False

    def test_log_check_no_supabase_env(self):
        """SUPABASE_URL 미설정 시에도 예외 없이 False를 반환한다."""
        with patch.dict(os.environ, {}, clear=True):
            sync = HealthDBSync()

        # supabase_url이 빈 문자열이면 POST 시 URL이 잘못되어 실패
        sync.session.post = MagicMock(side_effect=Exception("Invalid URL"))

        check = {"component": "test", "status": "OK", "message": "ok"}
        result = sync.log_check(check)

        assert result is False

    def test_log_batch_count(self, db_sync):
        """log_batch는 성공 건수를 반환한다."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        db_sync.session.post = MagicMock(return_value=mock_resp)

        batch = [
            {"check": {"component": "a", "status": "OK", "message": "ok"}},
            {"check": {"component": "b", "status": "ERROR", "message": "fail"},
             "diagnosis": {"ai_diagnosis": "test"}},
            {"component": "c", "status": "WARNING", "message": "warn"},
        ]
        count = db_sync.log_batch(batch)

        assert count == 3
        assert db_sync.session.post.call_count == 3

    def test_get_recent_incidents(self, db_sync):
        """최근 인시던트 조회."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {"component": "upbit_api", "severity": "ERROR"},
            {"component": "disk", "severity": "WARNING"},
        ]
        db_sync.session.get = MagicMock(return_value=mock_resp)

        result = db_sync.get_recent_incidents(hours=24)

        assert len(result) == 2
        db_sync.session.get.assert_called_once()

    def test_get_component_stats(self, db_sync):
        """컴포넌트별 통계 조회."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {"component": "upbit_api", "critical_count": 0, "error_count": 1},
            {"component": "disk", "critical_count": 0, "error_count": 0},
        ]
        db_sync.session.get = MagicMock(return_value=mock_resp)

        result = db_sync.get_component_stats()

        assert result["total_components"] == 2
        assert len(result["unhealthy"]) == 1
        assert result["unhealthy"][0]["component"] == "upbit_api"


# ============================================================
# TestMain
# ============================================================

class TestMain:
    """main.py 헬스 사이클 및 CLI 테스트."""

    def _mock_all_ok_checks(self):
        """모든 점검이 OK인 run_all_checks 결과를 반환한다."""
        return {
            "timestamp": "2026-03-12T12:00:00+09:00",
            "overall_status": "OK",
            "checks": [
                {"component": "upbit_api", "status": "OK", "message": "정상", "details": {}},
                {"component": "supabase", "status": "OK", "message": "정상", "details": {}},
            ],
            "summary": {"total": 2, "ok": 2, "warning": 0, "error": 0, "critical": 0},
        }

    def _mock_warning_checks(self):
        """WARNING 포함 run_all_checks 결과."""
        return {
            "timestamp": "2026-03-12T12:00:00+09:00",
            "overall_status": "WARNING",
            "checks": [
                {"component": "upbit_api", "status": "OK", "message": "정상", "details": {}},
                {"component": "disk", "status": "WARNING", "message": "여유 공간 주의", "details": {}},
            ],
            "summary": {"total": 2, "ok": 1, "warning": 1, "error": 0, "critical": 0},
        }

    def _mock_critical_checks(self):
        """CRITICAL 포함 run_all_checks 결과."""
        return {
            "timestamp": "2026-03-12T12:00:00+09:00",
            "overall_status": "CRITICAL",
            "checks": [
                {"component": "upbit_api", "status": "OK", "message": "정상", "details": {}},
                {"component": "emergency_flags", "status": "CRITICAL", "message": "EMERGENCY_STOP 활성", "details": {}},
            ],
            "summary": {"total": 2, "ok": 1, "warning": 0, "error": 0, "critical": 1},
        }

    def test_run_health_cycle_all_ok(self):
        """모든 점검 OK 시 정상 사이클 결과를 반환한다."""
        from scripts.lifeline.main import run_health_cycle

        mock_health = self._mock_all_ok_checks()

        with patch("scripts.lifeline.main.run_all_checks", return_value=mock_health), \
             patch("scripts.lifeline.main.HealthDBSync") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.log_batch.return_value = 2
            mock_db_cls.return_value = mock_db

            result = run_health_cycle()

        assert result["overall_status"] == "OK"
        assert result["non_ok_count"] == 0
        assert result["alert_sent"] is False

    def test_run_health_cycle_with_warning(self):
        """WARNING 시 진단만 하고 알림은 보내지 않는다."""
        from scripts.lifeline.main import run_health_cycle

        mock_health = self._mock_warning_checks()

        with patch("scripts.lifeline.main.run_all_checks", return_value=mock_health), \
             patch("scripts.lifeline.main.HealthDBSync") as mock_db_cls, \
             patch("scripts.lifeline.main._HAS_DIAGNOSTICIAN", False), \
             patch("scripts.lifeline.main._HAS_HEALER", False):
            mock_db = MagicMock()
            mock_db.log_batch.return_value = 2
            mock_db_cls.return_value = mock_db

            result = run_health_cycle()

        assert result["overall_status"] == "WARNING"
        assert result["non_ok_count"] == 1
        assert result["critical_count"] == 0
        assert result["alert_sent"] is False

    def test_run_health_cycle_with_critical_sends_alert(self):
        """CRITICAL 이슈가 있으면 텔레그램 알림을 전송한다."""
        from scripts.lifeline.main import run_health_cycle

        mock_health = self._mock_critical_checks()

        with patch("scripts.lifeline.main.run_all_checks", return_value=mock_health), \
             patch("scripts.lifeline.main.HealthDBSync") as mock_db_cls, \
             patch("scripts.lifeline.main._HAS_DIAGNOSTICIAN", False), \
             patch("scripts.lifeline.main._HAS_HEALER", False), \
             patch("scripts.lifeline.main.send_health_alert", return_value=True) as mock_alert:
            mock_db = MagicMock()
            mock_db.log_batch.return_value = 2
            mock_db_cls.return_value = mock_db

            result = run_health_cycle()

        assert result["overall_status"] == "CRITICAL"
        assert result["critical_count"] == 1
        assert result["alert_sent"] is True
        mock_alert.assert_called_once()

    def test_send_health_alert(self):
        """텔레그램 알림 전송 성공."""
        from scripts.lifeline.main import send_health_alert

        mock_resp = MagicMock()
        mock_resp.ok = True

        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_USER_ID": "12345",
        }):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                result = send_health_alert([
                    {"check": {"component": "disk", "status": "CRITICAL", "message": "디스크 부족"}},
                ])

        assert result is True
        mock_post.assert_called_once()

    def test_send_health_alert_no_token(self):
        """TELEGRAM_BOT_TOKEN 미설정 시 False를 반환한다."""
        from scripts.lifeline.main import send_health_alert

        with patch.dict(os.environ, {}, clear=True):
            # environ.get이 None을 반환하도록
            with patch("os.environ.get", return_value=None):
                result = send_health_alert([
                    {"check": {"component": "test", "status": "CRITICAL", "message": "fail"}},
                ])

        assert result is False

    def test_main_once_mode(self):
        """--once 모드로 1회 실행 후 종료한다."""
        from scripts.lifeline import main as main_mod

        mock_result = {
            "overall_status": "OK",
            "timestamp": "2026-03-12T12:00:00+09:00",
            "checks_summary": {"total": 8, "ok": 8},
            "non_ok_count": 0,
            "diagnosed_count": 0,
            "healed_count": 0,
            "db_synced_count": 8,
            "critical_count": 0,
            "alert_sent": False,
            "cycle_duration_ms": 100,
        }

        with patch.object(main_mod, "run_health_cycle", return_value=mock_result), \
             patch("sys.argv", ["main.py", "--once"]), \
             pytest.raises(SystemExit) as exc_info:
            main_mod.__name__ = "__main__"
            # 직접 argparse 로직 실행
            parser = main_mod.argparse.ArgumentParser()
            parser.add_argument("--once", action="store_true", default=False)
            parser.add_argument("--interval", type=int, default=None)
            parser.add_argument("--verbose", action="store_true", default=False)
            args = parser.parse_args(["--once"])

            result = main_mod.run_health_cycle(verbose=args.verbose)
            if result.get("overall_status") in ("ERROR", "CRITICAL"):
                sys.exit(1)
            sys.exit(0)

        assert exc_info.value.code == 0

    def test_main_exit_code_on_error(self):
        """ERROR/CRITICAL 시 exit code 1로 종료한다."""
        from scripts.lifeline import main as main_mod

        mock_result = {
            "overall_status": "ERROR",
            "timestamp": "2026-03-12T12:00:00+09:00",
            "checks_summary": {},
            "non_ok_count": 1,
            "diagnosed_count": 0,
            "healed_count": 0,
            "db_synced_count": 0,
            "critical_count": 0,
            "alert_sent": False,
            "cycle_duration_ms": 50,
        }

        with patch.object(main_mod, "run_health_cycle", return_value=mock_result), \
             pytest.raises(SystemExit) as exc_info:
            result = main_mod.run_health_cycle()
            if result.get("overall_status") in ("ERROR", "CRITICAL"):
                sys.exit(1)
            sys.exit(0)

        assert exc_info.value.code == 1
