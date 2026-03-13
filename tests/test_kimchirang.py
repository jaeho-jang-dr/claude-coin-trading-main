"""Kimchirang 유닛 테스트"""

import json
import os
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# 프로젝트 루트
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from kimchirang.config import KimchirangConfig, TradingParams, DBConfig
from kimchirang.kp_engine import KPSnapshot, KPEngine
from kimchirang.execution import (
    PositionSide, PositionState, LegResult, ExecutionResult, Executor,
)
from kimchirang.main import RLBridge, ACTION_HOLD, ACTION_ENTER, ACTION_EXIT
from kimchirang.db import KimchirangDB


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def config():
    with patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "test_key",
        "UPBIT_SECRET_KEY": "test_secret",
        "BINANCE_API_KEY": "test_binance",
        "BINANCE_API_SECRET": "test_binance_secret",
        "KR_DRY_RUN": "true",
        "KR_EMERGENCY_STOP": "false",
    }):
        return KimchirangConfig()


@pytest.fixture
def snapshot():
    return KPSnapshot(
        entry_kp=3.5,
        exit_kp=0.3,
        mid_kp=1.9,
        upbit_bid=85_000_000,
        upbit_ask=85_100_000,
        binance_bid=62_000.0,
        binance_ask=62_050.0,
        fx_rate=1_350.0,
        upbit_spread_pct=0.12,
        binance_spread_pct=0.08,
        funding_rate=0.0001,
        timestamp=time.time(),
    )


@pytest.fixture
def executor(config):
    with patch("kimchirang.state.load_position", return_value={
        "side": "none", "entry_kp": 0, "entry_time": 0,
        "upbit_qty": 0, "binance_qty": 0,
        "upbit_entry_price": 0, "binance_entry_price": 0,
        "trade_count_today": 0, "last_trade_time": 0,
    }):
        return Executor(config)


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def test_default_values(self, config):
        assert config.trading.dry_run is True
        assert config.trading.emergency_stop is False
        assert config.trading.leverage == 1
        assert config.trading.kp_entry_threshold == 3.0
        assert config.trading.kp_exit_threshold == 0.5

    def test_validate_missing_keys(self):
        with patch.dict(os.environ, {
            "UPBIT_ACCESS_KEY": "",
            "UPBIT_SECRET_KEY": "",
            "BINANCE_API_KEY": "",
            "BINANCE_API_SECRET": "",
        }, clear=False):
            c = KimchirangConfig()
            errors = c.validate()
            assert len(errors) >= 2

    def test_validate_leverage_limit(self, config):
        config.trading.leverage = 10
        errors = config.validate()
        assert any("레버리지" in e for e in errors)

    def test_summary_masks_keys(self, config):
        s = config.summary()
        assert "test_key" not in s
        assert "..." in s


# ============================================================
# KPSnapshot Tests
# ============================================================

class TestKPSnapshot:
    def test_valid_snapshot(self, snapshot):
        assert snapshot.is_valid is True

    def test_invalid_snapshot(self):
        s = KPSnapshot()
        assert s.is_valid is False

    def test_values(self, snapshot):
        assert snapshot.entry_kp == 3.5
        assert snapshot.fx_rate == 1350.0


# ============================================================
# PositionState Tests
# ============================================================

class TestPositionState:
    def test_is_open(self):
        p = PositionState()
        assert p.is_open is False
        p.side = PositionSide.LONG_KP
        assert p.is_open is True

    def test_hold_duration(self):
        p = PositionState(side=PositionSide.LONG_KP, entry_time=time.time() - 120)
        assert abs(p.hold_duration_min - 2.0) < 0.1

    def test_to_dict_from_dict_roundtrip(self):
        p = PositionState(
            side=PositionSide.LONG_KP,
            entry_kp=3.5,
            entry_time=1000.0,
            upbit_qty=0.001,
            binance_qty=0.001,
            upbit_entry_price=85_000_000,
            binance_entry_price=62_000.5,
            trade_count_today=2,
            last_trade_time=2000.0,
        )
        d = p.to_dict()
        p2 = PositionState.from_dict(d)
        assert p2.side == PositionSide.LONG_KP
        assert p2.entry_kp == 3.5
        assert p2.upbit_qty == 0.001
        assert p2.trade_count_today == 2

    def test_from_dict_invalid_side(self):
        p = PositionState.from_dict({"side": "invalid_value"})
        assert p.side == PositionSide.NONE


# ============================================================
# Executor Tests
# ============================================================

class TestExecutor:
    def test_safety_check_emergency_stop(self, executor):
        executor.config.trading.emergency_stop = True
        reason = executor._check_safety("enter")
        assert "EMERGENCY_STOP" in reason

    def test_safety_check_position_already_open(self, executor):
        executor.position.side = PositionSide.LONG_KP
        reason = executor._check_safety("enter")
        assert "포지션" in reason

    def test_safety_check_daily_limit(self, executor):
        executor.position.trade_count_today = 20
        reason = executor._check_safety("enter")
        assert "한도" in reason

    def test_safety_check_interval(self, executor):
        executor.position.last_trade_time = time.time()
        reason = executor._check_safety("enter")
        assert "간격" in reason

    def test_safety_check_exit_no_position(self, executor):
        reason = executor._check_safety("exit")
        assert "포지션" in reason

    @pytest.mark.asyncio
    async def test_enter_dry_run(self, executor, snapshot):
        with patch("kimchirang.state.save_position"):
            result = await executor.enter(snapshot)
            assert result.both_success
            assert result.dry_run
            assert executor.position.is_open

    @pytest.mark.asyncio
    async def test_exit_dry_run(self, executor, snapshot):
        # 먼저 진입
        with patch("kimchirang.state.save_position"):
            await executor.enter(snapshot)
            assert executor.position.is_open
            # 청산
            result = await executor.exit(snapshot)
            assert result.both_success
            assert not executor.position.is_open

    @pytest.mark.asyncio
    async def test_set_leverage_dry_run(self, executor):
        ok = await executor.set_leverage()
        assert ok is True

    def test_get_position_info(self, executor):
        info = executor.get_position_info()
        assert info["side"] == "none"
        assert info["is_open"] is False
        assert "trades_today" in info

    def test_calculate_pnl(self, executor, snapshot):
        executor.position.entry_kp = 3.5
        result = ExecutionResult(action="exit")
        pnl = executor._calculate_pnl(result, snapshot)
        # entry_kp(3.5) - exit_kp(0.3) - 수수료(0.18) = 3.02
        assert abs(pnl - 3.02) < 0.01


# ============================================================
# State Persistence Tests
# ============================================================

class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        from kimchirang import state as st
        original_file = st.STATE_FILE
        original_lock = st.LOCK_FILE

        st.STATE_FILE = tmp_path / "test_state.json"
        st.LOCK_FILE = st.STATE_FILE.with_suffix(".lock")
        try:
            data = {
                "side": "long_kp",
                "entry_kp": 3.5,
                "entry_time": 1000.0,
                "upbit_qty": 0.001,
                "binance_qty": 0.001,
                "upbit_entry_price": 85000000,
                "binance_entry_price": 62000.5,
                "trade_count_today": 2,
                "last_trade_time": 2000.0,
            }
            st.save_position(data)
            loaded = st.load_position()
            assert loaded["side"] == "long_kp"
            assert loaded["entry_kp"] == 3.5
            assert loaded["upbit_qty"] == 0.001
        finally:
            st.STATE_FILE = original_file
            st.LOCK_FILE = original_lock

    def test_load_missing_file(self, tmp_path):
        from kimchirang import state as st
        original_file = st.STATE_FILE
        original_lock = st.LOCK_FILE
        original_backup_file = getattr(st, 'BACKUP_FILE', None)
        original_memory = getattr(st, '_memory_backup', None)

        st.STATE_FILE = tmp_path / "nonexistent.json"
        st.LOCK_FILE = st.STATE_FILE.with_suffix(".lock")
        st.BACKUP_FILE = st.STATE_FILE.with_suffix(".bak")
        # 메모리 백업 초기화 (다른 테스트에서 오염 방지)
        st._memory_backup = {}
        try:
            loaded = st.load_position()
            assert loaded["side"] == "none"
        finally:
            st.STATE_FILE = original_file
            st.LOCK_FILE = original_lock
            if original_backup_file is not None:
                st.BACKUP_FILE = original_backup_file
            st._memory_backup = original_memory if original_memory else {}


# ============================================================
# Notifier Tests
# ============================================================

class TestNotifier:
    def test_notifier_disabled_without_env(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_USER_ID": ""}, clear=False):
            from importlib import reload
            import kimchirang.notifier as mod
            reload(mod)
            n = mod.KimchirangNotifier()
            assert n._enabled is False

    @pytest.mark.asyncio
    async def test_notify_error_does_not_crash(self):
        from kimchirang.notifier import KimchirangNotifier
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_USER_ID": ""}, clear=False):
            n = KimchirangNotifier()
            n._enabled = False
            # 비활성 상태에서 호출해도 예외 없이 통과
            await n.notify_error("test_phase", "test error")


# ============================================================
# DB Tests
# ============================================================

class TestDB:
    def test_db_disabled_without_config(self):
        from kimchirang.db import KimchirangDB
        from kimchirang.config import DBConfig
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            db = KimchirangDB(DBConfig())
            assert db._enabled is False

    @pytest.mark.asyncio
    async def test_record_trade_disabled_skips(self, snapshot):
        from kimchirang.db import KimchirangDB
        from kimchirang.config import DBConfig
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            db = KimchirangDB(DBConfig())
            result = ExecutionResult(action="enter")
            # 비활성 상태에서 호출해도 예외 없이 통과
            await db.record_trade(result, snapshot)

    @pytest.mark.asyncio
    async def test_record_trade_success(self, snapshot):
        from kimchirang.db import KimchirangDB
        from kimchirang.config import DBConfig
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test_key",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            mock_resp = MagicMock(status_code=201, text="")
            db._session.post = MagicMock(return_value=mock_resp)
            result = ExecutionResult(action="enter", kp_at_execution=3.5)
            await db.record_trade(result, snapshot, stats={"mid_kp": 1.9})
            db._session.post.assert_called_once()


# ============================================================
# ExecutionResult Tests
# ============================================================

class TestExecutionResult:
    def test_both_success(self):
        r = ExecutionResult(
            action="enter",
            upbit_leg=LegResult(exchange="upbit", success=True),
            binance_leg=LegResult(exchange="binance", success=True),
        )
        assert r.both_success is True
        assert r.partial_fill is False

    def test_partial_fill(self):
        r = ExecutionResult(
            action="enter",
            upbit_leg=LegResult(exchange="upbit", success=True),
            binance_leg=LegResult(exchange="binance", success=False),
        )
        assert r.both_success is False
        assert r.partial_fill is True

    def test_summary(self):
        r = ExecutionResult(
            action="enter",
            upbit_leg=LegResult(exchange="upbit", success=True, filled_price=85000000),
            binance_leg=LegResult(exchange="binance", success=True, filled_price=62000),
            kp_at_execution=3.5,
            dry_run=True,
        )
        s = r.summary()
        assert "DRY" in s
        assert "ENTER" in s
        assert "OK" in s


# ============================================================
# RLBridge 앙상블 Tests
# ============================================================

class TestRLBridge:
    """PPO + DQN 앙상블 RL 브릿지 테스트"""

    def test_mode_rule_when_disabled(self, config):
        config.rl.enabled = False
        bridge = RLBridge(config)
        bridge.load()
        assert bridge.mode == "rule"
        assert bridge.is_available is False

    def test_mode_rule_no_models(self, config):
        config.rl.enabled = True
        bridge = RLBridge(config)
        # load 없이도 rule 모드
        assert bridge.mode == "rule"
        assert bridge.is_available is False

    def test_get_action_hold_when_unavailable(self, config):
        bridge = RLBridge(config)
        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_HOLD

    def test_mode_ppo_only(self, config):
        config.rl.enabled = True
        bridge = RLBridge(config)
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(ACTION_ENTER), None)
        bridge._ppo_model = mock_ppo
        bridge._dqn_model = None
        bridge._available = True
        assert bridge.mode == "ppo"

    def test_mode_dqn_only(self, config):
        config.rl.enabled = True
        bridge = RLBridge(config)
        bridge._ppo_model = None
        bridge._dqn_model = MagicMock()
        bridge._available = True
        assert bridge.mode == "dqn"

    def test_mode_ensemble(self, config):
        config.rl.enabled = True
        bridge = RLBridge(config)
        bridge._ppo_model = MagicMock()
        bridge._dqn_model = MagicMock()
        bridge._available = True
        assert bridge.mode == "ensemble"

    def test_ensemble_agree_enter(self, config):
        """두 모델이 Enter에 합의하면 Enter"""
        bridge = RLBridge(config)
        ppo = MagicMock()
        ppo.predict.return_value = (np.array(ACTION_ENTER), None)
        dqn = MagicMock()
        dqn.predict.return_value = (np.array(ACTION_ENTER), None)
        bridge._ppo_model = ppo
        bridge._dqn_model = dqn
        bridge._available = True

        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_ENTER

    def test_ensemble_agree_exit(self, config):
        """두 모델이 Exit에 합의하면 Exit"""
        bridge = RLBridge(config)
        ppo = MagicMock()
        ppo.predict.return_value = (np.array(ACTION_EXIT), None)
        dqn = MagicMock()
        dqn.predict.return_value = (np.array(ACTION_EXIT), None)
        bridge._ppo_model = ppo
        bridge._dqn_model = dqn
        bridge._available = True

        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_EXIT

    def test_ensemble_disagree_hold(self, config):
        """두 모델이 불일치하면 Hold (신중)"""
        bridge = RLBridge(config)
        ppo = MagicMock()
        ppo.predict.return_value = (np.array(ACTION_ENTER), None)
        dqn = MagicMock()
        dqn.predict.return_value = (np.array(ACTION_EXIT), None)
        bridge._ppo_model = ppo
        bridge._dqn_model = dqn
        bridge._available = True

        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_HOLD

    def test_ensemble_disagree_enter_vs_hold(self, config):
        """Enter vs Hold 불일치 → Hold"""
        bridge = RLBridge(config)
        ppo = MagicMock()
        ppo.predict.return_value = (np.array(ACTION_ENTER), None)
        dqn = MagicMock()
        dqn.predict.return_value = (np.array(ACTION_HOLD), None)
        bridge._ppo_model = ppo
        bridge._dqn_model = dqn
        bridge._available = True

        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_HOLD

    def test_single_ppo_action(self, config):
        """PPO만 있으면 PPO 액션 그대로"""
        bridge = RLBridge(config)
        ppo = MagicMock()
        ppo.predict.return_value = (np.array(ACTION_ENTER), None)
        bridge._ppo_model = ppo
        bridge._dqn_model = None
        bridge._available = True

        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_ENTER

    def test_single_dqn_action(self, config):
        """DQN만 있으면 DQN 액션 그대로"""
        bridge = RLBridge(config)
        bridge._ppo_model = None
        dqn = MagicMock()
        dqn.predict.return_value = (np.array(ACTION_EXIT), None)
        bridge._dqn_model = dqn
        bridge._available = True

        state = np.zeros(12)
        assert bridge.get_action(state) == ACTION_EXIT

    def test_predict_error_returns_none(self, config):
        """모델 추론 실패 시 None → Hold"""
        bridge = RLBridge(config)
        ppo = MagicMock()
        ppo.predict.side_effect = RuntimeError("inference failed")
        bridge._ppo_model = ppo
        bridge._dqn_model = None
        bridge._available = True

        state = np.zeros(12)
        # _predict returns None for failed ppo, None for missing dqn → Hold
        assert bridge.get_action(state) == ACTION_HOLD

    def test_load_disabled(self, config):
        """RL 비활성화 시 load() → False"""
        config.rl.enabled = False
        bridge = RLBridge(config)
        result = bridge.load()
        assert result is False
        assert bridge.is_available is False


# ============================================================
# DB record_kp_snapshot / record_rl_model Tests
# ============================================================

class TestDBKPSnapshot:
    """record_kp_snapshot 이중 기록 테스트"""

    @pytest.mark.asyncio
    async def test_record_kp_snapshot_supabase_and_local(self, snapshot, tmp_path):
        """Supabase POST + 로컬 JSONL 이중 기록"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test_key",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            # 로컬 경로를 tmp_path로 변경
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                stats = {
                    "kp_ma_1m": 1.8,
                    "kp_ma_5m": 1.9,
                    "kp_z_score": 0.5,
                    "kp_velocity": 0.002,
                    "spread_cost": 0.2,
                    "funding_rate": 0.0001,
                }
                mock_resp = MagicMock(status_code=201, text="")
                db._session.post = MagicMock(return_value=mock_resp)
                await db.record_kp_snapshot(snapshot, stats)

                # Supabase POST 호출 확인
                db._session.post.assert_called_once()
                call_args = db._session.post.call_args
                assert "kimchirang_kp_history" in str(call_args)
                posted_row = call_args[1]["json"] if "json" in call_args[1] else call_args[0][0]
                # row가 json= kwarg으로 전달됨
                assert posted_row["mid_kp"] == snapshot.mid_kp

                # 로컬 JSONL 파일 확인
                local_file = os.path.join(str(tmp_path), "kp_history.jsonl")
                assert os.path.exists(local_file)
                with open(local_file, "r", encoding="utf-8") as f:
                    line = f.readline()
                    data = json.loads(line)
                    assert data["mid_kp"] == snapshot.mid_kp
                    assert "_saved_at" in data
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir

    @pytest.mark.asyncio
    async def test_record_kp_snapshot_disabled_local_only(self, snapshot, tmp_path):
        """Supabase 미설정 시 로컬만 기록"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                stats = {"kp_ma_5m": 1.5}
                await db.record_kp_snapshot(snapshot, stats)

                local_file = os.path.join(str(tmp_path), "kp_history.jsonl")
                assert os.path.exists(local_file)
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.loads(f.readline())
                    assert data["mid_kp"] == snapshot.mid_kp
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir


class TestDBRLModel:
    """record_rl_model 이중 기록 테스트"""

    @pytest.mark.asyncio
    async def test_record_rl_model_supabase_and_local(self, tmp_path):
        """RL 모델 성과 기록 (이중 저장)"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test_key",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                model_info = {
                    "model_type": "PPO",
                    "version": "v1.0",
                    "total_reward": 1523.4,
                    "sharpe_ratio": 0.85,
                    "max_drawdown": -3.2,
                    "train_steps": 500000,
                }
                mock_resp = MagicMock(status_code=201, text="")
                db._session.post = MagicMock(return_value=mock_resp)
                await db.record_rl_model(model_info)

                db._session.post.assert_called_once()
                call_args = db._session.post.call_args
                assert "kimchirang_rl_models" in str(call_args)

                local_file = os.path.join(str(tmp_path), "rl_models.jsonl")
                assert os.path.exists(local_file)
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.loads(f.readline())
                    assert data["model_type"] == "PPO"
                    assert data["train_steps"] == 500000
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir

    @pytest.mark.asyncio
    async def test_record_rl_model_disabled_local_only(self, tmp_path):
        """Supabase 미설정 시 로컬만 기록"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                model_info = {"model_type": "DQN", "version": "v2.0"}
                await db.record_rl_model(model_info)

                local_file = os.path.join(str(tmp_path), "rl_models.jsonl")
                assert os.path.exists(local_file)
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir


class TestDBLocalJSONL:
    """로컬 JSONL 이중 기록 공통 테스트"""

    @pytest.mark.asyncio
    async def test_local_jsonl_appends(self, snapshot, tmp_path):
        """여러 번 기록 시 JSONL에 행 추가"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                stats = {"kp_ma_5m": 1.0}
                await db.record_kp_snapshot(snapshot, stats)
                await db.record_kp_snapshot(snapshot, stats)
                await db.record_kp_snapshot(snapshot, stats)

                local_file = os.path.join(str(tmp_path), "kp_history.jsonl")
                with open(local_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    assert len(lines) == 3
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir

    @pytest.mark.asyncio
    async def test_record_trade_local_jsonl(self, snapshot, tmp_path):
        """record_trade도 로컬 JSONL에 기록"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                result = ExecutionResult(action="enter", kp_at_execution=3.5)
                await db.record_trade(result, snapshot)

                local_file = os.path.join(str(tmp_path), "trades.jsonl")
                assert os.path.exists(local_file)
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.loads(f.readline())
                    assert data["action"] == "enter"
                    assert data["kp_at_execution"] == 3.5
                    assert "_saved_at" in data
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir

    @pytest.mark.asyncio
    async def test_record_error_local_jsonl(self, tmp_path):
        """record_error도 로컬 JSONL에 기록"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            import kimchirang.db as db_mod
            original_dir = db_mod.LOCAL_DATA_DIR
            db_mod.LOCAL_DATA_DIR = str(tmp_path)
            try:
                await db.record_error("tick", "test error message")

                local_file = os.path.join(str(tmp_path), "errors.jsonl")
                assert os.path.exists(local_file)
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.loads(f.readline())
                    assert data["error_phase"] == "tick"
                    assert "test error" in data["error_message"]
            finally:
                db_mod.LOCAL_DATA_DIR = original_dir
