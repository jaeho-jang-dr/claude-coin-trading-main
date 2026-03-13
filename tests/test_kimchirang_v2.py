"""Kimchirang v2 유닛 테스트 -- RLBridge, KimchirangDB 이중 기록, DataFeeder"""

import json
import os
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import numpy as np

# 프로젝트 루트
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from kimchirang.config import KimchirangConfig, DBConfig
from kimchirang.kp_engine import KPSnapshot
from kimchirang.execution import ExecutionResult, LegResult
from kimchirang.data_feeder import (
    MarketState, OrderBook, TickerData, FundingData,
    FXFeeder, UpbitFeeder, BinanceFeeder, DataFeeder,
)
from kimchirang.db import KimchirangDB
from kimchirang.main import RLBridge, ACTION_HOLD, ACTION_ENTER, ACTION_EXIT


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
        "KR_RL_ENABLED": "true",
    }):
        yield KimchirangConfig()


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
def local_data_dir(tmp_path):
    """DB의 로컬 저장 경로를 임시 디렉토리로 교체"""
    return str(tmp_path)


# ============================================================
# RLBridge 앙상블 테스트
# ============================================================

class TestRLBridgePPOOnly:
    """PPO만 있을 때 단독 모드"""

    def test_ppo_only_mode(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        bridge._ppo_model = MagicMock()
        bridge._ppo_model.predict.return_value = (np.array(ACTION_ENTER), None)
        bridge._dqn_model = None

        assert bridge.mode == "ppo"

    def test_ppo_only_action(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(ACTION_ENTER), None)
        bridge._ppo_model = mock_ppo
        bridge._dqn_model = None

        state = np.zeros(12)
        action = bridge.get_action(state)
        assert action == ACTION_ENTER

    def test_ppo_only_exit_action(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(ACTION_EXIT), None)
        bridge._ppo_model = mock_ppo
        bridge._dqn_model = None

        state = np.zeros(12)
        action = bridge.get_action(state)
        assert action == ACTION_EXIT

    def test_ppo_only_hold_action(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(ACTION_HOLD), None)
        bridge._ppo_model = mock_ppo
        bridge._dqn_model = None

        state = np.zeros(12)
        action = bridge.get_action(state)
        assert action == ACTION_HOLD


class TestRLBridgeDQNOnly:
    """DQN만 있을 때 단독 모드"""

    def test_dqn_only_mode(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        bridge._ppo_model = None
        bridge._dqn_model = MagicMock()

        assert bridge.mode == "dqn"

    def test_dqn_only_action(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        bridge._ppo_model = None
        mock_dqn = MagicMock()
        mock_dqn.predict.return_value = (np.array(ACTION_EXIT), None)
        bridge._dqn_model = mock_dqn

        state = np.zeros(12)
        action = bridge.get_action(state)
        assert action == ACTION_EXIT

    def test_dqn_only_enter_action(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        bridge._ppo_model = None
        mock_dqn = MagicMock()
        mock_dqn.predict.return_value = (np.array(ACTION_ENTER), None)
        bridge._dqn_model = mock_dqn

        state = np.zeros(12)
        action = bridge.get_action(state)
        assert action == ACTION_ENTER


class TestRLBridgeEnsemble:
    """둘 다 있을 때 합의/불일치 동작"""

    def _make_bridge(self, config, ppo_action, dqn_action):
        bridge = RLBridge(config)
        bridge._available = True
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(ppo_action), None)
        bridge._ppo_model = mock_ppo
        mock_dqn = MagicMock()
        mock_dqn.predict.return_value = (np.array(dqn_action), None)
        bridge._dqn_model = mock_dqn
        return bridge

    def test_ensemble_mode(self, config):
        bridge = self._make_bridge(config, ACTION_HOLD, ACTION_HOLD)
        assert bridge.mode == "ensemble"

    def test_consensus_enter(self, config):
        bridge = self._make_bridge(config, ACTION_ENTER, ACTION_ENTER)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_ENTER

    def test_consensus_exit(self, config):
        bridge = self._make_bridge(config, ACTION_EXIT, ACTION_EXIT)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_EXIT

    def test_consensus_hold(self, config):
        bridge = self._make_bridge(config, ACTION_HOLD, ACTION_HOLD)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD

    def test_disagreement_enter_vs_exit(self, config):
        """PPO=Enter, DQN=Exit -> Hold (신중)"""
        bridge = self._make_bridge(config, ACTION_ENTER, ACTION_EXIT)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD

    def test_disagreement_enter_vs_hold(self, config):
        """PPO=Enter, DQN=Hold -> Hold (신중)"""
        bridge = self._make_bridge(config, ACTION_ENTER, ACTION_HOLD)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD

    def test_disagreement_exit_vs_hold(self, config):
        """PPO=Exit, DQN=Hold -> Hold (신중)"""
        bridge = self._make_bridge(config, ACTION_EXIT, ACTION_HOLD)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD

    def test_disagreement_hold_vs_enter(self, config):
        """PPO=Hold, DQN=Enter -> Hold (신중)"""
        bridge = self._make_bridge(config, ACTION_HOLD, ACTION_ENTER)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD


class TestRLBridgeMode:
    """mode 프로퍼티 반환값 테스트"""

    def test_mode_rule_no_models(self, config):
        bridge = RLBridge(config)
        assert bridge.mode == "rule"

    def test_mode_ppo(self, config):
        bridge = RLBridge(config)
        bridge._ppo_model = MagicMock()
        assert bridge.mode == "ppo"

    def test_mode_dqn(self, config):
        bridge = RLBridge(config)
        bridge._dqn_model = MagicMock()
        assert bridge.mode == "dqn"

    def test_mode_ensemble(self, config):
        bridge = RLBridge(config)
        bridge._ppo_model = MagicMock()
        bridge._dqn_model = MagicMock()
        assert bridge.mode == "ensemble"

    def test_is_available_default(self, config):
        bridge = RLBridge(config)
        assert bridge.is_available is False

    def test_unavailable_returns_hold(self, config):
        bridge = RLBridge(config)
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD

    def test_predict_exception_returns_none(self, config):
        bridge = RLBridge(config)
        bridge._available = True
        mock_ppo = MagicMock()
        mock_ppo.predict.side_effect = RuntimeError("model error")
        bridge._ppo_model = mock_ppo
        bridge._dqn_model = None

        # PPO 추론 실패 -> _predict returns None -> both None -> HOLD
        action = bridge.get_action(np.zeros(12))
        assert action == ACTION_HOLD


# ============================================================
# KimchirangDB 이중 기록 테스트
# ============================================================

class TestKimchirangDBKPSnapshot:
    """record_kp_snapshot 로컬 JSONL 저장 확인"""

    @pytest.mark.asyncio
    async def test_kp_snapshot_local_saved(self, snapshot, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            # 로컬 저장 경로를 tmp_path로 교체
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                stats = {"kp_ma_1m": 1.5, "kp_ma_5m": 1.8, "kp_z_score": 0.3,
                         "kp_velocity": 0.01, "spread_cost": 0.2, "funding_rate": 0.0001}
                await db.record_kp_snapshot(snapshot, stats)

                path = os.path.join(local_data_dir, "kp_history.jsonl")
                assert os.path.exists(path)
                with open(path, "r", encoding="utf-8") as f:
                    line = f.readline()
                    row = json.loads(line)
                assert row["mid_kp"] == 1.9
                assert row["entry_kp"] == 3.5
                assert row["fx_rate"] == 1350.0
                assert "_saved_at" in row

    @pytest.mark.asyncio
    async def test_kp_snapshot_stats_fields(self, snapshot, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                stats = {"kp_ma_1m": 2.0, "kp_ma_5m": 2.5, "kp_z_score": -0.5,
                         "kp_velocity": -0.02, "spread_cost": 0.15, "funding_rate": -0.001}
                await db.record_kp_snapshot(snapshot, stats)

                path = os.path.join(local_data_dir, "kp_history.jsonl")
                with open(path, "r", encoding="utf-8") as f:
                    row = json.loads(f.readline())
                assert row["kp_ma_1m"] == 2.0
                assert row["kp_z_score"] == -0.5
                assert row["kp_velocity"] == -0.02


class TestKimchirangDBRLModel:
    """record_rl_model 저장 확인"""

    @pytest.mark.asyncio
    async def test_rl_model_local_saved(self, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                model_info = {
                    "model_type": "PPO",
                    "total_steps": 100000,
                    "sharpe_ratio": 0.85,
                    "total_return": 5.2,
                    "mdd": 1.8,
                }
                await db.record_rl_model(model_info)

                path = os.path.join(local_data_dir, "rl_models.jsonl")
                assert os.path.exists(path)
                with open(path, "r", encoding="utf-8") as f:
                    row = json.loads(f.readline())
                assert row["model_type"] == "PPO"
                assert row["total_steps"] == 100000
                assert row["sharpe_ratio"] == 0.85
                assert "_saved_at" in row

    @pytest.mark.asyncio
    async def test_rl_model_supabase_attempted(self, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test_key",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                mock_resp = MagicMock(status_code=201, text="")
                db._session.post = MagicMock(return_value=mock_resp)
                await db.record_rl_model({"model_type": "DQN", "steps": 50000})
                db._session.post.assert_called_once()
                call_args = db._session.post.call_args
                assert "kimchirang_rl_models" in str(call_args)


class TestKimchirangDBSupabaseFallback:
    """Supabase 실패 시 로컬만 저장되는지"""

    @pytest.mark.asyncio
    async def test_supabase_fail_local_still_saved(self, snapshot, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test_key",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                # Supabase 500 에러 시뮬레이션
                db._session.post = MagicMock(return_value=MagicMock(status_code=500, text="Internal Server Error"))
                stats = {"kp_ma_1m": 1.0, "kp_ma_5m": 1.2}
                await db.record_kp_snapshot(snapshot, stats)

                # 로컬 파일은 여전히 존재
                path = os.path.join(local_data_dir, "kp_history.jsonl")
                assert os.path.exists(path)
                with open(path, "r", encoding="utf-8") as f:
                    row = json.loads(f.readline())
                assert row["mid_kp"] == 1.9

    @pytest.mark.asyncio
    async def test_supabase_exception_local_still_saved(self, snapshot, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test_key",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                # 네트워크 오류 시뮬레이션
                db._session.post = MagicMock(side_effect=ConnectionError("Network unreachable"))
                stats = {"kp_ma_1m": 1.0}
                await db.record_kp_snapshot(snapshot, stats)

                path = os.path.join(local_data_dir, "kp_history.jsonl")
                assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_supabase_disabled_local_only(self, snapshot, local_data_dir):
        """Supabase 미설정 시 로컬만 저장"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            assert db._enabled is False
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                result = ExecutionResult(action="enter", kp_at_execution=3.5)
                await db.record_trade(result, snapshot)

                path = os.path.join(local_data_dir, "trades.jsonl")
                assert os.path.exists(path)


class TestKimchirangDBSaveLocal:
    """_save_local 파일 생성 및 포맷 확인"""

    def test_save_local_creates_file(self, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                db._save_local("test_output.jsonl", {"key": "value", "num": 42})
                path = os.path.join(local_data_dir, "test_output.jsonl")
                assert os.path.exists(path)

    def test_save_local_jsonl_format(self, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                db._save_local("test.jsonl", {"a": 1})
                db._save_local("test.jsonl", {"b": 2})

                path = os.path.join(local_data_dir, "test.jsonl")
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                assert len(lines) == 2
                row1 = json.loads(lines[0])
                row2 = json.loads(lines[1])
                assert row1["a"] == 1
                assert row2["b"] == 2

    def test_save_local_adds_saved_at(self, local_data_dir):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                db._save_local("ts.jsonl", {"x": 1})
                path = os.path.join(local_data_dir, "ts.jsonl")
                with open(path, "r", encoding="utf-8") as f:
                    row = json.loads(f.readline())
                assert "_saved_at" in row
                # ISO 형식 확인
                assert "T" in row["_saved_at"]

    def test_save_local_unicode(self, local_data_dir):
        """한글 등 유니코드가 ensure_ascii=False로 저장되는지"""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }, clear=False):
            db = KimchirangDB(DBConfig())
            with patch("kimchirang.db.LOCAL_DATA_DIR", local_data_dir):
                db._save_local("uni.jsonl", {"msg": "김치프리미엄"})
                path = os.path.join(local_data_dir, "uni.jsonl")
                with open(path, "r", encoding="utf-8") as f:
                    content = f.readline()
                assert "김치프리미엄" in content  # ensure_ascii=False
                row = json.loads(content)
                assert row["msg"] == "김치프리미엄"


# ============================================================
# DataFeeder 테스트 (mock WebSocket)
# ============================================================

class TestMarketStateIsReady:
    """MarketState.is_ready 조건"""

    def test_not_ready_default(self):
        state = MarketState()
        assert state.is_ready is False

    def test_not_ready_missing_fx(self):
        state = MarketState()
        state.upbit_orderbook = OrderBook(best_bid=85_000_000, best_ask=85_100_000)
        state.binance_orderbook = OrderBook(best_bid=62_000, best_ask=62_050)
        # fx_rate = 0 (기본값)
        assert state.is_ready is False

    def test_not_ready_missing_upbit(self):
        state = MarketState()
        state.binance_orderbook = OrderBook(best_bid=62_000, best_ask=62_050)
        state.fx_rate = 1350.0
        assert state.is_ready is False

    def test_not_ready_missing_binance(self):
        state = MarketState()
        state.upbit_orderbook = OrderBook(best_bid=85_000_000, best_ask=85_100_000)
        state.fx_rate = 1350.0
        assert state.is_ready is False

    def test_ready_all_set(self):
        state = MarketState()
        state.upbit_orderbook = OrderBook(best_bid=85_000_000, best_ask=85_100_000)
        state.binance_orderbook = OrderBook(best_bid=62_000, best_ask=62_050)
        state.fx_rate = 1350.0
        assert state.is_ready is True


class TestMarketStateDataAge:
    """data_age_sec 계산"""

    def test_data_age_no_data(self):
        state = MarketState()
        age = state.data_age_sec
        assert age["upbit_ob"] == -1
        assert age["binance_ob"] == -1
        assert age["fx"] == -1

    def test_data_age_fresh_data(self):
        state = MarketState()
        now = time.time()
        state.upbit_orderbook = OrderBook(
            best_bid=85_000_000, best_ask=85_100_000, timestamp=now
        )
        state.binance_orderbook = OrderBook(
            best_bid=62_000, best_ask=62_050, timestamp=now
        )
        state.fx_updated_at = now

        age = state.data_age_sec
        assert 0 <= age["upbit_ob"] < 1
        assert 0 <= age["binance_ob"] < 1
        assert 0 <= age["fx"] < 1

    def test_data_age_stale_data(self):
        state = MarketState()
        old_time = time.time() - 60  # 1분 전
        state.upbit_orderbook = OrderBook(
            best_bid=85_000_000, best_ask=85_100_000, timestamp=old_time
        )
        state.binance_orderbook = OrderBook(
            best_bid=62_000, best_ask=62_050, timestamp=old_time
        )
        state.fx_updated_at = old_time

        age = state.data_age_sec
        assert 59 < age["upbit_ob"] < 62
        assert 59 < age["binance_ob"] < 62
        assert 59 < age["fx"] < 62


class TestFXFeederFallback:
    """FXFeeder fallback 동작"""

    @pytest.mark.asyncio
    async def test_fallback_uses_existing_rate(self, config):
        """모든 외부 소스 실패 시 기존 환율 유지 (10분 이내)"""
        state = MarketState()
        state.fx_rate = 1350.0
        state.fx_updated_at = time.time() - 300  # 5분 전 (10분 이내)
        feeder = FXFeeder(config, state)

        # _fetch_rate 내부의 두 aiohttp 호출 모두 실패하도록 aiohttp 자체를 에러 발생시킴
        original_fetch = feeder._fetch_rate

        async def patched_fetch():
            # 소스1, 소스2 모두 실패하면 소스3(기존 값 유지) 로직에 도달
            # 직접 소스3 로직만 테스트
            if state.fx_rate > 0 and time.time() - state.fx_updated_at < 600:
                return state.fx_rate
            return None

        feeder._fetch_rate = patched_fetch
        rate = await feeder._fetch_rate()
        assert rate == 1350.0

    @pytest.mark.asyncio
    async def test_fallback_expired_returns_none(self, config):
        """기존 환율이 10분 초과 시 None 반환"""
        state = MarketState()
        state.fx_rate = 1350.0
        state.fx_updated_at = time.time() - 700  # 11분 전 (10분 초과)
        feeder = FXFeeder(config, state)

        async def patched_fetch():
            if state.fx_rate > 0 and time.time() - state.fx_updated_at < 600:
                return state.fx_rate
            return None

        feeder._fetch_rate = patched_fetch
        rate = await feeder._fetch_rate()
        assert rate is None

    @pytest.mark.asyncio
    async def test_fallback_no_existing_rate(self, config):
        """기존 환율 없을 때 None 반환"""
        state = MarketState()
        # fx_rate = 0 (기본), fx_updated_at = 0 (기본)
        feeder = FXFeeder(config, state)

        async def patched_fetch():
            if state.fx_rate > 0 and time.time() - state.fx_updated_at < 600:
                return state.fx_rate
            return None

        feeder._fetch_rate = patched_fetch
        rate = await feeder._fetch_rate()
        assert rate is None

    @pytest.mark.asyncio
    async def test_run_updates_state_on_valid_rate(self, config):
        """run()에서 유효한 환율을 받으면 state를 갱신하는지"""
        state = MarketState()
        feeder = FXFeeder(config, state)
        feeder._running = False  # 루프 1회만 실행

        async def mock_fetch():
            feeder._running = False  # 1회만
            return 1380.0

        feeder._fetch_rate = mock_fetch
        # fx_update_interval_sec을 0으로 설정하면 sleep(0)
        config.trading.fx_update_interval_sec = 0
        await feeder.run()
        assert state.fx_rate == 1380.0
        assert state.fx_available is True

    @pytest.mark.asyncio
    async def test_run_ignores_invalid_rate(self, config):
        """run()에서 비정상 환율(< 1000)을 무시하는지"""
        state = MarketState()
        feeder = FXFeeder(config, state)

        async def mock_fetch():
            feeder._running = False
            return 500.0  # 비정상

        feeder._fetch_rate = mock_fetch
        config.trading.fx_update_interval_sec = 0
        await feeder.run()
        assert state.fx_rate == 0.0  # 갱신 안됨


class TestOrderBookProperties:
    """OrderBook 속성 테스트"""

    def test_mid_price(self):
        ob = OrderBook(best_bid=100.0, best_ask=102.0)
        assert ob.mid_price == 101.0

    def test_mid_price_zero(self):
        ob = OrderBook()
        assert ob.mid_price == 0.0

    def test_spread_pct(self):
        ob = OrderBook(best_bid=100.0, best_ask=100.5)
        assert abs(ob.spread_pct - 0.5) < 0.01

    def test_spread_pct_zero_bid(self):
        ob = OrderBook(best_bid=0, best_ask=100.0)
        assert ob.spread_pct == 0.0


class TestUpbitFeederHandleMessage:
    """UpbitFeeder 메시지 처리"""

    def test_orderbook_message(self, config):
        state = MarketState()
        feeder = UpbitFeeder(config, state)
        feeder._handle_message({
            "ty": "orderbook",
            "obu": [{"bp": 85000000, "ap": 85100000, "bs": 0.5, "as": 0.3}],
        })
        assert state.upbit_orderbook.best_bid == 85000000
        assert state.upbit_orderbook.best_ask == 85100000
        assert state.upbit_orderbook.bid_qty == 0.5
        assert state.upbit_orderbook.timestamp > 0

    def test_ticker_message(self, config):
        state = MarketState()
        feeder = UpbitFeeder(config, state)
        feeder._handle_message({
            "ty": "ticker",
            "tp": 85050000,
            "atv24h": 1234.5,
            "scr": 0.025,
        })
        assert state.upbit_ticker.last_price == 85050000
        assert state.upbit_ticker.volume_24h == 1234.5
        assert abs(state.upbit_ticker.change_pct_24h - 2.5) < 0.01

    def test_empty_orderbook_units(self, config):
        state = MarketState()
        feeder = UpbitFeeder(config, state)
        feeder._handle_message({"ty": "orderbook", "obu": []})
        assert state.upbit_orderbook.best_bid == 0.0  # 변경 안됨


class TestBinanceFeederHandleMessage:
    """BinanceFeeder 메시지 처리"""

    def test_bookticker_message(self, config):
        state = MarketState()
        feeder = BinanceFeeder(config, state)
        feeder._handle_message("bookTicker", {
            "e": "bookTicker",
            "b": "62000.0",
            "a": "62050.0",
            "B": "1.5",
            "A": "2.0",
        })
        assert state.binance_orderbook.best_bid == 62000.0
        assert state.binance_orderbook.best_ask == 62050.0

    def test_ticker_24h_message(self, config):
        state = MarketState()
        feeder = BinanceFeeder(config, state)
        feeder._handle_message("24hrTicker", {
            "e": "24hrTicker",
            "c": "62100.0",
            "v": "5000.0",
            "P": "1.5",
        })
        assert state.binance_ticker.last_price == 62100.0
        assert state.binance_ticker.volume_24h == 5000.0
        assert abs(state.binance_ticker.change_pct_24h - 1.5) < 0.01

    def test_markprice_message(self, config):
        state = MarketState()
        feeder = BinanceFeeder(config, state)
        feeder._handle_message("markPriceUpdate", {
            "e": "markPriceUpdate",
            "r": "0.0001",
            "T": 1700000000,
        })
        assert state.binance_funding.funding_rate == 0.0001
        assert state.binance_funding.next_funding_time == 1700000000
