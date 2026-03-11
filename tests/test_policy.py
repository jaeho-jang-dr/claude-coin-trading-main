"""Unit tests for rl_hybrid/rl/policy.py

Covers:
  1. PPOTrader initialization (with env, with model_path)
  2. SACTrader initialization
  3. TD3Trader initialization
  4. TradingMetricsCallback — _on_step logic
  5. predict() returns float in valid range
  6. save/load round-trip
  7. MODEL_DIR constant exists
  8. SB3_AVAILABLE flag
"""

import os
import tempfile

import numpy as np
import pytest

from rl_hybrid.rl.policy import (
    MODEL_DIR,
    PPOTrader,
    SACTrader,
    SB3_AVAILABLE,
    TD3Trader,
    TradingMetricsCallback,
)
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.state_encoder import OBSERVATION_DIM


# ---------------------------------------------------------------------------
# Skip everything if stable-baselines3 is not installed
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not SB3_AVAILABLE,
    reason="stable-baselines3 not installed",
)


# ---------------------------------------------------------------------------
# Helpers — minimal candle factory (mirrors test_environment.py)
# ---------------------------------------------------------------------------

def _make_candle(
    close: float = 50_000_000,
    change_rate: float = 0.0,
    volume: float = 100.0,
    **overrides,
) -> dict:
    candle = {
        "timestamp": "2026-01-01T00:00:00",
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
        "change_rate": change_rate,
        "sma_20": close,
        "sma_50": close,
        "rsi_14": 50.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_histogram": 0.0,
        "boll_upper": close * 1.02,
        "boll_middle": close,
        "boll_lower": close * 0.98,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
        "adx": 25.0,
        "adx_plus_di": 20.0,
        "adx_minus_di": 20.0,
        "atr": close * 0.01,
    }
    candle.update(overrides)
    return candle


def _make_candles(n: int = 60, base_price: float = 50_000_000, price_step: float = 1000) -> list[dict]:
    return [_make_candle(close=base_price + i * price_step) for i in range(n)]


def _make_env(n_candles: int = 60, lookback: int = 5) -> BitcoinTradingEnv:
    candles = _make_candles(n_candles)
    return BitcoinTradingEnv(
        candles=candles,
        initial_balance=10_000_000,
        lookback=lookback,
    )


# ===================================================================
# 1. MODULE-LEVEL CONSTANTS
# ===================================================================

class TestConstants:
    """MODEL_DIR and SB3_AVAILABLE exist and have expected types."""

    def test_model_dir_is_string(self):
        assert isinstance(MODEL_DIR, str)

    def test_model_dir_ends_with_rl_models(self):
        assert MODEL_DIR.endswith(os.path.join("data", "rl_models"))

    def test_sb3_available_is_bool(self):
        assert isinstance(SB3_AVAILABLE, bool)

    def test_sb3_available_true(self):
        # If we reached here (past pytestmark skip), SB3 must be available
        assert SB3_AVAILABLE is True


# ===================================================================
# 2. PPOTrader
# ===================================================================

class TestPPOTrader:
    """PPOTrader initialization, predict, save/load."""

    def test_init_with_env(self):
        env = _make_env()
        trader = PPOTrader(env=env)
        assert trader.model is not None
        assert trader.env is env

    def test_init_without_env_or_path(self):
        trader = PPOTrader()
        assert trader.model is None
        assert trader.env is None

    def test_default_model_path(self):
        trader = PPOTrader()
        assert "ppo_btc_latest" in trader.model_path

    def test_custom_model_path_nonexistent(self):
        """model_path set but .zip not on disk => no load, env present => create."""
        env = _make_env()
        trader = PPOTrader(env=env, model_path="/tmp/nonexistent_ppo_model")
        # model_path was set but file doesn't exist; env was provided so model created
        assert trader.model is not None

    def test_predict_returns_float_in_range(self):
        env = _make_env()
        trader = PPOTrader(env=env)
        obs, _ = env.reset()
        action = trader.predict(obs, deterministic=True)
        assert isinstance(action, float)
        assert -1.0 <= action <= 1.0

    def test_predict_stochastic(self):
        env = _make_env()
        trader = PPOTrader(env=env)
        obs, _ = env.reset()
        action = trader.predict(obs, deterministic=False)
        assert isinstance(action, float)
        assert -1.0 <= action <= 1.0

    def test_train_minimal(self):
        """Train for a tiny number of steps without errors."""
        env = _make_env(n_candles=80)
        trader = PPOTrader(env=env)
        with tempfile.TemporaryDirectory() as tmpdir:
            trader.model_path = os.path.join(tmpdir, "ppo_test")
            trader.train(total_timesteps=128)
            assert os.path.exists(trader.model_path + ".zip")

    def test_save_load_roundtrip(self):
        env = _make_env()
        trader = PPOTrader(env=env)
        obs, _ = env.reset()
        action_before = trader.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ppo_roundtrip")
            trader.save(path)
            assert os.path.exists(path + ".zip")

            # Load into new trader
            env2 = _make_env()
            trader2 = PPOTrader(env=env2, model_path=path)
            obs2, _ = env2.reset(seed=42)
            # Model was loaded successfully
            assert trader2.model is not None

    def test_get_weights(self):
        env = _make_env()
        trader = PPOTrader(env=env)
        weights = trader.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for v in weights.values():
            assert isinstance(v, np.ndarray)

    def test_set_weights(self):
        env = _make_env()
        trader = PPOTrader(env=env)
        weights = trader.get_weights()
        # Modify a weight
        key = list(weights.keys())[0]
        weights[key] = weights[key] + 0.001
        trader.set_weights(weights)
        new_weights = trader.get_weights()
        np.testing.assert_allclose(new_weights[key], weights[key], atol=1e-5)


# ===================================================================
# 3. SACTrader
# ===================================================================

class TestSACTrader:
    """SACTrader initialization, predict, save/load."""

    def test_init_with_env(self):
        env = _make_env()
        trader = SACTrader(env=env)
        assert trader.model is not None

    def test_init_without_env_or_path(self):
        trader = SACTrader()
        assert trader.model is None

    def test_default_model_path(self):
        trader = SACTrader()
        assert "sac_btc_latest" in trader.model_path

    def test_predict_returns_float_in_range(self):
        env = _make_env()
        trader = SACTrader(env=env)
        obs, _ = env.reset()
        action = trader.predict(obs, deterministic=True)
        assert isinstance(action, float)
        assert -1.0 <= action <= 1.0

    def test_save_load_roundtrip(self):
        env = _make_env()
        trader = SACTrader(env=env)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sac_roundtrip")
            trader.save(path)
            assert os.path.exists(path + ".zip")

            env2 = _make_env()
            trader2 = SACTrader(env=env2, model_path=path)
            assert trader2.model is not None

    def test_get_set_weights(self):
        env = _make_env()
        trader = SACTrader(env=env)
        weights = trader.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        # Round-trip
        trader.set_weights(weights)

    def test_train_minimal(self):
        env = _make_env(n_candles=80)
        trader = SACTrader(env=env)
        # SAC has learning_starts=1000 by default; we override model for quick test
        trader.model.learning_starts = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            trader.model_path = os.path.join(tmpdir, "sac_test")
            trader.train(total_timesteps=100)
            assert os.path.exists(trader.model_path + ".zip")


# ===================================================================
# 4. TD3Trader
# ===================================================================

class TestTD3Trader:
    """TD3Trader initialization, predict, save/load."""

    def test_init_with_env(self):
        env = _make_env()
        trader = TD3Trader(env=env)
        assert trader.model is not None

    def test_init_without_env_or_path(self):
        trader = TD3Trader()
        assert trader.model is None

    def test_default_model_path(self):
        trader = TD3Trader()
        assert "td3_btc_latest" in trader.model_path

    def test_predict_returns_float_in_range(self):
        env = _make_env()
        trader = TD3Trader(env=env)
        obs, _ = env.reset()
        action = trader.predict(obs, deterministic=True)
        assert isinstance(action, float)
        assert -1.0 <= action <= 1.0

    def test_save_load_roundtrip(self):
        env = _make_env()
        trader = TD3Trader(env=env)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "td3_roundtrip")
            trader.save(path)
            assert os.path.exists(path + ".zip")

            env2 = _make_env()
            trader2 = TD3Trader(env=env2, model_path=path)
            assert trader2.model is not None

    def test_get_set_weights(self):
        env = _make_env()
        trader = TD3Trader(env=env)
        weights = trader.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        trader.set_weights(weights)

    def test_train_minimal(self):
        env = _make_env(n_candles=80)
        trader = TD3Trader(env=env)
        trader.model.learning_starts = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            trader.model_path = os.path.join(tmpdir, "td3_test")
            trader.train(total_timesteps=100)
            assert os.path.exists(trader.model_path + ".zip")

    def test_action_noise_created(self):
        """TD3 should have action noise configured."""
        env = _make_env()
        trader = TD3Trader(env=env)
        assert trader.model.action_noise is not None


# ===================================================================
# 5. TradingMetricsCallback
# ===================================================================

class TestTradingMetricsCallback:
    """TradingMetricsCallback _on_step logic."""

    def test_init_defaults(self):
        cb = TradingMetricsCallback()
        assert cb.log_freq == 1000
        assert cb.episode_returns == []
        assert cb.episode_trades == []

    def test_init_custom_log_freq(self):
        cb = TradingMetricsCallback(log_freq=500)
        assert cb.log_freq == 500

    def test_on_step_collects_episode_info(self):
        """_on_step should collect episode returns from infos."""
        cb = TradingMetricsCallback(log_freq=1000)
        # Simulate the callback environment
        cb.locals = {
            "infos": [
                {"episode": {"r": 5.5, "l": 100}},
            ]
        }
        cb.num_timesteps = 500
        result = cb._on_step()
        assert result is True
        assert 5.5 in cb.episode_returns

    def test_on_step_collects_final_info(self):
        """_on_step should collect return_pct from final info."""
        cb = TradingMetricsCallback(log_freq=1000)
        cb.locals = {
            "infos": [
                {"return_pct": 2.3, "_final_info": True, "trade_count": 5},
            ]
        }
        cb.num_timesteps = 500
        cb._on_step()
        assert 2.3 in cb.episode_returns
        assert 5 in cb.episode_trades

    def test_on_step_no_collection_without_episode(self):
        """_on_step should not collect if info lacks episode/return_pct."""
        cb = TradingMetricsCallback()
        cb.locals = {"infos": [{"step": 10}]}
        cb.num_timesteps = 100
        cb._on_step()
        assert cb.episode_returns == []
        assert cb.episode_trades == []

    def test_on_step_returns_true(self):
        """_on_step must always return True to continue training."""
        cb = TradingMetricsCallback()
        cb.locals = {"infos": []}
        cb.num_timesteps = 1000
        assert cb._on_step() is True

    def test_on_step_logging_at_freq(self):
        """At log_freq intervals with data, logging branch is reached without error."""
        cb = TradingMetricsCallback(log_freq=100)
        cb.episode_returns = [1.0, 2.0, 3.0]
        cb.episode_trades = [5, 10]
        cb.locals = {"infos": []}
        cb.num_timesteps = 100  # divisible by log_freq
        result = cb._on_step()
        assert result is True

    def test_on_step_empty_infos(self):
        """Handle empty infos list gracefully."""
        cb = TradingMetricsCallback()
        cb.locals = {"infos": []}
        cb.num_timesteps = 1
        assert cb._on_step() is True

    def test_on_step_no_infos_key(self):
        """Handle missing 'infos' key in locals."""
        cb = TradingMetricsCallback()
        cb.locals = {}
        cb.num_timesteps = 1
        assert cb._on_step() is True

    def test_callback_integration_with_ppo(self):
        """Callback works when passed to PPO.learn()."""
        env = _make_env(n_candles=80)
        trader = PPOTrader(env=env)
        cb = TradingMetricsCallback(log_freq=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            trader.model_path = os.path.join(tmpdir, "ppo_cb_test")
            # Train with our custom callback
            trader.model.learn(total_timesteps=128, callback=[cb])
            # Callback ran without error
            assert True


# ===================================================================
# 6. Edge cases & error handling
# ===================================================================

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_ppo_load_nonexistent_path_creates_model_from_env(self):
        """If model_path doesn't exist but env is provided, create fresh model."""
        env = _make_env()
        trader = PPOTrader(env=env, model_path="/tmp/no_such_model_xyz")
        assert trader.model is not None

    def test_all_traders_share_env_interface(self):
        """All three traders expose the same public API."""
        env = _make_env()
        for TraderClass in [PPOTrader, SACTrader, TD3Trader]:
            trader = TraderClass(env=env)
            assert hasattr(trader, "predict")
            assert hasattr(trader, "train")
            assert hasattr(trader, "save")
            assert hasattr(trader, "load")
            assert hasattr(trader, "get_weights")
            assert hasattr(trader, "set_weights")

    def test_predict_with_batch_obs(self):
        """predict works with a single observation (not batched)."""
        env = _make_env()
        trader = PPOTrader(env=env)
        obs, _ = env.reset()
        # obs should be 1D
        assert obs.ndim == 1
        action = trader.predict(obs)
        assert isinstance(action, float)

    def test_save_creates_directory(self):
        """save() should create intermediate directories."""
        env = _make_env()
        trader = PPOTrader(env=env)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "model")
            trader.save(nested)
            assert os.path.exists(nested + ".zip")
