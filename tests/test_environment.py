"""Unit tests for rl_hybrid/rl/environment.py and rl_hybrid/rl/reward.py

Covers:
  - BitcoinTradingEnv initialization, observation/action spaces
  - reset() / step() with buy/sell/hold actions
  - get_episode_stats() structure
  - External signals integration
  - RewardCalculator correctness (profitable / losing trades)
  - Edge cases: single candle beyond lookback, zero volume
"""

import numpy as np
import pytest

from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST, UPBIT_FEE_RATE, SLIPPAGE_RATE
from rl_hybrid.rl.state_encoder import OBSERVATION_DIM
from rl_hybrid.rl.environment import BitcoinTradingEnv, BitcoinTradingEnvWithLLM


# ---------------------------------------------------------------------------
# Helpers — minimal candle factory
# ---------------------------------------------------------------------------

def _make_candle(
    close: float = 50_000_000,
    change_rate: float = 0.0,
    volume: float = 100.0,
    **overrides,
) -> dict:
    """Return a single candle dict with all indicator fields the env expects."""
    candle = {
        "timestamp": "2026-01-01T00:00:00",
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
        "change_rate": change_rate,
        # Technical indicators
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


def _make_candles(n: int = 30, base_price: float = 50_000_000, price_step: float = 0) -> list[dict]:
    """Generate *n* candles with optional price drift."""
    return [
        _make_candle(close=base_price + i * price_step)
        for i in range(n)
    ]


def _make_env(n_candles: int = 30, lookback: int = 5, **kwargs) -> BitcoinTradingEnv:
    """Convenience: create an env with minimal candles and small lookback."""
    candles = kwargs.pop("candles", None) or _make_candles(n_candles)
    return BitcoinTradingEnv(
        candles=candles,
        initial_balance=10_000_000,
        lookback=lookback,
        **kwargs,
    )


# ===================================================================
# 1. Initialization
# ===================================================================

class TestEnvInit:
    """BitcoinTradingEnv initialization with minimal candles."""

    def test_basic_creation(self):
        env = _make_env()
        assert env is not None
        assert env.initial_balance == 10_000_000
        assert env.lookback == 5

    def test_default_state_before_reset(self):
        env = _make_env()
        assert env.current_step == 0
        assert env.krw_balance == 0.0
        assert env.btc_balance == 0.0

    def test_start_end_indices(self):
        env = _make_env(n_candles=50, lookback=10)
        assert env.start_idx == 10
        assert env.end_idx == 49

    def test_max_steps_limits_end(self):
        env = _make_env(n_candles=50, lookback=5, max_steps=10)
        assert env.end_idx == 15  # start_idx(5) + max_steps(10)


# ===================================================================
# 2. Observation space shape matches StateEncoder (42,)
# ===================================================================

class TestObservationSpace:

    def test_shape_is_42(self):
        env = _make_env()
        assert env.observation_space.shape == (OBSERVATION_DIM,)
        assert OBSERVATION_DIM == 42

    def test_bounds(self):
        env = _make_env()
        np.testing.assert_array_equal(env.observation_space.low, np.zeros(42, dtype=np.float32))
        np.testing.assert_array_equal(env.observation_space.high, np.ones(42, dtype=np.float32))


# ===================================================================
# 3. Action space is correct
# ===================================================================

class TestActionSpace:

    def test_shape_and_bounds(self):
        env = _make_env()
        assert env.action_space.shape == (1,)
        assert float(env.action_space.low[0]) == pytest.approx(-1.0)
        assert float(env.action_space.high[0]) == pytest.approx(1.0)

    def test_sample(self):
        env = _make_env()
        sample = env.action_space.sample()
        assert sample.shape == (1,)
        assert -1.0 <= sample[0] <= 1.0


# ===================================================================
# 4. reset() returns valid observation
# ===================================================================

class TestReset:

    def test_returns_tuple(self):
        env = _make_env()
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_observation_shape_and_range(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBSERVATION_DIM,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_info_dict_keys(self):
        env = _make_env()
        _, info = env.reset(seed=42)
        for key in ("step", "price", "portfolio_value", "krw_balance", "btc_balance", "return_pct"):
            assert key in info

    def test_portfolio_reset(self):
        env = _make_env()
        env.reset(seed=42)
        assert env.krw_balance == 10_000_000
        assert env.btc_balance == 0.0
        assert env.trade_count == 0

    def test_start_idx_option(self):
        env = _make_env(n_candles=50, lookback=5)
        env.reset(seed=42, options={"start_idx": 10})
        assert env.current_step == 10

    def test_deterministic_with_seed(self):
        env = _make_env(n_candles=100, lookback=5)
        obs1, _ = env.reset(seed=123)
        step1 = env.current_step
        obs2, _ = env.reset(seed=123)
        step2 = env.current_step
        assert step1 == step2
        np.testing.assert_array_equal(obs1, obs2)


# ===================================================================
# 5. step() with buy/sell/hold actions
# ===================================================================

class TestStep:

    def _reset_env(self, **kwargs):
        env = _make_env(n_candles=30, lookback=5, **kwargs)
        env.reset(seed=0, options={"start_idx": 5})
        return env

    def test_hold_keeps_balance(self):
        env = self._reset_env()
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        # Hold (action=0) means target BTC ratio = 0.5 → will buy some
        # But with constant price, portfolio value should stay near initial
        assert obs.shape == (OBSERVATION_DIM,)
        assert isinstance(reward, float)
        assert not terminated

    def test_full_buy(self):
        env = self._reset_env()
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
        # action=1 → target BTC ratio = 1.0 → buy all
        assert env.krw_balance < 100  # nearly all spent
        assert env.btc_balance > 0

    def test_full_sell_from_btc_position(self):
        env = self._reset_env()
        # First buy all
        env.step(np.array([1.0]))
        # Then sell all
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0]))
        # action=-1 → target BTC ratio = 0 → sell all
        assert env.btc_balance == pytest.approx(0.0, abs=1e-10)
        assert env.krw_balance > 0

    def test_step_increments_current_step(self):
        env = self._reset_env()
        initial_step = env.current_step
        env.step(np.array([0.0]))
        assert env.current_step == initial_step + 1

    def test_step_returns_five_values(self):
        env = self._reset_env()
        result = env.step(np.array([0.0]))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "reward_components" in info

    def test_truncated_at_end(self):
        candles = _make_candles(10)
        env = BitcoinTradingEnv(candles=candles, lookback=2, initial_balance=10_000_000)
        env.reset(seed=0, options={"start_idx": 2})
        # Step until the end
        truncated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(np.array([0.0]))
            if truncated or terminated:
                break
        assert truncated

    def test_bankruptcy_terminates(self):
        """Simulate bankruptcy by manipulating balance."""
        env = self._reset_env()
        # Force portfolio value below 10% of initial
        env.krw_balance = 100  # nearly bankrupt
        env.btc_balance = 0.0
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated

    def test_trade_count_increments_on_buy(self):
        env = self._reset_env()
        assert env.trade_count == 0
        env.step(np.array([1.0]))  # full buy
        assert env.trade_count >= 1

    def test_transaction_cost_applied(self):
        """After buy+sell round trip, portfolio value should decrease due to fees."""
        env = self._reset_env()
        initial_value = env.krw_balance
        env.step(np.array([1.0]))   # buy all
        env.step(np.array([-1.0]))  # sell all
        # Fees should have eaten into the balance
        assert env.krw_balance < initial_value


# ===================================================================
# 6. get_episode_stats() returns correct structure
# ===================================================================

class TestEpisodeStats:

    def test_stat_keys(self):
        env = _make_env(n_candles=20, lookback=3)
        env.reset(seed=0, options={"start_idx": 3})
        for _ in range(5):
            env.step(np.array([0.0]))
        stats = env.get_episode_stats()
        expected_keys = {
            "total_return_pct", "total_trades", "avg_return", "std_return",
            "max_drawdown", "sharpe_ratio", "trade_count", "final_value", "steps",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_initial_return_near_zero(self):
        """With constant price and hold, return should be ~0."""
        env = _make_env(n_candles=20, lookback=3)
        env.reset(seed=0, options={"start_idx": 3})
        # action=-1 means hold all in KRW (target ratio 0)
        for _ in range(5):
            env.step(np.array([-1.0]))
        stats = env.get_episode_stats()
        # Staying all-KRW at constant price → ~0% return
        assert abs(stats["total_return_pct"]) < 1.0

    def test_steps_count(self):
        env = _make_env(n_candles=20, lookback=3)
        env.reset(seed=0, options={"start_idx": 3})
        n_steps = 5
        for _ in range(n_steps):
            env.step(np.array([0.0]))
        stats = env.get_episode_stats()
        assert stats["steps"] == n_steps


# ===================================================================
# 7. External signals integration
# ===================================================================

class TestExternalSignals:

    def test_without_signals_uses_defaults(self):
        env = _make_env(n_candles=20, lookback=3)
        obs, _ = env.reset(seed=0, options={"start_idx": 3})
        assert obs.shape == (OBSERVATION_DIM,)

    def test_with_signals(self):
        n = 20
        signals = [
            {
                "fgi_value": 25,
                "news_sentiment": "negative",
                "whale_score": 10,
                "funding_rate": 0.01,
                "long_short_ratio": 1.5,
                "kimchi_premium_pct": 2.0,
                "macro_score": -5,
                "fusion_score": 30,
                "nvt_signal": 80.0,
            }
            for _ in range(n)
        ]
        env = _make_env(n_candles=n, lookback=3, external_signals=signals)
        obs, _ = env.reset(seed=0, options={"start_idx": 3})
        assert obs.shape == (OBSERVATION_DIM,)
        # Observation should differ from default (no signals) env
        env_no_sig = _make_env(n_candles=n, lookback=3)
        obs_no_sig, _ = env_no_sig.reset(seed=0, options={"start_idx": 3})
        # They should NOT be identical since FGI, news etc. differ
        assert not np.allclose(obs, obs_no_sig)

    def test_numeric_news_sentiment(self):
        """news_sentiment can be a numeric value."""
        signals = [{"news_sentiment": 75.0} for _ in range(20)]
        env = _make_env(n_candles=20, lookback=3, external_signals=signals)
        obs, _ = env.reset(seed=0, options={"start_idx": 3})
        assert obs.shape == (OBSERVATION_DIM,)

    def test_sentiment_map_values(self):
        """Verify _SENTIMENT_MAP conversion for known sentiment labels."""
        env = _make_env(n_candles=5, lookback=1)
        assert env._SENTIMENT_MAP["very_positive"] == 80
        assert env._SENTIMENT_MAP["very_negative"] == -80
        assert env._SENTIMENT_MAP["neutral"] == 0

    def test_signals_shorter_than_candles(self):
        """If signals list is shorter, should fall back to defaults for later steps."""
        signals = [{"fgi_value": 10}]  # only 1 element
        env = _make_env(n_candles=20, lookback=3, external_signals=signals)
        obs, _ = env.reset(seed=0, options={"start_idx": 3})
        # step 3 > len(signals)=1, so defaults used — should not crash
        assert obs.shape == (OBSERVATION_DIM,)


# ===================================================================
# 8. Reward function correctness
# ===================================================================

class TestRewardCalculator:

    def test_reset(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        assert rc.peak_value == 10_000_000
        assert rc.total_trades == 0

    def test_positive_return_positive_reward(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        result = rc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_100_000,  # +1%
            action=0.5,
            prev_action=0.0,
            step=1,
        )
        assert result["reward"] > 0
        assert result["components"]["raw_return"] == pytest.approx(0.01, rel=1e-3)

    def test_negative_return_negative_reward(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        result = rc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=9_900_000,  # -1%
            action=0.0,
            prev_action=0.0,
            step=1,
        )
        assert result["reward"] < 0
        assert result["components"]["raw_return"] == pytest.approx(-0.01, rel=1e-3)

    def test_mdd_penalty_above_5pct(self):
        """Drawdown > 5% should incur penalty."""
        rc = RewardCalculator()
        rc.reset(10_000_000)
        # Build up some history first
        for i in range(5):
            rc.calculate(10_000_000, 10_000_000, 0.0, 0.0, i)
        # Now a 10% drop from peak
        result = rc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=9_000_000,
            action=0.0,
            prev_action=0.0,
            step=6,
        )
        assert result["components"]["mdd_penalty"] < 0
        assert result["components"]["drawdown"] == pytest.approx(0.1, rel=1e-2)

    def test_no_mdd_penalty_under_5pct(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        result = rc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=9_600_000,  # -4% from peak
            action=0.0,
            prev_action=0.0,
            step=1,
        )
        assert result["components"]["mdd_penalty"] == 0.0

    def test_profit_bonus_on_trade(self):
        """Trade with >0.1% profit should get bonus."""
        rc = RewardCalculator()
        rc.reset(10_000_000)
        # First trade
        result = rc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_020_000,  # +0.2%
            action=0.8,
            prev_action=0.0,  # action_change = 0.8 > 0.05
            step=1,
        )
        assert result["components"]["profit_bonus"] == pytest.approx(0.1)

    def test_overtrade_penalty(self):
        """Rapid successive trades should incur overtrade penalty."""
        rc = RewardCalculator()
        rc.reset(10_000_000)
        # First trade
        rc.calculate(10_000_000, 10_000_000, 0.8, 0.0, 1)
        # Quick second trade (steps_since_last_trade=1 < 4)
        result = rc.calculate(10_000_000, 10_000_000, -0.5, 0.8, 2)
        assert result["components"]["trade_penalty"] < 0

    def test_no_overtrade_penalty_after_cooldown(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        # First trade
        rc.calculate(10_000_000, 10_000_000, 0.8, 0.0, 1)
        # Wait 5 steps (no trade — small action changes)
        for i in range(5):
            rc.calculate(10_000_000, 10_000_000, 0.8, 0.8, 2 + i)
        # Now trade again — no overtrade penalty
        result = rc.calculate(10_000_000, 10_000_000, -0.5, 0.8, 7)
        assert result["components"]["trade_penalty"] == 0.0

    def test_sharpe_initial_scaling(self):
        """First 3 steps use simple return * 10 scaling."""
        rc = RewardCalculator()
        rc.reset(10_000_000)
        result = rc.calculate(10_000_000, 10_050_000, 0.0, 0.0, 1)  # +0.5%
        # raw_return = 0.005, sharpe_reward should be 0.005 * 10 = 0.05
        assert result["components"]["sharpe_reward"] == pytest.approx(0.05, rel=1e-2)

    def test_episode_stats_structure(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        for i in range(10):
            rc.calculate(10_000_000, 10_000_000 + i * 10_000, 0.0, 0.0, i)
        stats = rc.get_episode_stats(10_100_000, 10_000_000)
        expected_keys = {"total_return_pct", "total_trades", "avg_return", "std_return",
                         "max_drawdown", "sharpe_ratio"}
        assert expected_keys.issubset(set(stats.keys()))
        assert stats["total_return_pct"] == pytest.approx(1.0, rel=1e-2)

    def test_transaction_cost_constant(self):
        assert TRANSACTION_COST == UPBIT_FEE_RATE + SLIPPAGE_RATE
        assert TRANSACTION_COST == pytest.approx(0.0008)


# ===================================================================
# 9. Edge cases
# ===================================================================

class TestEdgeCases:

    def test_minimal_candles(self):
        """Env with exactly lookback+2 candles (minimum for 1 step)."""
        candles = _make_candles(n=7)
        env = BitcoinTradingEnv(candles=candles, lookback=5, initial_balance=10_000_000)
        obs, info = env.reset(seed=0, options={"start_idx": 5})
        assert obs.shape == (OBSERVATION_DIM,)
        obs2, reward, terminated, truncated, info2 = env.step(np.array([0.0]))
        assert truncated  # end_idx = 6, after step current_step = 6

    def test_zero_volume_candles(self):
        """Candles with volume=0 should not crash."""
        candles = _make_candles(n=15, base_price=50_000_000)
        for c in candles:
            c["volume"] = 0.0
        env = BitcoinTradingEnv(candles=candles, lookback=3, initial_balance=10_000_000)
        obs, info = env.reset(seed=0, options={"start_idx": 3})
        assert obs.shape == (OBSERVATION_DIM,)
        # Step through a few times
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
            assert not np.any(np.isnan(obs))

    def test_very_low_price(self):
        """Price near zero should not cause division errors."""
        candles = _make_candles(n=15, base_price=1.0)
        env = BitcoinTradingEnv(candles=candles, lookback=3, initial_balance=10_000_000)
        obs, _ = env.reset(seed=0, options={"start_idx": 3})
        assert not np.any(np.isnan(obs))

    def test_rising_price_profitable(self):
        """Buy in a rising market should yield positive return."""
        candles = _make_candles(n=15, base_price=50_000_000, price_step=500_000)
        env = BitcoinTradingEnv(candles=candles, lookback=3, initial_balance=10_000_000)
        env.reset(seed=0, options={"start_idx": 3})
        # Buy and hold
        env.step(np.array([1.0]))  # full buy
        for _ in range(5):
            env.step(np.array([1.0]))  # keep position
        stats = env.get_episode_stats()
        assert stats["total_return_pct"] > 0

    def test_falling_price_buy_loses(self):
        """Buy in a falling market should yield negative return."""
        candles = _make_candles(n=15, base_price=50_000_000, price_step=-500_000)
        env = BitcoinTradingEnv(candles=candles, lookback=3, initial_balance=10_000_000)
        env.reset(seed=0, options={"start_idx": 3})
        env.step(np.array([1.0]))  # full buy
        for _ in range(5):
            env.step(np.array([1.0]))
        stats = env.get_episode_stats()
        assert stats["total_return_pct"] < 0

    def test_render_ansi(self):
        env = _make_env(n_candles=20, lookback=3, render_mode="ansi")
        env.reset(seed=0, options={"start_idx": 3})
        output = env.render()
        assert isinstance(output, str)
        assert "BTC" in output

    def test_action_clipping(self):
        """Actions outside [-1,1] should be clipped."""
        env = _make_env(n_candles=20, lookback=3)
        env.reset(seed=0, options={"start_idx": 3})
        # Should not crash with out-of-bound actions
        obs, reward, _, _, _ = env.step(np.array([5.0]))
        assert obs.shape == (OBSERVATION_DIM,)
        obs2, reward2, _, _, _ = env.step(np.array([-5.0]))
        assert obs2.shape == (OBSERVATION_DIM,)


# ===================================================================
# 10. BitcoinTradingEnvWithLLM
# ===================================================================

class TestEnvWithLLM:

    def test_extended_observation_space(self):
        candles = _make_candles(20)
        env = BitcoinTradingEnvWithLLM(
            candles=candles, lookback=3, initial_balance=10_000_000, llm_embedding_dim=128
        )
        assert env.observation_space.shape == (OBSERVATION_DIM + 128,)

    def test_observation_includes_embedding(self):
        candles = _make_candles(20)
        env = BitcoinTradingEnvWithLLM(
            candles=candles, lookback=3, initial_balance=10_000_000, llm_embedding_dim=8
        )
        embedding = np.ones(8, dtype=np.float32) * 0.5
        env.set_llm_embedding(embedding)
        obs, _ = env.reset(seed=0, options={"start_idx": 3})
        assert obs.shape == (OBSERVATION_DIM + 8,)
        # Last 8 values should be the embedding
        np.testing.assert_array_almost_equal(obs[-8:], embedding)


# ===================================================================
# 11. RewardCalculator — Differential Sharpe after warmup
# ===================================================================

class TestSharpeReward:

    def test_sharpe_clipped(self):
        """Sharpe reward should be clipped to [-1.5, 1.5]."""
        rc = RewardCalculator()
        rc.reset(10_000_000)
        # Fill window with extreme returns
        for i in range(25):
            rc.calculate(10_000_000, 10_500_000, 0.0, 0.0, i)  # +5% each
        result = rc.calculate(10_000_000, 10_500_000, 0.0, 0.0, 26)
        sharpe = result["components"]["sharpe_reward"]
        assert -1.5 <= sharpe <= 1.5

    def test_zero_std_returns_mean_scaling(self):
        """When all returns are identical, std ~ 0 → use mean * 10."""
        rc = RewardCalculator()
        rc.reset(10_000_000)
        # All identical returns
        for i in range(5):
            rc.calculate(10_000_000, 10_000_000, 0.0, 0.0, i)
        result = rc.calculate(10_000_000, 10_000_000, 0.0, 0.0, 6)
        # std ≈ 0, so sharpe_reward = mean_return * 10 = 0
        assert result["components"]["sharpe_reward"] == pytest.approx(0.0, abs=1e-5)
