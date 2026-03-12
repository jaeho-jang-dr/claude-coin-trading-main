"""RL 훈련 파이프라인 End-to-End 테스트

전체 RL 파이프라인을 7개 시나리오로 검증:
  1. Full pipeline: 합성 캔들 → 지표 → 환경 → PPO 훈련 → 평가 → 통계
  2. Edge case pipeline: 시나리오 생성 → 지표 → 환경 → 훈련 → 평가
  3. Model save/load round-trip: 훈련 → 저장 → 로드 → 동일 shape 추론
  4. Multi-algorithm: PPO 훈련 후 모델 파일 존재 확인
  5. Environment reset/step loop: 100 랜덤 스텝 크래시 없음
  6. ScenarioGenerator → mix_with_real → 지표 → 환경 동작 E2E
  7. StateEncoder: 모든 시나리오 타입에서 유효한 관측 벡터 생성
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.scenario_generator import ScenarioGenerator
from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM

# SB3 availability check
try:
    from rl_hybrid.rl.policy import PPOTrader, SB3_AVAILABLE
except ImportError:
    SB3_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SB3_AVAILABLE,
    reason="stable-baselines3 not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_candles(n: int = 80, base_price: float = 100_000_000) -> list[dict]:
    """Generate minimal synthetic OHLCV candles for testing."""
    rng = np.random.RandomState(123)
    candles = []
    price = base_price
    for i in range(n):
        change = rng.uniform(-0.015, 0.015)
        open_p = price
        close_p = price * (1 + change)
        high_p = max(open_p, close_p) * (1 + rng.uniform(0, 0.005))
        low_p = min(open_p, close_p) * (1 - rng.uniform(0, 0.005))
        volume = rng.uniform(100, 1000)
        candles.append({
            "timestamp": f"2025-01-{(i // 24) + 1:02d}T{i % 24:02d}:00:00",
            "open": round(open_p),
            "high": round(high_p),
            "low": round(low_p),
            "close": round(close_p),
            "volume": round(volume, 4),
        })
        price = close_p
    return candles


def prepare_env_pair(candles: list[dict], balance: float = 10_000_000):
    """Split candles 80/20 and create train/eval environments."""
    loader = HistoricalDataLoader()
    enriched = loader.compute_indicators(candles)
    split = int(len(enriched) * 0.8)
    train_candles = enriched[:split]
    eval_candles = enriched[split:]
    # Ensure minimum candle count for environment lookback
    if len(eval_candles) < 30:
        eval_candles = enriched[-30:]
    train_env = BitcoinTradingEnv(candles=train_candles, initial_balance=balance)
    eval_env = BitcoinTradingEnv(candles=eval_candles, initial_balance=balance)
    return train_env, eval_env


# ---------------------------------------------------------------------------
# Test 1: Full pipeline — synthetic candles → indicators → env → PPO → eval
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_synthetic_to_eval():
    """Create synthetic candles, compute indicators, train PPO (500 steps),
    evaluate, and verify episode stats contain expected keys."""
    candles = make_synthetic_candles(100)
    train_env, eval_env = prepare_env_pair(candles)

    trader = PPOTrader(env=train_env)
    trader.train(total_timesteps=500, eval_env=eval_env, save_freq=250)

    # Evaluate 3 episodes
    all_stats = []
    for _ in range(3):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action = trader.predict(obs)
            obs, reward, terminated, truncated, info = eval_env.step(np.array([action]))
            done = terminated or truncated
        stats = eval_env.get_episode_stats()
        all_stats.append(stats)

    assert len(all_stats) == 3
    for stats in all_stats:
        assert "total_return_pct" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown" in stats
        assert "trade_count" in stats
        assert isinstance(stats["total_return_pct"], float)
        assert isinstance(stats["max_drawdown"], float)


# ---------------------------------------------------------------------------
# Test 2: Edge case pipeline — scenario generator → indicators → train → eval
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_edge_case_pipeline():
    """Generate edge-case scenarios, compute indicators, train, and evaluate."""
    gen = ScenarioGenerator(base_price=100_000_000, seed=99)
    scenario_candles = gen.generate_all(variations=1)

    # Take a subset (first 100 candles) for speed
    subset = scenario_candles[:100]
    assert len(subset) >= 50, f"Expected >= 50 candles, got {len(subset)}"

    train_env, eval_env = prepare_env_pair(subset)

    trader = PPOTrader(env=train_env)
    trader.train(total_timesteps=300, save_freq=150)

    # Single evaluation episode
    obs, _ = eval_env.reset()
    done = False
    step_count = 0
    while not done:
        action = trader.predict(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(np.array([action]))
        done = terminated or truncated
        step_count += 1

    stats = eval_env.get_episode_stats()
    assert step_count > 0
    assert "total_return_pct" in stats


# ---------------------------------------------------------------------------
# Test 3: Model save/load round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_model_save_load_roundtrip():
    """Train, save, load, and verify predict output shape matches."""
    candles = make_synthetic_candles(80)
    train_env, eval_env = prepare_env_pair(candles)

    trader = PPOTrader(env=train_env)
    trader.train(total_timesteps=200, save_freq=200)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")
        trader.save(model_path)

        # Verify file exists
        assert os.path.exists(model_path + ".zip"), "Model .zip file not found"

        # Load into a new trader
        loaded_trader = PPOTrader(env=eval_env)
        loaded_trader.load(model_path)

        # Get an observation and predict with both
        obs, _ = eval_env.reset()
        action_original = trader.predict(obs)
        action_loaded = loaded_trader.predict(obs)

        # Both should return float scalars in [-1, 1]
        assert isinstance(action_original, float)
        assert isinstance(action_loaded, float)
        assert -1.0 <= action_original <= 1.0
        assert -1.0 <= action_loaded <= 1.0


# ---------------------------------------------------------------------------
# Test 4: Multi-algorithm — train PPO and verify model files exist
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_multi_algorithm_ppo_files_exist():
    """Train PPO and verify model files are created."""
    candles = make_synthetic_candles(80)
    train_env, _ = prepare_env_pair(candles)

    with tempfile.TemporaryDirectory() as tmpdir:
        trader = PPOTrader(env=train_env)
        trader.train(total_timesteps=200, save_freq=200)

        model_path = os.path.join(tmpdir, "ppo_test")
        trader.save(model_path)

        assert os.path.exists(model_path + ".zip"), "PPO model file not found"

        # Verify the saved model can be loaded
        reload_env, _ = prepare_env_pair(candles)
        loaded = PPOTrader(env=reload_env)
        loaded.load(model_path)
        assert loaded.model is not None


# ---------------------------------------------------------------------------
# Test 5: Environment reset/step loop — 100 random steps without crash
# ---------------------------------------------------------------------------

def test_environment_100_random_steps():
    """Run 100 random actions in the environment without any crash."""
    candles = make_synthetic_candles(80)
    loader = HistoricalDataLoader()
    enriched = loader.compute_indicators(candles)
    env = BitcoinTradingEnv(candles=enriched, initial_balance=10_000_000)

    rng = np.random.RandomState(42)
    obs, info = env.reset()

    assert obs.shape == (OBSERVATION_DIM,), f"Expected shape ({OBSERVATION_DIM},), got {obs.shape}"
    assert np.all(np.isfinite(obs)), "Observation contains non-finite values"

    for step in range(100):
        action = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (OBSERVATION_DIM,), f"Step {step}: wrong obs shape"
        assert np.all(np.isfinite(obs)), f"Step {step}: non-finite obs"
        assert isinstance(reward, float), f"Step {step}: reward not float"
        assert np.isfinite(reward), f"Step {step}: non-finite reward"

        if terminated or truncated:
            obs, info = env.reset()

    # Verify episode stats are accessible
    stats = env.get_episode_stats()
    assert isinstance(stats, dict)
    assert "final_value" in stats


# ---------------------------------------------------------------------------
# Test 6: ScenarioGenerator → mix_with_real → indicators → environment E2E
# ---------------------------------------------------------------------------

def test_scenario_mix_with_real_to_env():
    """Generate scenarios, mix with real (synthetic) candles, compute
    indicators, and verify the environment can run a full episode."""
    real_candles = make_synthetic_candles(60)
    gen = ScenarioGenerator(base_price=real_candles[-1]["close"], seed=7)

    mixed = gen.mix_with_real(real_candles, synthetic_ratio=0.3, variations=1)
    assert len(mixed) > len(real_candles), (
        f"Mixed should be longer than real: {len(mixed)} vs {len(real_candles)}"
    )

    loader = HistoricalDataLoader()
    enriched = loader.compute_indicators(mixed)

    env = BitcoinTradingEnv(candles=enriched, initial_balance=10_000_000)
    obs, _ = env.reset()

    done = False
    steps = 0
    while not done and steps < 200:
        action = np.array([0.0], dtype=np.float32)  # Hold
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps > 0
    stats = env.get_episode_stats()
    assert "total_return_pct" in stats


# ---------------------------------------------------------------------------
# Test 7: StateEncoder produces valid observations for all scenario types
# ---------------------------------------------------------------------------

def test_state_encoder_all_scenario_types():
    """Verify StateEncoder produces valid (finite, [0,1] range) observations
    for candles from every scenario type."""
    gen = ScenarioGenerator(base_price=100_000_000, seed=42)
    loader = HistoricalDataLoader()
    encoder = StateEncoder()

    scenario_funcs = [
        gen.flash_crash,
        gen.dead_cat_bounce,
        gen.parabolic_pump,
        gen.whale_manipulation,
        gen.sideways_trap,
        gen.cascade_liquidation,
        gen.v_shape_recovery,
        gen.slow_bleed,
        gen.fomo_top,
        gen.black_swan,
    ]

    for func in scenario_funcs:
        name = func.__name__
        raw_candles = func()
        assert len(raw_candles) > 0, f"{name}: generated 0 candles"

        enriched = loader.compute_indicators(raw_candles)

        # Test the environment observation for the middle candle
        env = BitcoinTradingEnv(candles=enriched, initial_balance=10_000_000)
        obs, _ = env.reset()

        assert obs.shape == (OBSERVATION_DIM,), f"{name}: wrong obs shape {obs.shape}"
        assert np.all(np.isfinite(obs)), f"{name}: non-finite values in observation"
        assert np.all(obs >= 0.0), f"{name}: obs has values < 0"
        assert np.all(obs <= 1.0), f"{name}: obs has values > 1"

        # Take one step to verify step also works
        action = np.array([0.0], dtype=np.float32)
        obs2, reward, _, _, _ = env.step(action)
        assert obs2.shape == (OBSERVATION_DIM,), f"{name}: wrong obs shape after step"
        assert np.all(np.isfinite(obs2)), f"{name}: non-finite after step"
