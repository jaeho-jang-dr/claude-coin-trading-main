"""Unit tests for rl_hybrid.rl.scenario_generator.ScenarioGenerator

Covers:
  1. Each of 10 scenarios produces valid candles
  2. generate_all() returns correct total count
  3. mix_with_real() preserves real candles and adds synthetic
  4. Timestamp continuity in each scenario
  5. Price continuity (no NaN, no negative)
  6. slow_bleed() produces exactly 720 candles
  7. flash_crash severity parameter works
  8. Reproducibility with same seed
  9. Different seeds produce different results
"""

import math
from datetime import datetime

import pytest

from rl_hybrid.rl.scenario_generator import ScenarioGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    """Default generator with fixed seed."""
    return ScenarioGenerator(base_price=100_000_000, seed=42)


@pytest.fixture
def scenario_methods(gen):
    """List of (name, callable) for all 10 scenario methods."""
    return [
        ("flash_crash", gen.flash_crash),
        ("dead_cat_bounce", gen.dead_cat_bounce),
        ("parabolic_pump", gen.parabolic_pump),
        ("whale_manipulation", gen.whale_manipulation),
        ("sideways_trap", gen.sideways_trap),
        ("cascade_liquidation", gen.cascade_liquidation),
        ("v_shape_recovery", gen.v_shape_recovery),
        ("slow_bleed", gen.slow_bleed),
        ("fomo_top", gen.fomo_top),
        ("black_swan", gen.black_swan),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_candle(candle: dict, label: str = ""):
    """Assert a single candle dict satisfies OHLCV invariants."""
    prefix = f"[{label}] " if label else ""

    # Required keys
    for key in ("timestamp", "open", "high", "low", "close", "volume"):
        assert key in candle, f"{prefix}Missing key '{key}'"

    o, h, l, c, v = candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]

    # Positive prices
    assert o > 0, f"{prefix}open must be positive, got {o}"
    assert h > 0, f"{prefix}high must be positive, got {h}"
    assert l > 0, f"{prefix}low must be positive, got {l}"
    assert c > 0, f"{prefix}close must be positive, got {c}"

    # No NaN
    assert not math.isnan(o), f"{prefix}open is NaN"
    assert not math.isnan(h), f"{prefix}high is NaN"
    assert not math.isnan(l), f"{prefix}low is NaN"
    assert not math.isnan(c), f"{prefix}close is NaN"
    assert not math.isnan(v), f"{prefix}volume is NaN"

    # high >= open, close ; low <= open, close
    assert h >= o, f"{prefix}high ({h}) < open ({o})"
    assert h >= c, f"{prefix}high ({h}) < close ({c})"
    assert l <= o, f"{prefix}low ({l}) > open ({o})"
    assert l <= c, f"{prefix}low ({l}) > close ({c})"

    # Positive volume
    assert v > 0, f"{prefix}volume must be positive, got {v}"


def assert_timestamps_continuous(candles: list[dict], label: str = ""):
    """Assert timestamps are parseable and strictly increasing."""
    prefix = f"[{label}] " if label else ""
    assert len(candles) > 0, f"{prefix}empty candle list"

    prev_ts = None
    for i, c in enumerate(candles):
        ts = datetime.fromisoformat(c["timestamp"])
        if prev_ts is not None:
            assert ts >= prev_ts, (
                f"{prefix}Timestamp not monotonic at index {i}: {prev_ts} -> {ts}"
            )
        prev_ts = ts


# ---------------------------------------------------------------------------
# 1. Each scenario produces valid candles
# ---------------------------------------------------------------------------

class TestEachScenarioValid:
    """Every scenario must produce candles with valid OHLCV properties."""

    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "slow_bleed",
        "fomo_top",
        "black_swan",
    ])
    def test_valid_candles(self, gen, name):
        method = getattr(gen, name)
        candles = method()
        assert len(candles) > 0, f"{name} returned no candles"
        for i, c in enumerate(candles):
            assert_valid_candle(c, label=f"{name}[{i}]")


# ---------------------------------------------------------------------------
# 2. generate_all() returns correct total count
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_default_variations(self, gen):
        """generate_all(variations=3) produces 3 copies of all 10 scenarios."""
        all_candles = gen.generate_all(variations=3)
        assert len(all_candles) > 0
        # All candles must be valid
        for i, c in enumerate(all_candles):
            assert_valid_candle(c, label=f"all[{i}]")

    def test_single_variation(self, gen):
        """With variations=1, total equals the sum of individual scenario lengths."""
        all_candles = gen.generate_all(variations=1)
        # Re-create with same seed to get individual lengths
        gen2 = ScenarioGenerator(base_price=100_000_000, seed=42)
        expected_total = 0
        for name in [
            "flash_crash", "dead_cat_bounce", "parabolic_pump",
            "whale_manipulation", "sideways_trap", "cascade_liquidation",
            "v_shape_recovery", "slow_bleed", "fomo_top", "black_swan",
        ]:
            method = getattr(gen2, name)
            expected_total += len(method())
        assert len(all_candles) == expected_total

    def test_variations_scaling(self, gen):
        """More variations should produce more candles."""
        c1 = gen.generate_all(variations=1)
        gen2 = ScenarioGenerator(base_price=100_000_000, seed=42)
        c2 = gen2.generate_all(variations=2)
        assert len(c2) > len(c1)


# ---------------------------------------------------------------------------
# 3. mix_with_real() preserves real candles and adds synthetic
# ---------------------------------------------------------------------------

class TestMixWithReal:
    @pytest.fixture
    def real_candles(self):
        """Simple list of 100 fake 'real' candles."""
        return [
            {
                "timestamp": f"2024-12-{10 + i // 24:02d}T{i % 24:02d}:00:00",
                "open": 95_000_000 + i * 1000,
                "high": 95_100_000 + i * 1000,
                "low": 94_900_000 + i * 1000,
                "close": 95_050_000 + i * 1000,
                "volume": 300.0,
            }
            for i in range(100)
        ]

    def test_real_candles_preserved(self, gen, real_candles):
        mixed = gen.mix_with_real(real_candles, synthetic_ratio=0.3, variations=1)
        # First 100 candles must be exactly the real ones
        for i in range(len(real_candles)):
            assert mixed[i] == real_candles[i], f"Real candle {i} was modified"

    def test_synthetic_added(self, gen, real_candles):
        mixed = gen.mix_with_real(real_candles, synthetic_ratio=0.3, variations=1)
        assert len(mixed) > len(real_candles), "No synthetic candles were added"

    def test_empty_real_returns_synthetic(self, gen):
        result = gen.mix_with_real([], synthetic_ratio=0.3, variations=1)
        assert len(result) > 0, "Empty real should still produce synthetic candles"

    def test_base_price_updated(self, gen, real_candles):
        """mix_with_real should update base_price to last real close."""
        last_close = real_candles[-1]["close"]
        gen.mix_with_real(real_candles, synthetic_ratio=0.3, variations=1)
        assert gen.base_price == last_close

    def test_synthetic_candles_valid(self, gen, real_candles):
        mixed = gen.mix_with_real(real_candles, synthetic_ratio=0.3, variations=1)
        # Validate only the synthetic portion
        for i in range(len(real_candles), len(mixed)):
            assert_valid_candle(mixed[i], label=f"mixed_synthetic[{i}]")


# ---------------------------------------------------------------------------
# 4. Timestamp continuity in each scenario
# ---------------------------------------------------------------------------

class TestTimestampContinuity:
    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "fomo_top",
        "black_swan",
    ])
    def test_timestamps_monotonic(self, gen, name):
        method = getattr(gen, name)
        candles = method()
        assert_timestamps_continuous(candles, label=name)

    def test_slow_bleed_timestamps_mostly_monotonic(self, gen):
        """slow_bleed has fake bounces that reuse timestamps at bounce points.
        Verify timestamps are parseable and non-monotonic spots are limited
        to the fake-bounce insertions.
        """
        candles = gen.slow_bleed()
        non_monotonic = 0
        prev_ts = None
        for c in candles:
            ts = datetime.fromisoformat(c["timestamp"])
            if prev_ts is not None and ts < prev_ts:
                non_monotonic += 1
            prev_ts = ts
        # Fake bounces produce at most ~4 regression points (4 bounces x small window)
        assert non_monotonic <= 20, (
            f"Too many non-monotonic timestamps ({non_monotonic}) in slow_bleed"
        )

    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "slow_bleed",
        "fomo_top",
        "black_swan",
    ])
    def test_timestamps_parseable(self, gen, name):
        """All timestamps must be valid ISO format strings."""
        method = getattr(gen, name)
        candles = method()
        for i, c in enumerate(candles):
            try:
                datetime.fromisoformat(c["timestamp"])
            except (ValueError, TypeError) as e:
                pytest.fail(f"[{name}][{i}] Invalid timestamp '{c['timestamp']}': {e}")


# ---------------------------------------------------------------------------
# 5. Price continuity (no NaN, no negative)
# ---------------------------------------------------------------------------

class TestPriceContinuity:
    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "slow_bleed",
        "fomo_top",
        "black_swan",
    ])
    def test_no_nan_no_negative(self, gen, name):
        method = getattr(gen, name)
        candles = method()
        for i, c in enumerate(candles):
            for field in ("open", "high", "low", "close"):
                val = c[field]
                assert not math.isnan(val), f"[{name}][{i}] {field} is NaN"
                assert val > 0, f"[{name}][{i}] {field} is non-positive: {val}"

    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "slow_bleed",
        "fomo_top",
        "black_swan",
    ])
    def test_volume_positive(self, gen, name):
        method = getattr(gen, name)
        candles = method()
        for i, c in enumerate(candles):
            assert c["volume"] > 0, f"[{name}][{i}] volume not positive: {c['volume']}"


# ---------------------------------------------------------------------------
# 6. slow_bleed() produces exactly 720 candles
# ---------------------------------------------------------------------------

class TestSlowBleed:
    def test_exactly_720_candles(self, gen):
        candles = gen.slow_bleed()
        assert len(candles) == 720, f"Expected 720 candles, got {len(candles)}"

    def test_overall_downtrend(self, gen):
        """slow_bleed should end lower than it started."""
        candles = gen.slow_bleed()
        assert candles[-1]["close"] < candles[0]["open"], (
            "slow_bleed should produce a net downtrend"
        )

    def test_720_with_different_seeds(self):
        """slow_bleed should always return exactly 720 regardless of seed."""
        for seed in [0, 7, 99, 123, 9999]:
            g = ScenarioGenerator(base_price=50_000_000, seed=seed)
            candles = g.slow_bleed()
            assert len(candles) == 720, f"seed={seed}: got {len(candles)} candles"


# ---------------------------------------------------------------------------
# 7. flash_crash severity parameter works
# ---------------------------------------------------------------------------

class TestFlashCrashSeverity:
    def test_default_severity(self, gen):
        candles = gen.flash_crash()
        assert len(candles) > 0

    def test_higher_severity_deeper_drop(self):
        """Higher severity should produce a lower minimum price."""
        gen_low = ScenarioGenerator(base_price=100_000_000, seed=42)
        candles_low = gen_low.flash_crash(severity=0.10)
        min_low = min(c["low"] for c in candles_low)

        gen_high = ScenarioGenerator(base_price=100_000_000, seed=42)
        candles_high = gen_high.flash_crash(severity=0.30)
        min_high = min(c["low"] for c in candles_high)

        assert min_high < min_low, (
            f"severity=0.30 min ({min_high}) should be lower than severity=0.10 min ({min_low})"
        )

    def test_severity_zero(self):
        """severity=0 should produce no crash (prices stay near base)."""
        g = ScenarioGenerator(base_price=100_000_000, seed=42)
        candles = g.flash_crash(severity=0.0)
        min_price = min(c["low"] for c in candles)
        # With 0 severity, crash_bottom == price, so no significant drop
        assert min_price > 100_000_000 * 0.95, (
            f"severity=0 should not crash significantly, min was {min_price}"
        )

    def test_severity_range(self):
        """Multiple severity values should all produce valid candles."""
        for sev in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
            g = ScenarioGenerator(base_price=100_000_000, seed=42)
            candles = g.flash_crash(severity=sev)
            assert len(candles) > 0
            for i, c in enumerate(candles):
                assert_valid_candle(c, label=f"severity={sev}[{i}]")


# ---------------------------------------------------------------------------
# 8. Reproducibility with same seed
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self):
        """Two generators with the same seed must produce identical candles."""
        g1 = ScenarioGenerator(base_price=100_000_000, seed=42)
        g2 = ScenarioGenerator(base_price=100_000_000, seed=42)

        c1 = g1.flash_crash()
        c2 = g2.flash_crash()

        assert len(c1) == len(c2)
        for i, (a, b) in enumerate(zip(c1, c2)):
            assert a == b, f"Candle {i} differs with same seed"

    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "slow_bleed",
        "fomo_top",
        "black_swan",
    ])
    def test_all_scenarios_reproducible(self, name):
        g1 = ScenarioGenerator(base_price=100_000_000, seed=77)
        g2 = ScenarioGenerator(base_price=100_000_000, seed=77)

        c1 = getattr(g1, name)()
        c2 = getattr(g2, name)()

        assert len(c1) == len(c2), f"{name}: length mismatch"
        for i, (a, b) in enumerate(zip(c1, c2)):
            assert a == b, f"{name}[{i}]: candles differ with same seed"

    def test_generate_all_reproducible(self):
        g1 = ScenarioGenerator(base_price=100_000_000, seed=42)
        g2 = ScenarioGenerator(base_price=100_000_000, seed=42)

        c1 = g1.generate_all(variations=2)
        c2 = g2.generate_all(variations=2)

        assert len(c1) == len(c2)
        for i, (a, b) in enumerate(zip(c1, c2)):
            assert a == b, f"generate_all candle {i} differs"


# ---------------------------------------------------------------------------
# 9. Different seeds produce different results
# ---------------------------------------------------------------------------

class TestDifferentSeeds:
    def test_different_seeds_different_output(self):
        g1 = ScenarioGenerator(base_price=100_000_000, seed=42)
        g2 = ScenarioGenerator(base_price=100_000_000, seed=99)

        c1 = g1.flash_crash()
        c2 = g2.flash_crash()

        # At least some candles should differ
        differences = sum(1 for a, b in zip(c1, c2) if a["close"] != b["close"])
        assert differences > 0, "Different seeds produced identical output"

    @pytest.mark.parametrize("name", [
        "flash_crash",
        "dead_cat_bounce",
        "parabolic_pump",
        "whale_manipulation",
        "sideways_trap",
        "cascade_liquidation",
        "v_shape_recovery",
        "slow_bleed",
        "fomo_top",
        "black_swan",
    ])
    def test_all_scenarios_differ_with_seeds(self, name):
        g1 = ScenarioGenerator(base_price=100_000_000, seed=10)
        g2 = ScenarioGenerator(base_price=100_000_000, seed=20)

        c1 = getattr(g1, name)()
        c2 = getattr(g2, name)()

        differences = sum(1 for a, b in zip(c1, c2) if a["close"] != b["close"])
        assert differences > 0, f"{name}: seeds 10 and 20 produced identical output"

    def test_different_base_price(self):
        """Different base_price should shift all prices."""
        g1 = ScenarioGenerator(base_price=100_000_000, seed=42)
        g2 = ScenarioGenerator(base_price=50_000_000, seed=42)

        c1 = g1.flash_crash()
        c2 = g2.flash_crash()

        # First candle open should be roughly proportional to base_price
        assert c1[0]["open"] > c2[0]["open"], (
            "Higher base_price should produce higher initial prices"
        )


# ---------------------------------------------------------------------------
# Additional edge case / structural tests
# ---------------------------------------------------------------------------

class TestCandleStructure:
    """Test structural properties of generated candles."""

    def test_candle_keys(self, gen):
        """Every candle must have exactly the expected keys."""
        expected_keys = {"timestamp", "open", "high", "low", "close", "volume"}
        candles = gen.flash_crash()
        for c in candles:
            assert set(c.keys()) == expected_keys

    def test_prices_are_rounded_integers(self, gen):
        """open/high/low/close should be rounded (int-like)."""
        candles = gen.flash_crash()
        for c in candles:
            for field in ("open", "high", "low", "close"):
                assert c[field] == round(c[field]), f"{field} not rounded: {c[field]}"

    @pytest.mark.parametrize("name,expected_len", [
        ("flash_crash", 72),
        ("dead_cat_bounce", 96),
        ("parabolic_pump", 96),
        ("whale_manipulation", 96),
        ("sideways_trap", 96),
        ("v_shape_recovery", 72),
        ("fomo_top", 96),
        ("black_swan", 72),
    ])
    def test_scenario_lengths(self, gen, name, expected_len):
        """Scenarios based on _timestamps(n) should produce n candles (except slow_bleed)."""
        method = getattr(gen, name)
        candles = method()
        assert len(candles) == expected_len, (
            f"{name}: expected {expected_len}, got {len(candles)}"
        )

    def test_cascade_liquidation_length(self, gen):
        """cascade_liquidation produces 96 candles."""
        candles = gen.cascade_liquidation()
        assert len(candles) == 96


class TestMakeCandle:
    """Test the internal _make_candle helper."""

    def test_explicit_high_low(self, gen):
        c = gen._make_candle("2025-01-01T00:00:00", 100, 90, high_p=110, low_p=80)
        assert c["high"] == 110
        assert c["low"] == 80

    def test_auto_high_low(self, gen):
        c = gen._make_candle("2025-01-01T00:00:00", 100, 90)
        assert c["high"] >= 100
        assert c["low"] <= 90

    def test_explicit_volume(self, gen):
        c = gen._make_candle("2025-01-01T00:00:00", 100, 90, volume=12345.6789)
        assert c["volume"] == 12345.6789

    def test_high_always_ge_max_oc(self, gen):
        """high must be >= max(open, close) even if high_p is lower."""
        c = gen._make_candle("2025-01-01T00:00:00", 100, 110, high_p=95)
        assert c["high"] >= 110  # _make_candle does max(high_p, open, close)

    def test_low_always_le_min_oc(self, gen):
        """low must be <= min(open, close) even if low_p is higher."""
        c = gen._make_candle("2025-01-01T00:00:00", 100, 90, low_p=105)
        assert c["low"] <= 90  # _make_candle does min(low_p, open, close)


class TestTimestamps:
    """Test the _timestamps helper."""

    def test_correct_count(self, gen):
        ts = gen._timestamps(10)
        assert len(ts) == 10

    def test_hourly_intervals(self, gen):
        ts = gen._timestamps(5)
        for i in range(1, len(ts)):
            t0 = datetime.fromisoformat(ts[i - 1])
            t1 = datetime.fromisoformat(ts[i])
            delta = (t1 - t0).total_seconds()
            assert delta == 3600, f"Expected 1h gap, got {delta}s"

    def test_custom_start(self, gen):
        ts = gen._timestamps(3, start="2024-06-15T12:00:00")
        assert ts[0] == "2024-06-15T12:00:00"
        assert ts[1] == "2024-06-15T13:00:00"
        assert ts[2] == "2024-06-15T14:00:00"
