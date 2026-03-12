"""StateEncoder 단위 테스트

rl_hybrid/rl/state_encoder.py 의 FEATURE_SPEC, StateEncoder 를 검증한다.
"""

import numpy as np
import pytest

from rl_hybrid.rl.state_encoder import (
    FEATURE_SPEC,
    FEATURE_NAMES,
    OBSERVATION_DIM,
    StateEncoder,
)


# ---------------------------------------------------------------------------
# Helpers: 기본 입력 데이터 팩토리
# ---------------------------------------------------------------------------

def _make_market_data(**overrides):
    """정상적인 시장 데이터 dict 를 생성한다."""
    md = {
        "current_price": 90_000_000,
        "change_rate_24h": 0.02,
        "indicators": {
            "rsi_14": 55,
            "sma_20": 89_000_000,
            "sma_50": 87_000_000,
            "atr": 1_200_000,
            "macd": {"histogram": 120},
            "bollinger": {
                "upper": 92_000_000,
                "middle": 89_500_000,
                "lower": 87_000_000,
            },
            "stochastic": {"k": 65, "d": 60},
            "adx": {"adx": 30, "plus_di": 25, "minus_di": 18},
        },
        "indicators_4h": {"rsi_14": 58},
        "orderbook": {"ratio": 1.2},
        "trade_pressure": {"buy_volume": 150, "sell_volume": 100},
        "eth_btc_analysis": {"eth_btc_z_score": 0.5},
    }
    md.update(overrides)
    return md


def _make_external_data(**overrides):
    ext = {
        "sources": {
            "fear_greed": {"current": {"value": 45}},
            "news_sentiment": {"sentiment_score": 10},
            "whale_tracker": {"whale_score": {"score": 5}},
            "binance_sentiment": {
                "funding_rate": {"current_rate": 0.01},
                "top_trader_long_short": {"current_ratio": 1.1},
                "kimchi_premium": {"premium_pct": 1.5},
            },
            "macro": {"analysis": {"macro_score": 8}},
            "ai_signal": {"ai_composite_signal": {"score": 20}},
            "coinmarketcap": {"btc_dominance": 52},
        },
        "external_signal": {"total_score": 15},
    }
    ext.update(overrides)
    return ext


def _make_portfolio(**overrides):
    pf = {
        "total_eval": 10_000_000,
        "krw_balance": 5_000_000,
        "holdings": [
            {
                "currency": "BTC",
                "eval_amount": 5_000_000,
                "avg_buy_price": 88_000_000,
                "profit_loss_pct": 2.3,
            }
        ],
    }
    pf.update(overrides)
    return pf


def _make_agent_state(**overrides):
    ag = {
        "danger_score": 35,
        "opportunity_score": 40,
        "cascade_risk": 15,
        "consecutive_losses": 1,
        "hours_since_last_trade": 8,
        "daily_trade_count": 2,
    }
    ag.update(overrides)
    return ag


# ===========================================================================
# 1. FEATURE_SPEC 구조
# ===========================================================================


class TestFeatureSpec:
    def test_feature_count_is_42(self):
        assert len(FEATURE_SPEC) == 42

    def test_observation_dim_matches(self):
        assert OBSERVATION_DIM == 42

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 42

    def test_all_ranges_have_two_elements(self):
        for name, (lo, hi) in FEATURE_SPEC.items():
            assert lo <= hi, f"{name}: min({lo}) > max({hi})"

    def test_feature_names_order_matches_spec(self):
        assert FEATURE_NAMES == list(FEATURE_SPEC.keys())


# ===========================================================================
# 2. StateEncoder 초기화
# ===========================================================================


class TestStateEncoderInit:
    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_obs_dim(self, encoder):
        assert encoder.obs_dim == 42

    def test_feature_names(self, encoder):
        assert encoder.feature_names == FEATURE_NAMES

    def test_mins_shape(self, encoder):
        assert encoder._mins.shape == (42,)
        assert encoder._mins.dtype == np.float32

    def test_maxs_shape(self, encoder):
        assert encoder._maxs.shape == (42,)

    def test_ranges_shape_and_no_zeros(self, encoder):
        assert encoder._ranges.shape == (42,)
        # 0 나누기 방지 — 모든 range > 0
        assert np.all(encoder._ranges > 0)

    def test_mins_less_than_or_equal_maxs(self, encoder):
        assert np.all(encoder._mins <= encoder._maxs)


# ===========================================================================
# 3. encode() 정상 데이터
# ===========================================================================


class TestEncodeNormal:
    @pytest.fixture
    def obs(self):
        enc = StateEncoder()
        return enc.encode(
            _make_market_data(),
            _make_external_data(),
            _make_portfolio(),
            _make_agent_state(),
        )

    def test_output_shape(self, obs):
        assert obs.shape == (42,)

    def test_output_dtype(self, obs):
        assert obs.dtype == np.float32

    def test_values_in_0_1(self, obs):
        assert np.all(obs >= 0.0), f"min={obs.min()}"
        assert np.all(obs <= 1.0), f"max={obs.max()}"

    def test_not_all_zeros(self, obs):
        assert np.any(obs > 0)

    def test_not_all_ones(self, obs):
        assert np.any(obs < 1)


# ===========================================================================
# 4. encode() 누락/빈 데이터 — 크래시 없이 합리적 기본값
# ===========================================================================


class TestEncodeMissingData:
    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_empty_dicts(self, encoder):
        obs = encoder.encode({}, {}, {})
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_none_agent_state(self, encoder):
        obs = encoder.encode(
            _make_market_data(),
            _make_external_data(),
            _make_portfolio(),
            None,
        )
        assert obs.shape == (42,)
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_missing_indicators(self, encoder):
        md = _make_market_data()
        md["indicators"] = {}
        md["indicators_4h"] = {}
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_missing_orderbook_and_trade_pressure(self, encoder):
        md = _make_market_data()
        md.pop("orderbook", None)
        md.pop("trade_pressure", None)
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert obs.shape == (42,)

    def test_empty_holdings(self, encoder):
        pf = _make_portfolio()
        pf["holdings"] = []
        obs = encoder.encode(_make_market_data(), _make_external_data(), pf)
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_missing_sources_key(self, encoder):
        """external_data 에 'sources' 키가 없어도 동작한다."""
        ext = {"external_signal": {"total_score": 5}}
        obs = encoder.encode(_make_market_data(), ext, _make_portfolio())
        assert obs.shape == (42,)

    def test_all_none_indicators(self, encoder):
        """지표값이 None 일 때 기본값이 사용된다."""
        md = _make_market_data()
        md["indicators"] = {
            "rsi_14": None,
            "sma_20": None,
            "sma_50": None,
            "atr": None,
            "macd": {"histogram": None},
            "bollinger": {"upper": None, "middle": None, "lower": None},
            "stochastic": {"k": None, "d": None},
            "adx": {"adx": None, "plus_di": None, "minus_di": None},
        }
        md["indicators_4h"] = {"rsi_14": None}
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_missing_external_sub_sources(self, encoder):
        """외부 소스 하위 dict 가 비어있어도 동작한다."""
        ext = {
            "sources": {
                "fear_greed": {},
                "news_sentiment": {},
                "whale_tracker": {},
                "binance_sentiment": {},
                "macro": {},
                "ai_signal": {},
                "coinmarketcap": {},
            },
            "external_signal": {},
        }
        obs = encoder.encode(_make_market_data(), ext, _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))


# ===========================================================================
# 5. encode() 극단값 — [0,1] 클리핑
# ===========================================================================


class TestEncodeExtremeValues:
    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_very_high_values(self, encoder):
        md = _make_market_data(
            change_rate_24h=5.0,  # 500% — 극단
            current_price=999_000_000_000,
        )
        md["indicators"]["rsi_14"] = 999
        md["indicators"]["macd"]["histogram"] = 99999
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_very_low_values(self, encoder):
        md = _make_market_data(change_rate_24h=-5.0)
        md["indicators"]["rsi_14"] = -50
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_extreme_portfolio(self, encoder):
        pf = _make_portfolio()
        pf["total_eval"] = 1
        pf["krw_balance"] = 999_999_999
        pf["holdings"][0]["profit_loss_pct"] = -9999
        obs = encoder.encode(_make_market_data(), _make_external_data(), pf)
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_extreme_agent_state(self, encoder):
        ag = _make_agent_state(
            danger_score=999,
            opportunity_score=-50,
            cascade_risk=200,
            consecutive_losses=100,
        )
        obs = encoder.encode(
            _make_market_data(), _make_external_data(), _make_portfolio(), ag
        )
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_extreme_external_data(self, encoder):
        ext = _make_external_data()
        ext["sources"]["fear_greed"] = {"current": {"value": -100}}
        ext["sources"]["binance_sentiment"]["funding_rate"] = {"current_rate": 99}
        obs = encoder.encode(_make_market_data(), ext, _make_portfolio())
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)


# ===========================================================================
# 6. _normalize() 정확성
# ===========================================================================


class TestNormalize:
    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_min_maps_to_zero(self, encoder):
        result = encoder._normalize(encoder._mins.copy())
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_max_maps_to_one(self, encoder):
        result = encoder._normalize(encoder._maxs.copy())
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_midpoint_maps_to_half(self, encoder):
        mid = (encoder._mins + encoder._maxs) / 2
        result = encoder._normalize(mid)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_below_min_clips_to_zero(self, encoder):
        below = encoder._mins - 1000
        result = encoder._normalize(below)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_above_max_clips_to_one(self, encoder):
        above = encoder._maxs + 1000
        result = encoder._normalize(above)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_custom_value(self, encoder):
        """rsi_14 (idx 7): range [0, 100], raw=70 → 0.7"""
        idx = FEATURE_NAMES.index("rsi_14")
        raw = encoder._mins.copy()
        raw[idx] = 70.0
        result = encoder._normalize(raw)
        assert abs(result[idx] - 0.7) < 1e-5

    def test_output_dtype(self, encoder):
        result = encoder._normalize(encoder._mins.copy())
        assert result.dtype == np.float32


# ===========================================================================
# 7. decode_feature() 왕복 정확성
# ===========================================================================


class TestDecodeFeature:
    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_round_trip_rsi(self, encoder):
        """RSI 70 → encode → decode → 70"""
        idx = FEATURE_NAMES.index("rsi_14")
        raw = encoder._mins.copy()
        raw[idx] = 70.0
        obs = encoder._normalize(raw)
        decoded = encoder.decode_feature(obs, "rsi_14")
        assert abs(decoded - 70.0) < 1e-3

    def test_round_trip_fgi(self, encoder):
        idx = FEATURE_NAMES.index("fgi")
        raw = encoder._mins.copy()
        raw[idx] = 25.0
        obs = encoder._normalize(raw)
        decoded = encoder.decode_feature(obs, "fgi")
        assert abs(decoded - 25.0) < 1e-3

    def test_round_trip_all_features(self, encoder):
        """모든 feature 에 대해 min~max 사이 임의 값으로 왕복 검증"""
        rng = np.random.default_rng(42)
        for name, (lo, hi) in FEATURE_SPEC.items():
            raw_val = rng.uniform(lo, hi)
            idx = FEATURE_NAMES.index(name)
            raw = encoder._mins.copy()
            raw[idx] = raw_val
            obs = encoder._normalize(raw)
            decoded = encoder.decode_feature(obs, name)
            assert abs(decoded - raw_val) < 1e-2, (
                f"{name}: expected {raw_val}, got {decoded}"
            )

    def test_decode_at_min(self, encoder):
        obs = encoder._normalize(encoder._mins.copy())
        for name in FEATURE_NAMES:
            decoded = encoder.decode_feature(obs, name)
            expected = FEATURE_SPEC[name][0]
            assert abs(decoded - expected) < 1e-3, f"{name}: {decoded} != {expected}"

    def test_decode_at_max(self, encoder):
        obs = encoder._normalize(encoder._maxs.copy())
        for name in FEATURE_NAMES:
            decoded = encoder.decode_feature(obs, name)
            expected = FEATURE_SPEC[name][1]
            assert abs(decoded - expected) < 1e-3, f"{name}: {decoded} != {expected}"

    def test_invalid_feature_name_raises(self, encoder):
        obs = np.zeros(42, dtype=np.float32)
        with pytest.raises(ValueError):
            encoder.decode_feature(obs, "nonexistent_feature")


# ===========================================================================
# 8. Edge cases
# ===========================================================================


class TestEdgeCases:
    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_zero_price(self, encoder):
        """current_price=0 일 때 0 나누기 없이 동작한다."""
        md = _make_market_data(current_price=0)
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_zero_total_eval(self, encoder):
        """total_eval=0 — safe_div 기본값 사용."""
        pf = _make_portfolio()
        pf["total_eval"] = 0
        obs = encoder.encode(_make_market_data(), _make_external_data(), pf)
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_zero_avg_buy_price(self, encoder):
        """avg_buy_price=0 — safe_div 기본값 사용."""
        pf = _make_portfolio()
        pf["holdings"][0]["avg_buy_price"] = 0
        obs = encoder.encode(_make_market_data(), _make_external_data(), pf)
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_negative_price(self, encoder):
        md = _make_market_data(current_price=-1)
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_empty_portfolio_no_btc(self, encoder):
        """BTC 보유 없는 포트폴리오 — ETH 만 존재."""
        pf = {
            "total_eval": 10_000_000,
            "krw_balance": 10_000_000,
            "holdings": [
                {"currency": "ETH", "eval_amount": 0, "avg_buy_price": 0}
            ],
        }
        obs = encoder.encode(_make_market_data(), _make_external_data(), pf)
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_multiple_encodes_independent(self, encoder):
        """연속 encode 호출이 서로 영향을 주지 않는다."""
        obs1 = encoder.encode(
            _make_market_data(),
            _make_external_data(),
            _make_portfolio(),
            _make_agent_state(),
        )
        obs2 = encoder.encode(
            _make_market_data(change_rate_24h=-0.05),
            _make_external_data(),
            _make_portfolio(),
            _make_agent_state(danger_score=80),
        )
        # 서로 다른 입력이므로 관측도 달라야 한다
        assert not np.array_equal(obs1, obs2)

    def test_fgi_flat_format(self, encoder):
        """fear_greed 가 {'value': 50} 형태 (current 키 없음) 일 때."""
        ext = _make_external_data()
        ext["sources"]["fear_greed"] = {"value": 30}
        obs = encoder.encode(_make_market_data(), ext, _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    @pytest.mark.xfail(
        reason="whale_data.get() on line 179 crashes when whale_tracker is not a dict — known bug",
        raises=AttributeError,
        strict=True,
    )
    def test_whale_tracker_non_dict(self, encoder):
        """whale_tracker 가 dict 가 아닌 경우 — 현재 코드가 crash 한다."""
        ext = _make_external_data()
        ext["sources"]["whale_tracker"] = "error"
        obs = encoder.encode(_make_market_data(), ext, _make_portfolio())
        assert obs.shape == (42,)

    def test_sell_volume_zero(self, encoder):
        """sell_volume=0 — safe_div 로 0 나누기 방지."""
        md = _make_market_data()
        md["trade_pressure"] = {"buy_volume": 100, "sell_volume": 0}
        obs = encoder.encode(md, _make_external_data(), _make_portfolio())
        assert obs.shape == (42,)
        assert np.all(np.isfinite(obs))

    def test_encoder_is_reusable(self, encoder):
        """같은 인코더 인스턴스를 여러 번 사용할 수 있다."""
        for _ in range(10):
            obs = encoder.encode(
                _make_market_data(),
                _make_external_data(),
                _make_portfolio(),
            )
            assert obs.shape == (42,)
