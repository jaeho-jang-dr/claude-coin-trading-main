"""Unit tests — rl_hybrid/ RL 로직 (Team 2)

대상:
  - rl_hybrid/rl/reward.py      RewardCalculator (v6)
  - rl_hybrid/rl/reward_v7.py   RewardCalculatorV7
  - rl_hybrid/rl/reward_v8.py   RewardCalculatorV8
  - rl_hybrid/rl/environment.py  BitcoinTradingEnv
  - rl_hybrid/rl/data_loader.py  HistoricalDataLoader
"""

import math
import numpy as np
import pytest

from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST
from rl_hybrid.rl.reward_v7 import RewardCalculatorV7
from rl_hybrid.rl.reward_v8 import RewardCalculatorV8
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.state_encoder import OBSERVATION_DIM


# ============================================================
# Helpers
# ============================================================

def _make_candle(close: float = 50_000_000, change_rate: float = 0.0) -> dict:
    """최소 필드를 갖춘 캔들 dict 생성."""
    return {
        "timestamp": "2026-01-01T00:00:00",
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 100.0,
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


def _make_candles(n: int = 30, base_price: float = 50_000_000, price_step: float = 0) -> list:
    return [_make_candle(close=base_price + i * price_step) for i in range(n)]


def _make_env(n_candles: int = 30, lookback: int = 5, reward_version: str = "v6", **kwargs) -> BitcoinTradingEnv:
    candles = kwargs.pop("candles", None) or _make_candles(n_candles)
    return BitcoinTradingEnv(
        candles=candles,
        initial_balance=10_000_000,
        lookback=lookback,
        reward_version=reward_version,
        **kwargs,
    )


# ============================================================
# Section 1 — RewardCalculator v6: calculate() 출력 형식 및 클리핑
# ============================================================

class TestV6OutputFormat:
    """v6 calculate() 반환 구조 및 타입 검증."""

    @pytest.fixture
    def calc(self):
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        return rc

    def test_returns_dict_with_reward_and_components(self, calc):
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0)
        assert isinstance(result, dict)
        assert "reward" in result
        assert "components" in result

    def test_reward_is_float(self, calc):
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0)
        assert isinstance(result["reward"], float)

    def test_all_required_component_keys_present(self, calc):
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0)
        expected = {
            "raw_return", "sharpe_reward", "mdd_penalty",
            "profit_bonus", "drawdown", "trade_penalty", "total_trades",
        }
        assert expected == set(result["components"].keys())

    def test_all_float_components_except_total_trades(self, calc):
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0)
        for key, val in result["components"].items():
            if key == "total_trades":
                assert isinstance(val, int)
            else:
                assert isinstance(val, float), f"component '{key}' should be float, got {type(val)}"

    def test_sharpe_reward_clipped_to_1_5(self, calc):
        """샤프 보상은 [-1.5, 1.5] 범위로 클리핑.

        std > 0인 고수익 히스토리를 사용해야 Sharpe 계산 브랜치로 진입.
        mean 높고 std 낮은 패턴으로 sharpe >> 10 → 클리핑 확인.
        """
        vals = [0.10 + (i % 2) * 0.001 for i in range(20)]  # 교번하여 std > 0 보장
        for v in vals:
            calc.returns_history.append(v)
        sharpe = calc._compute_sharpe_reward(0.10)
        assert sharpe == pytest.approx(1.5)  # 클리핑 상한 도달

    def test_sharpe_reward_clipped_lower(self, calc):
        vals = [-0.10 + (i % 2) * 0.001 for i in range(20)]
        for v in vals:
            calc.returns_history.append(v)
        sharpe = calc._compute_sharpe_reward(-0.10)
        assert sharpe == pytest.approx(-1.5)  # 클리핑 하한 도달

    def test_reward_is_sum_of_numeric_components(self, calc):
        """reward == sharpe + mdd_penalty + profit_bonus + trade_penalty."""
        calc.steps_since_last_trade = 100  # 과매매 페널티 방지
        result = calc.calculate(10_000_000, 9_000_000, action=0.8, prev_action=0.0, step=0)
        c = result["components"]
        expected = c["sharpe_reward"] + c["mdd_penalty"] + c["profit_bonus"] + c["trade_penalty"]
        assert result["reward"] == pytest.approx(expected, abs=1e-9)

    def test_raw_return_calculated_correctly(self, calc):
        result = calc.calculate(100_000, 105_000, action=0.5, prev_action=0.5, step=0)
        assert result["components"]["raw_return"] == pytest.approx(0.05)


# ============================================================
# Section 2 — RewardCalculator v6: reset() 후 초기 상태
# ============================================================

class TestV6Reset:

    def test_returns_history_cleared(self):
        rc = RewardCalculator()
        rc.returns_history.append(0.01)
        rc.returns_history.append(0.02)
        rc.reset(10_000_000)
        assert len(rc.returns_history) == 0

    def test_peak_value_set_to_initial(self):
        rc = RewardCalculator()
        rc.peak_value = 999_999_999
        rc.reset(10_000_000)
        assert rc.peak_value == 10_000_000

    def test_max_drawdown_cleared(self):
        rc = RewardCalculator()
        rc.max_drawdown = 0.99
        rc.reset(10_000_000)
        assert rc.max_drawdown == 0.0

    def test_total_trades_cleared(self):
        rc = RewardCalculator()
        rc.total_trades = 42
        rc.reset(10_000_000)
        assert rc.total_trades == 0

    def test_steps_since_last_trade_cleared(self):
        rc = RewardCalculator()
        rc.steps_since_last_trade = 999
        rc.reset(10_000_000)
        assert rc.steps_since_last_trade == 0


# ============================================================
# Section 3 — sharpe_reward 계산 (히스토리 부족 / 충분)
# ============================================================

class TestSharpeReward:

    def test_insufficient_history_uses_simple_scaling_v6(self):
        """히스토리 < 3이면 latest_return * 10 반환."""
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        # 0개 수익 히스토리로 첫 번째 호출
        result = rc.calculate(10_000_000, 10_050_000, action=0.5, prev_action=0.5, step=0)
        # returns_history에 raw_return(0.005)이 추가된 직후 compute됨
        # n=1 < 3 → simple scaling: 0.005 * 10 = 0.05
        assert result["components"]["sharpe_reward"] == pytest.approx(0.005 * 10, rel=1e-3)

    def test_insufficient_history_uses_simple_scaling_v7(self):
        rc = RewardCalculatorV7(risk_free_rate=0.0)
        rc.reset(10_000_000)
        result = rc.calculate(10_000_000, 10_050_000, action=0.5, prev_action=0.5, step=0)
        # n=1 < 3 → simple scaling
        assert result["components"]["sharpe_reward"] == pytest.approx(0.005 * 10, rel=1e-3)

    def test_insufficient_history_uses_simple_scaling_v8(self):
        rc = RewardCalculatorV8(risk_free_rate=0.0)
        rc.reset(10_000_000)
        result = rc.calculate(10_000_000, 10_050_000, action=0.5, prev_action=0.5, step=0)
        # v8: n < 3 → latest_return * 15
        assert result["components"]["sharpe_reward"] == pytest.approx(0.005 * 15, rel=1e-3)

    def test_sufficient_history_uses_sharpe_v6(self):
        """히스토리 >= 3이면 실제 샤프 비율 계산."""
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        # 수익 히스토리 5개 삽입
        for r in [0.01, 0.02, 0.01, 0.02]:
            rc.returns_history.append(r)
        result = rc.calculate(10_000_000, 10_150_000, action=0.5, prev_action=0.5, step=4)
        sharpe = result["components"]["sharpe_reward"]
        # simple scaling이 아닌 실제 sharpe 계산이므로 범위 검증
        assert -1.5 <= sharpe <= 1.5
        # 양수 평균 수익 → 양수 sharpe
        assert sharpe > 0

    def test_sufficient_history_positive_mean_positive_sharpe_v7(self):
        rc = RewardCalculatorV7(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for r in [0.01, 0.02, 0.015, 0.01]:
            rc.returns_history.append(r)
        result = rc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=4)
        assert result["components"]["sharpe_reward"] > 0

    def test_sufficient_history_negative_mean_negative_sharpe_v8(self):
        rc = RewardCalculatorV8(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for r in [-0.01, -0.02, -0.015, -0.01]:
            rc.returns_history.append(r)
        result = rc.calculate(10_000_000, 9_900_000, action=0.5, prev_action=0.5, step=4)
        assert result["components"]["sharpe_reward"] < 0

    def test_zero_std_returns_mean_scaling_v6(self):
        """모든 수익률 동일 → std ≈ 0 → mean * 10."""
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for _ in range(5):
            rc.returns_history.append(0.01)
        sharpe = rc._compute_sharpe_reward(0.01)
        assert sharpe == pytest.approx(0.01 * 10, abs=1e-9)

    def test_zero_std_flat_returns_gives_zero_reward_v6(self):
        """수익률 0 연속 → sharpe_reward == 0."""
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for _ in range(5):
            rc.returns_history.append(0.0)
        sharpe = rc._compute_sharpe_reward(0.0)
        assert sharpe == pytest.approx(0.0, abs=1e-9)

    def test_v6_sharpe_clipped_at_positive_1_5(self):
        """mean 높고 std 낮은 교번 패턴 → Sharpe >> 10 → 상한 1.5 클리핑."""
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        vals = [0.10 + (i % 2) * 0.001 for i in range(20)]
        for v in vals:
            rc.returns_history.append(v)
        sharpe = rc._compute_sharpe_reward(0.10)
        assert sharpe == pytest.approx(1.5)

    def test_v6_sharpe_clipped_at_negative_1_5(self):
        """음수 mean 높고 std 낮은 패턴 → Sharpe << -10 → 하한 -1.5 클리핑."""
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        vals = [-0.10 + (i % 2) * 0.001 for i in range(20)]
        for v in vals:
            rc.returns_history.append(v)
        sharpe = rc._compute_sharpe_reward(-0.10)
        assert sharpe == pytest.approx(-1.5)


# ============================================================
# Section 4 — v8 고유 특성: trend_reward, holding_bonus, loss_penalty
# ============================================================

class TestV8UniqueFeatures:

    @pytest.fixture
    def calc(self):
        rc = RewardCalculatorV8(risk_free_rate=0.0)
        rc.reset(10_000_000)
        return rc

    # --- trend_reward ---

    def test_trend_reward_zero_when_insufficient_price_history(self, calc):
        """가격 히스토리 < 5이면 trend_reward == 0."""
        # price_history 비어있는 상태
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0, price=50_000_000)
        assert result["components"]["trend_reward"] == 0.0  # price_history len=1 < 5

    def test_trend_reward_positive_in_uptrend_with_long_position(self, calc):
        """상승 추세에서 long(양수 action) → trend_reward > 0."""
        prices = [50_000_000 + i * 100_000 for i in range(6)]
        for p in prices:
            calc.price_history.append(p)
        # 긍정적 행동 (long)
        reward = calc._compute_trend_reward(action=0.8)
        assert reward > 0

    def test_trend_reward_negative_in_uptrend_with_short_position(self, calc):
        """상승 추세에서 short(음수 action) → trend_reward < 0."""
        prices = [50_000_000 + i * 100_000 for i in range(6)]
        for p in prices:
            calc.price_history.append(p)
        reward = calc._compute_trend_reward(action=-0.8)
        assert reward < 0

    def test_trend_reward_positive_in_downtrend_with_short_position(self, calc):
        """하락 추세에서 short(음수 action) → trend_reward > 0."""
        prices = [50_000_000 - i * 100_000 for i in range(6)]
        for p in prices:
            calc.price_history.append(p)
        reward = calc._compute_trend_reward(action=-0.8)
        assert reward > 0

    def test_trend_reward_bounded(self, calc):
        """trend_reward 크기는 0.15 이하."""
        prices = [50_000_000 + i * 500_000 for i in range(10)]
        for p in prices:
            calc.price_history.append(p)
        reward = calc._compute_trend_reward(action=1.0)
        assert abs(reward) <= 0.15

    # --- holding_bonus ---

    def test_holding_bonus_on_profitable_held_position(self, calc):
        """수익 중이고 유의미한 포지션 유지 시 holding_bonus > 0."""
        # action_change <= 0.05 (거래 없음), raw_return > 0, abs(action) > 0.3
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_050_000,  # +0.5%
            action=0.8,
            prev_action=0.8,  # 변화 없음
            step=0,
        )
        assert result["components"]["holding_bonus"] == pytest.approx(0.04)

    def test_no_holding_bonus_when_losing(self, calc):
        """손실 중인 포지션 유지 → holding_bonus == 0."""
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=9_950_000,  # -0.5%
            action=0.8,
            prev_action=0.8,
            step=0,
        )
        assert result["components"]["holding_bonus"] == pytest.approx(0.0)

    def test_no_holding_bonus_when_position_too_small(self, calc):
        """포지션 abs(action) <= 0.3 → holding_bonus == 0."""
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_050_000,
            action=0.2,  # 포지션 작음
            prev_action=0.2,
            step=0,
        )
        assert result["components"]["holding_bonus"] == pytest.approx(0.0)

    def test_no_holding_bonus_when_trade_occurs(self, calc):
        """거래 발생 시 (action_change > 0.05) holding_bonus == 0."""
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_050_000,
            action=0.8,
            prev_action=0.0,  # 변화 큼 (거래 발생)
            step=0,
        )
        assert result["components"]["holding_bonus"] == pytest.approx(0.0)

    # --- loss_penalty ---

    def test_loss_penalty_on_loss_trade(self, calc):
        """손실 거래(raw_return < -0.1%) → loss_penalty < 0."""
        calc.steps_since_trade = 100  # 비활동 페널티 방지
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=9_900_000,  # -1%
            action=0.8,
            prev_action=0.0,  # action_change > 0.05 (거래 발생)
            step=0,
        )
        assert result["components"]["loss_penalty"] < 0
        assert result["components"]["loss_penalty"] == pytest.approx(-0.05)

    def test_no_loss_penalty_on_profit_trade(self, calc):
        """수익 거래 → loss_penalty == 0."""
        calc.steps_since_trade = 100
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_200_000,  # +2%
            action=0.8,
            prev_action=0.0,
            step=0,
        )
        assert result["components"]["loss_penalty"] == pytest.approx(0.0)

    def test_loss_penalty_escalates_on_consecutive_losses(self, calc):
        """연속 손실 3회 이상 → loss_penalty -0.15."""
        calc.consecutive_losses = 3  # 이미 3회 연속 손실 상태
        calc.steps_since_trade = 100
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=9_900_000,
            action=0.8,
            prev_action=0.0,
            step=0,
        )
        assert result["components"]["loss_penalty"] == pytest.approx(-0.15)

    def test_consecutive_losses_resets_on_profit(self, calc):
        """수익 거래 후 consecutive_losses 초기화."""
        calc.consecutive_losses = 5
        calc.steps_since_trade = 100
        calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_200_000,
            action=0.8,
            prev_action=0.0,
            step=0,
        )
        assert calc.consecutive_losses == 0

    def test_v8_reward_clipped_to_2(self, calc):
        """v8 보상 클리핑 [-2.0, 2.0]."""
        # 극단적 상황 시뮬레이션
        for _ in range(20):
            calc.returns_history.append(0.5)
        for p in [50_000_000 + i * 200_000 for i in range(10)]:
            calc.price_history.append(p)
        calc.steps_since_trade = 100
        result = calc.calculate(
            10_000_000, 15_000_000,
            action=1.0, prev_action=0.0, step=0,
        )
        assert result["reward"] <= 2.0
        assert result["reward"] >= -2.0

    def test_v8_components_keys(self, calc):
        """v8 컴포넌트 키 검증."""
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0)
        expected_keys = {
            "raw_return", "sharpe_reward", "trend_reward", "pnl_reward",
            "mdd_penalty", "profit_bonus", "trade_incentive", "holding_bonus",
            "loss_penalty", "inactivity_penalty", "drawdown", "total_trades",
        }
        assert expected_keys == set(result["components"].keys())


# ============================================================
# Section 5 — v7 고유 특성
# ============================================================

class TestV7Features:

    @pytest.fixture
    def calc(self):
        rc = RewardCalculatorV7(risk_free_rate=0.0)
        rc.reset(10_000_000)
        return rc

    def test_v7_components_keys(self, calc):
        result = calc.calculate(10_000_000, 10_100_000, action=0.5, prev_action=0.5, step=0)
        expected_keys = {
            "raw_return", "sharpe_reward", "trend_reward", "pnl_reward",
            "mdd_penalty", "profit_bonus", "holding_bonus", "drawdown",
            "dd_threshold", "total_trades",
        }
        assert expected_keys == set(result["components"].keys())

    def test_v7_reward_clipped_to_2(self, calc):
        for _ in range(20):
            calc.returns_history.append(0.5)
        for p in [50_000_000 + i * 200_000 for i in range(10)]:
            calc.price_history.append(p)
        calc.steps_since_trade = 100
        result = calc.calculate(10_000_000, 15_000_000, action=1.0, prev_action=0.0, step=0)
        assert -2.0 <= result["reward"] <= 2.0

    def test_v7_holding_bonus_on_profit_hold(self, calc):
        """수익 중 포지션 유지 → holding_bonus > 0."""
        result = calc.calculate(
            prev_portfolio_value=10_000_000,
            curr_portfolio_value=10_100_000,
            action=0.8,
            prev_action=0.8,  # no trade
            step=0,
        )
        assert result["components"]["holding_bonus"] == pytest.approx(0.05)

    def test_v7_adaptive_dd_threshold_changes_with_volatility(self, calc):
        """변동성 기반 dd_threshold가 고정 0.05가 아닌 동적 값."""
        # 낮은 변동성 히스토리
        for _ in range(5):
            calc.returns_history.append(0.001)
        result_low_vol = calc.calculate(10_000_000, 10_010_000, action=0.5, prev_action=0.5, step=5)
        threshold_low = result_low_vol["components"]["dd_threshold"]

        # 높은 변동성 히스토리
        calc.reset(10_000_000)
        for _ in range(5):
            calc.returns_history.append(0.05)
        result_high_vol = calc.calculate(10_000_000, 10_010_000, action=0.5, prev_action=0.5, step=5)
        threshold_high = result_high_vol["components"]["dd_threshold"]

        # 변동성이 높을수록 threshold도 높아야 함 (또는 한계값에 클리핑)
        assert threshold_high >= threshold_low

    def test_v7_pnl_reward_positive_on_gain(self, calc):
        result = calc.calculate(10_000_000, 10_200_000, action=0.5, prev_action=0.5, step=0)
        assert result["components"]["pnl_reward"] > 0

    def test_v7_pnl_reward_negative_on_loss(self, calc):
        result = calc.calculate(10_000_000, 9_800_000, action=0.5, prev_action=0.5, step=0)
        assert result["components"]["pnl_reward"] < 0


# ============================================================
# Section 6 — environment.py: reward_version 파라미터 동작
# ============================================================

class TestEnvironmentRewardVersion:

    def test_v6_uses_reward_calculator(self):
        env = _make_env(reward_version="v6")
        assert type(env.reward_calc).__name__ == "RewardCalculator"

    def test_v7_uses_reward_calculator_v7(self):
        env = _make_env(reward_version="v7")
        assert type(env.reward_calc).__name__ == "RewardCalculatorV7"

    def test_v8_uses_reward_calculator_v8(self):
        env = _make_env(reward_version="v8")
        assert type(env.reward_calc).__name__ == "RewardCalculatorV8"

    def test_default_reward_version_is_v6(self):
        candles = _make_candles(30)
        env = BitcoinTradingEnv(candles=candles, lookback=5, initial_balance=10_000_000)
        assert env.reward_version == "v6"
        assert type(env.reward_calc).__name__ == "RewardCalculator"

    def test_v6_step_produces_reward(self):
        env = _make_env(reward_version="v6")
        env.reset(seed=0, options={"start_idx": 5})
        _, reward, _, _, _ = env.step(np.array([0.5]))
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_v7_step_produces_reward(self):
        env = _make_env(reward_version="v7")
        env.reset(seed=0, options={"start_idx": 5})
        _, reward, _, _, _ = env.step(np.array([0.5]))
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_v8_step_produces_reward(self):
        env = _make_env(reward_version="v8")
        env.reset(seed=0, options={"start_idx": 5})
        _, reward, _, _, _ = env.step(np.array([0.5]))
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_v7_step_includes_price_in_calc(self):
        """v7/v8는 step에서 price를 reward_calc에 전달한다."""
        candles = _make_candles(30, price_step=10_000)
        env = BitcoinTradingEnv(
            candles=candles, lookback=5, initial_balance=10_000_000, reward_version="v7"
        )
        env.reset(seed=0, options={"start_idx": 5})
        # 여러 스텝 진행 후 price_history가 채워져야 함
        for _ in range(6):
            env.step(np.array([0.5]))
        assert len(env.reward_calc.price_history) >= 1

    def test_v6_does_not_have_price_history(self):
        """v6 RewardCalculator에는 price_history 속성이 없다."""
        env = _make_env(reward_version="v6")
        assert not hasattr(env.reward_calc, "price_history")

    def test_reward_components_in_info(self):
        """step info에 reward_components 포함."""
        for version in ("v6", "v7", "v8"):
            env = _make_env(reward_version=version)
            env.reset(seed=0, options={"start_idx": 5})
            _, _, _, _, info = env.step(np.array([0.0]))
            assert "reward_components" in info, f"{version}: reward_components missing from info"


# ============================================================
# Section 7 — get_episode_stats() 반환값 검증
# ============================================================

class TestGetEpisodeStats:

    def _run_env_steps(self, env, n_steps: int = 5, start_idx: int = 5):
        env.reset(seed=0, options={"start_idx": start_idx})
        for _ in range(n_steps):
            env.step(np.array([0.0]))
        return env.get_episode_stats()

    # --- v6 ---

    def test_v6_stats_keys(self):
        env = _make_env(reward_version="v6")
        stats = self._run_env_steps(env)
        required = {
            "total_return_pct", "total_trades", "avg_return",
            "std_return", "max_drawdown", "sharpe_ratio",
            "trade_count", "final_value", "steps",
        }
        assert required.issubset(set(stats.keys()))

    def test_v6_steps_matches_actual_steps(self):
        env = _make_env(reward_version="v6")
        n = 5
        stats = self._run_env_steps(env, n_steps=n)
        assert stats["steps"] == n

    def test_v6_final_value_is_positive(self):
        env = _make_env(reward_version="v6")
        stats = self._run_env_steps(env)
        assert stats["final_value"] > 0

    def test_v6_total_return_near_zero_on_hold_with_constant_price(self):
        """일정 가격 + 전량 현금 보유 → 수익률 ≈ 0."""
        env = _make_env(reward_version="v6", n_candles=20, lookback=3)
        env.reset(seed=0, options={"start_idx": 3})
        for _ in range(5):
            env.step(np.array([-1.0]))  # 전량 매도(KRW 유지)
        stats = env.get_episode_stats()
        assert abs(stats["total_return_pct"]) < 1.0

    def test_v6_trade_count_in_stats(self):
        candles = _make_candles(30)
        env = BitcoinTradingEnv(candles=candles, lookback=5, initial_balance=10_000_000, reward_version="v6")
        env.reset(seed=0, options={"start_idx": 5})
        env.step(np.array([1.0]))   # 전량 매수
        env.step(np.array([-1.0]))  # 전량 매도
        stats = env.get_episode_stats()
        assert stats["trade_count"] >= 1

    def test_v6_max_drawdown_non_negative(self):
        env = _make_env(reward_version="v6")
        stats = self._run_env_steps(env)
        assert stats["max_drawdown"] >= 0

    # --- v7 ---

    def test_v7_stats_keys(self):
        env = _make_env(reward_version="v7")
        stats = self._run_env_steps(env)
        required = {
            "total_return_pct", "total_trades", "avg_return",
            "std_return", "max_drawdown", "sharpe_ratio",
            "trade_count", "final_value", "steps",
        }
        assert required.issubset(set(stats.keys()))

    def test_v7_final_value_matches_portfolio(self):
        env = _make_env(reward_version="v7", n_candles=20, lookback=3)
        env.reset(seed=0, options={"start_idx": 3})
        for _ in range(5):
            env.step(np.array([-1.0]))  # KRW 유지
        stats = env.get_episode_stats()
        # KRW만 유지하므로 final_value ≈ initial_balance
        assert stats["final_value"] == pytest.approx(10_000_000, rel=0.01)

    # --- v8 ---

    def test_v8_stats_keys(self):
        env = _make_env(reward_version="v8")
        stats = self._run_env_steps(env)
        required = {
            "total_return_pct", "total_trades", "avg_return",
            "std_return", "max_drawdown", "sharpe_ratio",
            "trade_count", "final_value", "steps",
        }
        assert required.issubset(set(stats.keys()))

    def test_v8_rising_market_buy_shows_positive_return(self):
        """상승장 매수 → 양수 수익률."""
        candles = _make_candles(20, base_price=50_000_000, price_step=500_000)
        env = BitcoinTradingEnv(
            candles=candles, lookback=3, initial_balance=10_000_000, reward_version="v8"
        )
        env.reset(seed=0, options={"start_idx": 3})
        env.step(np.array([1.0]))  # 전량 매수
        for _ in range(5):
            env.step(np.array([1.0]))  # 유지
        stats = env.get_episode_stats()
        assert stats["total_return_pct"] > 0

    # --- 공통: get_episode_stats 직접 호출 (RewardCalculator) ---

    def test_reward_calc_v6_stats_structure(self):
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for i in range(10):
            rc.calculate(10_000_000, 10_000_000 + i * 5_000, action=0.5, prev_action=0.5, step=i)
        stats = rc.get_episode_stats(final_value=10_050_000, initial_value=10_000_000)
        assert set(stats.keys()) == {
            "total_return_pct", "total_trades", "avg_return",
            "std_return", "max_drawdown", "sharpe_ratio",
        }

    def test_reward_calc_v6_total_return_correct(self):
        rc = RewardCalculator(risk_free_rate=0.0)
        rc.reset(10_000_000)
        stats = rc.get_episode_stats(final_value=11_000_000, initial_value=10_000_000)
        assert stats["total_return_pct"] == pytest.approx(10.0)

    def test_reward_calc_v6_empty_history_no_crash(self):
        rc = RewardCalculator()
        rc.reset(10_000_000)
        stats = rc.get_episode_stats(final_value=10_000_000, initial_value=10_000_000)
        assert stats["total_return_pct"] == 0.0
        assert stats["sharpe_ratio"] == 0.0

    def test_reward_calc_v7_stats_structure(self):
        rc = RewardCalculatorV7(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for i in range(5):
            rc.calculate(10_000_000, 10_050_000, action=0.5, prev_action=0.5, step=i)
        stats = rc.get_episode_stats(final_value=10_050_000, initial_value=10_000_000)
        assert "total_return_pct" in stats
        assert "sharpe_ratio" in stats
        assert stats["total_return_pct"] == pytest.approx(0.5)

    def test_reward_calc_v8_stats_structure(self):
        rc = RewardCalculatorV8(risk_free_rate=0.0)
        rc.reset(10_000_000)
        for i in range(5):
            rc.calculate(10_000_000, 10_050_000, action=0.5, prev_action=0.5, step=i)
        stats = rc.get_episode_stats(final_value=10_050_000, initial_value=10_000_000)
        assert "total_return_pct" in stats
        assert stats["total_return_pct"] == pytest.approx(0.5)


# ============================================================
# Section 8 — HistoricalDataLoader (단위 테스트)
# ============================================================

class TestHistoricalDataLoader:

    def test_init_default_market(self):
        loader = HistoricalDataLoader()
        assert loader.market == "KRW-BTC"

    def test_init_custom_market(self):
        loader = HistoricalDataLoader(market="KRW-ETH")
        assert loader.market == "KRW-ETH"

    def test_compute_indicators_output_length(self):
        """compute_indicators 결과 길이는 입력과 동일."""
        loader = HistoricalDataLoader()
        candles = _make_candles(50)
        enriched = loader.compute_indicators(candles)
        assert len(enriched) == 50

    def test_compute_indicators_adds_required_keys(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(30)
        enriched = loader.compute_indicators(candles)
        required_keys = {
            "rsi_14", "sma_20", "sma_50",
            "macd", "macd_signal", "macd_histogram",
            "boll_upper", "boll_middle", "boll_lower",
            "stoch_k", "stoch_d", "atr", "adx", "change_rate",
        }
        for key in required_keys:
            assert key in enriched[0], f"missing indicator: {key}"

    def test_compute_indicators_rsi_range(self):
        """RSI는 0~100 범위."""
        loader = HistoricalDataLoader()
        candles = _make_candles(50, price_step=10_000)
        enriched = loader.compute_indicators(candles)
        rsi_values = [c["rsi_14"] for c in enriched]
        assert all(0 <= r <= 100 for r in rsi_values)

    def test_compute_indicators_sma20_greater_or_equal_single_val(self):
        """SMA20은 유한수."""
        loader = HistoricalDataLoader()
        candles = _make_candles(30)
        enriched = loader.compute_indicators(candles)
        for c in enriched:
            assert math.isfinite(c["sma_20"])

    def test_compute_indicators_bollinger_order(self):
        """볼린저: lower <= middle <= upper."""
        loader = HistoricalDataLoader()
        candles = _make_candles(50, price_step=1_000)
        enriched = loader.compute_indicators(candles)
        for c in enriched[20:]:  # 충분한 히스토리 이후
            assert c["boll_lower"] <= c["boll_middle"] + 1e-6
            assert c["boll_middle"] <= c["boll_upper"] + 1e-6

    def test_compute_indicators_change_rate_first_zero(self):
        """첫 캔들 change_rate == 0."""
        loader = HistoricalDataLoader()
        candles = _make_candles(10, price_step=5_000)
        enriched = loader.compute_indicators(candles)
        assert enriched[0]["change_rate"] == pytest.approx(0.0)

    def test_sma_static_method_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = HistoricalDataLoader._sma(data, 3)
        # SMA(3) at index 4 = (3+4+5)/3 = 4.0
        assert result[4] == pytest.approx(4.0)

    def test_ema_static_method_length(self):
        data = np.random.rand(20)
        result = HistoricalDataLoader._ema(data, 5)
        assert len(result) == 20

    def test_compute_rsi_output_shape(self):
        closes = np.array([50_000_000 + i * 10_000 for i in range(30)], dtype=float)
        rsi = HistoricalDataLoader._compute_rsi(closes, 14)
        assert rsi.shape == closes.shape

    def test_default_external_signal_keys(self):
        sig = HistoricalDataLoader._default_external_signal()
        required = {
            "fgi_value", "news_sentiment", "whale_score",
            "funding_rate", "long_short_ratio", "kimchi_premium_pct",
            "macro_score", "fusion_score",
        }
        assert required.issubset(set(sig.keys()))

    def test_align_external_no_signals_returns_defaults(self):
        loader = HistoricalDataLoader()
        candles = _make_candles(5)
        result = loader.align_external_to_candles(candles, signals=[])
        assert len(result) == 5
        assert result[0]["fgi_value"] == 50  # 기본값 확인

    def test_load_candles_unsupported_interval_raises(self):
        loader = HistoricalDataLoader()
        with pytest.raises(ValueError, match="지원하지 않는"):
            loader.load_candles(days=1, interval="2h")

    def test_cache_reused_on_second_call(self, monkeypatch):
        """동일 파라미터로 두 번 호출 시 캐시 사용 (requests 호출 없음)."""
        loader = HistoricalDataLoader()
        candles = _make_candles(5)
        loader._cache["KRW-BTC_1h_1"] = candles  # 수동 캐시 주입
        result = loader.load_candles(days=1, interval="1h")
        assert result is candles  # 동일 객체 반환 = 캐시 사용


# ============================================================
# Section 9 — 전체 통합: 환경 리셋 + 다수 스텝 + episode stats
# ============================================================

class TestIntegration:

    def test_full_episode_v6(self):
        """v6: 에피소드 전체 실행 후 stats 검증."""
        candles = _make_candles(30, price_step=50_000)
        env = BitcoinTradingEnv(candles=candles, lookback=5, initial_balance=10_000_000, reward_version="v6")
        env.reset(seed=42, options={"start_idx": 5})
        rewards = []
        for i in range(10):
            action = np.array([1.0 if i % 3 == 0 else -1.0])
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        stats = env.get_episode_stats()
        assert all(np.isfinite(rewards))
        assert stats["steps"] == len(rewards)
        assert stats["final_value"] > 0

    def test_full_episode_v7(self):
        """v7: 에피소드 전체 실행."""
        candles = _make_candles(30, price_step=50_000)
        env = BitcoinTradingEnv(candles=candles, lookback=5, initial_balance=10_000_000, reward_version="v7")
        env.reset(seed=42, options={"start_idx": 5})
        for i in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.5]))
            if terminated or truncated:
                break
        stats = env.get_episode_stats()
        assert "total_return_pct" in stats

    def test_full_episode_v8(self):
        """v8: 에피소드 전체 실행."""
        candles = _make_candles(30, price_step=50_000)
        env = BitcoinTradingEnv(candles=candles, lookback=5, initial_balance=10_000_000, reward_version="v8")
        env.reset(seed=42, options={"start_idx": 5})
        for i in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.5]))
            if terminated or truncated:
                break
        stats = env.get_episode_stats()
        assert "total_return_pct" in stats

    def test_multiple_resets_restore_state(self):
        """reset() 후 상태가 올바르게 복원되는지."""
        env = _make_env(n_candles=30, reward_version="v8")
        for _ in range(3):
            env.reset(seed=0, options={"start_idx": 5})
            assert env.krw_balance == 10_000_000
            assert env.btc_balance == 0.0
            assert env.trade_count == 0

    def test_v8_inactivity_penalty_triggers_after_36_steps(self):
        """36 스텝 이상 거래 없으면 inactivity_penalty < 0."""
        rc = RewardCalculatorV8(risk_free_rate=0.0)
        rc.reset(10_000_000)
        rc.steps_since_trade = 37  # 이미 37 스텝 비활동
        result = rc.calculate(
            10_000_000, 10_000_000,
            action=0.1, prev_action=0.1,  # 거래 없음 (change < 0.05)
            step=37,
        )
        assert result["components"]["inactivity_penalty"] < 0

    def test_v8_no_inactivity_penalty_before_36_steps(self):
        """36 스텝 이하 비활동 → inactivity_penalty == 0."""
        rc = RewardCalculatorV8(risk_free_rate=0.0)
        rc.reset(10_000_000)
        rc.steps_since_trade = 10
        result = rc.calculate(
            10_000_000, 10_000_000,
            action=0.1, prev_action=0.1,
            step=10,
        )
        assert result["components"]["inactivity_penalty"] == pytest.approx(0.0)
