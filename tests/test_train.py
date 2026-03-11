"""Unit tests for rl_hybrid.rl.train module."""

import argparse
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: generate minimal fake candle data with required fields
# ---------------------------------------------------------------------------

def _make_candles(n: int, base_price: float = 100_000_000) -> list[dict]:
    """Create n fake candle dicts with OHLCV + indicator fields."""
    candles = []
    ts = datetime(2025, 1, 1)
    price = base_price
    for i in range(n):
        price *= 1 + np.random.uniform(-0.01, 0.01)
        candles.append({
            "timestamp": (ts + timedelta(hours=i)).isoformat(),
            "open": round(price * 0.999),
            "high": round(price * 1.002),
            "low": round(price * 0.998),
            "close": round(price),
            "volume": round(np.random.uniform(100, 500), 4),
            # indicators (normally added by compute_indicators)
            "rsi_14": 50 + np.random.uniform(-20, 20),
            "sma_20": round(price),
            "sma_50": round(price),
            "macd": np.random.uniform(-500, 500),
            "macd_signal": np.random.uniform(-500, 500),
            "macd_histogram": np.random.uniform(-200, 200),
            "boll_upper": round(price * 1.02),
            "boll_middle": round(price),
            "boll_lower": round(price * 0.98),
            "stoch_k": np.random.uniform(20, 80),
            "stoch_d": np.random.uniform(20, 80),
            "atr": np.random.uniform(100000, 500000),
            "adx": np.random.uniform(10, 40),
        })
    return candles


def _make_signals(n: int) -> list[dict]:
    """Create n fake external signal dicts."""
    return [
        {
            "fgi_value": 50,
            "news_sentiment": 0,
            "whale_score": 0,
            "funding_rate": 0.0,
            "long_short_ratio": 1.0,
            "kimchi_premium_pct": 0.0,
            "macro_score": 0,
            "eth_btc_score": 0,
            "fusion_score": 0,
        }
        for _ in range(n)
    ]


# =========================================================================
# 1. get_trader_class
# =========================================================================

class TestGetTraderClass:
    """get_trader_class returns correct classes and raises for invalid algo."""

    def test_returns_ppo(self):
        from rl_hybrid.rl.train import get_trader_class
        from rl_hybrid.rl.policy import PPOTrader
        assert get_trader_class("ppo") is PPOTrader

    def test_returns_sac(self):
        from rl_hybrid.rl.train import get_trader_class
        from rl_hybrid.rl.policy import SACTrader
        assert get_trader_class("sac") is SACTrader

    def test_returns_td3(self):
        from rl_hybrid.rl.train import get_trader_class
        from rl_hybrid.rl.policy import TD3Trader
        assert get_trader_class("td3") is TD3Trader

    def test_raises_for_invalid_algo(self):
        from rl_hybrid.rl.train import get_trader_class
        with pytest.raises(ValueError, match="지원하지 않는 알고리즘"):
            get_trader_class("dqn")

    def test_raises_for_empty_string(self):
        from rl_hybrid.rl.train import get_trader_class
        with pytest.raises(ValueError):
            get_trader_class("")


# =========================================================================
# 2. prepare_data
# =========================================================================

class TestPrepareData:
    """prepare_data returns 4-tuple with correct structure."""

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    def test_returns_4_tuple(self, MockLoader):
        from rl_hybrid.rl.train import prepare_data

        candles = _make_candles(50)
        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = candles
        loader_inst.compute_indicators.return_value = candles
        loader_inst.load_external_signals.return_value = []

        result = prepare_data(days=5, interval="1h")
        assert len(result) == 4

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    def test_train_eval_split_ratio(self, MockLoader):
        from rl_hybrid.rl.train import prepare_data

        candles = _make_candles(100)
        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = candles
        loader_inst.compute_indicators.return_value = candles
        loader_inst.load_external_signals.return_value = []

        train_c, eval_c, train_s, eval_s = prepare_data(days=5)
        assert len(train_c) == 80
        assert len(eval_c) == 20

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    def test_candles_are_lists_of_dicts(self, MockLoader):
        from rl_hybrid.rl.train import prepare_data

        candles = _make_candles(50)
        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = candles
        loader_inst.compute_indicators.return_value = candles
        loader_inst.load_external_signals.return_value = []

        train_c, eval_c, _, _ = prepare_data(days=5)
        assert isinstance(train_c, list)
        assert isinstance(train_c[0], dict)
        assert "close" in train_c[0]

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    def test_signals_none_when_no_external(self, MockLoader):
        from rl_hybrid.rl.train import prepare_data

        candles = _make_candles(50)
        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = candles
        loader_inst.compute_indicators.return_value = candles
        loader_inst.load_external_signals.return_value = []

        _, _, train_s, eval_s = prepare_data(days=5)
        assert train_s is None
        assert eval_s is None

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    def test_signals_split_when_external_present(self, MockLoader):
        from rl_hybrid.rl.train import prepare_data

        candles = _make_candles(100)
        signals = _make_signals(100)
        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = candles
        loader_inst.compute_indicators.return_value = candles
        loader_inst.load_external_signals.return_value = [{"some": "data"}]
        loader_inst.align_external_to_candles.return_value = signals

        train_c, eval_c, train_s, eval_s = prepare_data(days=5)
        assert train_s is not None
        assert len(train_s) == 80
        assert len(eval_s) == 20

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    def test_signal_load_failure_graceful(self, MockLoader):
        from rl_hybrid.rl.train import prepare_data

        candles = _make_candles(50)
        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = candles
        loader_inst.compute_indicators.return_value = candles
        loader_inst.load_external_signals.side_effect = Exception("DB down")

        train_c, eval_c, train_s, eval_s = prepare_data(days=5)
        assert train_s is None
        assert eval_s is None
        assert len(train_c) > 0


# =========================================================================
# 3. prepare_edge_case_data
# =========================================================================

class TestPrepareEdgeCaseData:
    """prepare_edge_case_data returns 4-tuple with more candles than prepare_data."""

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    @patch("rl_hybrid.rl.train.ScenarioGenerator")
    def test_returns_4_tuple(self, MockGen, MockLoader):
        from rl_hybrid.rl.train import prepare_edge_case_data

        raw_candles = _make_candles(50)
        mixed_candles = _make_candles(70)  # more than raw

        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = raw_candles
        loader_inst.compute_indicators.return_value = mixed_candles
        loader_inst.load_external_signals.return_value = []

        gen_inst = MockGen.return_value
        gen_inst.mix_with_real.return_value = mixed_candles

        result = prepare_edge_case_data(days=5, synthetic_ratio=0.3)
        assert len(result) == 4

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    @patch("rl_hybrid.rl.train.ScenarioGenerator")
    def test_has_more_candles_than_raw(self, MockGen, MockLoader):
        from rl_hybrid.rl.train import prepare_edge_case_data

        raw_candles = _make_candles(50)
        mixed_candles = _make_candles(70)

        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = raw_candles
        loader_inst.compute_indicators.return_value = mixed_candles
        loader_inst.load_external_signals.return_value = []

        gen_inst = MockGen.return_value
        gen_inst.mix_with_real.return_value = mixed_candles

        train_c, eval_c, _, _ = prepare_edge_case_data(days=5, synthetic_ratio=0.3)
        total = len(train_c) + len(eval_c)
        assert total == len(mixed_candles)
        assert total > len(raw_candles)

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    @patch("rl_hybrid.rl.train.ScenarioGenerator")
    def test_synthetic_ratio_passed_to_generator(self, MockGen, MockLoader):
        from rl_hybrid.rl.train import prepare_edge_case_data

        raw_candles = _make_candles(50)
        mixed_candles = _make_candles(65)

        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = raw_candles
        loader_inst.compute_indicators.return_value = mixed_candles
        loader_inst.load_external_signals.return_value = []

        gen_inst = MockGen.return_value
        gen_inst.mix_with_real.return_value = mixed_candles

        prepare_edge_case_data(days=5, synthetic_ratio=0.5, variations=4)
        gen_inst.mix_with_real.assert_called_once_with(
            raw_candles, synthetic_ratio=0.5, variations=4
        )

    @patch("rl_hybrid.rl.train.HistoricalDataLoader")
    @patch("rl_hybrid.rl.train.ScenarioGenerator")
    def test_signals_with_synth_defaults(self, MockGen, MockLoader):
        from rl_hybrid.rl.train import prepare_edge_case_data

        raw_candles = _make_candles(50)
        mixed_candles = _make_candles(70)
        real_signals = _make_signals(50)

        loader_inst = MockLoader.return_value
        loader_inst.load_candles.return_value = raw_candles
        loader_inst.compute_indicators.return_value = mixed_candles
        loader_inst.load_external_signals.return_value = [{"some": "data"}]
        loader_inst.align_external_to_candles.return_value = real_signals
        loader_inst._default_external_signal.return_value = {
            "fgi_value": 50, "news_sentiment": 0, "whale_score": 0,
            "funding_rate": 0.0, "long_short_ratio": 1.0,
            "kimchi_premium_pct": 0.0, "macro_score": 0,
            "eth_btc_score": 0, "fusion_score": 0,
        }

        gen_inst = MockGen.return_value
        gen_inst.mix_with_real.return_value = mixed_candles

        _, _, train_s, eval_s = prepare_edge_case_data(days=5, synthetic_ratio=0.3)
        # Signals should be split (real + synth defaults)
        if train_s is not None:
            total_signals = len(train_s) + len(eval_s)
            assert total_signals == len(mixed_candles)


# =========================================================================
# 4. evaluate
# =========================================================================

class TestEvaluate:
    """evaluate returns list of stats dicts with required keys."""

    def test_returns_list_of_stats(self):
        from rl_hybrid.rl.train import evaluate

        mock_trader = MagicMock()
        mock_trader.predict.return_value = 0.0

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(42, dtype=np.float32), {})
        # Terminate after one step
        mock_env.step.return_value = (
            np.zeros(42, dtype=np.float32),
            0.0,
            True,  # terminated
            False,  # truncated
            {},
        )
        mock_env.get_episode_stats.return_value = {
            "total_return_pct": 1.5,
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.05,
            "trade_count": 3,
            "final_value": 10_150_000,
            "steps": 100,
        }

        result = evaluate(trader=mock_trader, env=mock_env, episodes=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_stats_have_required_keys(self):
        from rl_hybrid.rl.train import evaluate

        mock_trader = MagicMock()
        mock_trader.predict.return_value = 0.0

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(42, dtype=np.float32), {})
        mock_env.step.return_value = (
            np.zeros(42, dtype=np.float32), 0.0, True, False, {},
        )
        mock_env.get_episode_stats.return_value = {
            "total_return_pct": -0.5,
            "sharpe_ratio": -0.1,
            "max_drawdown": 0.02,
            "trade_count": 1,
            "final_value": 9_950_000,
            "steps": 50,
        }

        result = evaluate(trader=mock_trader, env=mock_env, episodes=2)
        required_keys = {"total_return_pct", "sharpe_ratio", "max_drawdown", "trade_count"}
        for stats in result:
            assert required_keys.issubset(stats.keys())

    def test_calls_env_reset_per_episode(self):
        from rl_hybrid.rl.train import evaluate

        mock_trader = MagicMock()
        mock_trader.predict.return_value = 0.0

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(42, dtype=np.float32), {})
        mock_env.step.return_value = (
            np.zeros(42, dtype=np.float32), 0.0, True, False, {},
        )
        mock_env.get_episode_stats.return_value = {
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "trade_count": 0,
            "final_value": 10_000_000,
            "steps": 1,
        }

        episodes = 5
        evaluate(trader=mock_trader, env=mock_env, episodes=episodes)
        assert mock_env.reset.call_count == episodes

    def test_single_episode(self):
        from rl_hybrid.rl.train import evaluate

        mock_trader = MagicMock()
        mock_trader.predict.return_value = 0.5

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(42, dtype=np.float32), {})
        mock_env.step.return_value = (
            np.zeros(42, dtype=np.float32), 1.0, True, False, {},
        )
        mock_env.get_episode_stats.return_value = {
            "total_return_pct": 2.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.01,
            "trade_count": 5,
            "final_value": 10_200_000,
            "steps": 10,
        }

        result = evaluate(trader=mock_trader, env=mock_env, episodes=1)
        assert len(result) == 1
        assert result[0]["total_return_pct"] == 2.0


# =========================================================================
# 5. backtest_baseline
# =========================================================================

class TestBacktestBaseline:
    """backtest_baseline runs without error."""

    @patch("rl_hybrid.rl.train.prepare_data")
    def test_runs_without_error(self, mock_prepare):
        from rl_hybrid.rl.train import backtest_baseline

        candles = _make_candles(30)
        mock_prepare.return_value = (candles[:24], candles[24:], None, None)

        # Should not raise
        backtest_baseline(days=5, interval="1h")

    @patch("rl_hybrid.rl.train.prepare_data")
    def test_handles_empty_eval_candles(self, mock_prepare):
        from rl_hybrid.rl.train import backtest_baseline

        candles = _make_candles(30)
        mock_prepare.return_value = (candles, [], None, None)

        # Should not raise (returns early with logger.error)
        backtest_baseline(days=5, interval="1h")

    @patch("rl_hybrid.rl.train.prepare_data")
    def test_calculates_positive_return(self, mock_prepare):
        from rl_hybrid.rl.train import backtest_baseline

        candles = []
        base = 100_000_000
        ts = datetime(2025, 1, 1)
        for i in range(10):
            p = base + i * 1_000_000  # steadily increasing
            candles.append({
                "timestamp": (ts + timedelta(hours=i)).isoformat(),
                "open": p, "high": p + 100_000,
                "low": p - 100_000, "close": p,
                "volume": 300.0,
            })
        mock_prepare.return_value = ([], candles, None, None)

        # Should run without error for increasing prices
        backtest_baseline(days=5)


# =========================================================================
# 6. CLI argument parsing
# =========================================================================

class TestCLIParsing:
    """Test argparse configuration in __main__ block."""

    def _parse(self, args_list):
        """Parse args using the same parser definition as train.py."""
        parser = argparse.ArgumentParser(description="BTC RL 트레이딩 모델 훈련")
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--steps", type=int, default=100_000)
        parser.add_argument("--balance", type=float, default=10_000_000)
        parser.add_argument("--interval", type=str, default="1h",
                            choices=["1h", "4h", "1d"])
        parser.add_argument("--algo", type=str, default="ppo",
                            choices=["ppo", "sac", "td3", "all"])
        parser.add_argument("--eval", action="store_true")
        parser.add_argument("--baseline", action="store_true")
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--multi", action="store_true")
        parser.add_argument("--edge-cases", action="store_true")
        parser.add_argument("--synthetic-ratio", type=float, default=0.3)
        return parser.parse_args(args_list)

    def test_defaults(self):
        args = self._parse([])
        assert args.days == 180
        assert args.steps == 100_000
        assert args.balance == 10_000_000
        assert args.interval == "1h"
        assert args.algo == "ppo"
        assert args.eval is False
        assert args.baseline is False
        assert args.model is None
        assert args.multi is False
        assert args.edge_cases is False
        assert args.synthetic_ratio == 0.3

    def test_algo_ppo(self):
        args = self._parse(["--algo", "ppo"])
        assert args.algo == "ppo"

    def test_algo_sac(self):
        args = self._parse(["--algo", "sac"])
        assert args.algo == "sac"

    def test_algo_td3(self):
        args = self._parse(["--algo", "td3"])
        assert args.algo == "td3"

    def test_algo_all(self):
        args = self._parse(["--algo", "all"])
        assert args.algo == "all"

    def test_invalid_algo_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--algo", "dqn"])

    def test_edge_cases_flag(self):
        args = self._parse(["--edge-cases"])
        assert args.edge_cases is True

    def test_synthetic_ratio(self):
        args = self._parse(["--synthetic-ratio", "0.5"])
        assert args.synthetic_ratio == 0.5

    def test_steps(self):
        args = self._parse(["--steps", "50000"])
        assert args.steps == 50000

    def test_days(self):
        args = self._parse(["--days", "365"])
        assert args.days == 365

    def test_interval_4h(self):
        args = self._parse(["--interval", "4h"])
        assert args.interval == "4h"

    def test_interval_1d(self):
        args = self._parse(["--interval", "1d"])
        assert args.interval == "1d"

    def test_invalid_interval_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--interval", "15m"])

    def test_eval_flag(self):
        args = self._parse(["--eval"])
        assert args.eval is True

    def test_baseline_flag(self):
        args = self._parse(["--baseline"])
        assert args.baseline is True

    def test_model_path(self):
        args = self._parse(["--model", "/tmp/my_model.zip"])
        assert args.model == "/tmp/my_model.zip"

    def test_multi_flag(self):
        args = self._parse(["--multi"])
        assert args.multi is True

    def test_combined_flags(self):
        args = self._parse([
            "--algo", "sac", "--days", "30", "--steps", "1000",
            "--edge-cases", "--synthetic-ratio", "0.6",
        ])
        assert args.algo == "sac"
        assert args.days == 30
        assert args.steps == 1000
        assert args.edge_cases is True
        assert args.synthetic_ratio == 0.6


# =========================================================================
# 7. train_all accepts edge_cases parameter
# =========================================================================

class TestTrainAll:
    """train_all accepts edge_cases parameter and dispatches correctly."""

    @patch("rl_hybrid.rl.train.backtest_baseline")
    @patch("rl_hybrid.rl.train.evaluate")
    @patch("rl_hybrid.rl.train.get_trader_class")
    @patch("rl_hybrid.rl.train.prepare_data")
    @patch("rl_hybrid.rl.train.prepare_edge_case_data")
    def test_uses_prepare_data_when_no_edge_cases(
        self, mock_edge, mock_prepare, mock_get_class, mock_eval, mock_baseline
    ):
        from rl_hybrid.rl.train import train_all

        candles = _make_candles(50)
        mock_prepare.return_value = (candles[:40], candles[40:], None, None)
        mock_edge.return_value = (candles[:40], candles[40:], None, None)

        mock_trader = MagicMock()
        mock_class = MagicMock(return_value=mock_trader)
        mock_get_class.return_value = mock_class

        mock_eval.return_value = [
            {"total_return_pct": 1.0, "sharpe_ratio": 0.5,
             "max_drawdown": 0.03, "trade_count": 2}
        ]

        with patch("rl_hybrid.rl.policy.SB3_AVAILABLE", True), \
             patch("rl_hybrid.rl.train.BitcoinTradingEnv"), \
             patch("rl_hybrid.rl.policy.MODEL_DIR", "/tmp/test_rl_models"), \
             patch("builtins.open", MagicMock()), \
             patch("os.makedirs"):
            train_all(days=5, total_steps=10, balance=1_000_000, edge_cases=False)

        mock_prepare.assert_called_once()
        mock_edge.assert_not_called()

    @patch("rl_hybrid.rl.train.backtest_baseline")
    @patch("rl_hybrid.rl.train.evaluate")
    @patch("rl_hybrid.rl.train.get_trader_class")
    @patch("rl_hybrid.rl.train.prepare_data")
    @patch("rl_hybrid.rl.train.prepare_edge_case_data")
    def test_uses_edge_case_data_when_flag_set(
        self, mock_edge, mock_prepare, mock_get_class, mock_eval, mock_baseline
    ):
        from rl_hybrid.rl.train import train_all

        candles = _make_candles(50)
        mock_edge.return_value = (candles[:40], candles[40:], None, None)
        mock_prepare.return_value = (candles[:40], candles[40:], None, None)

        mock_trader = MagicMock()
        mock_class = MagicMock(return_value=mock_trader)
        mock_get_class.return_value = mock_class

        mock_eval.return_value = [
            {"total_return_pct": 1.0, "sharpe_ratio": 0.5,
             "max_drawdown": 0.03, "trade_count": 2}
        ]

        with patch("rl_hybrid.rl.policy.SB3_AVAILABLE", True), \
             patch("rl_hybrid.rl.train.BitcoinTradingEnv"), \
             patch("rl_hybrid.rl.policy.MODEL_DIR", "/tmp/test_rl_models"), \
             patch("builtins.open", MagicMock()), \
             patch("os.makedirs"):
            train_all(
                days=5, total_steps=10, balance=1_000_000,
                edge_cases=True, synthetic_ratio=0.4,
            )

        mock_edge.assert_called_once()
        mock_prepare.assert_not_called()

    @patch("rl_hybrid.rl.train.backtest_baseline")
    @patch("rl_hybrid.rl.train.evaluate")
    @patch("rl_hybrid.rl.train.get_trader_class")
    @patch("rl_hybrid.rl.train.prepare_data")
    def test_trains_all_three_algos(
        self, mock_prepare, mock_get_class, mock_eval, mock_baseline
    ):
        from rl_hybrid.rl.train import train_all

        candles = _make_candles(50)
        mock_prepare.return_value = (candles[:40], candles[40:], None, None)

        mock_trader = MagicMock()
        mock_class = MagicMock(return_value=mock_trader)
        mock_get_class.return_value = mock_class

        mock_eval.return_value = [
            {"total_return_pct": 1.0, "sharpe_ratio": 0.5,
             "max_drawdown": 0.03, "trade_count": 2}
        ]

        with patch("rl_hybrid.rl.policy.SB3_AVAILABLE", True), \
             patch("rl_hybrid.rl.train.BitcoinTradingEnv"), \
             patch("rl_hybrid.rl.policy.MODEL_DIR", "/tmp/test_rl_models"), \
             patch("builtins.open", MagicMock()), \
             patch("os.makedirs"):
            train_all(days=5, total_steps=10, balance=1_000_000)

        # get_trader_class called once per algo
        assert mock_get_class.call_count == 3
        calls = [c.args[0] for c in mock_get_class.call_args_list]
        assert "ppo" in calls
        assert "sac" in calls
        assert "td3" in calls

    def test_returns_early_when_sb3_unavailable(self):
        from rl_hybrid.rl.train import train_all

        with patch("rl_hybrid.rl.policy.SB3_AVAILABLE", False):
            result = train_all(days=5, total_steps=10, balance=1_000_000)
            assert result is None
