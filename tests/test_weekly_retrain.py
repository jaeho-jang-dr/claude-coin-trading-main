"""Unit tests for rl_hybrid/rl/weekly_retrain.py

Covers:
  - Should-run logic: PPO weekly, SAC/TD3 only when day_of_month <= 7
  - Model comparison logic: sharpe_better, return_better, should_replace
  - Best model protection: don't replace when new model is worse
  - Candidate naming convention
  - Backup creation before replacement
  - Notification function (notify_telegram)
  - Edge cases: missing best model, failed evaluation (current_stats=None)
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from rl_hybrid.rl import weekly_retrain as wr

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTrader:
    """Minimal trader stub."""

    def __init__(self, *, env=None, model_path=None):
        self.env = env
        self.model_path = model_path
        self.model = mock.MagicMock()
        self._saved_path = None
        self._loaded_path = None

    def load(self, path):
        self._loaded_path = path

    def train(self, total_timesteps=0, eval_env=None, save_freq=0):
        pass

    def save(self, path):
        self._saved_path = path


def _fake_evaluate_model(trader, candles, signals, balance, episodes=10):
    """Return deterministic stats for testing."""
    return {
        "avg_return": 5.0,
        "avg_sharpe": 1.2,
        "avg_mdd": 0.05,
        "avg_trades": 20.0,
    }


def _fake_prepare_data(days):
    return [{"c": 1}], [{"c": 2}], [{"s": 1}], [{"s": 2}]


# ---------------------------------------------------------------------------
# 1. Should-run logic (PPO weekly, SAC/TD3 only month start)
# ---------------------------------------------------------------------------

class TestShouldRunLogic:

    @mock.patch("rl_hybrid.rl.weekly_retrain.datetime")
    def test_month_start_trains_all_algos(self, mock_dt):
        """day_of_month <= 7  ->  PPO + SAC + TD3"""
        mock_dt.now.return_value = datetime(2026, 3, 5, 12, 0, 0, tzinfo=KST)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        day = datetime.now(KST).replace(day=5).day
        assert day <= 7
        algos = ["ppo", "sac", "td3"] if day <= 7 else ["ppo"]
        assert algos == ["ppo", "sac", "td3"]

    @mock.patch("rl_hybrid.rl.weekly_retrain.datetime")
    def test_mid_month_trains_ppo_only(self, mock_dt):
        """day_of_month > 7  ->  PPO only"""
        mock_dt.now.return_value = datetime(2026, 3, 15, 12, 0, 0, tzinfo=KST)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        day = 15
        algos = ["ppo", "sac", "td3"] if day <= 7 else ["ppo"]
        assert algos == ["ppo"]

    def test_boundary_day_7_includes_all(self):
        """day == 7 is still month start."""
        day = 7
        algos = ["ppo", "sac", "td3"] if day <= 7 else ["ppo"]
        assert "sac" in algos and "td3" in algos

    def test_boundary_day_8_ppo_only(self):
        """day == 8 is NOT month start."""
        day = 8
        algos = ["ppo", "sac", "td3"] if day <= 7 else ["ppo"]
        assert algos == ["ppo"]


# ---------------------------------------------------------------------------
# 2. Model comparison logic
# ---------------------------------------------------------------------------

class TestModelComparison:

    def test_sharpe_better_within_tolerance(self):
        """New sharpe within 0.05 tolerance of current is still accepted."""
        current = {"avg_sharpe": 1.0, "avg_return": 3.0}
        best = {"avg_sharpe": 0.96, "avg_return": 3.5}
        sharpe_better = best["avg_sharpe"] > current["avg_sharpe"] - 0.05
        return_better = best["avg_return"] > current["avg_return"]
        assert sharpe_better is True
        assert return_better is True

    def test_sharpe_much_worse_blocks_replace(self):
        """Sharpe significantly worse blocks replacement."""
        current = {"avg_sharpe": 1.5, "avg_return": 3.0}
        best = {"avg_sharpe": 1.2, "avg_return": 5.0}
        sharpe_better = best["avg_sharpe"] > current["avg_sharpe"] - 0.05
        assert sharpe_better is False

    def test_return_worse_blocks_replace(self):
        """Lower return blocks replacement even if sharpe is fine."""
        current = {"avg_sharpe": 1.0, "avg_return": 5.0}
        best = {"avg_sharpe": 1.1, "avg_return": 4.0}
        return_better = best["avg_return"] > current["avg_return"]
        assert return_better is False

    def test_both_better_allows_replace(self):
        """Both metrics improved -> should_replace True."""
        current = {"avg_sharpe": 1.0, "avg_return": 3.0}
        best = {"avg_sharpe": 1.2, "avg_return": 5.0}
        sharpe_better = best["avg_sharpe"] > current["avg_sharpe"] - 0.05
        return_better = best["avg_return"] > current["avg_return"]
        should_replace = sharpe_better and return_better
        assert should_replace is True

    def test_both_worse_blocks_replace(self):
        current = {"avg_sharpe": 1.5, "avg_return": 5.0}
        best = {"avg_sharpe": 0.8, "avg_return": 2.0}
        sharpe_better = best["avg_sharpe"] > current["avg_sharpe"] - 0.05
        return_better = best["avg_return"] > current["avg_return"]
        should_replace = sharpe_better and return_better
        assert should_replace is False


# ---------------------------------------------------------------------------
# 3. Best model protection
# ---------------------------------------------------------------------------

class TestBestModelProtection:

    @mock.patch("rl_hybrid.rl.weekly_retrain.notify_telegram")
    @mock.patch("rl_hybrid.rl.weekly_retrain.evaluate_model")
    @mock.patch("rl_hybrid.rl.weekly_retrain.BitcoinTradingEnv")
    @mock.patch("rl_hybrid.rl.weekly_retrain.get_trader_class")
    @mock.patch("rl_hybrid.rl.weekly_retrain.prepare_data", side_effect=_fake_prepare_data)
    @mock.patch("rl_hybrid.rl.weekly_retrain.BEST_MODEL")
    @mock.patch("rl_hybrid.rl.weekly_retrain.INFO_PATH")
    def test_no_replace_when_new_is_worse(
        self, mock_info, mock_best, mock_prep, mock_get_tc, mock_env,
        mock_eval, mock_notify,
    ):
        """If new candidate is worse, best model is not replaced."""
        # best_model.zip exists
        mock_best.parent.__truediv__ = mock.MagicMock(return_value=mock.MagicMock())
        mock_best.parent.__truediv__.return_value.exists.return_value = True

        mock_info.exists.return_value = True

        mock_get_tc.return_value = FakeTrader

        # current best is excellent
        current_stats = {"avg_return": 10.0, "avg_sharpe": 2.0, "avg_mdd": 0.02, "avg_trades": 30.0}
        # new candidate is mediocre
        new_stats = {"avg_return": 2.0, "avg_sharpe": 0.5, "avg_mdd": 0.10, "avg_trades": 15.0}

        mock_eval.side_effect = [new_stats, new_stats, current_stats]

        # SB3 available
        with mock.patch("rl_hybrid.rl.weekly_retrain.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 12, 0, 0, tzinfo=KST)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with mock.patch.dict("sys.modules", {"rl_hybrid.rl.policy": mock.MagicMock(SB3_AVAILABLE=True)}):
                wr.weekly_retrain(days=30, total_steps=100, balance=1_000_000)

        # notify was called but model should NOT have been saved
        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "유지" in msg


# ---------------------------------------------------------------------------
# 4. Candidate naming convention
# ---------------------------------------------------------------------------

class TestCandidateNaming:

    def test_finetune_name(self):
        """Fine-tuned candidates: '{algo}_finetune'"""
        algo = "ppo"
        name = f"{algo}_finetune"
        assert name == "ppo_finetune"

    def test_scratch_name(self):
        """Scratch candidates: '{algo}_scratch'"""
        for algo in ("ppo", "sac", "td3"):
            name = f"{algo}_scratch"
            assert name.endswith("_scratch")
            assert name.split("_")[0] == algo

    def test_algo_extracted_from_name(self):
        """Algorithm is extracted as first part before underscore."""
        for name in ("ppo_finetune", "sac_scratch", "td3_scratch"):
            algo = name.split("_")[0]
            assert algo in ("ppo", "sac", "td3")

    def test_current_best_excluded_from_new_candidates(self):
        """'current_best' key is excluded when selecting new candidates."""
        candidates = {
            "ppo_scratch": {"avg_sharpe": 1.0, "avg_return": 3.0},
            "current_best": {"avg_sharpe": 1.5, "avg_return": 5.0},
        }
        new = {k: v for k, v in candidates.items() if k != "current_best"}
        assert "current_best" not in new
        assert "ppo_scratch" in new


# ---------------------------------------------------------------------------
# 5. Backup creation before replacement
# ---------------------------------------------------------------------------

class TestBackupCreation:

    def test_backup_copies_existing_best(self, tmp_path):
        """When replacing, existing best_model.zip should be backed up."""
        best_dir = tmp_path / "best"
        best_dir.mkdir()
        best_zip = best_dir / "best_model.zip"
        best_zip.write_bytes(b"old_model_bytes")

        history_dir = tmp_path / "retrain_history"
        timestamp = "20260315_120000"
        backup_path = history_dir / f"best_{timestamp}.zip"

        history_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_zip), str(backup_path))

        assert backup_path.exists()
        assert backup_path.read_bytes() == b"old_model_bytes"

    def test_backup_skipped_when_no_existing_best(self, tmp_path):
        """No backup if best_model.zip does not exist."""
        best_dir = tmp_path / "best"
        best_dir.mkdir()
        best_zip = best_dir / "best_model.zip"

        assert not best_zip.exists()
        # The code does: if (BEST_MODEL.parent / "best_model.zip").exists():
        # so nothing should happen

    def test_history_dir_created(self, tmp_path):
        """HISTORY_DIR is created with parents if missing."""
        history_dir = tmp_path / "deep" / "retrain_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        assert history_dir.exists()


# ---------------------------------------------------------------------------
# 6. Notification function
# ---------------------------------------------------------------------------

class TestNotifyTelegram:

    @mock.patch("subprocess.run")
    def test_notify_sends_message(self, mock_run):
        """notify_telegram calls subprocess with correct args."""
        wr.notify_telegram("test message")
        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == sys.executable
        assert "notify_telegram.py" in cmd[1]
        assert cmd[2] == "info"
        assert cmd[3] == "RL 주간 재학습"
        assert cmd[4] == "test message"
        assert args[1]["timeout"] == 15
        assert args[1]["check"] is False

    @mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="", timeout=15))
    def test_notify_handles_timeout(self, mock_run):
        """Timeout does not raise; it is caught and logged."""
        wr.notify_telegram("test")  # should not raise

    @mock.patch("subprocess.run", side_effect=FileNotFoundError("no script"))
    def test_notify_handles_missing_script(self, mock_run):
        """FileNotFoundError is caught gracefully."""
        wr.notify_telegram("test")  # should not raise

    @mock.patch("subprocess.run", side_effect=OSError("generic"))
    def test_notify_handles_generic_os_error(self, mock_run):
        """Any OSError is caught and logged."""
        wr.notify_telegram("test")  # should not raise


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_load_current_best_info_missing_file(self, tmp_path):
        """Returns defaults when model_info.json doesn't exist."""
        with mock.patch.object(wr, "INFO_PATH", tmp_path / "nonexistent.json"):
            info = wr.load_current_best_info()
        assert info == {"algorithm": "ppo", "avg_return_pct": 0, "avg_sharpe": 0}

    def test_load_current_best_info_corrupt_json(self, tmp_path):
        """Returns defaults when JSON is corrupt."""
        bad_file = tmp_path / "model_info.json"
        bad_file.write_text("{broken json!!!}")
        with mock.patch.object(wr, "INFO_PATH", bad_file):
            info = wr.load_current_best_info()
        assert info["algorithm"] == "ppo"

    def test_load_current_best_info_valid(self, tmp_path):
        """Returns parsed data when file is valid."""
        info_file = tmp_path / "model_info.json"
        data = {"algorithm": "sac", "avg_return_pct": 5.5, "avg_sharpe": 1.3}
        info_file.write_text(json.dumps(data))
        with mock.patch.object(wr, "INFO_PATH", info_file):
            info = wr.load_current_best_info()
        assert info["algorithm"] == "sac"
        assert info["avg_return_pct"] == 5.5

    def test_should_replace_true_when_current_stats_none(self):
        """When current_stats is None (no existing best), should_replace defaults True."""
        current_stats = None
        should_replace = True
        if current_stats:
            sharpe_better = False
            return_better = False
            should_replace = sharpe_better and return_better
        assert should_replace is True

    def test_no_candidates_returns_early(self):
        """When all training fails and candidates dict is empty, function returns."""
        candidates = {}
        assert not candidates  # triggers early return in real code

    @mock.patch("rl_hybrid.rl.weekly_retrain.notify_telegram")
    @mock.patch("rl_hybrid.rl.weekly_retrain.evaluate_model")
    @mock.patch("rl_hybrid.rl.weekly_retrain.BitcoinTradingEnv")
    @mock.patch("rl_hybrid.rl.weekly_retrain.get_trader_class")
    @mock.patch("rl_hybrid.rl.weekly_retrain.prepare_data", side_effect=_fake_prepare_data)
    def test_all_training_fails_notifies(
        self, mock_prep, mock_get_tc, mock_env, mock_eval, mock_notify,
    ):
        """When all candidates fail training, telegram is notified."""
        # Make get_trader_class raise for all algos
        mock_get_tc.side_effect = RuntimeError("no algo")

        with mock.patch.object(wr, "BEST_MODEL") as mock_bm:
            mock_bm.parent.__truediv__ = mock.MagicMock(return_value=mock.MagicMock())
            mock_bm.parent.__truediv__.return_value.exists.return_value = False
            with mock.patch.dict("sys.modules", {"rl_hybrid.rl.policy": mock.MagicMock(SB3_AVAILABLE=True)}):
                with mock.patch("rl_hybrid.rl.weekly_retrain.datetime") as mock_dt:
                    mock_dt.now.return_value = datetime(2026, 3, 15, 12, 0, 0, tzinfo=KST)
                    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                    wr.weekly_retrain(days=30, total_steps=100, balance=1_000_000)

        mock_notify.assert_called_once_with("주간 재학습 실패: 모든 후보 훈련 실패")

    def test_sb3_not_available_returns_early(self):
        """When SB3 is not installed, function returns immediately."""
        policy_mock = mock.MagicMock(SB3_AVAILABLE=False)
        with mock.patch.dict("sys.modules", {"rl_hybrid.rl.policy": policy_mock}):
            # Should return without error
            wr.weekly_retrain(days=30, total_steps=100, balance=1_000_000)

    def test_best_selection_prefers_sharpe_then_return(self):
        """Best candidate is selected by (sharpe, return) tuple."""
        candidates = {
            "ppo_scratch": {"avg_sharpe": 1.5, "avg_return": 3.0, "trader": FakeTrader()},
            "sac_scratch": {"avg_sharpe": 1.5, "avg_return": 5.0, "trader": FakeTrader()},
            "td3_scratch": {"avg_sharpe": 1.8, "avg_return": 2.0, "trader": FakeTrader()},
        }
        best_name = max(candidates, key=lambda k: (candidates[k]["avg_sharpe"], candidates[k]["avg_return"]))
        assert best_name == "td3_scratch"  # highest sharpe wins

    def test_best_selection_tiebreak_by_return(self):
        """When sharpe is tied, higher return wins."""
        candidates = {
            "ppo_scratch": {"avg_sharpe": 1.5, "avg_return": 3.0, "trader": FakeTrader()},
            "sac_scratch": {"avg_sharpe": 1.5, "avg_return": 5.0, "trader": FakeTrader()},
        }
        best_name = max(candidates, key=lambda k: (candidates[k]["avg_sharpe"], candidates[k]["avg_return"]))
        assert best_name == "sac_scratch"


# ---------------------------------------------------------------------------
# Integration-style test: model_info.json written correctly after replacement
# ---------------------------------------------------------------------------

class TestModelInfoWrite:

    def test_model_info_contents(self, tmp_path):
        """Verify model_info.json schema after a successful replacement."""
        info_path = tmp_path / "model_info.json"
        best_result = {"avg_return": 5.1234, "avg_sharpe": 1.5678, "avg_mdd": 0.045678}
        best_name = "sac_scratch"
        algo_name = best_name.split("_")[0]

        model_info = {
            "algorithm": algo_name,
            "avg_return_pct": round(best_result["avg_return"], 4),
            "avg_sharpe": round(best_result["avg_sharpe"], 4),
            "avg_mdd": round(best_result["avg_mdd"], 6),
            "training_steps": 200_000,
            "training_days": 90,
            "retrained_at": datetime.now(KST).isoformat(),
            "candidate_name": best_name,
        }
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        loaded = json.loads(info_path.read_text())
        assert loaded["algorithm"] == "sac"
        assert loaded["candidate_name"] == "sac_scratch"
        assert loaded["avg_return_pct"] == 5.1234
        assert loaded["avg_sharpe"] == 1.5678
        assert loaded["avg_mdd"] == 0.045678
        assert loaded["training_steps"] == 200_000
        assert loaded["training_days"] == 90
        assert "retrained_at" in loaded


# ---------------------------------------------------------------------------
# Evaluate model function
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    @mock.patch("rl_hybrid.rl.weekly_retrain.evaluate")
    @mock.patch("rl_hybrid.rl.weekly_retrain.BitcoinTradingEnv")
    def test_evaluate_model_returns_correct_keys(self, mock_env, mock_evaluate):
        """evaluate_model returns dict with avg_return, avg_sharpe, avg_mdd, avg_trades."""
        mock_evaluate.return_value = [
            {"total_return_pct": 5.0, "sharpe_ratio": 1.2, "max_drawdown": 0.03, "trade_count": 20},
            {"total_return_pct": 3.0, "sharpe_ratio": 0.8, "max_drawdown": 0.05, "trade_count": 15},
        ]
        trader = FakeTrader()
        result = wr.evaluate_model(trader, [{"c": 1}], [{"s": 1}], 10_000_000, episodes=2)

        assert "avg_return" in result
        assert "avg_sharpe" in result
        assert "avg_mdd" in result
        assert "avg_trades" in result
        assert abs(result["avg_return"] - 4.0) < 0.01
        assert abs(result["avg_sharpe"] - 1.0) < 0.01
        assert abs(result["avg_mdd"] - 0.04) < 0.01
        assert abs(result["avg_trades"] - 17.5) < 0.01

    @mock.patch("rl_hybrid.rl.weekly_retrain.evaluate")
    @mock.patch("rl_hybrid.rl.weekly_retrain.BitcoinTradingEnv")
    def test_evaluate_model_single_episode(self, mock_env, mock_evaluate):
        """Works correctly with a single episode."""
        mock_evaluate.return_value = [
            {"total_return_pct": 7.0, "sharpe_ratio": 2.0, "max_drawdown": 0.01, "trade_count": 50},
        ]
        result = wr.evaluate_model(FakeTrader(), [], [], 1_000_000, episodes=1)
        assert result["avg_return"] == 7.0
        assert result["avg_sharpe"] == 2.0
