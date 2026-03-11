"""Unit tests for rl_hybrid.rl.admin_review and rl_hybrid.rl.online_buffer

Covers:
  admin_review:
    - get_submissions_local() with empty dir / with JSON files
    - list_submissions() output formatting
    - show_leaderboard() grouping logic
    - promote_model() model file handling

  online_buffer:
    - OnlineExperienceBuffer initialization (empty / existing file)
    - add_experience() correctly stores data
    - should_train() triggers at TRIGGER_SIZE
    - Buffer persistence (save/load)
    - update_outcomes() correctly updates entries
"""

import json
import os
import sys
import tempfile
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch, call

import pytest

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rl_hybrid.rl import admin_review
from rl_hybrid.rl import online_buffer
from rl_hybrid.rl.online_buffer import OnlineExperienceBuffer, TRIGGER_SIZE


# ============================================================================
# admin_review: get_submissions_local()
# ============================================================================

class TestGetSubmissionsLocal:
    """get_submissions_local() tests"""

    def test_empty_dir_returns_empty_list(self, tmp_path):
        """When the training_submissions directory does not exist, return []."""
        fake_base = str(tmp_path / "data" / "training_submissions")
        with patch.object(admin_review.os.path, "exists", return_value=False):
            with patch.object(admin_review.os.path, "join", return_value=fake_base):
                result = admin_review.get_submissions_local()
        assert result == []

    def test_nonexistent_dir_returns_empty(self):
        """If base path doesn't exist at all, return []."""
        with patch.object(admin_review.os.path, "exists", return_value=False):
            result = admin_review.get_submissions_local()
        assert result == []

    def test_dir_with_json_files(self, tmp_path):
        """JSON files in the directory are loaded and sorted by avg_return_pct desc."""
        sub_dir = tmp_path / "data" / "training_submissions"
        sub_dir.mkdir(parents=True)

        file_a = {"trainer_id": "alice", "avg_return_pct": 5.0, "algorithm": "ppo"}
        file_b = {"trainer_id": "bob", "avg_return_pct": 12.3, "algorithm": "a2c"}
        file_c = {"trainer_id": "carol", "avg_return_pct": 8.1, "algorithm": "ppo"}

        (sub_dir / "a_result.json").write_text(json.dumps(file_a))
        (sub_dir / "b_result.json").write_text(json.dumps(file_b))
        (sub_dir / "c_result.json").write_text(json.dumps(file_c))
        (sub_dir / "not_json.txt").write_text("ignore me")

        # Patch os.path.abspath to redirect the base path calculation to our tmp dir
        original_join = os.path.join
        def patched_join(*args):
            # Intercept the specific join for "data", "training_submissions"
            if len(args) >= 3 and args[-2] == "data" and args[-1] == "training_submissions":
                return str(sub_dir)
            return original_join(*args)

        with patch("rl_hybrid.rl.admin_review.os.path.join", side_effect=patched_join):
            result = admin_review.get_submissions_local()

        # Should be sorted by avg_return_pct descending
        assert len(result) == 3
        assert result[0]["trainer_id"] == "bob"
        assert result[1]["trainer_id"] == "carol"
        assert result[2]["trainer_id"] == "alice"
        # _source and _file should be added
        for r in result:
            assert r["_source"] == "local"
            assert "_file" in r

    def test_dir_with_json_files_real_fs(self, tmp_path):
        """Integration-style test using real filesystem."""
        sub_dir = tmp_path / "data" / "training_submissions"
        sub_dir.mkdir(parents=True)

        (sub_dir / "sub1.json").write_text(json.dumps({
            "trainer_id": "t1", "avg_return_pct": 3.0
        }))
        (sub_dir / "sub2.json").write_text(json.dumps({
            "trainer_id": "t2", "avg_return_pct": 7.5
        }))

        original_join = os.path.join
        def patched_join(*args):
            if len(args) >= 3 and args[-2] == "data" and args[-1] == "training_submissions":
                return str(sub_dir)
            return original_join(*args)

        with patch("rl_hybrid.rl.admin_review.os.path.join", side_effect=patched_join):
            result = admin_review.get_submissions_local()

        assert len(result) == 2
        assert result[0]["avg_return_pct"] == 7.5
        assert result[1]["avg_return_pct"] == 3.0

    def test_non_json_files_ignored(self, tmp_path):
        """Non-.json files are skipped."""
        sub_dir = tmp_path / "submissions"
        sub_dir.mkdir()
        (sub_dir / "notes.txt").write_text("hello")
        (sub_dir / "readme.md").write_text("# readme")

        with patch.object(admin_review.os.path, "join", return_value=str(sub_dir)):
            with patch.object(admin_review.os.path, "exists", return_value=True):
                with patch("os.listdir", return_value=os.listdir(sub_dir)):
                    result = admin_review.get_submissions_local()

        assert result == []


# ============================================================================
# admin_review: list_submissions()
# ============================================================================

class TestListSubmissions:
    """list_submissions() output formatting tests"""

    def test_no_submissions_logs_info(self):
        """When no submissions exist, log an info message."""
        with patch.object(admin_review, "get_submissions_from_db", return_value=[]):
            with patch.object(admin_review.logger, "info") as mock_log:
                admin_review.list_submissions()
                mock_log.assert_called_once()
                assert "없습니다" in mock_log.call_args[0][0]

    def test_output_formatting(self, capsys):
        """Verify table formatting with sample data."""
        fake_subs = [
            {
                "trainer_id": "alice",
                "algorithm": "ppo",
                "avg_return_pct": 12.34,
                "avg_sharpe": 1.567,
                "avg_mdd": 0.089,
                "avg_trades": 45.2,
                "edge_cases": True,
                "status": "submitted",
            },
            {
                "trainer_id": "bob",
                "algorithm": "a2c",
                "avg_return_pct": 5.67,
                "avg_sharpe": 0.890,
                "avg_mdd": 0.123,
                "avg_trades": 30.0,
                "edge_cases": False,
                "status": "promoted",
            },
        ]
        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            admin_review.list_submissions()

        output = capsys.readouterr().out
        assert "2건" in output
        assert "alice" in output
        assert "bob" in output
        assert "12.34%" in output
        assert "5.67%" in output
        assert "ppo" in output
        assert "a2c" in output

    def test_edge_case_column(self, capsys):
        """Edge column shows Y/N correctly."""
        fake_subs = [
            {"trainer_id": "x", "algorithm": "ppo", "avg_return_pct": 1.0,
             "avg_sharpe": 0.5, "avg_mdd": 0.01, "avg_trades": 10.0,
             "edge_cases": True, "status": "ok"},
            {"trainer_id": "y", "algorithm": "ppo", "avg_return_pct": 2.0,
             "avg_sharpe": 0.6, "avg_mdd": 0.02, "avg_trades": 20.0,
             "edge_cases": None, "status": "ok"},
        ]
        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            admin_review.list_submissions()

        output = capsys.readouterr().out
        lines = [l for l in output.split("\n") if "|" in l and "Trainer" not in l and "---" not in l]
        assert len(lines) >= 2
        # First row has edge_cases=True → Y, second has None → N
        assert "Y" in lines[0]
        assert "N" in lines[1]


# ============================================================================
# admin_review: show_leaderboard()
# ============================================================================

class TestShowLeaderboard:
    """show_leaderboard() grouping logic tests"""

    def test_no_data_logs_info(self):
        """Empty submissions → log message."""
        with patch.object(admin_review, "get_submissions_from_db", return_value=[]):
            with patch.object(admin_review.logger, "info") as mock_log:
                admin_review.show_leaderboard()
                mock_log.assert_called_once()

    def test_grouping_by_trainer(self, capsys):
        """Should pick best result per trainer and rank them."""
        fake_subs = [
            {"trainer_id": "alice", "avg_return_pct": 10.0, "algorithm": "ppo"},
            {"trainer_id": "alice", "avg_return_pct": 15.0, "algorithm": "a2c"},
            {"trainer_id": "bob", "avg_return_pct": 12.0, "algorithm": "ppo"},
            {"trainer_id": "bob", "avg_return_pct": 8.0, "algorithm": "ppo"},
        ]
        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            admin_review.show_leaderboard()

        output = capsys.readouterr().out
        assert "리더보드" in output
        # alice's best is 15%, bob's best is 12%
        # alice should be ranked higher
        alice_pos = output.find("alice")
        bob_pos = output.find("bob")
        assert alice_pos < bob_pos

    def test_submission_count_per_trainer(self, capsys):
        """Count column shows correct submission count per trainer."""
        fake_subs = [
            {"trainer_id": "alice", "avg_return_pct": 10.0, "algorithm": "ppo"},
            {"trainer_id": "alice", "avg_return_pct": 15.0, "algorithm": "a2c"},
            {"trainer_id": "alice", "avg_return_pct": 5.0, "algorithm": "ppo"},
            {"trainer_id": "bob", "avg_return_pct": 12.0, "algorithm": "ppo"},
        ]
        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            admin_review.show_leaderboard()

        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        # Find data lines (contain trainer names)
        alice_line = [l for l in lines if "alice" in l][0]
        bob_line = [l for l in lines if "bob" in l][0]
        # alice has 3 submissions, bob has 1
        assert "3" in alice_line
        assert "1" in bob_line

    def test_single_trainer(self, capsys):
        """Single trainer still works."""
        fake_subs = [
            {"trainer_id": "solo", "avg_return_pct": 7.7, "algorithm": "ppo"},
        ]
        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            admin_review.show_leaderboard()

        output = capsys.readouterr().out
        assert "solo" in output
        assert "7.77%" in output or "7.70%" in output


# ============================================================================
# admin_review: promote_model()
# ============================================================================

class TestPromoteModel:
    """promote_model() model file handling tests"""

    def test_invalid_id_too_low(self):
        """submission_id < 1 logs error."""
        with patch.object(admin_review, "get_submissions_from_db",
                          return_value=[{"trainer_id": "a"}]):
            with patch.object(admin_review.logger, "error") as mock_err:
                admin_review.promote_model(0)
                mock_err.assert_called_once()
                assert "유효하지 않은" in mock_err.call_args[0][0]

    def test_invalid_id_too_high(self):
        """submission_id > len(subs) logs error."""
        with patch.object(admin_review, "get_submissions_from_db",
                          return_value=[{"trainer_id": "a"}]):
            with patch.object(admin_review.logger, "error") as mock_err:
                admin_review.promote_model(5)
                mock_err.assert_called_once()

    def test_model_not_found(self, tmp_path):
        """When model file doesn't exist, log error."""
        model_dir = tmp_path / "models"
        fake_subs = [
            {"trainer_id": "alice", "algorithm": "ppo", "avg_return_pct": 10.0},
        ]
        mock_policy = MagicMock()
        mock_policy.MODEL_DIR = str(model_dir)
        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            with patch.dict("sys.modules", {"rl_hybrid.rl.policy": mock_policy}):
                with patch.object(admin_review, "_update_db_status"):
                    with patch.object(admin_review.logger, "error") as mock_err:
                        admin_review.promote_model(1)
                        # Should log "모델 파일을 찾을 수 없습니다"
                        assert any("찾을 수 없" in str(c) for c in mock_err.call_args_list)

    def test_successful_promotion(self, tmp_path):
        """Full promotion flow: find model, backup old best, copy new."""
        model_dir = tmp_path / "rl_models"
        submissions_dir = model_dir / "submissions" / "alice_20260311"
        best_dir = model_dir / "best"
        submissions_dir.mkdir(parents=True)
        best_dir.mkdir(parents=True)

        # Create model + result files
        (submissions_dir / "model.zip").write_bytes(b"new_model_data")
        (submissions_dir / "result.json").write_text(json.dumps({
            "avg_return_pct": 10.0
        }))

        # Create existing best model to be backed up
        (best_dir / "best_model.zip").write_bytes(b"old_model_data")

        fake_subs = [
            {"trainer_id": "alice", "algorithm": "ppo", "avg_return_pct": 10.0,
             "avg_sharpe": 1.5, "avg_mdd": 0.05},
        ]

        mock_policy = MagicMock()
        mock_policy.MODEL_DIR = str(model_dir)

        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            with patch.dict("sys.modules", {"rl_hybrid.rl.policy": mock_policy}):
                with patch.object(admin_review, "_update_db_status") as mock_db:
                    admin_review.promote_model(1)

        # Verify new best model was copied
        assert (best_dir / "best_model.zip").read_bytes() == b"new_model_data"

        # Verify model_info.json was written
        info = json.loads((best_dir / "model_info.json").read_text())
        assert info["algorithm"] == "ppo"
        assert info["avg_return_pct"] == 10.0
        assert info["promoted_from"] == "alice"

        # Verify backup was created
        backup_parent = model_dir / "best_backup"
        assert backup_parent.exists()
        backup_dirs = list(backup_parent.iterdir())
        assert len(backup_dirs) == 1
        assert (backup_dirs[0] / "best_model.zip").read_bytes() == b"old_model_data"

        # Verify DB was updated
        mock_db.assert_called_once_with(fake_subs[0], "promoted")

    def test_promote_return_pct_mismatch(self, tmp_path):
        """If result.json avg_return_pct doesn't match, skip that directory."""
        model_dir = tmp_path / "rl_models"
        submissions_dir = model_dir / "submissions" / "alice_20260311"
        submissions_dir.mkdir(parents=True)

        (submissions_dir / "model.zip").write_bytes(b"model")
        (submissions_dir / "result.json").write_text(json.dumps({
            "avg_return_pct": 99.0  # Doesn't match 10.0
        }))

        fake_subs = [
            {"trainer_id": "alice", "algorithm": "ppo", "avg_return_pct": 10.0},
        ]

        mock_policy = MagicMock()
        mock_policy.MODEL_DIR = str(model_dir)

        with patch.object(admin_review, "get_submissions_from_db", return_value=fake_subs):
            with patch.dict("sys.modules", {"rl_hybrid.rl.policy": mock_policy}):
                with patch.object(admin_review, "_update_db_status"):
                    with patch.object(admin_review.logger, "error") as mock_err:
                        admin_review.promote_model(1)
                        assert any("찾을 수 없" in str(c) for c in mock_err.call_args_list)


# ============================================================================
# admin_review: _update_db_status()
# ============================================================================

class TestUpdateDbStatus:
    """_update_db_status() tests"""

    def test_no_env_vars_skips(self):
        """Without SUPABASE_URL/KEY, does nothing."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise
            admin_review._update_db_status({"id": 1}, "promoted")

    def test_no_record_id_skips(self):
        """Without record id, does nothing."""
        with patch.dict(os.environ, {"SUPABASE_URL": "http://x", "SUPABASE_SERVICE_KEY": "k"}):
            admin_review._update_db_status({}, "promoted")

    def test_successful_patch(self):
        """Sends PATCH request with correct parameters."""
        mock_requests = MagicMock()
        with patch.dict(os.environ, {"SUPABASE_URL": "http://x", "SUPABASE_SERVICE_KEY": "k"}):
            with patch.dict("sys.modules", {"requests": mock_requests}):
                admin_review._update_db_status({"id": 42}, "promoted")
                mock_requests.patch.assert_called_once()
                args, kwargs = mock_requests.patch.call_args
                assert "42" in args[0]
                assert kwargs["json"]["status"] == "promoted"


# ============================================================================
# online_buffer: OnlineExperienceBuffer initialization
# ============================================================================

class TestOnlineBufferInit:
    """OnlineExperienceBuffer initialization tests"""

    def test_init_no_file(self, tmp_path):
        """When buffer file doesn't exist, initialize with empty list."""
        with patch.object(online_buffer, "BUFFER_PATH", tmp_path / "nonexistent.json"):
            buf = OnlineExperienceBuffer()
        assert buf.buffer == []

    def test_init_with_existing_file(self, tmp_path):
        """Load existing buffer from disk."""
        buf_file = tmp_path / "online_buffer.json"
        existing = [
            {"timestamp": "2026-03-11T10:00:00", "rl_action": 0.5,
             "outcome_pct": None, "outcome_filled": False},
        ]
        buf_file.write_text(json.dumps(existing))

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        assert len(buf.buffer) == 1
        assert buf.buffer[0]["rl_action"] == 0.5

    def test_init_with_corrupt_file(self, tmp_path):
        """Corrupt JSON file → fallback to empty list."""
        buf_file = tmp_path / "online_buffer.json"
        buf_file.write_text("{invalid json!!!")

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        assert buf.buffer == []


# ============================================================================
# online_buffer: add_experience()
# ============================================================================

class TestAddExperience:
    """add_experience() correctly stores data"""

    def _make_buffer(self, tmp_path):
        buf_file = tmp_path / "buf.json"
        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf = OnlineExperienceBuffer()
        return buf, buf_file

    def test_adds_entry_with_correct_fields(self, tmp_path):
        buf_file = tmp_path / "buf.json"
        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf = OnlineExperienceBuffer()

                market_data = {
                    "ticker": {"trade_price": 50000000, "signed_change_rate": -0.02,
                               "acc_trade_volume_24h": 1234.5},
                    "indicators": {"rsi_14": 35.2, "sma_20": 48000000},
                    "fear_greed": {"value": 28},
                }
                buf.add_experience(
                    market_data=market_data,
                    external_data={},
                    portfolio={},
                    agent_state={},
                    rl_action=0.75,
                    agent_decision="buy",
                )

        assert len(buf.buffer) == 1
        entry = buf.buffer[0]
        assert entry["rl_action"] == 0.75
        assert entry["agent_decision"] == "buy"
        assert entry["outcome_pct"] is None
        assert entry["outcome_filled"] is False
        assert entry["market_state"]["price"] == 50000000
        assert entry["market_state"]["rsi_14"] == 35.2
        assert entry["market_state"]["fgi_value"] == 28
        assert "timestamp" in entry

    def test_multiple_experiences_accumulate(self, tmp_path):
        buf_file = tmp_path / "buf.json"
        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf = OnlineExperienceBuffer()
                for i in range(5):
                    buf.add_experience(
                        market_data={"ticker": {}, "indicators": {}, "fear_greed": {}},
                        external_data={}, portfolio={}, agent_state={},
                        rl_action=float(i), agent_decision="hold",
                    )
        assert len(buf.buffer) == 5
        assert [e["rl_action"] for e in buf.buffer] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_saves_to_disk_after_add(self, tmp_path):
        buf_file = tmp_path / "buf.json"
        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf = OnlineExperienceBuffer()
                buf.add_experience(
                    market_data={"ticker": {}, "indicators": {}, "fear_greed": {}},
                    external_data={}, portfolio={}, agent_state={},
                    rl_action=1.0, agent_decision="sell",
                )

        # Read back from disk
        saved = json.loads(buf_file.read_text())
        assert len(saved) == 1
        assert saved[0]["agent_decision"] == "sell"


# ============================================================================
# online_buffer: should_train()
# ============================================================================

class TestShouldTrain:
    """should_train() triggers at TRIGGER_SIZE"""

    def test_empty_buffer_returns_false(self, tmp_path):
        with patch.object(online_buffer, "BUFFER_PATH", tmp_path / "empty.json"):
            buf = OnlineExperienceBuffer()
        assert buf.should_train() is False

    def test_below_trigger_size_returns_false(self, tmp_path):
        """Even if all filled, below TRIGGER_SIZE → False."""
        entries = [
            {"outcome_filled": True, "outcome_pct": 1.0}
            for _ in range(TRIGGER_SIZE - 1)
        ]
        buf_file = tmp_path / "buf.json"
        buf_file.write_text(json.dumps(entries))

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        assert buf.should_train() is False

    def test_trigger_size_reached_but_not_enough_filled(self, tmp_path):
        """TRIGGER_SIZE entries but < 50% filled → False."""
        entries = []
        for i in range(TRIGGER_SIZE):
            entries.append({
                "outcome_filled": i < (TRIGGER_SIZE * 0.5 - 1),  # just under 50%
                "outcome_pct": 1.0 if i < (TRIGGER_SIZE * 0.5 - 1) else None,
            })
        buf_file = tmp_path / "buf.json"
        buf_file.write_text(json.dumps(entries))

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        assert buf.should_train() is False

    def test_trigger_size_reached_and_enough_filled(self, tmp_path):
        """TRIGGER_SIZE entries with >= 50% filled → True."""
        filled_count = int(TRIGGER_SIZE * 0.5)
        entries = []
        for i in range(TRIGGER_SIZE):
            entries.append({
                "outcome_filled": i < filled_count,
                "outcome_pct": 1.0 if i < filled_count else None,
            })
        buf_file = tmp_path / "buf.json"
        buf_file.write_text(json.dumps(entries))

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        assert buf.should_train() is True

    def test_all_filled_above_trigger(self, tmp_path):
        """More than TRIGGER_SIZE, all filled → True."""
        entries = [
            {"outcome_filled": True, "outcome_pct": 0.5}
            for _ in range(TRIGGER_SIZE + 10)
        ]
        buf_file = tmp_path / "buf.json"
        buf_file.write_text(json.dumps(entries))

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        assert buf.should_train() is True


# ============================================================================
# online_buffer: Buffer persistence (save/load)
# ============================================================================

class TestBufferPersistence:
    """Buffer save/load round-trip tests"""

    def test_save_creates_dir_and_file(self, tmp_path):
        """_save() creates MODEL_DIR if it doesn't exist."""
        model_dir = tmp_path / "new_dir"
        buf_file = model_dir / "online_buffer.json"

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", model_dir):
                buf = OnlineExperienceBuffer()
                buf.buffer = [{"test": 1}]
                buf._save()

        assert model_dir.exists()
        assert buf_file.exists()
        assert json.loads(buf_file.read_text()) == [{"test": 1}]

    def test_round_trip(self, tmp_path):
        """Data survives save → new instance load cycle."""
        buf_file = tmp_path / "buf.json"
        model_dir = tmp_path

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", model_dir):
                buf1 = OnlineExperienceBuffer()
                buf1.buffer = [
                    {"timestamp": "2026-03-11T10:00", "rl_action": 0.3,
                     "outcome_filled": False, "outcome_pct": None},
                    {"timestamp": "2026-03-11T14:00", "rl_action": 0.8,
                     "outcome_filled": True, "outcome_pct": 2.5},
                ]
                buf1._save()

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf2 = OnlineExperienceBuffer()

        assert len(buf2.buffer) == 2
        assert buf2.buffer[0]["rl_action"] == 0.3
        assert buf2.buffer[1]["outcome_pct"] == 2.5

    def test_empty_buffer_saves_empty_list(self, tmp_path):
        """Empty buffer serializes as []."""
        buf_file = tmp_path / "buf.json"
        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf = OnlineExperienceBuffer()
                buf._save()

        assert json.loads(buf_file.read_text()) == []


# ============================================================================
# online_buffer: update_outcomes()
# ============================================================================

class TestUpdateOutcomes:
    """update_outcomes() correctly updates entries"""

    def _make_buffer_with_entries(self, tmp_path, entries):
        buf_file = tmp_path / "buf.json"
        buf_file.write_text(json.dumps(entries))
        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf = OnlineExperienceBuffer()
        return buf, buf_file

    def test_matches_by_timestamp_prefix(self, tmp_path):
        """Matches entries where timestamp[:16] matches."""
        entries = [
            {"timestamp": "2026-03-11T10:00:15+09:00", "outcome_filled": False,
             "outcome_pct": None, "rl_action": 0.5},
            {"timestamp": "2026-03-11T14:00:30+09:00", "outcome_filled": False,
             "outcome_pct": None, "rl_action": 0.8},
        ]
        buf, buf_file = self._make_buffer_with_entries(tmp_path, entries)

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf.update_outcomes([
                    {"timestamp": "2026-03-11T10:00:00+09:00", "outcome_4h_pct": 2.5},
                ])

        assert buf.buffer[0]["outcome_pct"] == 2.5
        assert buf.buffer[0]["outcome_filled"] is True
        # Second entry unchanged
        assert buf.buffer[1]["outcome_pct"] is None
        assert buf.buffer[1]["outcome_filled"] is False

    def test_skips_already_filled(self, tmp_path):
        """Already-filled entries are not overwritten."""
        entries = [
            {"timestamp": "2026-03-11T10:00:15+09:00", "outcome_filled": True,
             "outcome_pct": 1.0, "rl_action": 0.5},
        ]
        buf, buf_file = self._make_buffer_with_entries(tmp_path, entries)

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf.update_outcomes([
                    {"timestamp": "2026-03-11T10:00:00+09:00", "outcome_4h_pct": 99.9},
                ])

        assert buf.buffer[0]["outcome_pct"] == 1.0  # unchanged

    def test_skips_none_pct(self, tmp_path):
        """Outcomes with pct=None are skipped."""
        entries = [
            {"timestamp": "2026-03-11T10:00:15+09:00", "outcome_filled": False,
             "outcome_pct": None, "rl_action": 0.5},
        ]
        buf, buf_file = self._make_buffer_with_entries(tmp_path, entries)

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf.update_outcomes([
                    {"timestamp": "2026-03-11T10:00:00+09:00", "outcome_4h_pct": None},
                ])

        assert buf.buffer[0]["outcome_filled"] is False

    def test_multiple_outcomes_update(self, tmp_path):
        """Multiple outcomes update multiple entries."""
        entries = [
            {"timestamp": "2026-03-11T10:00:15+09:00", "outcome_filled": False,
             "outcome_pct": None},
            {"timestamp": "2026-03-11T14:00:30+09:00", "outcome_filled": False,
             "outcome_pct": None},
            {"timestamp": "2026-03-11T18:00:45+09:00", "outcome_filled": False,
             "outcome_pct": None},
        ]
        buf, buf_file = self._make_buffer_with_entries(tmp_path, entries)

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf.update_outcomes([
                    {"timestamp": "2026-03-11T10:00:00", "outcome_4h_pct": 1.1},
                    {"timestamp": "2026-03-11T18:00:00", "outcome_4h_pct": -0.5},
                ])

        assert buf.buffer[0]["outcome_pct"] == 1.1
        assert buf.buffer[0]["outcome_filled"] is True
        assert buf.buffer[1]["outcome_filled"] is False  # no match
        assert buf.buffer[2]["outcome_pct"] == -0.5
        assert buf.buffer[2]["outcome_filled"] is True

    def test_saves_after_update(self, tmp_path):
        """Disk file is updated after successful outcome updates."""
        entries = [
            {"timestamp": "2026-03-11T10:00:15+09:00", "outcome_filled": False,
             "outcome_pct": None},
        ]
        buf, buf_file = self._make_buffer_with_entries(tmp_path, entries)

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                buf.update_outcomes([
                    {"timestamp": "2026-03-11T10:00:00", "outcome_4h_pct": 3.3},
                ])

        saved = json.loads(buf_file.read_text())
        assert saved[0]["outcome_pct"] == 3.3
        assert saved[0]["outcome_filled"] is True

    def test_no_matches_no_save(self, tmp_path):
        """If no entries match, _save() is not called."""
        entries = [
            {"timestamp": "2026-03-11T10:00:15+09:00", "outcome_filled": False,
             "outcome_pct": None},
        ]
        buf, buf_file = self._make_buffer_with_entries(tmp_path, entries)

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            with patch.object(online_buffer, "MODEL_DIR", tmp_path):
                with patch.object(buf, "_save") as mock_save:
                    buf.update_outcomes([
                        {"timestamp": "2099-01-01T00:00:00", "outcome_4h_pct": 5.0},
                    ])
                    mock_save.assert_not_called()


# ============================================================================
# online_buffer: get_stats()
# ============================================================================

class TestGetStats:
    """get_stats() returns correct statistics"""

    def test_empty_buffer_stats(self, tmp_path):
        with patch.object(online_buffer, "BUFFER_PATH", tmp_path / "empty.json"):
            buf = OnlineExperienceBuffer()
        stats = buf.get_stats()
        assert stats["total"] == 0
        assert stats["outcome_filled"] == 0
        assert stats["trigger_size"] == TRIGGER_SIZE
        assert stats["ready_to_train"] is False

    def test_partial_fill_stats(self, tmp_path):
        entries = [
            {"outcome_filled": True}, {"outcome_filled": True},
            {"outcome_filled": False}, {"outcome_filled": False},
        ]
        buf_file = tmp_path / "buf.json"
        buf_file.write_text(json.dumps(entries))

        with patch.object(online_buffer, "BUFFER_PATH", buf_file):
            buf = OnlineExperienceBuffer()
        stats = buf.get_stats()
        assert stats["total"] == 4
        assert stats["outcome_filled"] == 2


# ============================================================================
# admin_review: get_submissions_from_db()
# ============================================================================

class TestGetSubmissionsFromDb:
    """get_submissions_from_db() — DB vs local fallback"""

    def test_no_env_vars_falls_back_to_local(self):
        """Without SUPABASE env vars, falls back to local."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(admin_review, "get_submissions_local", return_value=[]) as mock_local:
                result = admin_review.get_submissions_from_db()
                mock_local.assert_called_once()
        assert result == []

    def test_db_success(self):
        """Successful DB call returns data."""
        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"trainer_id": "db_user"}]
        mock_requests.get.return_value = mock_resp
        with patch.dict(os.environ, {"SUPABASE_URL": "http://x", "SUPABASE_SERVICE_KEY": "k"}):
            with patch.dict("sys.modules", {"requests": mock_requests}):
                result = admin_review.get_submissions_from_db()
        assert len(result) == 1
        assert result[0]["trainer_id"] == "db_user"

    def test_db_failure_falls_back(self):
        """DB request failure falls back to local."""
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("connection error")
        with patch.dict(os.environ, {"SUPABASE_URL": "http://x", "SUPABASE_SERVICE_KEY": "k"}):
            with patch.dict("sys.modules", {"requests": mock_requests}):
                with patch.object(admin_review, "get_submissions_local",
                                  return_value=[{"trainer_id": "local"}]) as mock_local:
                    result = admin_review.get_submissions_from_db()
                    mock_local.assert_called_once()
        assert result[0]["trainer_id"] == "local"

    def test_db_non_200_falls_back(self):
        """Non-200 status falls back to local."""
        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_requests.get.return_value = mock_resp
        with patch.dict(os.environ, {"SUPABASE_URL": "http://x", "SUPABASE_SERVICE_KEY": "k"}):
            with patch.dict("sys.modules", {"requests": mock_requests}):
                with patch.object(admin_review, "get_submissions_local",
                                  return_value=[]) as mock_local:
                    result = admin_review.get_submissions_from_db()
                    mock_local.assert_called_once()
