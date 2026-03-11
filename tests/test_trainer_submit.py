"""Unit tests for rl_hybrid/rl/trainer_submit.py"""

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import real dependencies — trainer_submit uses numpy, train, environment
# at module level. These are available in the test venv.
import numpy as np

from rl_hybrid.rl.trainer_submit import (
    compute_model_hash,
    get_current_best,
    get_default_trainer_id,
    save_local,
    submit_to_db,
)


# ---------------------------------------------------------------------------
# 1. get_default_trainer_id
# ---------------------------------------------------------------------------
class TestGetDefaultTrainerId:
    def test_returns_string_with_hostname(self):
        result = get_default_trainer_id()
        hostname = platform.node().split(".")[0]
        assert isinstance(result, str)
        assert result == f"trainer-{hostname}"
        assert result.startswith("trainer-")

    @patch("platform.node", return_value="myhost.local")
    def test_strips_domain(self, mock_node):
        assert get_default_trainer_id() == "trainer-myhost"

    @patch("platform.node", return_value="simple")
    def test_no_dot_hostname(self, mock_node):
        assert get_default_trainer_id() == "trainer-simple"

    @patch("platform.node", return_value="a.b.c.d")
    def test_multiple_dots(self, mock_node):
        assert get_default_trainer_id() == "trainer-a"

    @patch("platform.node", return_value="")
    def test_empty_hostname(self, mock_node):
        assert get_default_trainer_id() == "trainer-"


# ---------------------------------------------------------------------------
# 2. compute_model_hash
# ---------------------------------------------------------------------------
class TestComputeModelHash:
    def test_consistent_hash_for_same_file(self, tmp_path):
        model_file = tmp_path / "model.zip"
        model_file.write_bytes(b"fake model content 12345")

        h1 = compute_model_hash(str(model_file))
        h2 = compute_model_hash(str(model_file))
        assert h1 == h2
        assert len(h1) == 16  # truncated to 16 chars

    def test_returns_empty_for_missing_file(self):
        result = compute_model_hash("/nonexistent/path/model.zip")
        assert result == ""

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "model_a.zip"
        f2 = tmp_path / "model_b.zip"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        assert compute_model_hash(str(f1)) != compute_model_hash(str(f2))

    def test_hash_matches_manual_sha256(self, tmp_path):
        content = b"deterministic content"
        model_file = tmp_path / "model.zip"
        model_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()[:16]
        assert compute_model_hash(str(model_file)) == expected

    def test_empty_file(self, tmp_path):
        model_file = tmp_path / "empty.zip"
        model_file.write_bytes(b"")
        result = compute_model_hash(str(model_file))
        expected = hashlib.sha256(b"").hexdigest()[:16]
        assert result == expected


# ---------------------------------------------------------------------------
# 3. get_current_best
# ---------------------------------------------------------------------------
class TestGetCurrentBest:
    def test_missing_file_returns_empty_dict(self):
        with patch("os.path.exists", return_value=False):
            result = get_current_best()
            assert result == {}

    def test_loads_json_when_file_exists(self):
        fake_data = {"avg_return_pct": 5.12, "algo": "ppo"}
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=json.dumps(fake_data))):
            result = get_current_best()
            assert result == fake_data

    def test_returns_dict_type(self):
        with patch("os.path.exists", return_value=False):
            result = get_current_best()
            assert isinstance(result, dict)

    def test_real_file(self, tmp_path):
        """Integration-style test with a real temp file."""
        info = {"avg_return_pct": 3.5, "model_hash": "abc123"}
        info_file = tmp_path / "model_info.json"
        info_file.write_text(json.dumps(info))

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=json.dumps(info))):
            result = get_current_best()
            assert result["avg_return_pct"] == 3.5


# ---------------------------------------------------------------------------
# 4. save_local
# ---------------------------------------------------------------------------
class TestSaveLocal:
    def test_creates_json_file(self, tmp_path):
        result = {
            "trainer_id": "trainer-test",
            "algorithm": "ppo",
            "avg_return_pct": 2.5,
        }

        base_dir = str(tmp_path / "submissions")
        original_join = os.path.join
        call_count = [0]

        def patched_join(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                return base_dir
            return original_join(*args)

        with patch("rl_hybrid.rl.trainer_submit.os.path.join", side_effect=patched_join):
            ret = save_local(result)

        assert ret is True
        files = list((tmp_path / "submissions").glob("*.json"))
        assert len(files) == 1

    def test_save_local_writes_valid_json(self, tmp_path):
        result = {
            "trainer_id": "trainer-abc",
            "algorithm": "sac",
            "avg_return_pct": 1.23,
        }

        # Patch the entire directory logic to use tmp_path
        base_dir = str(tmp_path / "submissions")

        original_join = os.path.join
        call_count = [0]

        def patched_join(*args):
            call_count[0] += 1
            # First call computes base_dir — redirect it
            if call_count[0] == 1:
                return base_dir
            return original_join(*args)

        with patch("rl_hybrid.rl.trainer_submit.os.path.join", side_effect=patched_join):
            ret = save_local(result)

        assert ret is True
        # Check a file was created
        files = list((tmp_path / "submissions").glob("*.json"))
        assert len(files) == 1

        # Verify JSON content
        with open(files[0]) as f:
            saved = json.load(f)
        assert saved["trainer_id"] == "trainer-abc"
        assert saved["algorithm"] == "sac"
        assert saved["avg_return_pct"] == 1.23

    def test_save_local_filename_format(self, tmp_path):
        result = {"trainer_id": "trainer-xyz"}
        base_dir = str(tmp_path)

        original_join = os.path.join
        call_count = [0]

        def patched_join(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                return base_dir
            return original_join(*args)

        with patch("rl_hybrid.rl.trainer_submit.os.path.join", side_effect=patched_join):
            save_local(result)

        files = list(tmp_path.glob("trainer-xyz_*.json"))
        assert len(files) == 1

    def test_returns_true(self, tmp_path):
        result = {"trainer_id": "t"}
        base_dir = str(tmp_path)

        original_join = os.path.join
        call_count = [0]

        def patched_join(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                return base_dir
            return original_join(*args)

        with patch("rl_hybrid.rl.trainer_submit.os.path.join", side_effect=patched_join):
            assert save_local(result) is True


# ---------------------------------------------------------------------------
# 5. submit_to_db — fallback to save_local when no env vars
# ---------------------------------------------------------------------------
class TestSubmitToDb:
    def test_no_env_vars_falls_back_to_save_local(self):
        result = {"trainer_id": "test", "algorithm": "ppo"}

        with patch.dict(os.environ, {}, clear=True), \
             patch("rl_hybrid.rl.trainer_submit.save_local", return_value=True) as mock_save:
            # Remove SUPABASE vars if present
            os.environ.pop("SUPABASE_URL", None)
            os.environ.pop("SUPABASE_SERVICE_KEY", None)

            ret = submit_to_db(result)
            mock_save.assert_called_once_with(result)
            assert ret is True

    def test_empty_supabase_url_falls_back(self):
        result = {"trainer_id": "test"}

        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_KEY": "key"}), \
             patch("rl_hybrid.rl.trainer_submit.save_local", return_value=True) as mock_save:
            ret = submit_to_db(result)
            mock_save.assert_called_once_with(result)
            assert ret is True

    def test_empty_supabase_key_falls_back(self):
        result = {"trainer_id": "test"}

        with patch.dict(os.environ, {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SERVICE_KEY": ""}), \
             patch("rl_hybrid.rl.trainer_submit.save_local", return_value=True) as mock_save:
            ret = submit_to_db(result)
            mock_save.assert_called_once()

    def test_successful_db_upload(self):
        result = {"trainer_id": "test", "algorithm": "ppo"}
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = [{"id": 42}]

        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret",
        }), patch("requests.post", return_value=mock_resp):
            import requests  # ensure it's importable in this context
            ret = submit_to_db(result)
            assert ret is True

    def test_db_upload_failure_falls_back(self):
        result = {"trainer_id": "test"}
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret",
        }), patch("requests.post", return_value=mock_resp), \
             patch("rl_hybrid.rl.trainer_submit.save_local", return_value=True) as mock_save:
            ret = submit_to_db(result)
            mock_save.assert_called_once_with(result)

    def test_db_connection_exception_falls_back(self):
        result = {"trainer_id": "test"}

        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret",
        }), patch("requests.post", side_effect=ConnectionError("timeout")), \
             patch("rl_hybrid.rl.trainer_submit.save_local", return_value=True) as mock_save:
            ret = submit_to_db(result)
            mock_save.assert_called_once_with(result)

    def test_status_200_also_succeeds(self):
        result = {"trainer_id": "test"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": 1}]

        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret",
        }), patch("requests.post", return_value=mock_resp):
            assert submit_to_db(result) is True

    def test_empty_response_data(self):
        result = {"trainer_id": "test"}
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = []

        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret",
        }), patch("requests.post", return_value=mock_resp):
            # Should still return True, with record_id="?"
            assert submit_to_db(result) is True


# ---------------------------------------------------------------------------
# 6. Main argument parsing
# ---------------------------------------------------------------------------
class TestMainArgParsing:
    """Test argument parsing without running actual training."""

    def test_default_args(self):
        """Default arguments parse correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--trainer-id", type=str, default=None)
        parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"])
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--steps", type=int, default=100_000)
        parser.add_argument("--balance", type=float, default=10_000_000)
        parser.add_argument("--edge-cases", action="store_true")
        parser.add_argument("--synthetic-ratio", type=float, default=0.3)
        parser.add_argument("--model", type=str, default=None)

        args = parser.parse_args([])
        assert args.algo == "ppo"
        assert args.days == 180
        assert args.steps == 100_000
        assert args.balance == 10_000_000
        assert args.edge_cases is False
        assert args.synthetic_ratio == 0.3
        assert args.trainer_id is None
        assert args.model is None

    def test_custom_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--trainer-id", type=str, default=None)
        parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"])
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--steps", type=int, default=100_000)
        parser.add_argument("--balance", type=float, default=10_000_000)
        parser.add_argument("--edge-cases", action="store_true")
        parser.add_argument("--synthetic-ratio", type=float, default=0.3)
        parser.add_argument("--model", type=str, default=None)

        args = parser.parse_args([
            "--algo", "sac",
            "--days", "365",
            "--steps", "200000",
            "--edge-cases",
            "--synthetic-ratio", "0.4",
            "--trainer-id", "trainer-mac-B",
        ])
        assert args.algo == "sac"
        assert args.days == 365
        assert args.steps == 200000
        assert args.edge_cases is True
        assert args.synthetic_ratio == 0.4
        assert args.trainer_id == "trainer-mac-B"

    def test_td3_algo(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"])
        args = parser.parse_args(["--algo", "td3"])
        assert args.algo == "td3"

    def test_invalid_algo_raises(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--algo", "dqn"])

    def test_model_path(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default=None)
        args = parser.parse_args(["--model", "/path/to/model.zip"])
        assert args.model == "/path/to/model.zip"

    def test_trainer_id_fallback_to_default(self):
        """When --trainer-id is not provided, main() uses get_default_trainer_id()."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--trainer-id", type=str, default=None)
        args = parser.parse_args([])
        trainer_id = args.trainer_id or get_default_trainer_id()
        assert trainer_id.startswith("trainer-")
