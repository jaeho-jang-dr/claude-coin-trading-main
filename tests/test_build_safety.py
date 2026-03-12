#!/usr/bin/env python3
"""
Build safety tests: verify safety guards, configuration integrity,
and absence of hardcoded secrets across the codebase.
"""

import os
import re
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
MIGRATIONS_DIR = PROJECT_DIR / "supabase" / "migrations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


EXECUTE_TRADE_SRC = _read(SCRIPTS_DIR / "execute_trade.py")
RUN_AGENTS_SRC = _read(SCRIPTS_DIR / "run_agents.py")
ENV_EXAMPLE_SRC = _read(PROJECT_DIR / ".env.example")


# ===========================================================================
# 1. execute_trade.py  DRY_RUN check before any trade execution
# ===========================================================================

class TestExecuteTradeDryRun:
    def test_dry_run_env_check_exists(self):
        """execute_trade.py must read DRY_RUN from environment."""
        assert "DRY_RUN" in EXECUTE_TRADE_SRC

    def test_dry_run_checked_before_order(self):
        """DRY_RUN check must appear before the actual POST /orders call."""
        dry_run_pos = EXECUTE_TRADE_SRC.index("DRY_RUN")
        order_pos = EXECUTE_TRADE_SRC.index("/orders")
        assert dry_run_pos < order_pos, (
            "DRY_RUN check must come before the API order call"
        )

    def test_dry_run_defaults_to_true_in_code(self):
        """Default value for DRY_RUN in code must be 'true' (safe default)."""
        assert re.search(
            r"""os\.environ\.get\(\s*["']DRY_RUN["']\s*,\s*["']true["']\s*\)""",
            EXECUTE_TRADE_SRC,
        ), "DRY_RUN must default to 'true' in execute_trade.py"


# ===========================================================================
# 2. execute_trade.py  EMERGENCY_STOP check
# ===========================================================================

class TestExecuteTradeEmergencyStop:
    def test_emergency_stop_env_check_exists(self):
        """execute_trade.py must read EMERGENCY_STOP from environment."""
        assert "EMERGENCY_STOP" in EXECUTE_TRADE_SRC

    def test_emergency_stop_checked_before_order(self):
        """EMERGENCY_STOP check must appear before the API order call."""
        es_pos = EXECUTE_TRADE_SRC.index("EMERGENCY_STOP")
        order_pos = EXECUTE_TRADE_SRC.index("/orders")
        assert es_pos < order_pos

    def test_auto_emergency_json_checked(self):
        """auto_emergency.json flag file must also be checked."""
        assert "auto_emergency.json" in EXECUTE_TRADE_SRC


# ===========================================================================
# 3. execute_trade.py  MAX_TRADE_AMOUNT limit
# ===========================================================================

class TestExecuteTradeMaxAmount:
    def test_max_trade_amount_env_check(self):
        """execute_trade.py must read MAX_TRADE_AMOUNT."""
        assert "MAX_TRADE_AMOUNT" in EXECUTE_TRADE_SRC

    def test_max_trade_amount_compared_before_order(self):
        """MAX_TRADE_AMOUNT comparison must occur before the order call."""
        max_pos = EXECUTE_TRADE_SRC.index("MAX_TRADE_AMOUNT")
        order_pos = EXECUTE_TRADE_SRC.index("/orders")
        assert max_pos < order_pos

    def test_max_trade_amount_defaults_to_100000(self):
        """Default MAX_TRADE_AMOUNT in code must be 100000."""
        assert re.search(
            r"""os\.environ\.get\(\s*["']MAX_TRADE_AMOUNT["']\s*,\s*["']100000["']\s*\)""",
            EXECUTE_TRADE_SRC,
        )


# ===========================================================================
# 4. execute_trade.py  MAX_DAILY_TRADES limit
# ===========================================================================

class TestExecuteTradeMaxDailyTrades:
    def test_max_daily_trades_in_env_example(self):
        """MAX_DAILY_TRADES must be defined in .env.example."""
        assert "MAX_DAILY_TRADES" in ENV_EXAMPLE_SRC


# ===========================================================================
# 5. run_agents.py  DRY_RUN awareness
# ===========================================================================

class TestRunAgentsDryRun:
    def test_dry_run_referenced(self):
        """run_agents.py must reference DRY_RUN."""
        assert "DRY_RUN" in RUN_AGENTS_SRC

    def test_dry_run_affects_execution_mode(self):
        """run_agents.py must use DRY_RUN to determine execution mode."""
        # The script checks DRY_RUN to set exec_mode to 'dry_run' vs 'execute'
        assert "dry_run" in RUN_AGENTS_SRC.lower()
        assert re.search(r"exec_mode\s*=\s*['\"]dry_run['\"]", RUN_AGENTS_SRC)

    def test_emergency_stop_checked(self):
        """run_agents.py must check EMERGENCY_STOP."""
        assert "EMERGENCY_STOP" in RUN_AGENTS_SRC

    def test_emergency_stop_causes_exit(self):
        """run_agents.py must exit when EMERGENCY_STOP is active."""
        # After EMERGENCY_STOP check, sys.exit(1) must follow
        es_match = re.search(r"EMERGENCY_STOP.*?sys\.exit\(1\)", RUN_AGENTS_SRC, re.DOTALL)
        assert es_match, "EMERGENCY_STOP must trigger sys.exit(1)"

    def test_dry_run_cli_flag(self):
        """run_agents.py must support --dry-run CLI flag."""
        assert "--dry-run" in RUN_AGENTS_SRC


# ===========================================================================
# 6. .env.example has all required safety parameters
# ===========================================================================

class TestEnvExampleSafetyParams:
    REQUIRED_PARAMS = [
        "DRY_RUN",
        "EMERGENCY_STOP",
        "MAX_TRADE_AMOUNT",
        "MAX_DAILY_TRADES",
        "MAX_POSITION_RATIO",
        "MIN_TRADE_INTERVAL_HOURS",
    ]

    @pytest.mark.parametrize("param", REQUIRED_PARAMS)
    def test_safety_param_present(self, param):
        """Each safety parameter must be defined in .env.example."""
        assert param in ENV_EXAMPLE_SRC, f"{param} missing from .env.example"


# ===========================================================================
# 7. .env.example DRY_RUN defaults to "true"
# ===========================================================================

class TestEnvExampleDryRunDefault:
    def test_dry_run_default_true(self):
        """DRY_RUN must default to true in .env.example."""
        match = re.search(r"^DRY_RUN\s*=\s*(.+)$", ENV_EXAMPLE_SRC, re.MULTILINE)
        assert match, "DRY_RUN line not found in .env.example"
        assert match.group(1).strip().lower() == "true", (
            f"DRY_RUN must default to 'true', got '{match.group(1).strip()}'"
        )

    def test_emergency_stop_default_false(self):
        """EMERGENCY_STOP must default to false in .env.example."""
        match = re.search(r"^EMERGENCY_STOP\s*=\s*(.+)$", ENV_EXAMPLE_SRC, re.MULTILINE)
        assert match, "EMERGENCY_STOP line not found in .env.example"
        assert match.group(1).strip().lower() == "false"


# ===========================================================================
# 8. No hardcoded API keys or secrets in Python files
# ===========================================================================

class TestNoHardcodedSecrets:
    # Patterns that suggest hardcoded secrets (with enough length to avoid
    # false positives on short illustrative strings)
    SECRET_PATTERNS = [
        # OpenAI / Anthropic style keys
        (r"""["']sk-[A-Za-z0-9]{20,}["']""", "OpenAI/Anthropic API key"),
        # Tavily keys
        (r"""["']tvly-[A-Za-z0-9]{20,}["']""", "Tavily API key"),
        # Long hex strings that look like secrets (40+ chars)
        (r"""["'][0-9a-f]{40,}["']""", "Hex secret"),
        # JWT tokens (eyJ...)
        (r"""["']eyJ[A-Za-z0-9_-]{50,}["']""", "JWT token"),
        # Hardcoded Bearer tokens (not dynamically constructed)
        (r"""["']Bearer\s+[A-Za-z0-9_.-]{30,}["']""", "Bearer token"),
    ]

    def _collect_python_files(self):
        """Collect all Python files under the project, excluding venv."""
        py_files = []
        for p in PROJECT_DIR.rglob("*.py"):
            rel = str(p.relative_to(PROJECT_DIR))
            # Skip virtual environments and node_modules
            if any(skip in rel for skip in ("venv/", ".venv/", "node_modules/", "__pycache__/")):
                continue
            py_files.append(p)
        return py_files

    @pytest.mark.parametrize(
        "pattern,label",
        SECRET_PATTERNS,
        ids=[label for _, label in SECRET_PATTERNS],
    )
    def test_no_hardcoded_secrets(self, pattern, label):
        """No Python file should contain hardcoded secret patterns."""
        violations = []
        for py_file in self._collect_python_files():
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            matches = re.findall(pattern, content)
            if matches:
                rel = py_file.relative_to(PROJECT_DIR)
                violations.append(f"  {rel}: {matches[0][:40]}...")

        assert not violations, (
            f"Hardcoded {label} found in:\n" + "\n".join(violations)
        )


# ===========================================================================
# 9. SQL migration files have valid syntax (basic check)
# ===========================================================================

class TestSqlMigrations:
    def test_migrations_directory_exists(self):
        """supabase/migrations/ directory must exist."""
        assert MIGRATIONS_DIR.exists()

    def test_at_least_one_migration(self):
        """There must be at least one SQL migration file."""
        sql_files = list(MIGRATIONS_DIR.glob("*.sql"))
        assert len(sql_files) > 0

    def test_migrations_contain_ddl(self):
        """Each SQL migration must contain CREATE or ALTER statements."""
        sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for sql_file in sql_files:
            content = sql_file.read_text(encoding="utf-8").upper()
            has_ddl = any(
                kw in content
                for kw in ("CREATE", "ALTER", "INSERT", "DROP", "ADD")
            )
            assert has_ddl, (
                f"{sql_file.name} does not contain any DDL statements "
                "(CREATE, ALTER, INSERT, DROP, ADD)"
            )

    def test_migrations_numbered_sequentially(self):
        """Migration files should be numbered (NNN_ prefix)."""
        sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for sql_file in sql_files:
            assert re.match(r"^\d{3}_", sql_file.name), (
                f"{sql_file.name} does not follow NNN_ numbering convention"
            )


# ===========================================================================
# 10. CODEOWNERS protects critical paths
# ===========================================================================

class TestCodeowners:
    CODEOWNERS_PATHS = [
        PROJECT_DIR / "CODEOWNERS",
        PROJECT_DIR / ".github" / "CODEOWNERS",
        PROJECT_DIR / "docs" / "CODEOWNERS",
    ]

    def _find_codeowners(self) -> str | None:
        for p in self.CODEOWNERS_PATHS:
            if p.exists():
                return p.read_text(encoding="utf-8")
        return None

    def test_codeowners_exists(self):
        """CODEOWNERS file must exist."""
        content = self._find_codeowners()
        assert content is not None, (
            "CODEOWNERS not found in project root, .github/, or docs/"
        )

    CRITICAL_PATHS = [
        "execute_trade.py",
        "run_agents.py",
        ".env",
        "supabase/",
    ]

    @pytest.mark.parametrize("path", CRITICAL_PATHS)
    def test_critical_path_protected(self, path):
        """Critical paths must be listed in CODEOWNERS."""
        content = self._find_codeowners()
        if content is None:
            pytest.skip("CODEOWNERS file not found")
        assert path in content, (
            f"'{path}' is not protected in CODEOWNERS"
        )

    def test_codeowners_has_owner(self):
        """CODEOWNERS entries must specify at least one owner (@user or @team)."""
        content = self._find_codeowners()
        if content is None:
            pytest.skip("CODEOWNERS file not found")
        # Find non-comment, non-empty lines and check for @ mention
        entries = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(entries) > 0, "CODEOWNERS has no entries"
        for entry in entries:
            assert "@" in entry, (
                f"CODEOWNERS entry missing owner: {entry}"
            )
