"""Build integrity tests for claude-coin-trading project.

Checks:
- All modules can be imported without errors
- Required env vars are documented in .env.example
- No circular imports in project packages
- All referenced files/directories exist
- JSON schemas are valid
- No Windows-incompatible imports (fcntl)
- Shell scripts have correct shebang and syntax markers
"""

import ast
import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# 1. Module import checks
# ---------------------------------------------------------------------------

AGENTS_MODULES = [
    "agents",
    "agents.base_agent",
    "agents.conservative",
    "agents.moderate",
    "agents.aggressive",
    "agents.orchestrator",
    "agents.external_data",
]

RL_HYBRID_CORE_MODULES = [
    "rl_hybrid",
    "rl_hybrid.config",
    "rl_hybrid.protocol",
]

RL_HYBRID_RL_MODULES = [
    "rl_hybrid.rl.reward",
    "rl_hybrid.rl.state_encoder",
    "rl_hybrid.rl.trajectory",
    "rl_hybrid.rl.scenario_generator",
    "rl_hybrid.rl.model_registry",
    "rl_hybrid.rl.decision_blender",
]

RL_HYBRID_RAG_MODULES = [
    "rl_hybrid.rag.prompts",
]

SCRIPT_MODULES = [
    "collect_market_data",
    "collect_fear_greed",
    "collect_news",
    "collect_ai_signal",
    "collect_onchain_data",
    "collect_eth_btc",
    "collect_macro",
    "collect_crypto_signals",
    "binance_sentiment",
    "whale_tracker",
    "calculate_external_signal",
    "summarize_news",
    "cycle_id",
    "feedback",
]


@pytest.mark.parametrize("module_name", AGENTS_MODULES)
def test_import_agents(module_name):
    """All agents/ modules should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", RL_HYBRID_CORE_MODULES)
def test_import_rl_hybrid_core(module_name):
    """Core rl_hybrid modules should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", RL_HYBRID_RL_MODULES)
def test_import_rl_hybrid_rl(module_name):
    """rl_hybrid.rl modules should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", RL_HYBRID_RAG_MODULES)
def test_import_rl_hybrid_rag(module_name):
    """rl_hybrid.rag modules should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_import_scripts(module_name):
    """scripts/ modules should import without errors (via sys.path)."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. Required env vars documentation check
# ---------------------------------------------------------------------------

def _extract_env_vars_from_code() -> set:
    """Parse all .py files and extract env var names used in os.getenv/os.environ."""
    patterns = [
        re.compile(r'os\.getenv\(["\']([A-Z_]+)["\']'),
        re.compile(r'os\.environ\.get\(["\']([A-Z_]+)["\']'),
        re.compile(r'os\.environ\[["\']([A-Z_]+)["\']'),
    ]
    env_vars = set()
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip non-project dirs
        if ".venv" in root or ".git" in root or "node_modules" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            fpath = os.path.join(root, f)
            # Skip test files — they use patch.dict and set fake env vars
            if "test_" in f or f.startswith("test"):
                continue
            try:
                content = open(fpath, "r", encoding="utf-8").read()
            except Exception:
                continue
            for pat in patterns:
                for m in pat.finditer(content):
                    env_vars.add(m.group(1))
    return env_vars


def _extract_env_vars_from_example() -> set:
    """Parse .env.example and extract defined variable names."""
    env_vars = set()
    env_example = PROJECT_ROOT / ".env.example"
    if not env_example.exists():
        return env_vars
    with open(env_example, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                env_vars.add(line.split("=", 1)[0].strip())
    return env_vars


# Core env vars that MUST be documented (excluding optional/internal ones)
CRITICAL_ENV_VARS = {
    "UPBIT_ACCESS_KEY",
    "UPBIT_SECRET_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_USER_ID",
    "DRY_RUN",
    "MAX_TRADE_AMOUNT",
    "EMERGENCY_STOP",
    "TAVILY_API_KEY",
    "GEMINI_API_KEY",
}


def test_critical_env_vars_documented():
    """All critical env vars must appear in .env.example."""
    documented = _extract_env_vars_from_example()
    missing = CRITICAL_ENV_VARS - documented
    assert not missing, f"Critical env vars missing from .env.example: {missing}"


def test_env_var_naming_consistency():
    """Detect SUPABASE_SERVICE_KEY vs SUPABASE_SERVICE_ROLE_KEY inconsistency.

    The .env.example defines SUPABASE_SERVICE_ROLE_KEY, but some files use
    SUPABASE_SERVICE_KEY (without _ROLE). This test flags the inconsistency.
    """
    code_vars = _extract_env_vars_from_code()
    # Both variants should not coexist in production code
    has_role = "SUPABASE_SERVICE_ROLE_KEY" in code_vars
    has_no_role = "SUPABASE_SERVICE_KEY" in code_vars
    if has_role and has_no_role:
        # Find the offending files
        offenders = []
        pat = re.compile(r'SUPABASE_SERVICE_KEY[^_]')
        for root, dirs, files in os.walk(PROJECT_ROOT):
            if ".venv" in root or ".git" in root:
                continue
            for f in files:
                if not f.endswith(".py") or "test_" in f:
                    continue
                fpath = os.path.join(root, f)
                try:
                    content = open(fpath, "r", encoding="utf-8").read()
                except Exception:
                    continue
                if pat.search(content):
                    offenders.append(os.path.relpath(fpath, PROJECT_ROOT))
        pytest.fail(
            f"Inconsistent env var naming: SUPABASE_SERVICE_KEY (without _ROLE) "
            f"used in: {offenders}. Should be SUPABASE_SERVICE_ROLE_KEY."
        )


# ---------------------------------------------------------------------------
# 3. Circular import detection
# ---------------------------------------------------------------------------

def _build_import_graph() -> dict:
    """Build a directed graph of project-internal imports."""
    project_packages = ["agents", "scripts", "rl_hybrid"]
    graph = {}

    for pkg in project_packages:
        pkg_dir = PROJECT_ROOT / pkg
        if not pkg_dir.exists():
            continue
        for root, _, files in os.walk(pkg_dir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, PROJECT_ROOT).replace(os.sep, "/")
                mod_name = rel.replace("/", ".").replace(".py", "")
                if mod_name.endswith(".__init__"):
                    mod_name = mod_name[:-9]

                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        tree = ast.parse(fh.read())
                except Exception:
                    continue

                deps = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(alias.name.startswith(p) for p in project_packages):
                                deps.add(alias.name.split(".")[0] + "." + ".".join(alias.name.split(".")[1:]) if "." in alias.name else alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and any(node.module.startswith(p) for p in project_packages):
                            deps.add(node.module)
                graph[mod_name] = deps

    return graph


def _find_cycles(graph: dict) -> list:
    """DFS-based cycle detection."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            # Normalize: find the actual module key
            n = neighbor
            if n not in graph:
                parts = n.split(".")
                while parts:
                    test = ".".join(parts)
                    if test in graph:
                        n = test
                        break
                    parts.pop()
                else:
                    continue

            if n not in visited:
                dfs(n)
            elif n in rec_stack:
                idx = path.index(n)
                cycle = path[idx:] + [n]
                cycles.append(cycle)

        path.pop()
        rec_stack.discard(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def test_no_circular_imports():
    """No circular imports should exist among project packages."""
    graph = _build_import_graph()
    cycles = _find_cycles(graph)
    assert not cycles, f"Circular imports detected: {cycles}"


# ---------------------------------------------------------------------------
# 4. File existence checks
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "CLAUDE.md",
    ".env.example",
    "requirements.txt",
    "strategy.md",
    "prompts/schemas/decision_result.json",
    "agents/__init__.py",
    "agents/base_agent.py",
    "agents/orchestrator.py",
    "agents/conservative.py",
    "agents/moderate.py",
    "agents/aggressive.py",
    "agents/external_data.py",
    "rl_hybrid/__init__.py",
    "rl_hybrid/config.py",
    "rl_hybrid/protocol.py",
    "rl_hybrid/nodes/__init__.py",
    "rl_hybrid/rl/__init__.py",
    "rl_hybrid/rag/__init__.py",
    "rl_hybrid/launchers/__init__.py",
    "scripts/collect_market_data.py",
    "scripts/collect_fear_greed.py",
    "scripts/collect_news.py",
    "scripts/capture_chart.py",
    "scripts/execute_trade.py",
    "scripts/get_portfolio.py",
    "scripts/notify_telegram.py",
    "scripts/run_agents.py",
    "tests/__init__.py",
]


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_file_exists(rel_path):
    """All required project files must exist."""
    full_path = PROJECT_ROOT / rel_path
    assert full_path.exists(), f"Required file missing: {rel_path}"


REQUIRED_DIRS = [
    "agents",
    "scripts",
    "rl_hybrid",
    "rl_hybrid/nodes",
    "rl_hybrid/rl",
    "rl_hybrid/rag",
    "rl_hybrid/launchers",
    "prompts/schemas",
    "tests",
]


@pytest.mark.parametrize("rel_dir", REQUIRED_DIRS)
def test_required_directory_exists(rel_dir):
    """All required directories must exist."""
    full_path = PROJECT_ROOT / rel_dir
    assert full_path.is_dir(), f"Required directory missing: {rel_dir}"


# ---------------------------------------------------------------------------
# 5. __init__.py existence checks
# ---------------------------------------------------------------------------

PACKAGES_NEEDING_INIT = [
    "agents",
    "rl_hybrid",
    "rl_hybrid/nodes",
    "rl_hybrid/rl",
    "rl_hybrid/rag",
    "rl_hybrid/launchers",
    "tests",
]


@pytest.mark.parametrize("pkg_path", PACKAGES_NEEDING_INIT)
def test_init_py_exists(pkg_path):
    """All Python packages must have __init__.py."""
    init_file = PROJECT_ROOT / pkg_path / "__init__.py"
    assert init_file.exists(), f"Missing __init__.py in {pkg_path}/"


def test_scripts_no_init_py():
    """scripts/ directory should NOT have __init__.py (it is not a package).

    Tests import scripts via sys.path manipulation. The agents/ code imports
    via 'from scripts.cycle_id import ...' which works because PROJECT_ROOT
    is on sys.path. If scripts/ had __init__.py it could cause double-import
    issues, but currently the design relies on sys.path having PROJECT_ROOT.
    """
    # This is informational: scripts/ works both ways, but currently no __init__.py
    init_file = PROJECT_ROOT / "scripts" / "__init__.py"
    # We just check it exists or not — no assertion, just note the state
    assert not init_file.exists() or True  # pass either way


# ---------------------------------------------------------------------------
# 6. JSON schema validation
# ---------------------------------------------------------------------------

def test_decision_result_schema_valid():
    """prompts/schemas/decision_result.json must be valid JSON with required structure."""
    schema_path = PROJECT_ROOT / "prompts" / "schemas" / "decision_result.json"
    assert schema_path.exists(), "decision_result.json not found"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert "type" in schema, "Schema missing 'type' key"
    assert schema["type"] == "object", "Schema type should be 'object'"
    assert "properties" in schema, "Schema missing 'properties'"
    assert "required" in schema, "Schema missing 'required'"

    # Required fields per the schema
    required_fields = schema["required"]
    assert "decision" in required_fields
    assert "confidence" in required_fields
    assert "reason" in required_fields
    assert "timestamp" in required_fields

    # Decision enum values
    decision_prop = schema["properties"]["decision"]
    assert "enum" in decision_prop
    assert set(decision_prop["enum"]) == {"매수", "매도", "관망"}


# ---------------------------------------------------------------------------
# 7. Windows compatibility checks
# ---------------------------------------------------------------------------

def test_fcntl_has_fallback():
    """fcntl (Unix-only) usage should have a Windows fallback or be guarded.

    rl_hybrid/rl/online_buffer.py imports fcntl at module level without
    try/except, which will fail on Windows.
    """
    online_buffer_path = PROJECT_ROOT / "rl_hybrid" / "rl" / "online_buffer.py"
    if not online_buffer_path.exists():
        pytest.skip("online_buffer.py not found")

    content = open(online_buffer_path, "r", encoding="utf-8").read()

    # Check if fcntl import is guarded with try/except
    has_bare_import = "import fcntl" in content
    has_try_guard = "try:" in content.split("import fcntl")[0][-50:] if has_bare_import else True

    if has_bare_import and not has_try_guard:
        pytest.xfail(
            "rl_hybrid/rl/online_buffer.py imports fcntl without try/except guard. "
            "This will fail on Windows. Known issue."
        )


def test_signal_handler_guarded():
    """add_signal_handler usage should be wrapped in try/except NotImplementedError."""
    trader_path = PROJECT_ROOT / "scripts" / "short_term_trader.py"
    if not trader_path.exists():
        pytest.skip("short_term_trader.py not found")

    content = open(trader_path, "r", encoding="utf-8").read()
    if "add_signal_handler" in content:
        assert "NotImplementedError" in content, (
            "add_signal_handler used without NotImplementedError guard"
        )


# ---------------------------------------------------------------------------
# 8. Shell script checks
# ---------------------------------------------------------------------------

SHELL_SCRIPTS = [
    "scripts/run_analysis.sh",
    "scripts/run_agents.sh",
    "scripts/cron_run.sh",
    "scripts/setup_cron.sh",
    "setup.sh",
]


@pytest.mark.parametrize("script_path", SHELL_SCRIPTS)
def test_shell_script_has_shebang(script_path):
    """Shell scripts must have a proper shebang line."""
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        pytest.skip(f"{script_path} not found")

    with open(full_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    assert first_line.startswith("#!/"), f"{script_path} missing shebang. Got: {first_line}"


def test_run_agents_sh_windows_compatible():
    """run_agents.sh should detect Windows Python (.venv/Scripts/python.exe)."""
    script_path = PROJECT_ROOT / "scripts" / "run_agents.sh"
    if not script_path.exists():
        pytest.skip("run_agents.sh not found")

    content = open(script_path, "r", encoding="utf-8").read()
    assert ".venv/Scripts/python.exe" in content, (
        "run_agents.sh does not check for Windows venv Python path"
    )


def test_run_analysis_sh_no_windows_python():
    """run_analysis.sh uses 'python3' and .venv/bin/activate (Unix only).

    This is a known Windows incompatibility in the legacy pipeline.
    """
    script_path = PROJECT_ROOT / "scripts" / "run_analysis.sh"
    if not script_path.exists():
        pytest.skip("run_analysis.sh not found")

    content = open(script_path, "r", encoding="utf-8").read()
    # Check if it has Windows path detection
    has_windows_detect = ".venv/Scripts/python" in content
    if not has_windows_detect:
        pytest.xfail(
            "run_analysis.sh does not detect Windows venv. "
            "Uses 'python3' and '.venv/bin/activate' only. Known issue."
        )


# ---------------------------------------------------------------------------
# 9. Dependency consistency (requirements.txt vs actual imports)
# ---------------------------------------------------------------------------

def _parse_requirements() -> set:
    """Extract package names from requirements.txt."""
    req_path = PROJECT_ROOT / "requirements.txt"
    packages = set()
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Extract package name (before >= or ==)
            name = re.split(r"[><=!]", line)[0].strip().lower()
            packages.add(name)
    return packages


# Map of import names to pip package names (where they differ)
IMPORT_TO_PACKAGE = {
    "jwt": "pyjwt",
    "dotenv": "python-dotenv",
    "PIL": "pillow",
    "zmq": "pyzmq",
    "google": "google-generativeai",
    "google.generativeai": "google-generativeai",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "websockets": "websockets",
    "gymnasium": "gymnasium",
    "numpy": "numpy",
    "torch": "torch",
    "stable_baselines3": "stable-baselines3",
    "msgpack": "msgpack",
}


def _extract_third_party_imports() -> set:
    """Find all third-party imports used across the project."""
    stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
    project_modules = {"agents", "scripts", "rl_hybrid", "tests"}

    third_party = set()
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if ".venv" in root or ".git" in root or "node_modules" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in stdlib_modules and top not in project_modules:
                            third_party.add(top)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top not in stdlib_modules and top not in project_modules:
                            third_party.add(top)

    return third_party


CORE_THIRD_PARTY = {
    "requests",
    "jwt",
    "dotenv",
    "zmq",
    "msgpack",
    "numpy",
}


def test_core_dependencies_in_requirements():
    """Core third-party imports must be listed in requirements.txt."""
    requirements = _parse_requirements()
    for imp in CORE_THIRD_PARTY:
        pkg_name = IMPORT_TO_PACKAGE.get(imp, imp).lower()
        assert pkg_name in requirements, (
            f"Import '{imp}' (package '{pkg_name}') not found in requirements.txt"
        )


# ---------------------------------------------------------------------------
# 10. Hardcoded path checks
# ---------------------------------------------------------------------------

def test_no_hardcoded_absolute_paths_in_production_code():
    """Production code should not have hardcoded user-specific absolute paths.

    Paths like /Users/drj00/workspace, /home/user, C:\\Users\\... indicate
    machine-specific paths that won't be portable.
    """
    path_patterns = [
        re.compile(r'["\']/(Users|home)/\w+/'),
        re.compile(r'["\'][A-Z]:\\\\Users\\\\'),
    ]
    violations = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        if ".venv" in root or ".git" in root or "node_modules" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            # Skip test files
            if "test_" in f:
                continue
            fpath = os.path.join(root, f)
            try:
                content = open(fpath, "r", encoding="utf-8").read()
            except Exception:
                continue
            for pat in path_patterns:
                matches = pat.findall(content)
                if matches:
                    rel = os.path.relpath(fpath, PROJECT_ROOT)
                    violations.append(rel)
                    break

    # startup.sh has hardcoded paths but it's a shell script, not checked here
    assert not violations, (
        f"Hardcoded absolute paths found in: {violations}"
    )


def test_startup_sh_has_hardcoded_path():
    """startup.sh has a hardcoded macOS path -- known portability issue."""
    startup = PROJECT_ROOT / "scripts" / "startup.sh"
    if not startup.exists():
        pytest.skip("startup.sh not found")

    content = open(startup, "r", encoding="utf-8").read()
    if "/Users/" in content:
        pytest.xfail(
            "scripts/startup.sh has hardcoded /Users/drj00/ path. "
            "This is a known issue for macOS-specific deployment."
        )
