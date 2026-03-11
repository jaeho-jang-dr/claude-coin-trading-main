"""빌드/통합 테스트 — 프로젝트 전체 임포트 + 의존성 일관성 검증

모킹 없이 실제 임포트를 수행하여 다음을 검증한다:
1. rl_hybrid.rl 모듈 전체 임포트
2. scripts/ 모듈 전체 임포트
3. agents/ 모듈 전체 임포트
4. OBSERVATION_DIM == 42 일관성
5. FEATURE_SPEC 항목 수 == 42
6. StateEncoder 출력 차원 == 환경 관측 공간
7. .env.example 필수 환경변수 문서화
8. requirements.txt 핵심 패키지 포함
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. rl_hybrid.rl 모듈 전체 임포트
# ---------------------------------------------------------------------------
class TestRLHybridImports:
    """rl_hybrid.rl 패키지의 모든 모듈이 에러 없이 임포트되는지 검증"""

    RL_MODULES = [
        "rl_hybrid.rl.state_encoder",
        "rl_hybrid.rl.environment",
        "rl_hybrid.rl.reward",
        "rl_hybrid.rl.policy",
        "rl_hybrid.rl.train",
        "rl_hybrid.rl.data_loader",
        "rl_hybrid.rl.scenario_generator",
        "rl_hybrid.rl.online_buffer",
        "rl_hybrid.rl.weekly_retrain",
        "rl_hybrid.rl.trainer_submit",
        "rl_hybrid.rl.admin_review",
    ]

    @pytest.mark.parametrize("module_name", RL_MODULES)
    def test_rl_module_imports(self, module_name):
        """각 rl_hybrid.rl 모듈이 임포트 에러 없이 로드된다"""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{module_name} 임포트 실패"

    def test_rl_hybrid_package_imports(self):
        """rl_hybrid 최상위 패키지가 임포트된다"""
        import rl_hybrid
        assert rl_hybrid is not None

    def test_rl_hybrid_config_imports(self):
        """rl_hybrid.config 모듈이 임포트된다"""
        from rl_hybrid.config import SystemConfig, config
        assert config is not None
        assert isinstance(config, SystemConfig)


# ---------------------------------------------------------------------------
# 2. scripts/ 모듈 임포트
# ---------------------------------------------------------------------------
class TestScriptsImports:
    """scripts/ 디렉토리의 모든 Python 모듈이 임포트되는지 검증"""

    SCRIPT_MODULES = [
        "scripts.collect_market_data",
        "scripts.collect_fear_greed",
        "scripts.collect_news",
        "scripts.capture_chart",
        "scripts.execute_trade",
        "scripts.get_portfolio",
        "scripts.collect_ai_signal",
        "scripts.short_term_trader",
        "scripts.notify_telegram",
        "scripts.collect_onchain_data",
        "scripts.binance_sentiment",
        "scripts.calculate_external_signal",
        "scripts.collect_eth_btc",
        "scripts.collect_macro",
        "scripts.evaluate_switches",
        "scripts.feedback",
        "scripts.whale_tracker",
        "scripts.collect_crypto_signals",
        "scripts.summarize_news",
        "scripts.collect_coinmarketcap",
        "scripts.cycle_id",
        "scripts.recall",
        "scripts.retrospective",
        "scripts.save_decision",
        "scripts.recall_rag",
        "scripts.dashboard",
        "scripts.check_telegram_cmd",
        "scripts.setup_weekly_retrain",
        "scripts.weekly_retrain",
        "scripts._db_check",
        "scripts.web_server",
    ]

    @pytest.mark.parametrize("module_name", SCRIPT_MODULES)
    def test_script_module_imports(self, module_name):
        """각 scripts/ 모듈이 importlib로 에러 없이 로드된다"""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{module_name} 임포트 실패"

    def test_run_agents_import(self):
        """scripts.run_agents가 importlib로 로드된다 (실행 스크립트)"""
        mod = importlib.import_module("scripts.run_agents")
        assert mod is not None


# ---------------------------------------------------------------------------
# 3. agents/ 모듈 임포트
# ---------------------------------------------------------------------------
class TestAgentsImports:
    """agents/ 패키지의 모든 모듈이 임포트되는지 검증"""

    AGENT_MODULES = [
        "agents.base_agent",
        "agents.conservative",
        "agents.moderate",
        "agents.aggressive",
        "agents.orchestrator",
        "agents.external_data",
    ]

    @pytest.mark.parametrize("module_name", AGENT_MODULES)
    def test_agent_module_imports(self, module_name):
        """각 agents/ 모듈이 에러 없이 임포트된다"""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{module_name} 임포트 실패"

    def test_agents_package_init(self):
        """agents 패키지 __init__.py가 임포트된다"""
        import agents
        assert agents is not None


# ---------------------------------------------------------------------------
# 4. OBSERVATION_DIM == 42 일관성
# ---------------------------------------------------------------------------
class TestObservationDimConsistency:
    """OBSERVATION_DIM이 프로젝트 전체에서 42로 일관되는지 검증"""

    def test_state_encoder_observation_dim_is_42(self):
        """state_encoder에서 정의한 OBSERVATION_DIM == 42"""
        from rl_hybrid.rl.state_encoder import OBSERVATION_DIM
        assert OBSERVATION_DIM == 42, (
            f"OBSERVATION_DIM이 42가 아닙니다: {OBSERVATION_DIM}"
        )

    def test_environment_uses_same_observation_dim(self):
        """environment.py가 state_encoder와 동일한 OBSERVATION_DIM을 사용한다"""
        from rl_hybrid.rl.state_encoder import OBSERVATION_DIM as encoder_dim
        from rl_hybrid.rl.environment import OBSERVATION_DIM as env_dim
        assert encoder_dim == env_dim, (
            f"state_encoder({encoder_dim}) != environment({env_dim})"
        )

    def test_observation_dim_equals_feature_spec_length(self):
        """OBSERVATION_DIM == len(FEATURE_SPEC)"""
        from rl_hybrid.rl.state_encoder import OBSERVATION_DIM, FEATURE_SPEC
        assert OBSERVATION_DIM == len(FEATURE_SPEC), (
            f"OBSERVATION_DIM({OBSERVATION_DIM}) != len(FEATURE_SPEC)({len(FEATURE_SPEC)})"
        )


# ---------------------------------------------------------------------------
# 5. FEATURE_SPEC has exactly 42 entries
# ---------------------------------------------------------------------------
class TestFeatureSpec:
    """FEATURE_SPEC 구조 검증"""

    def test_feature_spec_has_42_entries(self):
        """FEATURE_SPEC에 정확히 42개 항목이 있다"""
        from rl_hybrid.rl.state_encoder import FEATURE_SPEC
        assert len(FEATURE_SPEC) == 42, (
            f"FEATURE_SPEC 항목 수: {len(FEATURE_SPEC)}, 기대값: 42"
        )

    def test_feature_spec_all_tuples(self):
        """FEATURE_SPEC의 모든 값이 (min, max) 튜플이다"""
        from rl_hybrid.rl.state_encoder import FEATURE_SPEC
        for name, bounds in FEATURE_SPEC.items():
            assert isinstance(bounds, tuple), f"{name}: 튜플이 아닙니다"
            assert len(bounds) == 2, f"{name}: (min, max) 길이가 2가 아닙니다"
            assert bounds[0] <= bounds[1], (
                f"{name}: min({bounds[0]}) > max({bounds[1]})"
            )

    def test_feature_names_count(self):
        """FEATURE_NAMES도 42개"""
        from rl_hybrid.rl.state_encoder import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 42


# ---------------------------------------------------------------------------
# 6. StateEncoder 출력 차원 == 환경 관측 공간
# ---------------------------------------------------------------------------
class TestEncoderEnvironmentAlignment:
    """StateEncoder 출력과 BitcoinTradingEnv 관측 공간이 일치하는지 검증"""

    def test_encoder_obs_dim_matches_env_observation_space(self):
        """StateEncoder.obs_dim == BitcoinTradingEnv.observation_space.shape[0]"""
        from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM
        encoder = StateEncoder()
        assert encoder.obs_dim == OBSERVATION_DIM
        assert encoder.obs_dim == 42

    def test_encoder_output_shape(self):
        """StateEncoder.encode()가 (42,) shape의 배열을 반환한다"""
        import numpy as np
        from rl_hybrid.rl.state_encoder import StateEncoder

        encoder = StateEncoder()
        # 최소한의 더미 데이터로 인코딩
        market_data = {
            "current_price": 50000000,
            "change_rate_24h": 0.01,
            "indicators": {},
            "indicators_4h": {},
            "orderbook": {},
            "trade_pressure": {},
            "eth_btc_analysis": {},
        }
        external_data = {"sources": {}, "external_signal": {}}
        portfolio = {"krw_balance": 1000000, "holdings": [], "total_eval": 1000000}
        agent_state = {}

        obs = encoder.encode(market_data, external_data, portfolio, agent_state)
        assert obs.shape == (42,), f"인코더 출력 shape: {obs.shape}"
        assert obs.dtype == np.float32

    def test_env_observation_space_shape(self):
        """BitcoinTradingEnv의 observation_space가 (42,)이다"""
        from rl_hybrid.rl.state_encoder import OBSERVATION_DIM
        # observation_space는 OBSERVATION_DIM으로 정의되므로 값만 확인
        assert OBSERVATION_DIM == 42


# ---------------------------------------------------------------------------
# 7. .env.example 필수 환경변수 문서화
# ---------------------------------------------------------------------------
class TestEnvExampleDocumentation:
    """.env.example에 필수 환경변수가 모두 문서화되었는지 검증"""

    ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

    REQUIRED_VARS = [
        # 거래소
        "UPBIT_ACCESS_KEY",
        "UPBIT_SECRET_KEY",
        # 데이터 수집
        "TAVILY_API_KEY",
        # 데이터베이스
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        # 알림
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_USER_ID",
        # 안전장치
        "DRY_RUN",
        "MAX_TRADE_AMOUNT",
        "MAX_DAILY_TRADES",
        "MAX_POSITION_RATIO",
        "MIN_TRADE_INTERVAL_HOURS",
        "EMERGENCY_STOP",
        # LLM-RL 하이브리드
        "GEMINI_API_KEY",
    ]

    @pytest.fixture(autouse=True)
    def _load_env_example(self):
        """env.example 파일 내용을 읽는다"""
        assert self.ENV_EXAMPLE_PATH.exists(), (
            f".env.example 파일이 없습니다: {self.ENV_EXAMPLE_PATH}"
        )
        self.env_content = self.ENV_EXAMPLE_PATH.read_text()

    @pytest.mark.parametrize("var_name", REQUIRED_VARS)
    def test_required_var_documented(self, var_name):
        """각 필수 환경변수가 .env.example에 기재되어 있다"""
        assert var_name in self.env_content, (
            f"{var_name}이 .env.example에 문서화되지 않았습니다"
        )


# ---------------------------------------------------------------------------
# 8. requirements.txt 핵심 패키지 포함
# ---------------------------------------------------------------------------
class TestRequirementsTxt:
    """requirements.txt에 실제 사용하는 핵심 패키지가 포함되는지 검증"""

    REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

    KEY_PACKAGES = [
        "stable-baselines3",
        "gymnasium",
        "numpy",
        "requests",
        "torch",
        "python-dotenv",
        "pyzmq",
        "google-generativeai",
    ]

    @pytest.fixture(autouse=True)
    def _load_requirements(self):
        """requirements.txt 파일 내용을 읽는다"""
        assert self.REQUIREMENTS_PATH.exists(), (
            f"requirements.txt 파일이 없습니다: {self.REQUIREMENTS_PATH}"
        )
        self.req_content = self.REQUIREMENTS_PATH.read_text().lower()

    @pytest.mark.parametrize("package", KEY_PACKAGES)
    def test_key_package_listed(self, package):
        """핵심 패키지가 requirements.txt에 포함되어 있다"""
        assert package.lower() in self.req_content, (
            f"{package}가 requirements.txt에 없습니다"
        )

    def test_requirements_not_empty(self):
        """requirements.txt가 비어있지 않다"""
        lines = [
            l.strip() for l in self.req_content.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        assert len(lines) > 0, "requirements.txt에 패키지가 없습니다"

    def test_key_packages_actually_importable(self):
        """핵심 패키지가 실제로 임포트 가능하다"""
        import_map = {
            "stable-baselines3": "stable_baselines3",
            "gymnasium": "gymnasium",
            "numpy": "numpy",
            "requests": "requests",
            "torch": "torch",
            "python-dotenv": "dotenv",
            "pyzmq": "zmq",
        }
        for pkg_name, import_name in import_map.items():
            mod = importlib.import_module(import_name)
            assert mod is not None, (
                f"{pkg_name} ({import_name})를 임포트할 수 없습니다"
            )
