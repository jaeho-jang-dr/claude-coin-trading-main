"""분산 시스템 설정 — 환경변수 및 기본값 관리"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class ZMQConfig:
    """ZeroMQ 통신 설정"""
    main_brain_host: str = os.getenv("ZMQ_MAIN_BRAIN_HOST", "127.0.0.1")
    router_port: int = int(os.getenv("ZMQ_MAIN_BRAIN_PORT", "5555"))
    pub_port: int = int(os.getenv("ZMQ_PUB_PORT", "5556"))
    heartbeat_interval: int = int(os.getenv("ZMQ_HEARTBEAT_INTERVAL", "30"))
    request_timeout_ms: int = int(os.getenv("ZMQ_REQUEST_TIMEOUT", "30000"))
    max_missed_heartbeats: int = 3

    @property
    def router_addr(self) -> str:
        return f"tcp://{self.main_brain_host}:{self.router_port}"

    @property
    def pub_addr(self) -> str:
        return f"tcp://{self.main_brain_host}:{self.pub_port}"


@dataclass
class GeminiConfig:
    """Gemini API 설정"""
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    analysis_model: str = os.getenv("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dim: int = int(os.getenv("RAG_EMBEDDING_DIM", "3072"))
    max_retries: int = 3
    rpm_limit: int = 10  # requests per minute


@dataclass
class RAGConfig:
    """RAG 파이프라인 설정"""
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    similarity_threshold: float = 0.7
    embedding_dim: int = int(os.getenv("RAG_EMBEDDING_DIM", "3072"))


@dataclass
class SupabaseConfig:
    """Supabase 설정 (기존 .env 재사용)"""
    url: str = os.getenv("SUPABASE_URL", "")
    service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    db_url: str = os.getenv("SUPABASE_DB_URL", "")


@dataclass
class MultiObjectiveRLConfig:
    """다중 목표 RL 설정"""
    # 목표 가중치 (profit, risk, efficiency, sharpe, tail_risk)
    weights: dict = field(default_factory=lambda: {
        "profit": 0.2, "risk": 0.2, "efficiency": 0.2,
        "sharpe": 0.2, "tail_risk": 0.2,
    })
    # 제약 임계값
    max_mdd: float = float(os.getenv("MORL_MAX_MDD", "0.10"))
    max_trades_per_day: float = float(os.getenv("MORL_MAX_TRADES_PER_DAY", "20"))
    # CVaR 설정
    cvar_alpha: float = 0.05
    cvar_window: int = 200
    # Pareto frontier
    pareto_max_k: int = 10
    pareto_save_dir: str = "data/rl_models/pareto"
    # Envelope MORL
    envelope_morl: bool = os.getenv("MORL_ENVELOPE", "false").lower() == "true"
    # Adaptive weights
    adaptive_weights: bool = os.getenv("MORL_ADAPTIVE", "true").lower() == "true"
    weight_adjustment_rate: float = 0.1
    weight_update_interval: int = 1000
    # Evaluation
    eval_freq: int = 10000


@dataclass
class LLMStateConfig:
    """LLM State Encoder 설정"""
    enabled: bool = os.getenv("LLM_STATE_ENCODER_ENABLED", "false").lower() == "true"
    projected_dim: int = int(os.getenv("LLM_STATE_PROJECTED_DIM", "64"))
    cache_ttl_seconds: int = int(os.getenv("LLM_STATE_CACHE_TTL", "3600"))


@dataclass
class SelfTuningConfig:
    """Self-Tuning RL 설정"""
    enabled: bool = os.getenv("SELF_TUNING_ENABLED", "false").lower() == "true"
    tuning_frequency_hours: int = int(os.getenv("SELF_TUNING_FREQUENCY_HOURS", "24"))
    auto_apply: bool = os.getenv("SELF_TUNING_AUTO_APPLY", "true").lower() == "true"
    approval_threshold_pct: float = float(os.getenv("SELF_TUNING_APPROVAL_THRESHOLD", "30.0"))
    rollback_sharpe_drop: float = float(os.getenv("SELF_TUNING_ROLLBACK_SHARPE_DROP", "-0.05"))


@dataclass
class MultiAgentConsensusConfig:
    """Multi-Agent Consensus RL 설정"""
    enabled: bool = os.getenv("MULTI_AGENT_ENABLED", "true").lower() == "true"
    model_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rl_models", "multi_agent",
    )
    scalping_steps: int = int(os.getenv("MA_SCALPING_STEPS", "200000"))
    swing_steps: int = int(os.getenv("MA_SWING_STEPS", "500000"))
    weight_learner_steps: int = int(os.getenv("MA_WEIGHT_LEARNER_STEPS", "50000"))
    veto_threshold: float = float(os.getenv("MA_VETO_THRESHOLD", "0.7"))
    default_scalp_weight: float = float(os.getenv("MA_SCALP_WEIGHT", "0.4"))
    default_swing_weight: float = float(os.getenv("MA_SWING_WEIGHT", "0.6"))


@dataclass
class SystemConfig:
    """전체 시스템 설정"""
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    multi_objective_rl: MultiObjectiveRLConfig = field(default_factory=MultiObjectiveRLConfig)
    llm_state: LLMStateConfig = field(default_factory=LLMStateConfig)
    self_tuning: SelfTuningConfig = field(default_factory=SelfTuningConfig)
    multi_agent: MultiAgentConsensusConfig = field(default_factory=MultiAgentConsensusConfig)
    log_dir: str = "logs/nodes"
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 싱글톤 인스턴스
config = SystemConfig()
