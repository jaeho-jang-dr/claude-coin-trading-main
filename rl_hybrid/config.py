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
class SystemConfig:
    """전체 시스템 설정"""
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    log_dir: str = "logs/nodes"
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 싱글톤 인스턴스
config = SystemConfig()
