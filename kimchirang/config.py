"""Kimchirang 설정 -- 환경변수 + 시스템 파라미터

모든 API 키는 .env에서 로드한다. 직접 하드코딩 금지.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# API Keys (환경변수에서 로드)
# ============================================================

@dataclass(frozen=True)
class UpbitConfig:
    access_key: str = field(default_factory=lambda: os.getenv("UPBIT_ACCESS_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("UPBIT_SECRET_KEY", ""))
    # Upbit WebSocket
    ws_url: str = "wss://api.upbit.com/websocket/v1"
    # Upbit REST
    rest_url: str = "https://api.upbit.com/v1"


@dataclass(frozen=True)
class BinanceConfig:
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    # Binance USDM Futures WebSocket
    ws_url: str = "wss://fstream.binance.com/ws"
    # Binance USDM Futures REST
    rest_url: str = "https://fapi.binance.com"


# ============================================================
# Trading Parameters
# ============================================================

@dataclass
class TradingParams:
    # 대상 심볼
    upbit_symbol: str = "KRW-BTC"
    binance_symbol: str = "BTC/USDT"
    binance_futures_symbol: str = "BTCUSDT"

    # 김치프리미엄 진입/청산 임계값 (%)
    kp_entry_threshold: float = 3.0      # KP >= 3% 이면 진입
    kp_exit_threshold: float = 0.5       # KP <= 0.5% 이면 청산
    kp_stop_loss: float = 8.0            # KP >= 8% 이면 손절 (역방향 확대)

    # 포지션 크기
    trade_amount_krw: int = 100_000      # 1회 매매 금액 (KRW)
    max_position_krw: int = 1_000_000    # 최대 포지션 (KRW)
    leverage: int = 1                    # Binance Futures 레버리지 (1x 권장)

    # 주문 실행
    max_slippage_pct: float = 0.3        # 최대 허용 슬리피지 (%)
    order_timeout_sec: float = 5.0       # 주문 타임아웃
    retry_count: int = 2                 # 실패 시 재시도 횟수

    # FX Rate
    fx_update_interval_sec: int = 60     # USD/KRW 갱신 주기 (초)

    # 안전장치
    dry_run: bool = field(
        default_factory=lambda: os.getenv("KR_DRY_RUN", os.getenv("DRY_RUN", "true")).lower() == "true"
    )
    emergency_stop: bool = field(
        default_factory=lambda: os.getenv("KR_EMERGENCY_STOP", "false").lower() == "true"
    )
    max_daily_trades: int = 20
    min_trade_interval_sec: int = 60     # 최소 매매 간격 (초)


# ============================================================
# RL Integration
# ============================================================

@dataclass
class RLConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("KR_RL_ENABLED", "true").lower() == "true"
    )
    # RL 상태 벡터 차원 (KP 전용 피처)
    state_dim: int = 12
    # RL 액션: 0=Hold, 1=Enter, 2=Exit
    action_dim: int = 3
    # RL 모델 경로
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "KR_RL_MODEL_PATH",
            os.path.join("data", "rl_models", "kimchirang", "best_model.zip"),
        )
    )


# ============================================================
# Logging & DB
# ============================================================

@dataclass(frozen=True)
class DBConfig:
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))


# ============================================================
# Singleton Config
# ============================================================

class KimchirangConfig:
    """전체 설정을 하나로 묶는 컨테이너"""

    def __init__(self):
        self.upbit = UpbitConfig()
        self.binance = BinanceConfig()
        self.trading = TradingParams()
        self.rl = RLConfig()
        self.db = DBConfig()

    def validate(self) -> list[str]:
        """설정 검증 -- 누락된 필수 항목 반환"""
        errors = []
        if not self.upbit.access_key:
            errors.append("UPBIT_ACCESS_KEY 미설정")
        if not self.upbit.secret_key:
            errors.append("UPBIT_SECRET_KEY 미설정")
        if not self.binance.api_key:
            errors.append("BINANCE_API_KEY 미설정")
        if not self.binance.api_secret:
            errors.append("BINANCE_API_SECRET 미설정")
        if self.trading.leverage > 5:
            errors.append(f"레버리지 {self.trading.leverage}x 과다 (최대 5x)")
        return errors

    def summary(self) -> str:
        """현재 설정 요약 (API 키 마스킹)"""
        def mask(s: str) -> str:
            return f"{s[:4]}...{s[-4:]}" if len(s) > 8 else "***"

        return (
            f"=== Kimchirang Config ===\n"
            f"DRY_RUN: {self.trading.dry_run}\n"
            f"Upbit: {mask(self.upbit.access_key)}\n"
            f"Binance: {mask(self.binance.api_key)}\n"
            f"KP Entry: {self.trading.kp_entry_threshold}%\n"
            f"KP Exit: {self.trading.kp_exit_threshold}%\n"
            f"Trade Amount: {self.trading.trade_amount_krw:,} KRW\n"
            f"Leverage: {self.trading.leverage}x\n"
            f"RL Enabled: {self.rl.enabled}\n"
        )
