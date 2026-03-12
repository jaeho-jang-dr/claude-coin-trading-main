"""Kimchirang KP Engine -- 실시간 김치프리미엄 계산 + RL 상태 벡터 생성

호가 기반으로 Entry/Exit KP를 계산하고,
이동평균/표준편차/속도 등 통계 피처를 추적한다.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from kimchirang.config import KimchirangConfig
from kimchirang.data_feeder import MarketState

logger = logging.getLogger("kimchirang.engine")


@dataclass
class KPSnapshot:
    """단일 시점의 김치프리미엄 데이터"""
    # 호가 기반 KP (슬리피지 반영)
    entry_kp: float = 0.0     # Upbit Ask / (Binance Bid * FX) - 1
    exit_kp: float = 0.0      # Upbit Bid / (Binance Ask * FX) - 1
    mid_kp: float = 0.0       # 중간값 (모니터링용)

    # 원시 가격
    upbit_bid: float = 0.0
    upbit_ask: float = 0.0
    binance_bid: float = 0.0
    binance_ask: float = 0.0
    fx_rate: float = 0.0

    # 스프레드
    upbit_spread_pct: float = 0.0
    binance_spread_pct: float = 0.0

    # 펀딩비
    funding_rate: float = 0.0

    # 타임스탬프
    timestamp: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.upbit_bid > 0 and self.binance_bid > 0 and self.fx_rate > 0


class KPEngine:
    """김치프리미엄 계산 엔진 + 통계 추적

    실시간 호가 데이터로 Entry/Exit KP를 계산하고,
    이동평균, 표준편차, 속도 등 통계 피처를 유지한다.
    """

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state

        # KP 히스토리 (최근 1시간, 1초 간격 가정 = 3600개)
        self._history: deque[KPSnapshot] = deque(maxlen=3600)

        # 통계 캐시
        self._last_snapshot: KPSnapshot = KPSnapshot()
        self._stats_cache: dict = {}
        self._stats_updated_at: float = 0

    def calculate(self) -> KPSnapshot:
        """현재 시점의 KP 스냅샷 계산"""
        s = self.state
        now = time.time()

        if not s.is_ready:
            return KPSnapshot(timestamp=now)

        upbit_bid = s.upbit_orderbook.best_bid
        upbit_ask = s.upbit_orderbook.best_ask
        binance_bid = s.binance_orderbook.best_bid
        binance_ask = s.binance_orderbook.best_ask
        fx = s.fx_rate

        # 김치프리미엄 계산 (호가 기반)
        # Entry: Upbit에서 사고(Ask) + Binance에서 숏(Bid에 팔아야 함)
        entry_kp = (upbit_ask / (binance_bid * fx) - 1) * 100
        # Exit: Upbit에서 팔고(Bid) + Binance에서 롱커버(Ask에 사야 함)
        exit_kp = (upbit_bid / (binance_ask * fx) - 1) * 100
        # Mid: 모니터링용 중간값
        mid_kp = ((upbit_bid + upbit_ask) / 2) / (((binance_bid + binance_ask) / 2) * fx) - 1
        mid_kp *= 100

        snapshot = KPSnapshot(
            entry_kp=round(entry_kp, 4),
            exit_kp=round(exit_kp, 4),
            mid_kp=round(mid_kp, 4),
            upbit_bid=upbit_bid,
            upbit_ask=upbit_ask,
            binance_bid=binance_bid,
            binance_ask=binance_ask,
            fx_rate=fx,
            upbit_spread_pct=s.upbit_orderbook.spread_pct,
            binance_spread_pct=s.binance_orderbook.spread_pct,
            funding_rate=s.binance_funding.funding_rate,
            timestamp=now,
        )

        self._history.append(snapshot)
        self._last_snapshot = snapshot
        return snapshot

    def get_stats(self) -> dict:
        """KP 통계 (이동평균, 표준편차, 속도)"""
        now = time.time()
        # 1초 캐시
        if now - self._stats_updated_at < 1.0 and self._stats_cache:
            return self._stats_cache

        if len(self._history) < 2:
            return {
                "mid_kp": self._last_snapshot.mid_kp,
                "entry_kp": self._last_snapshot.entry_kp,
                "exit_kp": self._last_snapshot.exit_kp,
                "kp_ma_1m": 0, "kp_ma_5m": 0, "kp_ma_15m": 0,
                "kp_std_5m": 0, "kp_z_score": 0,
                "kp_velocity": 0, "kp_acceleration": 0,
                "spread_cost": 0, "funding_rate": 0,
                "n_samples": len(self._history),
            }

        # numpy 배열로 변환
        kps = np.array([s.mid_kp for s in self._history])
        timestamps = np.array([s.timestamp for s in self._history])

        # 이동평균 (최근 N개 샘플 기준)
        ma_1m = float(np.mean(kps[-60:])) if len(kps) >= 60 else float(np.mean(kps))
        ma_5m = float(np.mean(kps[-300:])) if len(kps) >= 300 else float(np.mean(kps))
        ma_15m = float(np.mean(kps[-900:])) if len(kps) >= 900 else float(np.mean(kps))

        # 표준편차 (5분 윈도우)
        window_5m = kps[-300:] if len(kps) >= 300 else kps
        std_5m = float(np.std(window_5m))

        # Z-Score: 현재 KP가 5분 평균 대비 몇 표준편차인지
        z_score = (kps[-1] - ma_5m) / std_5m if std_5m > 0.001 else 0

        # KP 속도 (최근 10개 샘플의 기울기)
        velocity = 0.0
        acceleration = 0.0
        if len(kps) >= 10:
            recent = kps[-10:]
            recent_t = timestamps[-10:]
            dt = recent_t[-1] - recent_t[0]
            if dt > 0:
                velocity = (recent[-1] - recent[0]) / dt * 60  # %/분
            if len(kps) >= 20:
                prev = kps[-20:-10]
                prev_t = timestamps[-20:-10]
                prev_dt = prev_t[-1] - prev_t[0]
                if prev_dt > 0:
                    prev_vel = (prev[-1] - prev[0]) / prev_dt * 60
                    acceleration = velocity - prev_vel

        # 총 스프레드 비용 (양쪽 슬리피지 합산)
        spread_cost = (
            self._last_snapshot.upbit_spread_pct
            + self._last_snapshot.binance_spread_pct
        )

        self._stats_cache = {
            "mid_kp": round(self._last_snapshot.mid_kp, 4),
            "entry_kp": round(self._last_snapshot.entry_kp, 4),
            "exit_kp": round(self._last_snapshot.exit_kp, 4),
            "kp_ma_1m": round(ma_1m, 4),
            "kp_ma_5m": round(ma_5m, 4),
            "kp_ma_15m": round(ma_15m, 4),
            "kp_std_5m": round(std_5m, 4),
            "kp_z_score": round(z_score, 4),
            "kp_velocity": round(velocity, 4),
            "kp_acceleration": round(acceleration, 4),
            "spread_cost": round(spread_cost, 4),
            "funding_rate": self._last_snapshot.funding_rate,
            "n_samples": len(self._history),
        }
        self._stats_updated_at = now
        return self._stats_cache

    def build_rl_state(self) -> np.ndarray:
        """RL 에이전트용 상태 벡터 (12차원)

        각 피처는 대략 [-1, 1] 범위로 정규화.

        차원:
          0: mid_kp / 10 (보통 -5~10% -> -0.5~1.0)
          1: entry_kp / 10
          2: exit_kp / 10
          3: kp_ma_1m / 10
          4: kp_ma_5m / 10
          5: kp_z_score / 3 (클리핑)
          6: kp_velocity / 1 (클리핑)
          7: spread_cost / 1 (0~1)
          8: funding_rate * 100 (보통 -0.1~0.1 -> -10~10 -> 클리핑)
          9: upbit_volume 변화 (placeholder, 0)
         10: binance_volume 변화 (placeholder, 0)
         11: 포지션 상태 (외부에서 주입, 기본 0)
        """
        stats = self.get_stats()

        state = np.array([
            np.clip(stats["mid_kp"] / 10, -1, 1),
            np.clip(stats["entry_kp"] / 10, -1, 1),
            np.clip(stats["exit_kp"] / 10, -1, 1),
            np.clip(stats["kp_ma_1m"] / 10, -1, 1),
            np.clip(stats["kp_ma_5m"] / 10, -1, 1),
            np.clip(stats["kp_z_score"] / 3, -1, 1),
            np.clip(stats["kp_velocity"] / 1, -1, 1),
            np.clip(stats["spread_cost"] / 1, 0, 1),
            np.clip(stats["funding_rate"] * 100, -1, 1),
            0.0,  # upbit volume (추후 확장)
            0.0,  # binance volume (추후 확장)
            0.0,  # position state (외부 주입)
        ], dtype=np.float32)

        return state

    def should_enter(self, snapshot: KPSnapshot = None) -> bool:
        """규칙 기반 진입 조건 (RL 없을 때 fallback)"""
        snap = snapshot or self._last_snapshot
        return (
            snap.is_valid
            and snap.entry_kp >= self.config.trading.kp_entry_threshold
        )

    def should_exit(self, snapshot: KPSnapshot = None) -> bool:
        """규칙 기반 청산 조건 (RL 없을 때 fallback)"""
        snap = snapshot or self._last_snapshot
        return (
            snap.is_valid
            and snap.exit_kp <= self.config.trading.kp_exit_threshold
        )

    def should_stop_loss(self, snapshot: KPSnapshot = None) -> bool:
        """손절 조건: KP가 역방향으로 크게 확대"""
        snap = snapshot or self._last_snapshot
        return (
            snap.is_valid
            and snap.mid_kp >= self.config.trading.kp_stop_loss
        )
