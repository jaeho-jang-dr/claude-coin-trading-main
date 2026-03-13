"""Kimchirang KP Engine -- 실시간 김치프리미엄 계산 + RL 상태 벡터 생성

호가 기반으로 Entry/Exit KP를 계산하고,
이동평균/표준편차/속도 등 통계 피처를 추적한다.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from kimchirang.config import KimchirangConfig
from kimchirang.data_feeder import MarketState

logger = logging.getLogger("kimchirang.engine")

# KP 극단값 필터링 범위 (%)
# 김치프리미엄이 -20% ~ +30% 밖이면 데이터 오류로 판단
KP_MIN = -20.0
KP_MAX = 30.0


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


def _is_finite(v: float) -> bool:
    """NaN/Inf 체크"""
    return not (math.isnan(v) or math.isinf(v))


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
        # 별도 float deque: numpy 변환 없이 슬라이싱 가능
        self._kp_values: deque[float] = deque(maxlen=3600)
        self._ts_values: deque[float] = deque(maxlen=3600)

        # 통계 캐시
        self._last_snapshot: KPSnapshot = KPSnapshot()
        self._stats_cache: dict = {}
        self._stats_updated_at: float = 0

        # RL 상태 벡터 재사용 (매 호출마다 새 배열 생성 방지)
        self._rl_state_buf: np.ndarray = np.zeros(12, dtype=np.float32)

        # 비정상 데이터 연속 카운터 (로깅 빈도 제어)
        self._invalid_data_count = 0

    def calculate(self) -> KPSnapshot:
        """현재 시점의 KP 스냅샷 계산

        비정상 데이터(음수 가격, NaN, 극단값) 필터링 포함.
        """
        s = self.state
        now = time.time()

        if not s.is_ready:
            return KPSnapshot(timestamp=now)

        upbit_bid = s.upbit_orderbook.best_bid
        upbit_ask = s.upbit_orderbook.best_ask
        binance_bid = s.binance_orderbook.best_bid
        binance_ask = s.binance_orderbook.best_ask
        fx = s.fx_rate

        # NaN/Inf 필터링
        prices = [upbit_bid, upbit_ask, binance_bid, binance_ask, fx]
        if not all(_is_finite(p) for p in prices):
            self._invalid_data_count += 1
            if self._invalid_data_count <= 3 or self._invalid_data_count % 30 == 0:
                logger.warning(
                    f"NaN/Inf 가격 데이터 감지 ({self._invalid_data_count}회) -- "
                    f"upbit={upbit_bid}/{upbit_ask}, binance={binance_bid}/{binance_ask}, fx={fx}"
                )
            return KPSnapshot(timestamp=now)

        # 0 또는 음수 가격 필터링
        if upbit_ask <= 0 or upbit_bid <= 0 or binance_ask <= 0 or binance_bid <= 0 or fx <= 0:
            self._invalid_data_count += 1
            if self._invalid_data_count <= 3 or self._invalid_data_count % 30 == 0:
                logger.warning(
                    f"비양수 가격 데이터 ({self._invalid_data_count}회) -- "
                    f"upbit_ask={upbit_ask}, binance_ask={binance_ask}, fx={fx}"
                )
            return KPSnapshot(timestamp=now)

        # 호가 역전 체크 (ask < bid)
        if upbit_ask < upbit_bid:
            self._invalid_data_count += 1
            if self._invalid_data_count % 30 == 0:
                logger.warning(f"Upbit 호가 역전: bid={upbit_bid} > ask={upbit_ask}")
            return KPSnapshot(timestamp=now)
        if binance_ask < binance_bid:
            self._invalid_data_count += 1
            if self._invalid_data_count % 30 == 0:
                logger.warning(f"Binance 호가 역전: bid={binance_bid} > ask={binance_ask}")
            return KPSnapshot(timestamp=now)

        # 김치프리미엄 계산 (호가 기반)
        # Entry: Upbit에서 사고(Ask) + Binance에서 숏(Bid에 팔아야 함)
        entry_kp = (upbit_ask / (binance_bid * fx) - 1) * 100
        # Exit: Upbit에서 팔고(Bid) + Binance에서 롱커버(Ask에 사야 함)
        exit_kp = (upbit_bid / (binance_ask * fx) - 1) * 100
        # Mid: 모니터링용 중간값
        mid_kp = ((upbit_bid + upbit_ask) / 2) / (((binance_bid + binance_ask) / 2) * fx) - 1
        mid_kp *= 100

        # KP 결과 NaN/Inf 체크
        if not all(_is_finite(v) for v in [entry_kp, exit_kp, mid_kp]):
            self._invalid_data_count += 1
            if self._invalid_data_count % 10 == 0:
                logger.warning(f"KP 계산 결과 NaN/Inf: entry={entry_kp}, exit={exit_kp}, mid={mid_kp}")
            return KPSnapshot(timestamp=now)

        # KP 극단값 필터링
        if not (KP_MIN <= mid_kp <= KP_MAX):
            self._invalid_data_count += 1
            if self._invalid_data_count <= 3 or self._invalid_data_count % 30 == 0:
                logger.warning(
                    f"KP 극단값 ({self._invalid_data_count}회): mid={mid_kp:.2f}% "
                    f"(허용 범위: {KP_MIN}~{KP_MAX}%)"
                )
            return KPSnapshot(timestamp=now)

        # 유효한 데이터 -- 카운터 리셋
        self._invalid_data_count = 0

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
        self._kp_values.append(snapshot.mid_kp)
        self._ts_values.append(now)
        self._last_snapshot = snapshot
        return snapshot

    @staticmethod
    def _slice_deque(d: deque, n: int) -> list:
        """deque의 마지막 n개 요소를 리스트로 반환 (전체 변환 회피)"""
        length = len(d)
        if n >= length:
            return list(d)
        return [d[i] for i in range(length - n, length)]

    def get_stats(self) -> dict:
        """KP 통계 (이동평균, 표준편차, 속도)

        최적화: 전체 deque->numpy 변환 대신, 필요한 윈도우만 슬라이싱.
        속도/가속도 계산은 10~20개만 필요하므로 순수 Python으로 처리.
        """
        now = time.time()
        # 1초 캐시
        if now - self._stats_updated_at < 1.0 and self._stats_cache:
            return self._stats_cache

        n = len(self._kp_values)

        if n < 2:
            return {
                "mid_kp": self._last_snapshot.mid_kp,
                "entry_kp": self._last_snapshot.entry_kp,
                "exit_kp": self._last_snapshot.exit_kp,
                "kp_ma_1m": 0, "kp_ma_5m": 0, "kp_ma_15m": 0,
                "kp_std_5m": 0, "kp_z_score": 0,
                "kp_velocity": 0, "kp_acceleration": 0,
                "spread_cost": 0, "funding_rate": 0,
                "n_samples": n,
            }

        # 가장 큰 윈도우 크기만 numpy 변환 (최대 900개, 전체 3600개 대비 절약)
        max_window = min(n, 900)
        kps_window = np.array(self._slice_deque(self._kp_values, max_window))

        # NaN 체크: 혹시라도 NaN이 들어왔으면 제거
        if np.any(np.isnan(kps_window)):
            logger.warning("KP 히스토리에 NaN 값 발견 -- 제거 후 계산")
            kps_window = kps_window[~np.isnan(kps_window)]
            if len(kps_window) < 2:
                return self._stats_cache if self._stats_cache else {
                    "mid_kp": self._last_snapshot.mid_kp,
                    "entry_kp": self._last_snapshot.entry_kp,
                    "exit_kp": self._last_snapshot.exit_kp,
                    "kp_ma_1m": 0, "kp_ma_5m": 0, "kp_ma_15m": 0,
                    "kp_std_5m": 0, "kp_z_score": 0,
                    "kp_velocity": 0, "kp_acceleration": 0,
                    "spread_cost": 0, "funding_rate": 0,
                    "n_samples": n,
                }

        kw_len = len(kps_window)

        # 이동평균 (각 윈도우의 끝부분만 사용)
        if kw_len >= 60:
            ma_1m = float(np.mean(kps_window[-60:]))
        else:
            ma_1m = float(np.mean(kps_window))
        if kw_len >= 300:
            ma_5m = float(np.mean(kps_window[-300:]))
        else:
            ma_5m = float(np.mean(kps_window))
        ma_15m = float(np.mean(kps_window))  # kps_window는 이미 최대 900개

        # 표준편차 (5분 윈도우)
        window_5m = kps_window[-300:] if kw_len >= 300 else kps_window
        std_5m = float(np.std(window_5m))

        # Z-Score
        current_kp = self._kp_values[-1]
        z_score = (current_kp - ma_5m) / std_5m if std_5m > 0.001 else 0.0
        # Z-Score 극단값 클리핑
        z_score = max(-10.0, min(10.0, z_score))

        # KP 속도/가속도 (10~20개만 -- 순수 Python, numpy 불필요)
        velocity = 0.0
        acceleration = 0.0
        if n >= 10:
            kp_last = self._kp_values[-1]
            kp_10ago = self._kp_values[-10]
            ts_last = self._ts_values[-1]
            ts_10ago = self._ts_values[-10]
            dt = ts_last - ts_10ago
            if dt > 0:
                velocity = (kp_last - kp_10ago) / dt * 60  # %/분
            if n >= 20:
                kp_11ago = self._kp_values[-11]
                kp_20ago = self._kp_values[-20]
                ts_11ago = self._ts_values[-11]
                ts_20ago = self._ts_values[-20]
                prev_dt = ts_11ago - ts_20ago
                if prev_dt > 0:
                    prev_vel = (kp_11ago - kp_20ago) / prev_dt * 60
                    acceleration = velocity - prev_vel

        # 총 스프레드 비용 (양쪽 슬리피지 합산)
        spread_cost = (
            self._last_snapshot.upbit_spread_pct
            + self._last_snapshot.binance_spread_pct
        )

        # 최종 결과 NaN 방어
        def _safe_round(v, digits):
            return round(v, digits) if _is_finite(v) else 0.0

        self._stats_cache = {
            "mid_kp": _safe_round(self._last_snapshot.mid_kp, 4),
            "entry_kp": _safe_round(self._last_snapshot.entry_kp, 4),
            "exit_kp": _safe_round(self._last_snapshot.exit_kp, 4),
            "kp_ma_1m": _safe_round(ma_1m, 4),
            "kp_ma_5m": _safe_round(ma_5m, 4),
            "kp_ma_15m": _safe_round(ma_15m, 4),
            "kp_std_5m": _safe_round(std_5m, 4),
            "kp_z_score": _safe_round(z_score, 4),
            "kp_velocity": _safe_round(velocity, 4),
            "kp_acceleration": _safe_round(acceleration, 4),
            "spread_cost": _safe_round(spread_cost, 4),
            "funding_rate": self._last_snapshot.funding_rate if _is_finite(self._last_snapshot.funding_rate) else 0.0,
            "n_samples": n,
        }
        self._stats_updated_at = now
        return self._stats_cache

    def build_rl_state(self) -> np.ndarray:
        """RL 에이전트용 상태 벡터 (12차원)

        각 피처는 대략 [-1, 1] 범위로 정규화.
        사전 할당된 버퍼를 재사용하여 매 호출마다 배열 생성을 방지한다.

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
        buf = self._rl_state_buf

        # min/max로 클리핑 (np.clip 개별 호출보다 빠름)
        def _clip(v, lo, hi):
            return lo if v < lo else (hi if v > hi else v)

        buf[0] = _clip(stats["mid_kp"] / 10, -1, 1)
        buf[1] = _clip(stats["entry_kp"] / 10, -1, 1)
        buf[2] = _clip(stats["exit_kp"] / 10, -1, 1)
        buf[3] = _clip(stats["kp_ma_1m"] / 10, -1, 1)
        buf[4] = _clip(stats["kp_ma_5m"] / 10, -1, 1)
        buf[5] = _clip(stats["kp_z_score"] / 3, -1, 1)
        buf[6] = _clip(stats["kp_velocity"], -1, 1)
        buf[7] = _clip(stats["spread_cost"], 0, 1)
        buf[8] = _clip(stats["funding_rate"] * 100, -1, 1)
        buf[9] = 0.0   # upbit volume (추후 확장)
        buf[10] = 0.0  # binance volume (추후 확장)
        buf[11] = 0.0  # position state (외부 주입)

        # NaN 방어: RL 모델에 NaN 입력 방지
        if np.any(np.isnan(buf)):
            logger.warning("RL 상태 벡터에 NaN 감지 -- 0으로 대체")
            np.nan_to_num(buf, copy=False, nan=0.0)

        return buf

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
