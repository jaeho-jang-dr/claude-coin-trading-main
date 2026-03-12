"""Kimchirang RL Environment -- 김치프리미엄 차익거래 학습 환경

히스토리컬 Upbit/Binance 가격 데이터로 KP를 시뮬레이션하고,
에이전트가 Enter/Hold/Exit 판단을 학습한다.

State (12차원): kp_engine.py와 동일
  0: mid_kp / 10
  1: entry_kp / 10
  2: exit_kp / 10
  3: kp_ma_1m / 10  (여기서는 kp_ma_short)
  4: kp_ma_5m / 10  (여기서는 kp_ma_long)
  5: kp_z_score / 3
  6: kp_velocity / 1
  7: spread_cost / 1
  8: funding_rate * 100
  9: volume_change_upbit (placeholder)
  10: volume_change_binance (placeholder)
  11: position_state (0 or 1)

Action (continuous [-1, 1]):
  > 0.3  → Enter (Upbit 매도 + Binance 숏)
  < -0.3 → Exit (청산)
  else   → Hold
"""

import logging
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import requests

logger = logging.getLogger("kimchirang.rl_env")

# 수수료 (편도)
UPBIT_FEE = 0.0005       # 0.05%
BINANCE_FEE = 0.0004     # 0.04%
TOTAL_ROUND_TRIP = (UPBIT_FEE + BINANCE_FEE) * 2  # 진입+청산 왕복


class KPHistoricalData:
    """Upbit + Binance 히스토리컬 가격으로 KP 시계열 생성"""

    def __init__(self, days: int = 90):
        self.days = days
        self.kp_series = []       # mid_kp 시계열
        self.entry_kp_series = []
        self.exit_kp_series = []
        self.upbit_prices = []
        self.binance_prices = []
        self.fx_rates = []
        self.timestamps = []

    def collect(self) -> bool:
        """Upbit 1시간봉 + Binance 1시간봉 수집 후 KP 계산"""
        logger.info(f"히스토리컬 KP 데이터 수집 시작 ({self.days}일)")

        upbit_candles = self._fetch_upbit_candles()
        binance_candles = self._fetch_binance_candles()

        if not upbit_candles or not binance_candles:
            logger.error("캔들 데이터 수집 실패")
            return False

        # 타임스탬프 기준 매칭
        upbit_map = {c["ts"]: c for c in upbit_candles}
        binance_map = {c["ts"]: c for c in binance_candles}

        common_ts = sorted(set(upbit_map.keys()) & set(binance_map.keys()))
        logger.info(f"공통 타임스탬프: {len(common_ts)}개")

        if len(common_ts) < 100:
            logger.error(f"데이터 부족: {len(common_ts)}개")
            return False

        # FX rate (현재 환율 고정 — 히스토리컬 FX는 별도 소스 필요)
        fx = self._get_fx_rate()

        for ts in common_ts:
            u = upbit_map[ts]
            b = binance_map[ts]

            upbit_mid = (u["high"] + u["low"]) / 2
            binance_mid = (b["high"] + b["low"]) / 2

            # KP 계산
            binance_krw = binance_mid * fx
            mid_kp = (upbit_mid / binance_krw - 1) * 100

            # 스프레드 시뮬레이션 (호가 없으므로 0.05% 가정)
            spread = 0.05
            entry_kp = mid_kp + spread
            exit_kp = mid_kp - spread

            self.kp_series.append(mid_kp)
            self.entry_kp_series.append(entry_kp)
            self.exit_kp_series.append(exit_kp)
            self.upbit_prices.append(upbit_mid)
            self.binance_prices.append(binance_mid)
            self.fx_rates.append(fx)
            self.timestamps.append(ts)

        logger.info(
            f"KP 데이터 생성 완료: {len(self.kp_series)}개, "
            f"범위 {min(self.kp_series):.2f}% ~ {max(self.kp_series):.2f}%"
        )
        return True

    def _fetch_upbit_candles(self) -> list:
        """Upbit 1시간봉 수집 (최대 200개씩 페이지네이션)"""
        candles = []
        url = "https://api.upbit.com/v1/candles/minutes/60"
        to = None
        total_needed = self.days * 24

        while len(candles) < total_needed:
            params = {"market": "KRW-BTC", "count": 200}
            if to:
                params["to"] = to

            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if not data:
                    break

                for c in data:
                    # 시간 단위로 정규화 (분 제거)
                    ts = c["candle_date_time_utc"][:13]  # "2026-03-11T22"
                    candles.append({
                        "ts": ts,
                        "open": c["opening_price"],
                        "high": c["high_price"],
                        "low": c["low_price"],
                        "close": c["trade_price"],
                        "volume": c["candle_acc_trade_volume"],
                    })

                to = data[-1]["candle_date_time_utc"]
                logger.info(f"  Upbit: {len(candles)}/{total_needed}개")

                import time
                time.sleep(0.2)  # rate limit

            except Exception as e:
                logger.error(f"Upbit 캔들 수집 오류: {e}")
                break

        return candles

    def _fetch_binance_candles(self) -> list:
        """Binance 1시간봉 수집"""
        candles = []
        url = "https://api.binance.com/api/v3/klines"
        total_needed = self.days * 24
        end_time = None

        while len(candles) < total_needed:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "limit": 1000,
            }
            if end_time:
                params["endTime"] = end_time

            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if not data:
                    break

                for c in data:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
                    ts = dt.strftime("%Y-%m-%dT%H")

                    candles.append({
                        "ts": ts,
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })

                end_time = data[0][0] - 1  # 이전 페이지
                logger.info(f"  Binance: {len(candles)}/{total_needed}개")

                import time
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Binance 캔들 수집 오류: {e}")
                break

        return candles

    def _get_fx_rate(self) -> float:
        """현재 USD/KRW 환율"""
        try:
            r = requests.get(
                "https://api.upbit.com/v1/ticker?markets=KRW-USDT",
                timeout=5,
            )
            data = r.json()
            return data[0]["trade_price"]
        except Exception:
            return 1450.0  # fallback


class KimchirangEnv(gym.Env):
    """김치프리미엄 RL 환경

    에피소드: 히스토리컬 KP 데이터를 순차적으로 재생
    보상: 수익률 기반 (수수료 포함)
    """

    metadata = {"render_modes": []}

    def __init__(self, kp_data: KPHistoricalData, window: int = 20):
        super().__init__()

        self.kp_data = kp_data
        self.window = window  # 이동평균 윈도우

        # 12차원 관측 공간
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        # Discrete action: 0=Hold, 1=Enter, 2=Exit
        self.action_space = gym.spaces.Discrete(3)

        self._idx = 0
        self._position_open = False
        self._entry_kp = 0.0
        self._entry_idx = 0
        self._total_pnl = 0.0
        self._trade_count = 0
        self._last_trade_idx = 0
        self._kp_history = deque(maxlen=max(window * 5, 100))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = self.window  # 초기 윈도우만큼 건너뜀
        self._position_open = False
        self._entry_kp = 0.0
        self._entry_idx = 0
        self._total_pnl = 0.0
        self._trade_count = 0
        self._kp_history.clear()

        # 초기 히스토리 채움
        for i in range(self.window):
            self._kp_history.append(self.kp_data.kp_series[i])

        return self._get_obs(), {}

    def step(self, action):
        action_val = int(action)  # Discrete: 0=Hold, 1=Enter, 2=Exit
        reward = 0.0

        mid_kp = self.kp_data.kp_series[self._idx]
        entry_kp = self.kp_data.entry_kp_series[self._idx]
        exit_kp = self.kp_data.exit_kp_series[self._idx]

        self._kp_history.append(mid_kp)

        steps_since_last = self._idx - self._last_trade_idx

        # === 액션 처리 ===
        if action_val == 1 and not self._position_open:
            # ENTER: KP 높을 때 진입이 좋다는 것을 즉시 보상으로 알려줌
            self._position_open = True
            self._entry_kp = entry_kp
            self._entry_idx = self._idx
            self._last_trade_idx = self._idx
            # 진입 타이밍 보상: KP가 MA 위면 좋은 진입
            history = list(self._kp_history)
            kp_ma = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
            if mid_kp > kp_ma + 0.3:
                reward = 0.1   # 평균 위에서 진입 = 좋음
            elif mid_kp > kp_ma:
                reward = 0.02
            else:
                reward = -0.05  # 평균 아래에서 진입 = 나쁨

            # 과매매 패널티
            if steps_since_last < 4:
                reward -= 0.2

        elif action_val == 2 and self._position_open:
            # EXIT: 청산 PnL = 핵심 보상
            kp_profit = self._entry_kp - exit_kp
            net_profit = kp_profit - TOTAL_ROUND_TRIP * 100
            reward = net_profit  # 순수 PnL
            self._total_pnl += net_profit
            self._trade_count += 1
            self._position_open = False
            self._last_trade_idx = self._idx

        elif action_val == 1 and self._position_open:
            # 이미 포지션 있는데 또 진입 시도 → 작은 패널티
            reward = -0.02

        elif action_val == 2 and not self._position_open:
            # 포지션 없는데 청산 시도 → 작은 패널티
            reward = -0.02

        elif self._position_open:
            # HOLD (보유 중): 방향 힌트
            unrealized = self._entry_kp - mid_kp
            reward = unrealized * 0.005

            hold_hours = self._idx - self._entry_idx
            if hold_hours > 48:
                reward -= 0.003

        else:
            # HOLD (대기): 0 보상 (중립)
            reward = 0.0

        # 다음 스텝
        self._idx += 1
        terminated = self._idx >= len(self.kp_data.kp_series) - 1
        truncated = False

        # 강제 청산 (에피소드 끝)
        if terminated and self._position_open:
            kp_profit = self._entry_kp - exit_kp
            net_profit = kp_profit - TOTAL_ROUND_TRIP * 100
            reward += net_profit
            self._total_pnl += net_profit
            self._position_open = False

        obs = self._get_obs() if not terminated else np.zeros(12, dtype=np.float32)

        return obs, reward, terminated, truncated, {
            "total_pnl": self._total_pnl,
            "trade_count": self._trade_count,
        }

    def _get_obs(self) -> np.ndarray:
        """12차원 관측 벡터 (kp_engine.py와 동일 구조)"""
        kp = self.kp_data.kp_series[self._idx]
        entry_kp = self.kp_data.entry_kp_series[self._idx]
        exit_kp = self.kp_data.exit_kp_series[self._idx]

        history = list(self._kp_history)
        arr = np.array(history) if len(history) > 0 else np.array([kp])

        # 이동평균
        short_window = min(5, len(arr))
        long_window = min(20, len(arr))
        kp_ma_short = np.mean(arr[-short_window:])
        kp_ma_long = np.mean(arr[-long_window:])

        # Z-score
        std = np.std(arr[-long_window:]) if len(arr) >= 2 else 1.0
        z_score = (kp - kp_ma_long) / max(std, 0.01)

        # Velocity (최근 변화율)
        if len(arr) >= 2:
            velocity = arr[-1] - arr[-2]
        else:
            velocity = 0.0

        # 스프레드 비용
        spread_cost = 0.1  # 고정 추정

        # 펀딩비 (히스토리컬 없으므로 0)
        funding_rate = 0.0

        state = np.array([
            kp / 10,                    # 0: mid_kp
            entry_kp / 10,              # 1: entry_kp
            exit_kp / 10,               # 2: exit_kp
            kp_ma_short / 10,           # 3: MA short
            kp_ma_long / 10,            # 4: MA long
            np.clip(z_score / 3, -1, 1),  # 5: z-score
            np.clip(velocity, -1, 1),   # 6: velocity
            spread_cost,                # 7: spread cost
            funding_rate * 100,         # 8: funding rate
            0.0,                        # 9: upbit volume (placeholder)
            0.0,                        # 10: binance volume (placeholder)
            1.0 if self._position_open else 0.0,  # 11: position state
        ], dtype=np.float32)

        return state
