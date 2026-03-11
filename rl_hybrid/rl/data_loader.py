"""과거 데이터 로더 — Upbit API + Supabase에서 훈련 데이터 구성

히스토리컬 캔들 데이터를 로드하고, 기술 지표를 계산하여
Gymnasium 환경에 공급한다.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("rl.data_loader")

# Upbit API 설정
UPBIT_API = "https://api.upbit.com/v1"
CANDLE_LIMIT = 200  # API 1회 최대


class HistoricalDataLoader:
    """과거 시장 데이터 로더"""

    def __init__(self, market: str = "KRW-BTC"):
        self.market = market
        self._cache: dict[str, np.ndarray] = {}

    def load_candles(
        self,
        days: int = 365,
        interval: str = "1h",
    ) -> list[dict]:
        """Upbit에서 과거 캔들 데이터 로드

        Args:
            days: 수집 기간 (일)
            interval: "1h", "4h", "1d"

        Returns:
            [{"timestamp", "open", "high", "low", "close", "volume"}, ...]
            시간순 정렬 (과거 → 최근)
        """
        cache_key = f"{self.market}_{interval}_{days}"
        if cache_key in self._cache:
            logger.info(f"캐시 사용: {cache_key}")
            return self._cache[cache_key]

        interval_map = {
            "1h": ("minutes/60", days * 24),
            "4h": ("minutes/240", days * 6),
            "1d": ("days", days),
        }

        if interval not in interval_map:
            raise ValueError(f"지원하지 않는 interval: {interval}")

        endpoint, total_candles = interval_map[interval]
        url = f"{UPBIT_API}/candles/{endpoint}"

        all_candles = []
        to = None  # 마지막 캔들 이전부터 수집

        while len(all_candles) < total_candles:
            count = min(CANDLE_LIMIT, total_candles - len(all_candles))
            params = {"market": self.market, "count": count}
            if to:
                params["to"] = to

            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                candles = resp.json()

                if not candles:
                    break

                for c in candles:
                    all_candles.append({
                        "timestamp": c["candle_date_time_kst"],
                        "open": float(c["opening_price"]),
                        "high": float(c["high_price"]),
                        "low": float(c["low_price"]),
                        "close": float(c["trade_price"]),
                        "volume": float(c["candle_acc_trade_volume"]),
                        "volume_krw": float(c.get("candle_acc_trade_price", 0)),
                    })

                # 다음 페이지
                to = candles[-1]["candle_date_time_utc"]
                time.sleep(0.15)  # rate limit

            except Exception as e:
                logger.error(f"캔들 로드 에러: {e}")
                break

        # 시간순 정렬 (오래된 것 먼저)
        all_candles.reverse()
        logger.info(f"캔들 로드 완료: {len(all_candles)}개 ({interval}, {days}일)")

        self._cache[cache_key] = all_candles
        return all_candles

    def compute_indicators(self, candles: list[dict]) -> list[dict]:
        """캔들 데이터에 기술 지표 계산하여 추가

        추가되는 지표: rsi_14, sma_20, sma_50, macd, macd_signal,
        macd_histogram, boll_upper, boll_middle, boll_lower,
        stoch_k, stoch_d, atr, adx
        """
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c["volume"] for c in candles])

        n = len(candles)

        # RSI-14
        rsi = self._compute_rsi(closes, 14)

        # SMA
        sma20 = self._sma(closes, 20)
        sma50 = self._sma(closes, 50)

        # EMA (MACD 구성)
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        macd_signal = self._ema(macd_line, 9)
        macd_hist = macd_line - macd_signal

        # Bollinger Bands
        boll_mid = sma20.copy()
        boll_std = self._rolling_std(closes, 20)
        boll_upper = boll_mid + 2 * boll_std
        boll_lower = boll_mid - 2 * boll_std

        # Stochastic
        stoch_k, stoch_d = self._stochastic(highs, lows, closes, 14, 3)

        # ATR
        atr = self._compute_atr(highs, lows, closes, 14)

        # ADX
        adx, plus_di, minus_di = self._compute_adx(highs, lows, closes, 14)

        # 변동률
        change_rates = np.zeros(n)
        change_rates[1:] = (closes[1:] - closes[:-1]) / closes[:-1]

        # Volume SMA (한 번만 계산)
        volume_sma20 = self._sma(volumes, 20)

        # 결과에 추가
        enriched = []
        for i in range(n):
            c = candles[i].copy()
            c.update({
                "rsi_14": float(rsi[i]),
                "sma_20": float(sma20[i]),
                "sma_50": float(sma50[i]),
                "ema_12": float(ema12[i]),
                "ema_26": float(ema26[i]),
                "macd": float(macd_line[i]),
                "macd_signal": float(macd_signal[i]),
                "macd_histogram": float(macd_hist[i]),
                "boll_upper": float(boll_upper[i]),
                "boll_middle": float(boll_mid[i]),
                "boll_lower": float(boll_lower[i]),
                "stoch_k": float(stoch_k[i]),
                "stoch_d": float(stoch_d[i]),
                "atr": float(atr[i]),
                "adx": float(adx[i]),
                "adx_plus_di": float(plus_di[i]),
                "adx_minus_di": float(minus_di[i]),
                "change_rate": float(change_rates[i]),
                "volume_sma20": float(volume_sma20[i]),
            })
            enriched.append(c)

        return enriched

    # === 지표 계산 헬퍼 ===

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        cumsum = np.cumsum(data)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        result[period - 1:] = cumsum[period - 1:] / period
        # 초기값 채우기
        for i in range(period - 1):
            result[i] = data[:i + 1].mean()
        return result

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.zeros_like(data)
        alpha = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        result = np.full_like(data, np.nan)
        if n == 0:
            return result

        # Use stride_tricks to create rolling windows, then vectorized std
        # This avoids catastrophic cancellation from cumsum-of-squares
        # on large values (e.g., BTC prices ~50,000,000)
        if n >= period:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(data, period)
            result[period - 1:] = windows.std(axis=1)

        # Fill initial values (matching original behavior)
        result[0] = 0
        for i in range(1, min(period - 1, n)):
            result[i] = data[:i + 1].std()
        return result

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> np.ndarray:
        if len(closes) <= period:
            return np.full_like(closes, 50.0)
        deltas = np.diff(closes, prepend=closes[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros_like(closes)
        avg_loss = np.zeros_like(closes)

        # 초기 SMA
        avg_gain[period] = gains[1:period + 1].mean()
        avg_loss[period] = losses[1:period + 1].mean()

        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - 100 / (1 + rs)
        rsi[:period] = 50  # 데이터 부족 구간
        return rsi

    @staticmethod
    def _stochastic(highs, lows, closes, k_period, d_period):
        n = len(closes)
        stoch_k = np.full(n, 50.0)
        for i in range(k_period - 1, n):
            h = highs[i - k_period + 1:i + 1].max()
            l = lows[i - k_period + 1:i + 1].min()
            if h != l:
                stoch_k[i] = (closes[i] - l) / (h - l) * 100
        # %D = SMA(%K, d_period)
        stoch_d = HistoricalDataLoader._sma(stoch_k, d_period)
        return stoch_k, stoch_d

    @staticmethod
    def _compute_atr(highs, lows, closes, period):
        n = len(closes)
        tr = np.empty(n)
        tr[0] = highs[0] - lows[0]
        if n > 1:
            prev_close = closes[:-1]
            tr[1:] = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - prev_close),
                    np.abs(lows[1:] - prev_close),
                ),
            )
        return HistoricalDataLoader._ema(tr, period)

    @staticmethod
    def _compute_adx(highs, lows, closes, period):
        n = len(closes)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        if n > 1:
            up = highs[1:] - highs[:-1]
            down = lows[:-1] - lows[1:]
            plus_dm[1:] = np.where((up > down) & (up > 0), up, 0)
            minus_dm[1:] = np.where((down > up) & (down > 0), down, 0)

        atr = HistoricalDataLoader._compute_atr(highs, lows, closes, period)
        atr = np.where(atr > 0, atr, 1)

        plus_di = 100 * HistoricalDataLoader._ema(plus_dm, period) / atr
        minus_di = 100 * HistoricalDataLoader._ema(minus_dm, period) / atr

        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.where(
                (plus_di + minus_di) > 0,
                100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
                0,
            )
        adx = HistoricalDataLoader._ema(dx, period)

        return adx, plus_di, minus_di

    # === 외부 시그널 로드 ===

    def load_external_signals(self, days: int = 365) -> list[dict]:
        """Supabase external_signal_log에서 외부 시그널 로드

        Args:
            days: 수집 기간 (일)

        Returns:
            [{"recorded_at", "fgi_value", "news_sentiment", "whale_score",
              "funding_rate", "long_short_ratio", "kimchi_premium_pct",
              "macro_score", "eth_btc_score", "fusion_score"}, ...]
            시간순 정렬 (과거 → 최근)
        """
        db_url = os.environ.get("SUPABASE_DB_URL")
        if not db_url:
            logger.warning("SUPABASE_DB_URL 미설정 — 외부 시그널 로드 불가")
            return []

        try:
            import psycopg2
        except ImportError:
            logger.warning("psycopg2 미설치 — 외부 시그널 로드 불가")
            return []

        since = datetime.utcnow() - timedelta(days=days)

        query = """
            SELECT recorded_at, fgi_value, news_sentiment,
                   whale_score, funding_rate, long_short_ratio,
                   kimchi_premium_pct, macro_score, eth_btc_score,
                   fusion_score, fusion_signal
            FROM external_signal_log
            WHERE recorded_at >= %s
            ORDER BY recorded_at ASC
        """

        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(query, (since,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            conn.close()

            signals = [dict(zip(columns, row)) for row in rows]
            logger.info(f"외부 시그널 로드 완료: {len(signals)}건 ({days}일)")
            return signals

        except Exception as e:
            logger.warning(f"외부 시그널 로드 실패: {e}")
            return []

    def align_external_to_candles(
        self, candles: list[dict], signals: list[dict]
    ) -> list[dict]:
        """캔들 타임스탬프에 외부 시그널을 forward-fill 정렬

        각 캔들 시점에 해당하는 가장 최근 외부 시그널을 매핑한다.
        캔들은 1h, 외부 시그널은 ~4h 간격이므로 한 시그널이 여러 캔들에 적용된다.

        Args:
            candles: 시간순 정렬된 캔들 데이터 (timestamp 키 포함)
            signals: 시간순 정렬된 외부 시그널 (recorded_at 키 포함)

        Returns:
            캔들과 동일 길이의 외부 시그널 리스트. 매핑 불가 시 기본값 dict.
        """
        if not signals:
            return [self._default_external_signal() for _ in candles]

        # 시그널 타임스탬프를 파싱
        parsed_signals = []
        for sig in signals:
            ts = sig["recorded_at"]
            if isinstance(ts, str):
                # ISO format 파싱
                ts = ts.replace("T", " ").replace("Z", "")
                if "+" in ts:
                    ts = ts.split("+")[0]
                ts = datetime.fromisoformat(ts)
            parsed_signals.append((ts, sig))

        result = []
        sig_idx = 0

        for candle in candles:
            # 캔들 타임스탬프 파싱 (Upbit KST → UTC 변환)
            candle_ts_str = candle.get("timestamp", "")
            if isinstance(candle_ts_str, str) and candle_ts_str:
                candle_ts_str = candle_ts_str.replace("T", " ")
                candle_ts = datetime.fromisoformat(candle_ts_str)
                # Upbit candle_date_time_kst → UTC (KST = UTC+9)
                candle_ts_utc = candle_ts - timedelta(hours=9)
            else:
                # 파싱 불가 시 기본값
                result.append(self._default_external_signal())
                continue

            # Forward-fill: 캔들 시점 이전의 가장 최근 시그널 찾기
            while (
                sig_idx < len(parsed_signals) - 1
                and parsed_signals[sig_idx + 1][0] <= candle_ts_utc
            ):
                sig_idx += 1

            if parsed_signals[sig_idx][0] <= candle_ts_utc:
                result.append(parsed_signals[sig_idx][1])
            else:
                # 아직 시그널이 없는 초기 구간
                result.append(self._default_external_signal())

        logger.info(
            f"외부 시그널 정렬 완료: {len(result)}건 "
            f"(원본 시그널 {len(signals)}건)"
        )
        return result

    @staticmethod
    def _default_external_signal() -> dict:
        """외부 시그널 기본값 (데이터 없을 때 사용)"""
        return {
            "fgi_value": 50,
            "news_sentiment": 0,
            "whale_score": 0,
            "funding_rate": 0.0,
            "long_short_ratio": 1.0,
            "kimchi_premium_pct": 0.0,
            "macro_score": 0,
            "eth_btc_score": 0,
            "fusion_score": 0,
            "fusion_signal": "neutral",
            "nvt_signal": 100.0,
        }
