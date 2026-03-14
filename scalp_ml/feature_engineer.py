#!/usr/bin/env python3
"""
피처 엔지니어링

scalp_market_snapshot과 signal_attempt_log 데이터를
ML 학습용 피처 매트릭스로 변환한다.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

KST = timezone(timedelta(hours=9))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")

log = logging.getLogger("feature_eng")


class FeatureEngineer:
    """시그널 데이터를 ML 피처로 변환"""

    # 기본 피처 목록
    FEATURES = [
        # 가격 기반
        "btc_price",
        "rsi_1h",
        "sma20_diff_pct",       # (price - sma20) / sma20 * 100
        # 감성
        "fgi",
        "news_score",
        # 고래
        "whale_buy_ratio",
        "whale_buy_count",
        "whale_sell_count",
        # 체결 강도
        "trade_pressure_ratio",
        # 모멘텀
        "momentum_1m_pct",
        "momentum_5m_pct",
        "momentum_15m_pct",
        # 변동성
        "volatility_5m",
        # 시장 상태 (one-hot)
        "trend_uptrend",
        "trend_downtrend",
        "trend_sideways",
        # 시간 (cyclical encoding)
        "hour_sin",
        "hour_cos",
        # 포지션 상태
        "open_positions",
        "daily_trade_count",
    ]

    def __init__(self, db_client=None):
        self.db = db_client
        self.headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        if WORKER_TOKEN:
            self.headers["x-worker-token"] = WORKER_TOKEN

    def fetch_snapshots(self, hours: int = 24) -> list[dict]:
        """최근 N시간의 시장 스냅샷 조회"""
        cutoff = (datetime.now(KST) - timedelta(hours=hours)).isoformat()
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/scalp_market_snapshot",
                params={
                    "select": "*",
                    "recorded_at": f"gt.{cutoff}",
                    "order": "recorded_at.asc",
                },
                headers=self.headers,
                timeout=15,
            )
            return resp.json() if resp.ok else []
        except Exception as e:
            log.warning(f"스냅샷 조회 실패: {e}")
            return []

    def fetch_signals_with_outcome(self, days: int = 7) -> list[dict]:
        """사후 추적 완료된 시그널 조회 (학습 데이터)"""
        cutoff = (datetime.now(KST) - timedelta(days=days)).isoformat()
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/signal_attempt_log",
                params={
                    "select": "*",
                    "signal_type": "neq.no_signal",
                    "outcome_5m_pct": "not.is.null",
                    "recorded_at": f"gt.{cutoff}",
                    "order": "recorded_at.asc",
                },
                headers=self.headers,
                timeout=15,
            )
            return resp.json() if resp.ok else []
        except Exception as e:
            log.warning(f"시그널 조회 실패: {e}")
            return []

    def transform_signal(self, signal: dict) -> dict | None:
        """시그널 1건을 피처 벡터로 변환"""
        try:
            import math

            price = signal.get("btc_price")
            if not price:
                return None

            # SMA20 차이 (%)
            # signal_attempt_log에는 sma20이 직접 없으므로 근사 계산
            sma20_diff = 0  # 기본값

            # 시간 cyclical encoding
            recorded_at = signal.get("recorded_at", "")
            hour = 12  # 기본값
            if recorded_at:
                try:
                    dt = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
                    hour = dt.hour
                except Exception:
                    pass

            hour_rad = 2 * math.pi * hour / 24

            # market_trend one-hot
            trend = signal.get("market_trend", "sideways")

            features = {
                "btc_price": price,
                "rsi_1h": signal.get("rsi_value", 50),
                "sma20_diff_pct": sma20_diff,
                "fgi": signal.get("fgi_value", 50),
                "news_score": signal.get("news_score", 0),
                "whale_buy_ratio": signal.get("whale_buy_ratio", 0.5),
                "whale_buy_count": 0,  # signal_attempt_log에 없음
                "whale_sell_count": 0,
                "trade_pressure_ratio": signal.get("trade_pressure_ratio", 0.5),
                "momentum_1m_pct": signal.get("momentum_1m_pct", 0),
                "momentum_5m_pct": 0,  # signal에는 1m만 있음
                "momentum_15m_pct": 0,
                "volatility_5m": signal.get("volatility_5m", 0),
                "trend_uptrend": 1 if trend == "uptrend" else 0,
                "trend_downtrend": 1 if trend == "downtrend" else 0,
                "trend_sideways": 1 if trend == "sideways" else 0,
                "hour_sin": round(math.sin(hour_rad), 4),
                "hour_cos": round(math.cos(hour_rad), 4),
                "open_positions": 0,
                "daily_trade_count": 0,
            }

            # 타겟 (5분 후 수익 여부)
            outcome_5m = signal.get("outcome_5m_pct")
            if outcome_5m is not None:
                features["target_win_5m"] = 1 if outcome_5m >= 0.15 else 0
                features["target_pct_5m"] = outcome_5m

            outcome_15m = signal.get("outcome_15m_pct")
            if outcome_15m is not None:
                features["target_win_15m"] = 1 if outcome_15m >= 0.15 else 0

            return features

        except Exception as e:
            log.debug(f"피처 변환 실패: {e}")
            return None

    def build_dataset(self, days: int = 7) -> list[dict]:
        """학습용 데이터셋 생성"""
        signals = self.fetch_signals_with_outcome(days)
        log.info(f"시그널 {len(signals)}건 조회")

        dataset = []
        for sig in signals:
            features = self.transform_signal(sig)
            if features and "target_win_5m" in features:
                dataset.append(features)

        log.info(f"학습 데이터셋: {len(dataset)}건 ({len(dataset)/max(len(signals),1)*100:.0f}%)")
        return dataset
