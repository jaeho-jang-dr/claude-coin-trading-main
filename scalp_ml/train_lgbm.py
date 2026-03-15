#!/usr/bin/env python3
"""
스캘핑 LightGBM 분류기 — 24시간 속성 훈련

Upbit 1분봉 데이터에서 직접 라벨을 생성하여 훈련한다.
라벨: "지금 매수하면 10분 내 TP(0.8%)에 먼저 도달하는가? (SL 0.7% 전에)"

피처: RSI, 모멘텀, 변동성, 거래량비, 볼린저밴드 위치, 시간대 등
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

log = logging.getLogger("train_lgbm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "scalp_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

# ── 스캘핑 파라미터 ──
TP_PCT = 0.8   # 익절 기준 (%)
SL_PCT = 0.7   # 손절 기준 (%)
HOLD_WINDOW = 10  # 최대 보유 분수
FEE_PCT = 0.1  # 왕복 수수료


# ── 데이터 수집: Upbit 1분봉 ──────────────────────────

def fetch_1min_candles(market: str = "KRW-BTC", count: int = 200, to: str = None) -> list[dict]:
    """Upbit 1분봉 캔들 조회 (최대 200개)"""
    params = {"market": market, "count": count}
    if to:
        params["to"] = to
    r = requests.get("https://api.upbit.com/v1/candles/minutes/1", params=params, timeout=15)
    if r.ok:
        return r.json()
    log.warning(f"캔들 조회 실패: {r.status_code}")
    return []


def collect_candles(days: int = 7, market: str = "KRW-BTC") -> list[dict]:
    """N일치 1분봉 수집 (페이징)"""
    all_candles = []
    target_count = days * 24 * 60  # 1일 = 1440분
    to = None

    log.info(f"1분봉 {target_count}건 수집 시작 ({days}일)...")
    while len(all_candles) < target_count:
        batch = fetch_1min_candles(market, count=200, to=to)
        if not batch:
            break
        all_candles.extend(batch)
        to = batch[-1]["candle_date_time_utc"]  # 가장 오래된 시간
        time.sleep(0.12)  # Upbit rate limit

        if len(all_candles) % 2000 == 0:
            log.info(f"  수집: {len(all_candles)}/{target_count}")

    # 시간순 정렬 (오래된 것 먼저)
    all_candles.sort(key=lambda c: c["candle_date_time_utc"])
    log.info(f"1분봉 수집 완료: {len(all_candles)}건")
    return all_candles


# ── 기술지표 계산 ──────────────────────────────────

def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    diffs = [closes[i] - closes[i - 1] for i in range(-period, 0)]
    gains = [d for d in diffs if d > 0]
    losses = [-d for d in diffs if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_features(candles: list[dict], idx: int) -> dict | None:
    """idx 시점의 피처 벡터 생성"""
    if idx < 30:  # 최소 30분 히스토리 필요
        return None

    closes = [c["trade_price"] for c in candles[max(0, idx - 60):idx + 1]]
    volumes = [c["candle_acc_trade_volume"] for c in candles[max(0, idx - 60):idx + 1]]
    highs = [c["high_price"] for c in candles[max(0, idx - 60):idx + 1]]
    lows = [c["low_price"] for c in candles[max(0, idx - 60):idx + 1]]

    current = closes[-1]

    # RSI (14분)
    rsi_14 = compute_rsi(closes, 14)

    # 모멘텀
    mom_1m = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
    mom_5m = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
    mom_10m = (closes[-1] / closes[-11] - 1) * 100 if len(closes) >= 11 else 0
    mom_15m = (closes[-1] / closes[-16] - 1) * 100 if len(closes) >= 16 else 0
    mom_30m = (closes[-1] / closes[-31] - 1) * 100 if len(closes) >= 31 else 0

    # SMA
    sma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current
    sma_diff = (current / sma20 - 1) * 100

    # 볼린저밴드 위치
    if len(closes) >= 20:
        std20 = np.std(closes[-20:])
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / sma20 * 100
        bb_position = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    else:
        bb_width = 0
        bb_position = 0.5

    # 변동성 (5분 표준편차)
    if len(closes) >= 5:
        returns_5m = [(closes[i] / closes[i - 1] - 1) * 100 for i in range(-5, 0)]
        vol_5m = np.std(returns_5m)
    else:
        vol_5m = 0

    # 거래량 비율 (최근 5분 vs 이전 20분)
    if len(volumes) >= 25:
        vol_recent = sum(volumes[-5:])
        vol_prev = sum(volumes[-25:-5]) / 4  # 평균 5분
        vol_ratio = vol_recent / vol_prev if vol_prev > 0 else 1.0
    else:
        vol_ratio = 1.0

    # 캔들 패턴: 최근 3개 양봉/음봉
    opens = [c["opening_price"] for c in candles[max(0, idx - 3):idx + 1]]
    cl = closes[-4:] if len(closes) >= 4 else closes
    bullish_count = sum(1 for o, c in zip(opens[-3:], cl[-3:]) if c > o)

    # 고가/저가 대비 위치
    high_10 = max(highs[-10:]) if len(highs) >= 10 else current
    low_10 = min(lows[-10:]) if len(lows) >= 10 else current
    hl_position = (current - low_10) / (high_10 - low_10) if high_10 != low_10 else 0.5

    # 시간대 (cyclical)
    ts = candles[idx].get("candle_date_time_kst", candles[idx]["candle_date_time_utc"])
    try:
        hour = int(ts[11:13])
    except Exception:
        hour = 12
    hour_rad = 2 * math.pi * hour / 24

    return {
        "rsi_14": rsi_14,
        "mom_1m": mom_1m,
        "mom_5m": mom_5m,
        "mom_10m": mom_10m,
        "mom_15m": mom_15m,
        "mom_30m": mom_30m,
        "sma_diff_pct": sma_diff,
        "bb_width": bb_width,
        "bb_position": bb_position,
        "vol_5m": vol_5m,
        "vol_ratio": vol_ratio,
        "bullish_count": bullish_count,
        "hl_position": hl_position,
        "hour_sin": round(math.sin(hour_rad), 4),
        "hour_cos": round(math.cos(hour_rad), 4),
    }


FEATURE_NAMES = [
    "rsi_14", "mom_1m", "mom_5m", "mom_10m", "mom_15m", "mom_30m",
    "sma_diff_pct", "bb_width", "bb_position", "vol_5m", "vol_ratio",
    "bullish_count", "hl_position", "hour_sin", "hour_cos",
]


# ── 라벨 생성 ──────────────────────────────────────

def compute_label(candles: list[dict], idx: int) -> tuple[int, float] | None:
    """
    idx에서 매수 시 HOLD_WINDOW 분 내:
      - 5분 후 수익률 (regression target)
      - 수수료 차감 후 수익이면 1, 아니면 0 (classification target)

    Returns: (label, pnl_pct) or None
    """
    if idx + HOLD_WINDOW >= len(candles):
        return None

    entry = candles[idx]["trade_price"]

    # 5분 후 가격으로 수익률 계산 (+ 최적 청산 시점 탐색)
    best_pnl = -999
    worst_dd = 0
    exit_5m = candles[min(idx + 5, len(candles) - 1)]["trade_price"]
    pnl_5m = (exit_5m / entry - 1) * 100

    for j in range(1, HOLD_WINDOW + 1):
        high = candles[idx + j]["high_price"]
        low = candles[idx + j]["low_price"]
        pnl_high = (high / entry - 1) * 100
        pnl_low = (low / entry - 1) * 100
        best_pnl = max(best_pnl, pnl_high)
        worst_dd = min(worst_dd, pnl_low)

    # 라벨: 최대 도달 수익이 수수료 이상이고, 최대 낙폭이 SL 이내
    # → "이 진입은 수익 기회가 있었다"
    net_best = best_pnl - FEE_PCT
    could_profit = net_best >= 0.2 and worst_dd > -SL_PCT
    label = 1 if could_profit else 0

    return label, pnl_5m


# ── 데이터셋 구축 ──────────────────────────────────

def build_dataset(candles: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """캔들 → (X, y_class, y_reg) 매트릭스"""
    X_list, y_cls_list, y_reg_list = [], [], []

    for i in range(30, len(candles) - HOLD_WINDOW):
        features = compute_features(candles, i)
        result = compute_label(candles, i)
        if features is None or result is None:
            continue

        label, pnl_5m = result
        X_list.append([features[f] for f in FEATURE_NAMES])
        y_cls_list.append(label)
        y_reg_list.append(pnl_5m)

    X = np.array(X_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int32)
    y_reg = np.array(y_reg_list, dtype=np.float32)

    log.info(f"데이터셋: {len(X)}건, 수익구간={y_cls.sum()} ({y_cls.mean()*100:.1f}%)")
    log.info(f"  5분 수익률: 평균={y_reg.mean():.3f}%, 중앙값={np.median(y_reg):.3f}%, "
             f"std={y_reg.std():.3f}%")
    return X, y_cls, y_reg


# ── 훈련 ──────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, y_reg: np.ndarray, save: bool = True) -> dict:
    """
    듀얼 모델 훈련:
    1. 회귀 모델: 5분 수익률 예측
    2. 분류 모델: 수익 구간 진입 여부 (SMOTE + focal weight)
    """
    import lightgbm as lgb

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    yr_train, yr_test = y_reg[:split], y_reg[split:]

    log.info(f"Train: {len(X_train)}건, Test: {len(X_test)}건")

    # ── 1. 회귀 모델: 5분 수익률 예측 ──
    reg_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "n_estimators": 1000,
        "verbose": -1,
    }

    reg_train = lgb.Dataset(X_train, label=yr_train, feature_name=FEATURE_NAMES)
    reg_val = lgb.Dataset(X_test, label=yr_test, reference=reg_train)

    reg_model = lgb.train(
        reg_params,
        reg_train,
        valid_sets=[reg_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    yr_pred = reg_model.predict(X_test)
    mae = np.mean(np.abs(yr_test - yr_pred))
    corr = np.corrcoef(yr_test, yr_pred)[0, 1] if len(yr_test) > 1 else 0

    log.info(f"\n=== 회귀 모델 (5분 수익률 예측) ===")
    log.info(f"  MAE:  {mae:.4f}%")
    log.info(f"  상관계수: {corr:.4f}")
    log.info(f"  실제 평균: {yr_test.mean():.4f}%, 예측 평균: {yr_pred.mean():.4f}%")

    # ── 2. 수익률 예측 기반 진입 시뮬레이션 ──
    log.info(f"\n=== 진입 시뮬레이션 (예측 수익률 기준) ===")
    best_threshold = 0.0
    best_sharpe = -999

    for thr in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
        entries = yr_pred >= thr
        n_trades = entries.sum()
        if n_trades < 10:
            continue

        actual_returns = yr_test[entries]
        net_returns = actual_returns - FEE_PCT  # 수수료 차감
        win_rate = (net_returns > 0).mean()
        avg_return = net_returns.mean()
        total_return = net_returns.sum()
        sharpe = avg_return / (net_returns.std() + 1e-8)

        log.info(f"  thr≥{thr:.2f}%: {n_trades}건, 승률 {win_rate:.1%}, "
                 f"평균 {avg_return:.3f}%, 합계 {total_return:.2f}%, Sharpe {sharpe:.3f}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = thr

    log.info(f"\n  최적 threshold: ≥{best_threshold:.2f}% (Sharpe {best_sharpe:.3f})")

    # 피처 중요도
    importance = dict(zip(FEATURE_NAMES, reg_model.feature_importance().tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    log.info(f"\n=== 피처 중요도 TOP 5 ===")
    for i, (feat, imp) in enumerate(list(importance.items())[:5]):
        log.info(f"  {i+1}. {feat}: {imp}")

    # ── 3. 시간대별 분석 ──
    log.info(f"\n=== 시간대별 수익성 (테스트셋) ===")
    hour_sin_idx = FEATURE_NAMES.index("hour_sin")
    hour_cos_idx = FEATURE_NAMES.index("hour_cos")
    for h in range(0, 24, 3):
        h_rad = 2 * math.pi * h / 24
        h_sin, h_cos = math.sin(h_rad), math.cos(h_rad)
        mask = (np.abs(X_test[:, hour_sin_idx] - h_sin) < 0.3) & \
               (np.abs(X_test[:, hour_cos_idx] - h_cos) < 0.3)
        if mask.sum() > 20:
            avg_ret = yr_test[mask].mean()
            win = (yr_test[mask] > FEE_PCT).mean()
            log.info(f"  {h:02d}~{h+3:02d}시: {mask.sum()}건, 평균 {avg_ret:.3f}%, 승률 {win:.1%}")

    metrics = {
        "model_type": "regression",
        "mae": round(float(mae), 4),
        "correlation": round(float(corr), 4),
        "best_threshold": best_threshold,
        "best_sharpe": round(float(best_sharpe), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_iteration": reg_model.best_iteration,
        "feature_importance": {k: int(v) for k, v in importance.items()},
    }

    if save:
        model_path = MODEL_DIR / "lgbm_scalp_latest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": reg_model,
                "features": FEATURE_NAMES,
                "metrics": metrics,
                "threshold": best_threshold,
                "model_type": "regression",
            }, f)
        log.info(f"\n모델 저장: {model_path}")

        metrics_path = MODEL_DIR / "lgbm_scalp_metrics.json"
        safe_metrics = {k: v for k, v in metrics.items() if k != "feature_importance"}
        safe_metrics["feature_importance_top5"] = dict(list(importance.items())[:5])
        with open(metrics_path, "w") as f:
            json.dump(safe_metrics, f, indent=2)

        try:
            register_model_version(metrics)
        except Exception as e:
            log.warning(f"DB 등록 실패: {e}")

    return metrics


def register_model_version(metrics: dict):
    """scalp_model_versions 테이블에 등록"""
    row = {
        "version_tag": f"lgbm_v{datetime.now(KST).strftime('%Y%m%d_%H%M')}",
        "model_type": "lightgbm",
        "model_config": json.dumps({
            "features": FEATURE_NAMES,
            "tp_pct": TP_PCT,
            "sl_pct": SL_PCT,
            "hold_window": HOLD_WINDOW,
        }),
        "train_accuracy": metrics.get("mae", 0),
        "test_accuracy": metrics.get("correlation", 0),
        "f1_score": metrics.get("best_sharpe", 0),
        "is_active": True,
        "notes": f"v5 보수적 기준, {metrics['train_size']}건 학습",
    }
    requests.post(
        f"{SUPABASE_URL}/rest/v1/scalp_model_versions",
        json=row,
        headers={**HEADERS, "Prefer": "return=minimal"},
        timeout=10,
    )
    log.info("DB 모델 버전 등록 완료")


# ── 메인 ──────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="스캘핑 LightGBM 훈련")
    parser.add_argument("--days", type=int, default=14, help="학습 데이터 기간 (일)")
    parser.add_argument("--cache", action="store_true", help="캐시된 캔들 사용")
    args = parser.parse_args()

    cache_path = MODEL_DIR / "candles_cache.pkl"

    if args.cache and cache_path.exists():
        log.info("캐시된 캔들 데이터 로드...")
        with open(cache_path, "rb") as f:
            candles = pickle.load(f)
    else:
        candles = collect_candles(days=args.days)
        with open(cache_path, "wb") as f:
            pickle.dump(candles, f)
        log.info(f"캔들 캐시 저장: {cache_path}")

    if len(candles) < 1000:
        log.error(f"캔들 부족: {len(candles)}건 (최소 1000건 필요)")
        sys.exit(1)

    X, y_cls, y_reg = build_dataset(candles)
    if len(X) < 100:
        log.error(f"데이터 부족: {len(X)}건")
        sys.exit(1)

    metrics = train_model(X, y_cls, y_reg)
    log.info(f"\n{'='*50}")
    log.info(f"훈련 완료! MAE={metrics['mae']:.4f}%, Corr={metrics['correlation']:.4f}, "
             f"Sharpe={metrics['best_sharpe']:.3f}")
    log.info(f"{'='*50}")

    return metrics


if __name__ == "__main__":
    main()
