#!/usr/bin/env python3
"""
Upbit 시장 데이터 수집 스크립트

수집 항목:
  - 현재가 (ticker)
  - 일봉 캔들 30일 / 4시간봉 캔들 42개
  - 호가창 (orderbook)
  - 최근 체결 100건
  - 기술적 지표: SMA(20), EMA(10), RSI(14), MACD, 볼린저밴드, 스토캐스틱

출력: JSON (stdout)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests

UPBIT_API = "https://api.upbit.com/v1"


# ── API 호출 (Exponential Backoff 포함) ─────────────────
def api_get(path: str, params: dict | None = None, max_retries: int = 3) -> dict | list:
    url = f"{UPBIT_API}{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(max_retries):
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[rate_limit] 429 received, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


# ── 기술적 지표 계산 ────────────────────────────────────
def sma(prices: list[float], period: int) -> float:
    window = prices[-period:]
    return sum(window) / len(window)


def ema(prices: list[float], period: int) -> float:
    k = 2 / (period + 1)
    value = prices[0]
    for p in prices[1:]:
        value = p * k + value * (1 - k)
    return value


def rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = prices[i] - prices[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    ag, al = gains / period, losses / period
    for i in range(period + 1, len(prices)):
        d = prices[i] - prices[i - 1]
        if d >= 0:
            ag = (ag * (period - 1) + d) / period
            al = (al * (period - 1)) / period
        else:
            ag = (ag * (period - 1)) / period
            al = (al * (period - 1) - d) / period
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def macd(prices: list[float]) -> dict:
    """MACD = EMA12 - EMA26, Signal = MACD의 9일 EMA"""
    if len(prices) < 26:
        return {"macd": 0, "signal": 0, "histogram": 0}
    # 각 시점의 MACD 값을 계산하여 시그널 EMA를 구한다
    k12 = 2 / 13
    k26 = 2 / 27
    ema12_val = prices[0]
    ema26_val = prices[0]
    macd_series = []
    for p in prices[1:]:
        ema12_val = p * k12 + ema12_val * (1 - k12)
        ema26_val = p * k26 + ema26_val * (1 - k26)
        macd_series.append(ema12_val - ema26_val)
    m = macd_series[-1]
    # Signal = MACD 시리즈의 9일 EMA
    k9 = 2 / 10
    s = macd_series[0]
    for v in macd_series[1:]:
        s = v * k9 + s * (1 - k9)
    return {"macd": round(m, 2), "signal": round(s, 2), "histogram": round(m - s, 2)}


def bollinger(prices: list[float], period: int = 20) -> dict:
    mid = sma(prices, period)
    window = prices[-period:]
    var = sum((p - mid) ** 2 for p in window) / period
    sd = var**0.5
    return {
        "upper": round(mid + 2 * sd, 2),
        "middle": round(mid, 2),
        "lower": round(mid - 2 * sd, 2),
    }


def stochastic(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> dict:
    """스토캐스틱 %K (Fast) 및 %D (3-period SMA of %K)."""
    if len(closes) < period + 2:
        return {"k": 50.0, "d": 50.0}
    # 최근 3개 봉의 %K를 계산하여 %D = 3-period SMA(%K)
    k_values = []
    for offset in range(3):
        idx = len(closes) - 1 - offset
        if idx < period - 1:
            break
        h = max(highs[idx - period + 1 : idx + 1])
        l = min(lows[idx - period + 1 : idx + 1])
        c = closes[idx]
        k_val = 50.0 if h == l else ((c - l) / (h - l)) * 100
        k_values.append(k_val)
    k = k_values[0]
    d = sum(k_values) / len(k_values)
    return {"k": round(k, 2), "d": round(d, 2)}


def calc_adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> dict:
    """ADX (Average Directional Index) — 추세 강도 측정.

    Returns: {"adx": float, "plus_di": float, "minus_di": float, "regime": str}
    regime: "trending" (ADX >= 25), "ranging" (ADX < 20), "transitioning" (20-25)
    """
    n = len(closes)
    if n < period + 1:
        return {"adx": 0, "plus_di": 0, "minus_di": 0, "regime": "unknown"}

    # True Range, +DM, -DM
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, n):
        hi, lo, prev_c = highs[i], lows[i], closes[i - 1]
        tr_list.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm_list.append(up if up > down and up > 0 else 0)
        minus_dm_list.append(down if down > up and down > 0 else 0)

    # Wilder smoothing (initial SMA then EMA-like)
    atr = sum(tr_list[:period]) / period
    plus_dm_s = sum(plus_dm_list[:period]) / period
    minus_dm_s = sum(minus_dm_list[:period]) / period

    dx_list = []
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_dm_s = (plus_dm_s * (period - 1) + plus_dm_list[i]) / period
        minus_dm_s = (minus_dm_s * (period - 1) + minus_dm_list[i]) / period

        plus_di = (plus_dm_s / atr * 100) if atr > 0 else 0
        minus_di = (minus_dm_s / atr * 100) if atr > 0 else 0
        di_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0
        dx_list.append(dx)

    if not dx_list:
        return {"adx": 0, "plus_di": 0, "minus_di": 0, "regime": "unknown"}

    # ADX = SMA of DX (first period), then Wilder smooth
    adx = sum(dx_list[:period]) / min(period, len(dx_list))
    for i in range(period, len(dx_list)):
        adx = (adx * (period - 1) + dx_list[i]) / period

    # Final +DI, -DI
    plus_di = (plus_dm_s / atr * 100) if atr > 0 else 0
    minus_di = (minus_dm_s / atr * 100) if atr > 0 else 0

    regime = "trending" if adx >= 25 else "ranging" if adx < 20 else "transitioning"

    return {
        "adx": round(adx, 2),
        "plus_di": round(plus_di, 2),
        "minus_di": round(minus_di, 2),
        "regime": regime,
    }


def calc_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """ATR (Average True Range) — 변동성 측정."""
    if len(closes) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(closes)):
        hi, lo, prev_c = highs[i], lows[i], closes[i - 1]
        tr_list.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
    atr = sum(tr_list[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
    return round(atr, 2)


# ── 메인 ────────────────────────────────────────────────
def collect_eth_btc_ratio() -> dict:
    """ETH/BTC 비율 및 시장 구조 데이터를 수집한다."""
    try:
        eth_ticker = api_get("/ticker", {"markets": "KRW-ETH"})[0]
        btc_ticker = api_get("/ticker", {"markets": "KRW-BTC"})[0]
        time.sleep(0.15)

        btc_daily = api_get("/candles/days", {"market": "KRW-BTC", "count": "60"})
        time.sleep(0.15)
        eth_daily = api_get("/candles/days", {"market": "KRW-ETH", "count": "60"})

        btc_prices = [c["trade_price"] for c in reversed(btc_daily)]
        eth_prices = [c["trade_price"] for c in reversed(eth_daily)]
        n = min(len(btc_prices), len(eth_prices))

        ratios = [eth_prices[i] / btc_prices[i] for i in range(n)]
        mean_r = statistics.mean(ratios)
        std_r = statistics.stdev(ratios) if len(ratios) > 1 else 0.001
        if std_r == 0:
            std_r = 0.001
        z_score = (ratios[-1] - mean_r) / std_r

        # ETH RSI
        eth_closes = [c["trade_price"] for c in reversed(eth_daily)]
        eth_rsi = rsi(eth_closes, 14)

        return {
            "eth_price": eth_ticker["trade_price"],
            "eth_change_24h": round(eth_ticker["signed_change_rate"] * 100, 2),
            "eth_volume_24h_krw": round(eth_ticker["acc_trade_price_24h"]),
            "eth_rsi_14": round(eth_rsi, 2),
            "eth_btc_ratio": round(ratios[-1], 6),
            "eth_btc_ratio_avg60": round(mean_r, 6),
            "eth_btc_ratio_min60": round(min(ratios), 6),
            "eth_btc_ratio_max60": round(max(ratios), 6),
            "eth_btc_z_score": round(z_score, 2),
            "eth_btc_signal": (
                "ETH 극단적 저평가" if z_score < -2
                else "ETH 저평가" if z_score < -1
                else "ETH 극단적 고평가" if z_score > 2
                else "ETH 고평가" if z_score > 1
                else "정상 범위"
            ),
        }
    except Exception as e:
        return {"error": f"ETH 데이터 수집 실패: {e}"}


def main(market: str = "KRW-BTC"):
    ticker = api_get("/ticker", {"markets": market})[0]
    # 220일봉으로 EMA(200), ADX 등 장기 지표 지원
    daily = api_get("/candles/days", {"market": market, "count": "220"})
    four_h = api_get("/candles/minutes/240", {"market": market, "count": "42"})
    ob = api_get("/orderbook", {"markets": market})[0]
    trades = api_get("/trades/ticks", {"market": market, "count": "100"})

    daily.reverse()  # 오래된 순 정렬
    closes = [c["trade_price"] for c in daily]
    highs = [c["high_price"] for c in daily]
    lows = [c["low_price"] for c in daily]

    buy_vol = sum(t["trade_volume"] for t in trades if t["ask_bid"] == "BID")
    sell_vol = sum(t["trade_volume"] for t in trades if t["ask_bid"] == "ASK")

    # ETH/BTC 비율 및 시장 구조
    time.sleep(0.15)
    eth_btc = collect_eth_btc_ratio()

    # ADX / ATR
    adx_data = calc_adx(highs, lows, closes)
    atr_val = calc_atr(highs, lows, closes)

    # 4시간봉 지표 계산
    four_h.reverse()  # 오래된 순 정렬
    closes_4h = [c["trade_price"] for c in four_h]
    highs_4h = [c["high_price"] for c in four_h]
    lows_4h = [c["low_price"] for c in four_h]

    # 일봉 요약 (raw 데이터 대신 최근 5일 OHLCV 요약만)
    daily_summary = []
    for c in daily[-5:]:
        daily_summary.append({
            "date": c.get("candle_date_time_kst", "")[:10],
            "open": c["opening_price"],
            "high": c["high_price"],
            "low": c["low_price"],
            "close": c["trade_price"],
            "change_pct": round((c["trade_price"] - c["opening_price"]) / max(c["opening_price"], 1) * 100, 2),
            "volume": round(c["candle_acc_trade_volume"], 2),
        })

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "market": market,
        "current_price": ticker["trade_price"],
        "change_rate_24h": ticker["signed_change_rate"],
        "volume_24h": ticker["acc_trade_volume_24h"],
        "indicators": {
            "sma_20": round(sma(closes, 20), 2),
            "sma_50": round(sma(closes, 50), 2) if len(closes) >= 50 else None,
            "sma_200": round(sma(closes, 200), 2) if len(closes) >= 200 else None,
            "ema_10": round(ema(closes, 10), 2),
            "ema_50": round(ema(closes, 50), 2) if len(closes) >= 50 else None,
            "ema_200": round(ema(closes, 200), 2) if len(closes) >= 200 else None,
            "rsi_14": round(rsi(closes, 14), 2),
            "macd": macd(closes),
            "bollinger": bollinger(closes, 20),
            "stochastic": stochastic(highs, lows, closes, 14),
            "adx": adx_data,
            "atr": atr_val,
        },
        "indicators_4h": {
            "rsi_14": round(rsi(closes_4h, 14), 2) if len(closes_4h) > 14 else None,
            "macd": macd(closes_4h) if len(closes_4h) >= 26 else None,
            "stochastic": stochastic(highs_4h, lows_4h, closes_4h, 14),
        },
        "orderbook": {
            "bid_total": ob["total_bid_size"],
            "ask_total": ob["total_ask_size"],
            "ratio": round(ob["total_bid_size"] / max(ob["total_ask_size"], 1e-8), 4),
        },
        "trade_pressure": {"buy_volume": buy_vol, "sell_volume": sell_vol},
        "eth_btc_analysis": eth_btc,
        "daily_summary_5d": daily_summary,
    }
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
