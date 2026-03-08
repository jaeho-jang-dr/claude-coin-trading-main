#!/usr/bin/env python3
"""
온체인 데이터 수집 스크립트

무료 API를 활용하여 BTC 온체인 지표를 수집한다:
  1. 펀딩레이트 (Binance Futures)
  2. 롱/숏 비율 (Binance Futures)
  3. 오픈 인터레스트 (Binance Futures)
  4. 거래소 BTC 유입/유출 추정 (mempool.space)

출력: JSON (stdout)
"""

from __future__ import annotations

import json
import sys
import time

import requests

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
TIMEOUT = 10


def safe_get(url: str, params: dict | None = None) -> dict | list | None:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def collect_funding_rate() -> dict:
    """Binance BTCUSDT 무기한 선물 펀딩레이트."""
    data = safe_get(f"{BINANCE_FAPI}/fundingRate", {
        "symbol": "BTCUSDT", "limit": "10"
    })
    if not data:
        return {"error": "funding rate 수집 실패"}

    latest = data[-1]
    rate = float(latest.get("fundingRate", 0))
    rates = [float(d["fundingRate"]) for d in data]
    avg_rate = sum(rates) / len(rates)

    # 펀딩레이트 해석 (극단값 먼저 체크)
    if rate > 0.01:
        signal = "과열_롱"
    elif rate > 0.005:
        signal = "롱_우세"
    elif rate < -0.01:
        signal = "과열_숏"
    elif rate < -0.005:
        signal = "숏_우세"
    else:
        signal = "중립"

    return {
        "current_rate": round(rate * 100, 4),
        "avg_rate_10": round(avg_rate * 100, 4),
        "signal": signal,
        "interpretation": (
            "롱 과열 → 하락 조정 가능" if signal == "과열_롱"
            else "숏 과열 → 숏스퀴즈 가능" if signal == "과열_숏"
            else "롱 우세" if signal == "롱_우세"
            else "숏 우세" if signal == "숏_우세"
            else "균형 상태"
        ),
    }


def collect_long_short_ratio() -> dict:
    """Binance 롱/숏 비율 (Top Traders)."""
    data = safe_get(f"{BINANCE_FAPI}/topLongShortAccountRatio", {
        "symbol": "BTCUSDT", "period": "1h", "limit": "5"
    })
    if not data:
        return {"error": "롱숏 비율 수집 실패"}

    latest = data[-1]
    long_ratio = float(latest.get("longAccount", 0.5))
    short_ratio = float(latest.get("shortAccount", 0.5))
    ls_ratio = round(long_ratio / max(short_ratio, 0.01), 2)

    return {
        "long_pct": round(long_ratio * 100, 1),
        "short_pct": round(short_ratio * 100, 1),
        "long_short_ratio": ls_ratio,
        "signal": (
            "극단적_롱" if ls_ratio > 2.0
            else "롱_우세" if ls_ratio > 1.2
            else "극단적_숏" if ls_ratio < 0.5
            else "숏_우세" if ls_ratio < 0.8
            else "균형"
        ),
    }


def collect_open_interest() -> dict:
    """Binance BTCUSDT 오픈 인터레스트."""
    data = safe_get(f"{BINANCE_FAPI}/openInterest", {"symbol": "BTCUSDT"})
    if not data:
        return {"error": "OI 수집 실패"}

    oi = float(data.get("openInterest", 0))

    # 최근 OI 변화 추이 (24시간)
    hist = safe_get("https://fapi.binance.com/futures/data/openInterestHist", {
        "symbol": "BTCUSDT", "period": "1h", "limit": "24"
    })
    oi_change_24h = 0
    if hist and len(hist) >= 2:
        first_oi = float(hist[0].get("sumOpenInterest", oi))
        oi_change_24h = round((oi - first_oi) / max(first_oi, 1) * 100, 2)

    return {
        "open_interest_btc": round(oi, 2),
        "oi_change_24h_pct": oi_change_24h,
        "signal": (
            "급증" if oi_change_24h > 5
            else "증가" if oi_change_24h > 2
            else "급감" if oi_change_24h < -5
            else "감소" if oi_change_24h < -2
            else "안정"
        ),
    }


def collect_mempool_fees() -> dict:
    """mempool.space에서 비트코인 네트워크 수수료 및 트랜잭션 데이터."""
    fees = safe_get("https://mempool.space/api/v1/fees/recommended")
    if not fees:
        return {"error": "mempool 수집 실패"}

    # 높은 수수료 = 네트워크 활동 증가 = 강세 신호
    fast_fee = fees.get("fastestFee", 0)
    return {
        "fastest_fee_sat_vb": fast_fee,
        "half_hour_fee": fees.get("halfHourFee", 0),
        "hour_fee": fees.get("hourFee", 0),
        "network_activity": (
            "매우_활발" if fast_fee > 50
            else "활발" if fast_fee > 20
            else "한산" if fast_fee < 5
            else "보통"
        ),
    }


def main():
    result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00")}

    result["funding_rate"] = collect_funding_rate()
    time.sleep(0.2)
    result["long_short_ratio"] = collect_long_short_ratio()
    time.sleep(0.2)
    result["open_interest"] = collect_open_interest()
    time.sleep(0.2)
    result["network_activity"] = collect_mempool_fees()

    # 종합 시그널
    signals = []
    fr = result["funding_rate"]
    if not fr.get("error"):
        if fr["signal"] == "과열_롱":
            signals.append(-2)
        elif fr["signal"] == "과열_숏":
            signals.append(2)
        elif fr["signal"] == "숏_우세":
            signals.append(1)
        elif fr["signal"] == "롱_우세":
            signals.append(-1)
        else:
            signals.append(0)

    ls = result["long_short_ratio"]
    if not ls.get("error"):
        if ls["signal"] == "극단적_롱":
            signals.append(-2)  # 역발상: 롱 과열 → 하락
        elif ls["signal"] == "극단적_숏":
            signals.append(2)   # 역발상: 숏 과열 → 상승
        elif ls["signal"] == "숏_우세":
            signals.append(1)
        elif ls["signal"] == "롱_우세":
            signals.append(-1)
        else:
            signals.append(0)

    oi = result["open_interest"]
    if not oi.get("error"):
        if oi["signal"] == "급증":
            signals.append(1)
        elif oi["signal"] == "급감":
            signals.append(-1)
        else:
            signals.append(0)

    if signals:
        avg = sum(signals) / len(signals)
        if avg > 1:
            result["onchain_signal"] = "bullish"
        elif avg > 0.3:
            result["onchain_signal"] = "slightly_bullish"
        elif avg < -1:
            result["onchain_signal"] = "bearish"
        elif avg < -0.3:
            result["onchain_signal"] = "slightly_bearish"
        else:
            result["onchain_signal"] = "neutral"
    else:
        result["onchain_signal"] = "no_data"

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
