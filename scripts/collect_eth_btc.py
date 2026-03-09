#!/usr/bin/env python3
"""
ETH/BTC 비율 및 시장 구조 분석

Upbit 공개 API로 ETH/BTC 60일 비율, z-score, 거래량 비교를 수집한다.
에이전트 모드에서 시장 구조 변화를 감지하는 데 사용.

API 키 불필요 (Upbit 공개 시세 조회)
출력: JSON (stdout)
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import time

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import requests

UPBIT_API = "https://api.upbit.com/v1"
TIMEOUT = 10


def _get(url: str, params: dict = None) -> dict | list | None:
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[eth_btc] {url} 오류: {e}", file=sys.stderr)
    return None


def main():
    # 현재가
    btc_ticker = _get(f"{UPBIT_API}/ticker", {"markets": "KRW-BTC"})
    eth_ticker = _get(f"{UPBIT_API}/ticker", {"markets": "KRW-ETH"})

    if not btc_ticker or not eth_ticker:
        json.dump({"error": "ticker 조회 실패"}, sys.stdout, ensure_ascii=False)
        return

    btc_t = btc_ticker[0]
    eth_t = eth_ticker[0]

    time.sleep(0.15)

    # 60일 일봉
    btc_days = _get(f"{UPBIT_API}/candles/days", {"market": "KRW-BTC", "count": 60})
    time.sleep(0.15)
    eth_days = _get(f"{UPBIT_API}/candles/days", {"market": "KRW-ETH", "count": 60})

    if not btc_days or not eth_days:
        json.dump({"error": "일봉 조회 실패"}, sys.stdout, ensure_ascii=False)
        return

    # 시간순 정렬 (역순 → 정순)
    btc_prices = [c["trade_price"] for c in reversed(btc_days)]
    eth_prices = [c["trade_price"] for c in reversed(eth_days)]

    min_len = min(len(btc_prices), len(eth_prices))
    ratios = [e / b for b, e in zip(btc_prices[:min_len], eth_prices[:min_len])]

    if len(ratios) < 10:
        json.dump({"error": "데이터 부족"}, sys.stdout, ensure_ascii=False)
        return

    current_ratio = ratios[-1]
    mean_ratio = statistics.mean(ratios)
    std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0001
    z_score = (current_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0

    # 추세: 최근 7일 vs 이전 7일
    recent_7 = statistics.mean(ratios[-7:]) if len(ratios) >= 7 else current_ratio
    prev_7 = statistics.mean(ratios[-14:-7]) if len(ratios) >= 14 else mean_ratio
    trend_pct = ((recent_7 - prev_7) / prev_7 * 100) if prev_7 > 0 else 0

    # 거래량 비교
    btc_vol = round(btc_t["acc_trade_price_24h"] / 1e8)  # 억원 단위
    eth_vol = round(eth_t["acc_trade_price_24h"] / 1e8)
    vol_ratio = eth_vol / btc_vol if btc_vol > 0 else 0

    # 시그널 판단
    if z_score <= -2:
        signal = "eth_undervalued"
        signal_note = f"ETH/BTC z-score {z_score:.2f} ≤ -2 — ETH 상대 저평가, 시장 구조 변화 가능"
    elif z_score >= 2:
        signal = "eth_overvalued"
        signal_note = f"ETH/BTC z-score {z_score:.2f} ≥ 2 — ETH 상대 고평가, 자금 이동 주시"
    elif abs(z_score) >= 1.5:
        signal = "diverging"
        signal_note = f"ETH/BTC z-score {z_score:.2f} — 괴리 확대 중"
    else:
        signal = "normal"
        signal_note = f"ETH/BTC z-score {z_score:.2f} — 정상 범위"

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "source": "Upbit public API (free)",
        "btc_price": btc_t["trade_price"],
        "eth_price": eth_t["trade_price"],
        "btc_change_24h": round(btc_t["signed_change_rate"] * 100, 2),
        "eth_change_24h": round(eth_t["signed_change_rate"] * 100, 2),
        "eth_btc_ratio": round(current_ratio, 6),
        "eth_btc_ratio_avg60": round(mean_ratio, 6),
        "eth_btc_z_score": round(z_score, 2),
        "eth_btc_trend_7d_pct": round(trend_pct, 2),
        "btc_volume_24h_억": btc_vol,
        "eth_volume_24h_억": eth_vol,
        "eth_btc_volume_ratio": round(vol_ratio, 3),
        "signal": signal,
        "signal_note": signal_note,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stdout, ensure_ascii=False)
        sys.exit(1)
