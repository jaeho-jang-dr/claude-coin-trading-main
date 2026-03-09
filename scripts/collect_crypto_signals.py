#!/usr/bin/env python3
"""CoinGecko 거래량 이상 감지 스크립트.

CoinGecko 무료 API로 거래량 상위 50개 토큰을 조회하고,
거래량/시총 비율(vol_mcap_ratio)로 이상 거래량을 감지한다.
ExternalDataAgent + run_analysis.sh 양쪽에서 사용.

출력: BTC/ETH 분석 + 거래량 이상 토큰 목록 (JSON)
"""

import json
import sys
import time

import requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"


def fetch_signals():
    """CoinGecko에서 거래량 상위 50개 토큰 조회."""
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": "50",
        "page": "1",
    }
    for attempt in range(3):
        try:
            resp = requests.get(COINGECKO_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                print(json.dumps({"error": str(e)}))
                sys.exit(1)
            time.sleep(2 ** attempt)
    else:
        print(json.dumps({"error": "CoinGecko rate limit exceeded"}))
        sys.exit(1)

    try:
        raw = resp.json()
    except (ValueError, json.JSONDecodeError):
        print(json.dumps({"error": "CoinGecko returned non-JSON response"}))
        sys.exit(1)
    if not isinstance(raw, list):
        print(json.dumps({"error": f"CoinGecko unexpected response type: {type(raw).__name__}"}))
        sys.exit(1)
    signals = []
    for coin in raw:
        mc = coin.get("market_cap") or 0
        vol = coin.get("total_volume") or 0
        ratio = round(vol / mc * 100, 2) if mc > 0 else 0
        signals.append({
            "symbol": (coin.get("symbol") or "").upper(),
            "price": coin.get("current_price"),
            "change_24h": coin.get("price_change_percentage_24h"),
            "volume": vol,
            "market_cap": mc,
            "vol_mcap_ratio": ratio,
        })
    signals.sort(key=lambda s: s["vol_mcap_ratio"], reverse=True)
    return signals


def classify_anomaly(ratio):
    if ratio > 20:
        return "CRITICAL"
    elif ratio > 5:
        return "HIGH"
    elif ratio > 2:
        return "MODERATE"
    return "LOW"


def main():
    signals = fetch_signals()

    # BTC/ETH 개별 분석
    btc = next((s for s in signals if s["symbol"] == "BTC"), None)
    eth = next((s for s in signals if s["symbol"] == "ETH"), None)

    for token in [btc, eth]:
        if token:
            token["anomaly_level"] = classify_anomaly(token["vol_mcap_ratio"])

    # 이상 거래량 알림 (ratio > 5%)
    high_alerts = [s for s in signals if s["vol_mcap_ratio"] > 5.0]
    for a in high_alerts:
        a["anomaly_level"] = classify_anomaly(a["vol_mcap_ratio"])

    result = {
        "source": "coingecko",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "btc": btc,
        "eth": eth,
        "anomaly_alerts": {
            "count": len(high_alerts),
            "threshold": "vol_mcap_ratio > 5.0%",
            "tokens": high_alerts[:10],  # 상위 10개만
        },
        "top_volume_tokens": [
            {"symbol": s["symbol"], "ratio": s["vol_mcap_ratio"], "change_24h": s["change_24h"]}
            for s in signals[:10]
        ],
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
