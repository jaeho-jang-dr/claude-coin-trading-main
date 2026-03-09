#!/usr/bin/env python3
"""
AI 복합 시그널 수집 스크립트

사람이 동시에 처리할 수 없는 6가지 분석을 0.5초 만에 수행한다:
  1. 호가창 심층 분석 (Orderbook Imbalance)
  2. 고래 거래 감지 (Whale Detection)
  3. 멀티 타임프레임 다이버전스 (Multi-TF Divergence)
  4. 변동성 레짐 감지 (Volatility Regime)
  5. 거래량 이상 감지 (Volume Anomaly)
  6. 복합 시그널 점수 (Multi-Signal Score: -100 ~ +100)

출력: JSON (stdout)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests

UPBIT_API = "https://api.upbit.com/v1"
RATE_LIMIT_WAIT = 0.15  # Upbit API rate limit 대응


def api_get(path: str, params: dict | None = None, max_retries: int = 3) -> dict | list:
    url = f"{UPBIT_API}{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(max_retries):
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"[rate_limit] 429 received, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


def calc_rsi(candles: list[dict], period: int = 14) -> float | None:
    prices = [c["trade_price"] for c in candles]
    if len(prices) < period + 1:
        return None
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
    return 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 2)


def trend_pct(candles: list[dict], count: int = 5) -> float:
    if len(candles) < count:
        return 0.0
    first = candles[count - 1]["trade_price"]
    last = candles[0]["trade_price"]
    return round((last - first) / first * 100, 2) if first else 0.0


# ── 분석 모듈 ──────────────────────────────────────────


def analyze_orderbook(market: str) -> dict:
    """#1 호가창 심층 분석"""
    ob = api_get("/orderbook", {"markets": market})[0]
    units = ob.get("orderbook_units", [])

    total_bid = ob["total_bid_size"]
    total_ask = ob["total_ask_size"]
    imbalance = round(total_bid / max(total_ask, 1e-8), 4)

    avg_bid = total_bid / max(len(units), 1)
    avg_ask = total_ask / max(len(units), 1)

    bid_walls = [
        {"price": u["bid_price"], "size": u["bid_size"],
         "multiple": round(u["bid_size"] / avg_bid, 1)}
        for u in units if u["bid_size"] > avg_bid * 2
    ]
    ask_walls = [
        {"price": u["ask_price"], "size": u["ask_size"],
         "multiple": round(u["ask_size"] / avg_ask, 1)}
        for u in units if u["ask_size"] > avg_ask * 2
    ]

    return {
        "total_bid": round(total_bid, 4),
        "total_ask": round(total_ask, 4),
        "imbalance_ratio": imbalance,
        "signal": "buy" if imbalance > 1.1 else "sell" if imbalance < 0.9 else "neutral",
        "bid_walls": bid_walls,
        "ask_walls": ask_walls,
    }


def analyze_whale_trades(market: str) -> dict:
    """#2 고래 거래 감지 + CVD (Cumulative Volume Delta)"""
    trades = api_get("/trades/ticks", {"market": market, "count": "200"})

    buy_vol = sell_vol = 0.0
    buy_count = sell_count = 0
    whale_trades = []       # 1,000만원+ (고래)
    mega_whale_trades = []  # 5,000만원+ (대형 고래)
    whale_threshold_krw = 10_000_000       # 1,000만원 (상위 ~3.5%)
    mega_whale_threshold_krw = 50_000_000  # 5,000만원 (상위 ~0.5%)

    # CVD: 누적 체결 델타 (매수체결량 - 매도체결량)
    cvd = 0.0
    cvd_series = []

    for t in trades:
        vol = abs(t.get("trade_volume", 0))
        price = t.get("trade_price", 0)
        krw = vol * price
        side = t.get("ask_bid", "")

        if side == "BID":
            buy_vol += vol
            buy_count += 1
            cvd += vol
        else:
            sell_vol += vol
            sell_count += 1
            cvd -= vol

        cvd_series.append(cvd)

        if krw >= whale_threshold_krw:
            entry = {
                "side": side,
                "volume": round(vol, 8),
                "price": price,
                "krw_amount": round(krw),
                "time": t.get("trade_time_utc", ""),
                "is_mega": krw >= mega_whale_threshold_krw,
            }
            whale_trades.append(entry)
            if krw >= mega_whale_threshold_krw:
                mega_whale_trades.append(entry)

    total_vol = buy_vol + sell_vol
    buy_ratio = round(buy_vol / max(total_vol, 1e-8) * 100, 1)
    whale_buy = sum(1 for t in whale_trades if t["side"] == "BID")
    whale_sell = len(whale_trades) - whale_buy
    mega_buy = sum(1 for t in mega_whale_trades if t["side"] == "BID")
    mega_sell = len(mega_whale_trades) - mega_buy

    # CVD 추세 분석: 후반 50건 vs 전반 50건
    cvd_trend = "neutral"
    if len(cvd_series) >= 100:
        first_half = cvd_series[49] - cvd_series[0]
        second_half = cvd_series[-1] - cvd_series[49]
        if second_half > first_half + total_vol * 0.01:
            cvd_trend = "accumulating"  # 매수 가속
        elif second_half < first_half - total_vol * 0.01:
            cvd_trend = "distributing"  # 매도 가속

    return {
        "total_trades": len(trades),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
        "buy_ratio_pct": buy_ratio,
        "trade_pressure": (
            "buy" if buy_ratio > 55 else "sell" if buy_ratio < 45 else "neutral"
        ),
        "cvd": round(cvd, 8),
        "cvd_trend": cvd_trend,
        "whale_threshold_krw": whale_threshold_krw,
        "whale_trades_count": len(whale_trades),
        "whale_buy": whale_buy,
        "whale_sell": whale_sell,
        "whale_signal": (
            "buy" if whale_buy > whale_sell
            else "sell" if whale_sell > whale_buy
            else "neutral"
        ),
        "mega_whale_threshold_krw": mega_whale_threshold_krw,
        "mega_whale_count": len(mega_whale_trades),
        "mega_whale_buy": mega_buy,
        "mega_whale_sell": mega_sell,
        "top_whale_trades": whale_trades[:5],
    }


def analyze_multi_timeframe(market: str) -> dict:
    """#3 멀티 타임프레임 다이버전스"""
    daily = api_get("/candles/days", {"market": market, "count": "30"})
    time.sleep(RATE_LIMIT_WAIT)
    h4 = api_get("/candles/minutes/240", {"market": market, "count": "42"})
    time.sleep(RATE_LIMIT_WAIT)
    h1 = api_get("/candles/minutes/60", {"market": market, "count": "24"})

    # 오래된 순 정렬 후 RSI 계산
    daily_asc = list(reversed(daily))
    h4_asc = list(reversed(h4))
    h1_asc = list(reversed(h1))

    rsi_d = calc_rsi(daily_asc)
    rsi_4h = calc_rsi(h4_asc)
    rsi_1h = calc_rsi(h1_asc)

    # 추세 (최신순 candles 사용)
    trend_d = trend_pct(daily, 5)
    trend_4h = trend_pct(h4, 6)
    trend_1h = trend_pct(h1, 6)

    divergence = None
    divergence_type = None
    if rsi_d is not None and rsi_1h is not None:
        gap = abs(rsi_d - rsi_1h)
        if gap > 15:
            if rsi_1h < rsi_d:
                divergence_type = "short_term_oversold"
                divergence = (
                    f"1h RSI({rsi_1h}) << daily RSI({rsi_d}): "
                    "단기 과매도, 반등 가능"
                )
            else:
                divergence_type = "short_term_overbought"
                divergence = (
                    f"1h RSI({rsi_1h}) >> daily RSI({rsi_d}): "
                    "단기 과매수, 조정 가능"
                )

    # 멀티 타임프레임 추세 정렬 (Trend Alignment)
    trends = [trend_d, trend_4h, trend_1h]
    valid_trends = [t for t in trends if t != 0.0]
    if len(valid_trends) >= 2:
        bullish = sum(1 for t in valid_trends if t > 0)
        bearish = sum(1 for t in valid_trends if t < 0)
        if bullish == len(valid_trends):
            trend_alignment = "all_bullish"
        elif bearish == len(valid_trends):
            trend_alignment = "all_bearish"
        elif bullish > bearish:
            trend_alignment = "mostly_bullish"
        elif bearish > bullish:
            trend_alignment = "mostly_bearish"
        else:
            trend_alignment = "mixed"
    else:
        trend_alignment = "insufficient_data"

    return {
        "timeframes": {
            "daily": {"rsi": rsi_d, "trend_pct": trend_d},
            "4h": {"rsi": rsi_4h, "trend_pct": trend_4h},
            "1h": {"rsi": rsi_1h, "trend_pct": trend_1h},
        },
        "trend_alignment": trend_alignment,
        "divergence": divergence,
        "divergence_type": divergence_type,
    }


def analyze_volatility(daily_candles: list[dict]) -> dict:
    """#4 변동성 레짐 감지"""
    if len(daily_candles) < 20:
        return {"error": "insufficient data"}

    ranges = [
        (c["high_price"] - c["low_price"]) / max(c["low_price"], 1) * 100
        for c in daily_candles[:20]
    ]

    avg_20 = sum(ranges) / 20
    avg_5 = sum(ranges[:5]) / 5
    avg_3 = sum(ranges[:3]) / 3
    vol_ratio = round(avg_3 / max(avg_20, 0.01), 2)

    if vol_ratio > 1.5:
        regime = "high_volatility"
    elif vol_ratio > 1.0:
        regime = "expanding"
    elif vol_ratio < 0.5:
        regime = "low_volatility"
    else:
        regime = "normal"

    return {
        "avg_range_20d": round(avg_20, 2),
        "avg_range_5d": round(avg_5, 2),
        "avg_range_3d": round(avg_3, 2),
        "vol_ratio_3d_20d": vol_ratio,
        "regime": regime,
    }


def analyze_volume(daily_candles: list[dict]) -> dict:
    """#5 거래량 이상 감지"""
    if len(daily_candles) < 20:
        return {"error": "insufficient data"}

    volumes = [c["candle_acc_trade_volume"] for c in daily_candles[:20]]
    avg_vol = sum(volumes) / 20

    recent_5 = []
    for c in daily_candles[:5]:
        vol = c["candle_acc_trade_volume"]
        ratio = round(vol / max(avg_vol, 1), 2)
        change = round(
            (c["trade_price"] - c["opening_price"]) / max(c["opening_price"], 1) * 100,
            1,
        )
        anomaly = (
            "spike" if ratio > 2.0
            else "high" if ratio > 1.5
            else "low" if ratio < 0.5
            else "normal"
        )
        recent_5.append({
            "date": c["candle_date_time_kst"][:10],
            "volume": round(vol),
            "ratio_vs_avg": ratio,
            "price_change_pct": change,
            "anomaly": anomaly,
        })

    return {
        "avg_volume_20d": round(avg_vol),
        "recent_5d": recent_5,
    }


def compute_composite_score(
    orderbook: dict,
    whale: dict,
    multi_tf: dict,
    volatility: dict,
    volume: dict,
    daily_candles: list[dict],
) -> dict:
    """#6 복합 시그널 점수 산출 (-100 ~ +100)"""
    score = 0
    components = []

    # 1) 호가 불균형 (최대 +/-15)
    imb = orderbook.get("imbalance_ratio", 1.0)
    if imb > 1.1:
        pts = min(int((imb - 1.0) * 50), 15)
        score += pts
        components.append({"name": "orderbook_imbalance", "score": pts,
                           "detail": f"매수 우세 ({imb}x)"})
    elif imb < 0.9:
        pts = max(int((imb - 1.0) * 50), -15)
        score += pts
        components.append({"name": "orderbook_imbalance", "score": pts,
                           "detail": f"매도 우세 ({imb}x)"})

    # 2) 체결 강도 (최대 +/-20)
    buy_r = whale.get("buy_ratio_pct", 50)
    if buy_r > 60:
        pts = min(int((buy_r - 50) * 2), 20)
        score += pts
        components.append({"name": "trade_pressure", "score": pts,
                           "detail": f"매수 체결 {buy_r}%"})
    elif buy_r < 40:
        pts = max(int((buy_r - 50) * 2), -20)
        score += pts
        components.append({"name": "trade_pressure", "score": pts,
                           "detail": f"매도 체결 {100-buy_r}%"})

    # 3) 고래 방향 (최대 +/-15, 1000만원+ 기준)
    wb = whale.get("whale_buy", 0)
    ws = whale.get("whale_sell", 0)
    mega_b = whale.get("mega_whale_buy", 0)
    mega_s = whale.get("mega_whale_sell", 0)

    # 대형 고래(5000만원+)는 가중치 2배
    weighted_buy = wb + mega_b  # 대형 고래는 이미 wb에 포함 + 추가 가중
    weighted_sell = ws + mega_s

    if weighted_buy > weighted_sell and (wb + ws) >= 2:
        pts = min(int((weighted_buy - weighted_sell) * 5), 15)
        score += pts
        detail = f"고래 매수 {wb}건 vs 매도 {ws}건"
        if mega_b > 0:
            detail += f" (대형고래 매수 {mega_b}건)"
        components.append({"name": "whale_direction", "score": pts, "detail": detail})
    elif weighted_sell > weighted_buy and (wb + ws) >= 2:
        pts = max(int((weighted_buy - weighted_sell) * 5), -15)
        score += pts
        detail = f"고래 매도 {ws}건 vs 매수 {wb}건"
        if mega_s > 0:
            detail += f" (대형고래 매도 {mega_s}건)"
        components.append({"name": "whale_direction", "score": pts, "detail": detail})

    # 3.5) CVD 추세 (최대 +/-10)
    cvd_trend = whale.get("cvd_trend", "neutral")
    if cvd_trend == "accumulating":
        score += 10
        components.append({"name": "cvd_trend", "score": 10,
                           "detail": "CVD 매수 가속 (숨겨진 매집)"})
    elif cvd_trend == "distributing":
        score -= 10
        components.append({"name": "cvd_trend", "score": -10,
                           "detail": "CVD 매도 가속 (숨겨진 분배)"})

    # 4) 다이버전스 (최대 +/-10)
    div_type = multi_tf.get("divergence_type")
    if div_type == "short_term_oversold":
        score += 10
        components.append({"name": "tf_divergence", "score": 10,
                           "detail": "단기 과매도 다이버전스 (반등 가능)"})
    elif div_type == "short_term_overbought":
        score -= 10
        components.append({"name": "tf_divergence", "score": -10,
                           "detail": "단기 과매수 다이버전스 (조정 가능)"})

    # 4.5) 멀티TF 추세 정렬 (최대 +/-10)
    alignment = multi_tf.get("trend_alignment", "mixed")
    if alignment == "all_bullish":
        score += 10
        components.append({"name": "trend_alignment", "score": 10,
                           "detail": "전 타임프레임 상승 정렬"})
    elif alignment == "all_bearish":
        score -= 10
        components.append({"name": "trend_alignment", "score": -10,
                           "detail": "전 타임프레임 하락 정렬"})

    # 5) 변동성 레짐 (최대 +/-10)
    regime = volatility.get("regime", "normal")
    if regime == "low_volatility":
        score += 10
        components.append({"name": "volatility_regime", "score": 10,
                           "detail": "저변동성 축적 구간 (큰 움직임 전조)"})
    elif regime == "high_volatility":
        score -= 10
        components.append({"name": "volatility_regime", "score": -10,
                           "detail": "고변동성 위험 구간"})

    # 6) 거래량 이상 + 방향 (최대 +/-15)
    if daily_candles and len(daily_candles) >= 20:
        vol_today = daily_candles[0]["candle_acc_trade_volume"]
        avg_vol = sum(c["candle_acc_trade_volume"] for c in daily_candles[:20]) / 20
        vol_ratio = vol_today / max(avg_vol, 1)
        price_chg = (
            daily_candles[0]["trade_price"] - daily_candles[0]["opening_price"]
        )
        if vol_ratio > 1.5 and price_chg > 0:
            pts = min(int(vol_ratio * 7.5), 15)
            score += pts
            components.append({"name": "volume_anomaly", "score": pts,
                               "detail": f"거래량 급증({vol_ratio:.1f}x) + 양봉"})
        elif vol_ratio > 1.5 and price_chg < 0:
            pts = max(-int(vol_ratio * 7.5), -15)
            score += pts
            components.append({"name": "volume_anomaly", "score": pts,
                               "detail": f"거래량 급증({vol_ratio:.1f}x) + 음봉"})

    # 점수 클램프
    score = max(-100, min(100, score))

    if score >= 30:
        interpretation = "strong_buy"
    elif score >= 10:
        interpretation = "weak_buy"
    elif score <= -30:
        interpretation = "strong_sell"
    elif score <= -10:
        interpretation = "weak_sell"
    else:
        interpretation = "neutral"

    return {
        "score": score,
        "interpretation": interpretation,
        "max_possible": 100,
        "min_possible": -100,
        "components": components,
    }


# ── 메인 ──────────────────────────────────────────────


def main(market: str = "KRW-BTC"):
    # 데이터 수집
    orderbook = analyze_orderbook(market)
    time.sleep(RATE_LIMIT_WAIT)

    whale = analyze_whale_trades(market)
    time.sleep(RATE_LIMIT_WAIT)

    multi_tf = analyze_multi_timeframe(market)
    # multi_timeframe 내부에서 3번 API 호출 + sleep 포함

    # 일봉 데이터 (멀티TF에서 이미 호출했지만 변동성/거래량 분석용으로 재호출)
    daily = api_get("/candles/days", {"market": market, "count": "30"})

    volatility = analyze_volatility(daily)
    volume = analyze_volume(daily)

    # 복합 점수 산출
    composite = compute_composite_score(
        orderbook, whale, multi_tf, volatility, volume, daily
    )

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "market": market,
        "ai_composite_signal": composite,
        "details": {
            "orderbook_imbalance": orderbook,
            "whale_detection": whale,
            "multi_timeframe": multi_tf,
            "volatility_regime": volatility,
            "volume_anomaly": volume,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
