#!/usr/bin/env python3
"""
바이낸스 파생상품 심리 지표 + 김치 프리미엄 (무료 API — 키 불필요)

수집 항목:
  1. 롱/숏 비율 (Top Trader Long/Short Position Ratio)
  2. 글로벌 롱/숏 비율 (Global Long/Short Account Ratio)
  3. 펀딩비 (Funding Rate)
  4. 미결제 약정 (Open Interest)
  5. 김치 프리미엄 (Upbit KRW vs Binance USDT + 환율)

출력: JSON (stdout)

사용법:
    python3 scripts/binance_sentiment.py
"""

from __future__ import annotations

import io
import json
import sys
import time

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── 설정 ──────────────────────────────────────────────

BINANCE_FUTURES = "https://fapi.binance.com"
BINANCE_DATA = "https://fapi.binance.com/futures/data"
UPBIT_API = "https://api.upbit.com/v1"

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 0.2

# 커넥션 재사용을 위한 세션
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """모듈 레벨 requests.Session을 반환한다 (커넥션 풀 재사용)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Accept": "application/json"})
    return _session


# ── HTTP 헬퍼 ─────────────────────────────────────────


def _get(url: str, params: dict | None = None, label: str = "") -> dict | list | None:
    """Exponential backoff 포함 GET 요청 (Session 재사용)."""
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(
                    f"[binance_sentiment] {label} 429. {wait}초 후 재시도",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(
                f"[binance_sentiment] {label} HTTP {resp.status_code}: {resp.text[:200]}",
                file=sys.stderr,
            )
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(
                f"[binance_sentiment] {label} 오류: {e}. {wait}초 후 재시도",
                file=sys.stderr,
            )
            time.sleep(wait)
    return None


# ── 바이낸스 파생상품 데이터 ──────────────────────────


def fetch_top_long_short_ratio(symbol: str = "BTCUSDT", period: str = "1h") -> dict:
    """상위 트레이더 롱/숏 포지션 비율 (최근 데이터)."""
    data = _get(
        f"{BINANCE_DATA}/topLongShortPositionRatio",
        {"symbol": symbol, "period": period, "limit": "5"},
        "top_ls_ratio",
    )
    if not data or not isinstance(data, list):
        return {"error": "조회 실패"}

    latest = data[0]
    history = []
    for d in data:
        history.append({
            "timestamp": d.get("timestamp", 0),
            "long_ratio": float(d.get("longAccount", 0)),
            "short_ratio": float(d.get("shortAccount", 0)),
            "long_short_ratio": float(d.get("longShortRatio", 0)),
        })

    lr = float(latest.get("longShortRatio", 1.0))
    if lr > 1.5:
        signal = "overleveraged_long"
    elif lr > 1.2:
        signal = "long_dominant"
    elif lr < 0.7:
        signal = "overleveraged_short"
    elif lr < 0.85:
        signal = "short_dominant"
    else:
        signal = "balanced"

    return {
        "current_ratio": lr,
        "long_pct": round(float(latest.get("longAccount", 0)) * 100, 1),
        "short_pct": round(float(latest.get("shortAccount", 0)) * 100, 1),
        "signal": signal,
        "history": history,
    }


def fetch_global_long_short_ratio(symbol: str = "BTCUSDT", period: str = "1h") -> dict:
    """글로벌 롱/숏 계좌 비율."""
    data = _get(
        f"{BINANCE_DATA}/globalLongShortAccountRatio",
        {"symbol": symbol, "period": period, "limit": "5"},
        "global_ls_ratio",
    )
    if not data or not isinstance(data, list):
        return {"error": "조회 실패"}

    latest = data[0]
    lr = float(latest.get("longShortRatio", 1.0))

    return {
        "current_ratio": lr,
        "long_pct": round(float(latest.get("longAccount", 0)) * 100, 1),
        "short_pct": round(float(latest.get("shortAccount", 0)) * 100, 1),
    }


def fetch_funding_rate(symbol: str = "BTCUSDT") -> dict:
    """현재 펀딩비."""
    data = _get(
        f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": "5"},
        "funding_rate",
    )
    if not data or not isinstance(data, list):
        return {"error": "조회 실패"}

    latest = data[-1]  # 가장 최근
    rate = float(latest.get("fundingRate", 0))

    # 펀딩비 해석
    if rate > 0.001:
        signal = "extremely_bullish_crowd"  # 롱 과밀 → 조정 경고
    elif rate > 0.0005:
        signal = "bullish_crowd"
    elif rate < -0.001:
        signal = "extremely_bearish_crowd"  # 숏 과밀 → 반등 가능
    elif rate < -0.0005:
        signal = "bearish_crowd"
    else:
        signal = "neutral"

    history = []
    for d in data:
        history.append({
            "funding_rate": float(d.get("fundingRate", 0)),
            "funding_time": d.get("fundingTime", 0),
        })

    return {
        "current_rate": rate,
        "current_rate_pct": round(rate * 100, 4),
        "signal": signal,
        "note": "양수=롱 과밀(조정 경고), 음수=숏 과밀(반등 가능)",
        "history": history,
    }


def fetch_open_interest(symbol: str = "BTCUSDT") -> dict:
    """미결제 약정."""
    data = _get(
        f"{BINANCE_FUTURES}/fapi/v1/openInterest",
        {"symbol": symbol},
        "open_interest",
    )
    if not data:
        return {"error": "조회 실패"}

    oi = float(data.get("openInterest", 0))

    # 과거 OI 변화율은 history API로 보충
    time.sleep(RATE_LIMIT_WAIT)
    hist = _get(
        f"{BINANCE_DATA}/openInterestHist",
        {"symbol": symbol, "period": "1h", "limit": "24"},
        "oi_history",
    )

    oi_history = []
    oi_change_24h = 0.0
    if hist and isinstance(hist, list) and len(hist) >= 2:
        for h in hist:
            oi_history.append({
                "timestamp": h.get("timestamp", 0),
                "oi_btc": round(float(h.get("sumOpenInterest", 0)), 2),
                "oi_usd": round(float(h.get("sumOpenInterestValue", 0))),
            })
        first_oi = float(hist[0].get("sumOpenInterest", 0))
        last_oi = float(hist[-1].get("sumOpenInterest", 0))
        if first_oi > 0:
            oi_change_24h = round((last_oi - first_oi) / first_oi * 100, 2)

    return {
        "current_oi_btc": oi,
        "oi_change_24h_pct": oi_change_24h,
        "note": "OI 급증 + 가격상승 = 롱 신규진입, OI 급증 + 가격하락 = 숏 신규진입",
        "history": oi_history[-6:],  # 최근 6시간
    }


# ── 김치 프리미엄 ─────────────────────────────────────


def fetch_kimchi_premium() -> dict:
    """
    업비트 KRW 가격 vs 바이낸스 USDT 가격 + 환율로 김치 프리미엄 계산.
    환율 소스: 한국 수출입은행 or 바이낸스 USDT/KRW P2P 간접 추정.
    """
    # 업비트 BTC/KRW
    upbit = _get(
        f"{UPBIT_API}/ticker", {"markets": "KRW-BTC"}, "upbit_btc"
    )
    time.sleep(RATE_LIMIT_WAIT)

    # 바이낸스 BTC/USDT
    binance = _get(
        "https://api.binance.com/api/v3/ticker/price",
        {"symbol": "BTCUSDT"},
        "binance_btc",
    )
    time.sleep(RATE_LIMIT_WAIT)

    # 환율: 업비트 USDT/KRW로 간접 추정 (가장 실시간)
    usdt_krw_data = _get(
        f"{UPBIT_API}/ticker", {"markets": "KRW-USDT"}, "upbit_usdt"
    )

    if not upbit or not binance:
        return {"error": "가격 데이터 조회 실패"}

    upbit_price = float(upbit[0]["trade_price"]) if isinstance(upbit, list) else 0
    binance_price = float(binance.get("price", 0))

    # USDT/KRW 환율
    if usdt_krw_data and isinstance(usdt_krw_data, list):
        usdt_krw = float(usdt_krw_data[0]["trade_price"])
    else:
        usdt_krw = 1450.0  # 폴백 추정값

    # 김치 프리미엄 계산
    binance_in_krw = binance_price * usdt_krw
    if binance_in_krw > 0:
        premium_pct = round((upbit_price - binance_in_krw) / binance_in_krw * 100, 2)
    else:
        premium_pct = 0.0

    # 김치 프리미엄 시그널
    if premium_pct > 5.0:
        signal = "extreme_fomo"  # 국내 과열 — 조정 경고
    elif premium_pct > 3.0:
        signal = "high_premium"
    elif premium_pct > 1.0:
        signal = "moderate_premium"
    elif premium_pct < -1.0:
        signal = "discount"  # 국내 할인 — 매수 기회 가능
    elif premium_pct < -3.0:
        signal = "deep_discount"
    else:
        signal = "normal"

    return {
        "upbit_btc_krw": upbit_price,
        "binance_btc_usdt": binance_price,
        "usdt_krw_rate": usdt_krw,
        "binance_btc_in_krw": round(binance_in_krw),
        "premium_pct": premium_pct,
        "signal": signal,
        "note": "양수=국내 프리미엄(과열), 음수=국내 디스카운트(매수 기회 가능)",
    }


# ── 종합 심리 점수 ────────────────────────────────────


def compute_sentiment_score(
    top_ls: dict,
    funding: dict,
    oi: dict,
    kimchi: dict,
) -> dict:
    """
    파생상품 심리 종합 점수: -30 ~ +30
    양수 = 시장 과열 (조정 경고)
    음수 = 시장 비관 (반등 가능)
    """
    score = 0
    components: list[dict] = []

    # 1) 롱/숏 비율 (최대 ±10)
    if "current_ratio" in top_ls:
        lr = top_ls["current_ratio"]
        if lr > 1.5:
            pts = 10
            components.append({"name": "long_short_ratio", "score": pts,
                               "detail": f"롱 과밀 ({lr:.2f}) — 조정 경고"})
        elif lr > 1.2:
            pts = 5
            components.append({"name": "long_short_ratio", "score": pts,
                               "detail": f"롱 우세 ({lr:.2f})"})
        elif lr < 0.7:
            pts = -10
            components.append({"name": "long_short_ratio", "score": pts,
                               "detail": f"숏 과밀 ({lr:.2f}) — 반등 가능"})
        elif lr < 0.85:
            pts = -5
            components.append({"name": "long_short_ratio", "score": pts,
                               "detail": f"숏 우세 ({lr:.2f})"})
        else:
            pts = 0
        score += pts

    # 2) 펀딩비 (최대 ±10)
    if "current_rate" in funding:
        rate = funding["current_rate"]
        if rate > 0.001:
            pts = 10
            components.append({"name": "funding_rate", "score": pts,
                               "detail": f"극단적 양수 펀딩 ({rate*100:.3f}%) — 롱 과열"})
        elif rate > 0.0005:
            pts = 5
            components.append({"name": "funding_rate", "score": pts,
                               "detail": f"양수 펀딩 ({rate*100:.3f}%)"})
        elif rate < -0.001:
            pts = -10
            components.append({"name": "funding_rate", "score": pts,
                               "detail": f"극단적 음수 펀딩 ({rate*100:.3f}%) — 숏 과열"})
        elif rate < -0.0005:
            pts = -5
            components.append({"name": "funding_rate", "score": pts,
                               "detail": f"음수 펀딩 ({rate*100:.3f}%)"})
        else:
            pts = 0
        score += pts

    # 3) 김치 프리미엄 (최대 ±10)
    if "premium_pct" in kimchi:
        p = kimchi["premium_pct"]
        if p > 5.0:
            pts = 10
            components.append({"name": "kimchi_premium", "score": pts,
                               "detail": f"극단 프리미엄 ({p:.1f}%) — 국내 FOMO"})
        elif p > 3.0:
            pts = 5
            components.append({"name": "kimchi_premium", "score": pts,
                               "detail": f"높은 프리미엄 ({p:.1f}%)"})
        elif p < -3.0:
            pts = -10
            components.append({"name": "kimchi_premium", "score": pts,
                               "detail": f"깊은 디스카운트 ({p:.1f}%) — 매수 기회"})
        elif p < -1.0:
            pts = -5
            components.append({"name": "kimchi_premium", "score": pts,
                               "detail": f"디스카운트 ({p:.1f}%)"})
        else:
            pts = 0
        score += pts

    score = max(-30, min(30, score))

    # 종합 해석
    if score >= 15:
        interpretation = "overheated"  # 과열 → 조정 대비
    elif score >= 5:
        interpretation = "warming"
    elif score <= -15:
        interpretation = "capitulation"  # 항복 → 반등 준비
    elif score <= -5:
        interpretation = "cooling"
    else:
        interpretation = "neutral"

    return {
        "score": score,
        "max_score": 30,
        "interpretation": interpretation,
        "components": components,
        "note": "양수=시장 과열(조정 경고), 음수=시장 비관(반등 가능). Data Fusion 원칙 준수.",
    }


# ── 메인 ──────────────────────────────────────────────


def main() -> None:
    # 독립적인 API 호출을 병렬로 실행하여 지연 시간 절감
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_top_long_short_ratio): "top_ls",
            executor.submit(fetch_global_long_short_ratio): "global_ls",
            executor.submit(fetch_funding_rate): "funding",
            executor.submit(fetch_open_interest): "oi",
            executor.submit(fetch_kimchi_premium): "kimchi",
        }
        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"[binance_sentiment] {key} 실패: {e}", file=sys.stderr)
                results[key] = {"error": str(e)}

    top_ls = results["top_ls"]
    global_ls = results["global_ls"]
    funding = results["funding"]
    oi = results["oi"]
    kimchi = results["kimchi"]

    sentiment = compute_sentiment_score(top_ls, funding, oi, kimchi)

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "source": "binance_futures + upbit (free, no API key)",
        "top_trader_long_short": top_ls,
        "global_long_short": global_ls,
        "funding_rate": funding,
        "open_interest": oi,
        "kimchi_premium": kimchi,
        "sentiment_score": sentiment,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
