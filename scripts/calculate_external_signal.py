#!/usr/bin/env python3
"""
외부 지표 종합 점수 산출 (calculate_external_signal)

입력: 파이프라인에서 수집한 3개 JSON 파일
  - whale_tracker.json    (블록체인 온체인 고래)
  - binance_sentiment.json (롱숏/펀딩비/OI/김치프리미엄)
  - ai_signal.json        (Upbit 호가/체결/거래소내 고래)

출력: JSON (stdout) — 종합 외부 시그널 점수 + Data Fusion 판정

점수 체계:
  총점 범위: -100 ~ +100
  양수 = 매수 우호적 (외부 환경이 강세)
  음수 = 매도 우호적 (외부 환경이 약세)

사용법:
    python3 scripts/calculate_external_signal.py data/snapshots/20260309_153000/
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def load_json(path: Path) -> dict:
    """JSON 파일 로드. 실패 시 빈 dict 반환."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[external_signal] {path.name} 로드 실패: {e}", file=sys.stderr)
        return {}


# ── 개별 점수 산출 함수 ───────────────────────────────


def score_onchain_whale(whale_data: dict) -> dict:
    """
    블록체인 온체인 고래 활동 → -15 ~ +15
    이제 방향 추정 포함: 거래소 입출금 패턴 분석.
    """
    ws = whale_data.get("whale_score", {})
    raw_score = ws.get("score", 0)  # -20 ~ +20
    activity = ws.get("whale_activity_level", "low")
    direction = ws.get("direction", "neutral")
    net_flow = ws.get("net_flow_btc", 0)

    # -20~+20을 -15~+15으로 리스케일 (비중 상향: 방향성 추가됨)
    score = max(-15, min(15, int(raw_score * 0.75)))

    return {
        "name": "onchain_whale",
        "score": score,
        "max": 15,
        "activity_level": activity,
        "direction": direction,
        "net_flow_btc": net_flow,
        "detail": ws.get("reasons", []),
    }


def score_binance_derivatives(binance_data: dict) -> dict:
    """
    바이낸스 파생상품 심리 → -30 ~ +30

    핵심 로직:
    - 롱 과밀 + 양수 펀딩 = 시장 과열 → 음수 (조정 경고)
    - 숏 과밀 + 음수 펀딩 = 시장 비관 → 양수 (반등 가능 = 매수 우호)
    - OI 급증 = 레버리지 축적 → 변동성 증폭 신호
    """
    score = 0
    details: list[str] = []

    # 1) 롱/숏 비율 (±10)
    top_ls = binance_data.get("top_trader_long_short", {})
    lr = top_ls.get("current_ratio", 1.0)
    if isinstance(lr, (int, float)):
        if lr > 1.5:
            score -= 10
            details.append(f"롱 과밀 ({lr:.2f}) → 조정 경고")
        elif lr > 1.2:
            score -= 5
            details.append(f"롱 우세 ({lr:.2f})")
        elif lr < 0.7:
            score += 10
            details.append(f"숏 과밀 ({lr:.2f}) → 숏스퀴즈 가능")
        elif lr < 0.85:
            score += 5
            details.append(f"숏 우세 ({lr:.2f})")

    # 2) 펀딩비 (±10)
    funding = binance_data.get("funding_rate", {})
    rate = funding.get("current_rate", 0)
    if isinstance(rate, (int, float)):
        if rate > 0.001:
            score -= 10
            details.append(f"극단 양수 펀딩 ({rate*100:.3f}%) → 롱 과열")
        elif rate > 0.0005:
            score -= 5
            details.append(f"양수 펀딩 ({rate*100:.3f}%)")
        elif rate < -0.001:
            score += 10
            details.append(f"극단 음수 펀딩 ({rate*100:.3f}%) → 숏 과열, 반등 가능")
        elif rate < -0.0005:
            score += 5
            details.append(f"음수 펀딩 ({rate*100:.3f}%)")

    # 3) 김치 프리미엄 (±10) — 한국 시장 특화 핵심 지표
    kimchi = binance_data.get("kimchi_premium", {})
    premium = kimchi.get("premium_pct", 0)
    if isinstance(premium, (int, float)):
        if premium > 5.0:
            score -= 10
            details.append(f"극단 김치P ({premium:.1f}%) → 국내 FOMO 과열")
        elif premium > 3.0:
            score -= 5
            details.append(f"높은 김치P ({premium:.1f}%)")
        elif premium < -3.0:
            score += 10
            details.append(f"깊은 디스카운트 ({premium:.1f}%) → 매수 기회")
        elif premium < -1.0:
            score += 5
            details.append(f"디스카운트 ({premium:.1f}%)")

    score = max(-30, min(30, score))

    return {
        "name": "binance_derivatives",
        "score": score,
        "max": 30,
        "long_short_ratio": lr if isinstance(lr, (int, float)) else None,
        "funding_rate_pct": round(rate * 100, 4) if isinstance(rate, (int, float)) else None,
        "kimchi_premium_pct": premium if isinstance(premium, (int, float)) else None,
        "details": details,
    }


def score_exchange_whale(ai_signal_data: dict) -> dict:
    """
    거래소 내부 고래 체결 (Upbit 1000만원+) → -15 ~ +15
    """
    whale = ai_signal_data.get("details", {}).get("whale_detection", {})
    wb = whale.get("whale_buy", 0)
    ws = whale.get("whale_sell", 0)
    mega_b = whale.get("mega_whale_buy", 0)
    mega_s = whale.get("mega_whale_sell", 0)
    total = wb + ws

    if total < 2:
        return {
            "name": "exchange_whale",
            "score": 0,
            "max": 15,
            "details": ["거래소 내 고래 체결 부족 (2건 미만)"],
        }

    # 대형 고래(5000만원+)는 가중치 2배
    weighted_buy = wb + mega_b
    weighted_sell = ws + mega_s
    total_weighted = weighted_buy + weighted_sell

    if total_weighted == 0:
        ratio = 0
    else:
        ratio = (weighted_buy - weighted_sell) / total_weighted  # -1 ~ +1

    score = int(round(ratio * 15))
    score = max(-15, min(15, score))

    details = [f"고래 매수 {wb}건 vs 매도 {ws}건 (1000만원+ 기준)"]
    if mega_b + mega_s > 0:
        details.append(f"대형고래 매수 {mega_b}건 vs 매도 {mega_s}건 (5000만원+)")

    return {
        "name": "exchange_whale",
        "score": score,
        "max": 15,
        "details": details,
    }


def score_market_microstructure(ai_signal_data: dict) -> dict:
    """
    AI 복합 시그널 (호가 불균형 + 체결 강도 + 다이버전스 + 변동성 + 거래량)
    → -15 ~ +15
    AI 복합 시그널은 프롬프트에도 직접 주입되므로 이중 계산 완화 위해 비중 하향.
    """
    composite = ai_signal_data.get("ai_composite_signal", {})
    raw_score = composite.get("score", 0)  # -100 ~ +100

    # -100~+100을 -15~+15로 리스케일
    score = max(-15, min(15, int(raw_score * 0.15)))

    return {
        "name": "market_microstructure",
        "score": score,
        "max": 15,
        "raw_composite_score": raw_score,
        "interpretation": composite.get("interpretation", "unknown"),
        "components": [c.get("name", "") for c in composite.get("components", [])],
    }


# ── Data Fusion 종합 ─────────────────────────────────


def calculate_external_signal(
    whale_data: dict,
    binance_data: dict,
    ai_signal_data: dict,
) -> dict:
    """
    모든 외부 지표를 종합하여 단일 점수와 Data Fusion 판정을 산출한다.

    총점: -75 ~ +75 (이론적 최대/최소)
    가중치 배분:
      - 바이낸스 파생상품:  ±30 (최대 비중 — 스마트머니 + 김치P)
      - 거래소 내부 고래:   ±15 (실시간 매매 압력, 방향성 있음)
      - 시장 미시구조:      ±15 (호가/체결/거래량, 이중 계산 완화)
      - 온체인 고래:        ±15 (활동량 + 거래소 입출금 방향 추정)
    """
    onchain = score_onchain_whale(whale_data)
    derivatives = score_binance_derivatives(binance_data)
    exchange = score_exchange_whale(ai_signal_data)
    micro = score_market_microstructure(ai_signal_data)

    components = [derivatives, micro, exchange, onchain]
    total_score = sum(c["score"] for c in components)
    max_possible = sum(c["max"] for c in components)

    # Data Fusion 신호 겹침 판정
    bearish_count = sum(1 for c in components if c["score"] < -3)
    bullish_count = sum(1 for c in components if c["score"] > 3)

    if bullish_count >= 3:
        fusion_signal = "strong_buy"
        fusion_note = f"{bullish_count}개 지표 동시 강세 — 매수 가중치 강화"
    elif bullish_count >= 2:
        fusion_signal = "buy"
        fusion_note = f"{bullish_count}개 지표 강세 겹침"
    elif bearish_count >= 3:
        fusion_signal = "strong_sell"
        fusion_note = f"{bearish_count}개 지표 동시 약세 — 매도/조정 경고"
    elif bearish_count >= 2:
        fusion_signal = "sell"
        fusion_note = f"{bearish_count}개 지표 약세 겹침"
    elif bullish_count == 1 and bearish_count == 1:
        fusion_signal = "mixed"
        fusion_note = "강세/약세 혼재 — 방향성 불분명"
    else:
        fusion_signal = "neutral"
        fusion_note = "외부 지표 중립"

    # 전략 보너스 점수 산출 (strategy.md 점수제에 직접 적용, ±20점)
    if total_score >= 30:
        strategy_bonus = 20
    elif total_score >= 20:
        strategy_bonus = 15
    elif total_score >= 10:
        strategy_bonus = 10
    elif total_score >= 5:
        strategy_bonus = 5
    elif total_score <= -30:
        strategy_bonus = -20
    elif total_score <= -20:
        strategy_bonus = -15
    elif total_score <= -10:
        strategy_bonus = -10
    elif total_score <= -5:
        strategy_bonus = -5
    else:
        strategy_bonus = 0

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "total_score": total_score,
        "max_possible": max_possible,
        "strategy_bonus": strategy_bonus,
        "strategy_bonus_note": (
            f"매수 점수에 +{strategy_bonus}점 적용"
            if strategy_bonus > 0
            else f"매수 점수에 {strategy_bonus}점 적용"
            if strategy_bonus < 0
            else "보너스 없음 (중립)"
        ),
        "fusion": {
            "signal": fusion_signal,
            "bullish_indicators": bullish_count,
            "bearish_indicators": bearish_count,
            "note": fusion_note,
        },
        "components": {c["name"]: c for c in components},
    }


# ── 메인 ──────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python3 scripts/calculate_external_signal.py <snapshot_dir>", file=sys.stderr)
        sys.exit(1)

    snapshot_dir = Path(sys.argv[1])

    whale_data = load_json(snapshot_dir / "whale_tracker.json")
    binance_data = load_json(snapshot_dir / "binance_sentiment.json")
    ai_signal_data = load_json(snapshot_dir / "ai_signal.json")

    result = calculate_external_signal(whale_data, binance_data, ai_signal_data)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)
