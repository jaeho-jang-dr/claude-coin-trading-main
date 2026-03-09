"""
전략 에이전트 추상 기본 클래스

모든 전략 에이전트(보수적/보통/공격적)는 이 클래스를 상속하고,
자기만의 임계값과 규칙으로 매매 판단을 내린다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Decision:
    """매매 결정 결과."""
    decision: str           # "buy" | "sell" | "hold"
    confidence: float       # 0.0 ~ 1.0
    reason: str
    buy_score: dict         # 점수 내역
    trade_params: dict      # {"side": "bid"|"ask", "market": "KRW-BTC", "amount": int}
    external_signal: dict   # Data Fusion 결과
    agent_name: str         # 결정을 내린 에이전트
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S+09:00"))

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "buy_score": self.buy_score,
            "trade_params": self.trade_params,
            "external_signal_summary": {
                "total_score": self.external_signal.get("total_score", 0),
                "strategy_bonus": self.external_signal.get("strategy_bonus", 0),
                "fusion_signal": self.external_signal.get("fusion", {}).get("signal", "unknown"),
            },
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
        }


class BaseStrategyAgent(ABC):
    """전략 에이전트 공통 인터페이스."""

    # 서브클래스에서 반드시 정의
    name: str = ""
    emoji: str = ""
    description: str = ""

    # ── 매수 조건 임계값 (서브클래스에서 오버라이드) ──
    fgi_threshold: int = 30
    rsi_threshold: int = 30
    sma_deviation_pct: float = -5.0
    buy_score_threshold: int = 70
    macd_bonus: bool = False

    # ── 매도 조건 ──
    target_profit_pct: float = 15.0
    stop_loss_pct: float = -5.0
    forced_stop_loss_pct: float = -10.0
    sell_fgi_threshold: int = 75
    sell_rsi_threshold: int = 70

    # ── 매매 규모 ──
    max_trade_ratio: float = 0.10      # 총 자산 대비 1회 매매
    max_daily_trades: int = 3
    weekend_reduction: float = 0.50    # 주말 축소 비율
    dca_max_ratio: float = 0.50        # DCA 최대 비율

    # ── 점수 배점 ──
    fgi_points: int = 30
    rsi_points: int = 25
    sma_points: int = 25
    news_points: int = 20

    def calculate_buy_score(
        self,
        fgi: int,
        rsi: float,
        sma_deviation: float,
        news_negative: bool,
        external_bonus: int,
        macd_golden_cross: bool = False,
    ) -> dict:
        """매수 점수를 계산한다."""
        score = 0
        breakdown: dict = {}

        # 1) FGI
        if fgi <= self.fgi_threshold:
            pts = self.fgi_points
            if fgi <= self.fgi_threshold * 0.5:
                pts += 5  # 극단 보너스
            breakdown["fgi"] = {"score": pts, "value": fgi, "threshold": self.fgi_threshold}
            score += pts
        elif fgi <= self.fgi_threshold + 10:
            # 부분 충족: 근접
            pts = int(self.fgi_points * 0.5)
            breakdown["fgi"] = {"score": pts, "value": fgi, "partial": True}
            score += pts
        else:
            breakdown["fgi"] = {"score": 0, "value": fgi}

        # 2) RSI
        if rsi <= self.rsi_threshold:
            pts = self.rsi_points
            breakdown["rsi"] = {"score": pts, "value": round(rsi, 2), "threshold": self.rsi_threshold}
            score += pts
        elif rsi <= self.rsi_threshold + 5:
            pts = 15  # 부분 점수
            breakdown["rsi"] = {"score": pts, "value": round(rsi, 2), "partial": True}
            score += pts
        else:
            breakdown["rsi"] = {"score": 0, "value": round(rsi, 2)}

        # 3) SMA 이탈
        if sma_deviation <= self.sma_deviation_pct:
            pts = self.sma_points
            breakdown["sma"] = {"score": pts, "value": round(sma_deviation, 2), "threshold": self.sma_deviation_pct}
            score += pts
        elif sma_deviation <= self.sma_deviation_pct * 0.5:
            pts = 15
            breakdown["sma"] = {"score": pts, "value": round(sma_deviation, 2), "partial": True}
            score += pts
        else:
            breakdown["sma"] = {"score": 0, "value": round(sma_deviation, 2)}

        # 4) 뉴스 감성
        if not news_negative:
            breakdown["news"] = {"score": self.news_points, "negative": False}
            score += self.news_points
        else:
            breakdown["news"] = {"score": 0, "negative": True}

        # 5) MACD 보너스 (보통 전략만)
        if self.macd_bonus and macd_golden_cross:
            pts = 10
            breakdown["macd"] = {"score": pts, "golden_cross": True}
            score += pts

        # 6) 외부 지표 Data Fusion 보너스
        breakdown["external"] = {"score": external_bonus}
        score += external_bonus

        breakdown["total"] = score
        breakdown["threshold"] = self.buy_score_threshold
        breakdown["result"] = "buy" if score >= self.buy_score_threshold else "hold"

        return breakdown

    def evaluate_sell(
        self,
        profit_pct: float,
        current_fgi: int,
        current_rsi: float,
        buy_score: dict,
        ai_signal_score: int,
        drop_context: dict | None = None,
    ) -> dict | None:
        """
        매도 조건을 평가한다. 매도해야 하면 dict 반환, 아니면 None.

        drop_context 키:
          - cascade_risk (0~100): 캐스케이딩 하락 위험도
          - price_change_4h: 최근 4시간 가격 변동률(%)
          - volume_ratio: 최근 거래량/평균 거래량 비율
          - external_bearish_count: 외부 약세 지표 겹침 수
          - dca_already_done: 이미 DCA 1회 실행 여부
          - trend_falling: 하락 추세 지속 여부
        """
        dc = drop_context or {}

        # 목표 수익 달성
        if profit_pct >= self.target_profit_pct:
            # AI 시그널이 강세면 매도 유예 (1회만)
            if ai_signal_score > 20:
                return {
                    "action": "hold_defer",
                    "reason": f"목표 수익 {profit_pct:.1f}% 달성이나 AI 시그널 강세({ai_signal_score}) → 1회 유예",
                }
            return {
                "action": "sell",
                "reason": f"목표 수익 달성: {profit_pct:.1f}% >= {self.target_profit_pct}%",
                "type": "target_profit",
            }

        # FGI 과열
        if current_fgi >= self.sell_fgi_threshold:
            return {
                "action": "sell",
                "reason": f"FGI 과열: {current_fgi} >= {self.sell_fgi_threshold}",
                "type": "fgi_overbought",
            }

        # RSI 과매수
        if current_rsi >= self.sell_rsi_threshold:
            return {
                "action": "sell",
                "reason": f"RSI 과매수: {current_rsi:.1f} >= {self.sell_rsi_threshold}",
                "type": "rsi_overbought",
            }

        # 강제 손절 — 어떤 상황에서도 무조건 매도
        if profit_pct <= self.forced_stop_loss_pct:
            return {
                "action": "sell",
                "reason": f"강제 손절: {profit_pct:.1f}% <= {self.forced_stop_loss_pct}%",
                "type": "forced_stop",
            }

        # ── 강화된 하이브리드 손절 ──
        if profit_pct <= self.stop_loss_pct:
            conditions_met = sum(1 for k in ["fgi", "rsi", "sma", "news"]
                                 if buy_score.get(k, {}).get("score", 0) > 0)

            cascade_risk = dc.get("cascade_risk", 0)
            dca_already_done = dc.get("dca_already_done", False)
            volume_ratio = dc.get("volume_ratio", 1.0)
            external_bearish = dc.get("external_bearish_count", 0)
            price_change_4h = dc.get("price_change_4h", 0)
            trend_falling = dc.get("trend_falling", False)

            # ① DCA 이미 1회 실행 → 추가 물타기 금지
            if dca_already_done:
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), "
                        f"이미 DCA 1회 실행됨 → 추가 물타기 금지, 즉시 손절"
                    ),
                    "type": "dca_exhausted",
                }

            # ② 캐스케이딩 위험 극심 (70+) → 무조건 손절
            if cascade_risk >= 70:
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), "
                        f"캐스케이딩 위험 {cascade_risk}점 "
                        f"(4h {price_change_4h:+.1f}%, 거래량 {volume_ratio:.1f}x, "
                        f"약세지표 {external_bearish}개) → 즉시 손절"
                    ),
                    "type": "cascade_stop",
                }

            # ③ 캐스케이딩 위험 중간 (40~69) → 더 강한 바닥 근거 요구
            if cascade_risk >= 40:
                if conditions_met >= 4 and ai_signal_score >= 0:
                    return {
                        "action": "dca",
                        "reason": (
                            f"손절선 도달({profit_pct:.1f}%), "
                            f"캐스케이딩 위험 중간({cascade_risk}점)이나 "
                            f"강한 바닥 시그널 {conditions_met}개 + AI({ai_signal_score}) → 신중 DCA"
                        ),
                        "type": "hybrid_dca_cautious",
                    }
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), "
                        f"캐스케이딩 위험 {cascade_risk}점, "
                        f"바닥 시그널 부족({conditions_met}개) → 즉시 손절"
                    ),
                    "type": "cascade_moderate_stop",
                }

            # ④ 외부 약세 지표 3개+ 겹침 → 바닥 시그널 있어도 매도
            if external_bearish >= 3 and conditions_met >= 3:
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), "
                        f"바닥 시그널 {conditions_met}개이나 "
                        f"외부 약세 지표 {external_bearish}개 동시 겹침 "
                        f"(고래매도+펀딩+롱과밀+뉴스 등) → 즉시 손절"
                    ),
                    "type": "external_bearish_override",
                }

            # ⑤ 하락 추세 지속 중 → DCA 요건 강화
            if trend_falling and conditions_met >= 3 and ai_signal_score >= 0:
                if conditions_met >= 4 or ai_signal_score >= 15:
                    return {
                        "action": "dca",
                        "reason": (
                            f"손절선 도달({profit_pct:.1f}%), "
                            f"하락 추세 지속이나 강한 바닥 시그널 "
                            f"{conditions_met}개 + AI({ai_signal_score}) → 신중 DCA"
                        ),
                        "type": "hybrid_dca_trend",
                    }
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), "
                        f"하락 추세 지속 + 바닥 시그널 {conditions_met}개 "
                        f"부족(추세 하락 시 4개+ 필요) → 즉시 손절"
                    ),
                    "type": "trend_stop",
                }

            # ⑥ 기본 하이브리드 (캐스케이딩/추세 위험 낮을 때)
            if conditions_met >= 3 and ai_signal_score >= 0:
                return {
                    "action": "dca",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%)이나 "
                        f"바닥 시그널 {conditions_met}개 + AI({ai_signal_score}) → DCA"
                    ),
                    "type": "hybrid_dca",
                }
            elif conditions_met >= 3 and ai_signal_score < -20:
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), 바닥 시그널이나 "
                        f"AI 매도 압력 극심({ai_signal_score}) → 즉시 손절"
                    ),
                    "type": "hybrid_forced",
                }
            else:
                return {
                    "action": "sell",
                    "reason": (
                        f"손절선 도달({profit_pct:.1f}%), "
                        f"바닥 시그널 부족({conditions_met}개) → 즉시 손절"
                    ),
                    "type": "stop_loss",
                }

        return None

    @abstractmethod
    def decide(
        self,
        market_data: dict,
        external_signal: dict,
        portfolio: dict,
        drop_context: dict | None = None,
    ) -> Decision:
        """시장 데이터와 외부 시그널을 받아 매매 결정을 내린다."""
        ...

    def _extract_indicators(self, market_data: dict) -> dict:
        """market_data JSON에서 주요 지표를 추출한다."""
        indicators = market_data.get("indicators", {})
        ticker = market_data.get("ticker", {})

        return {
            "current_price": ticker.get("trade_price", 0),
            "rsi": indicators.get("rsi_14", 50),
            "sma20": indicators.get("sma_20", 0),
            "sma_deviation": indicators.get("sma_20_deviation_pct", 0),
            "macd": indicators.get("macd", {}),
            "bollinger": indicators.get("bollinger", {}),
            "price_change_24h": ticker.get("signed_change_rate", 0) * 100,
        }

    def _is_weekend(self) -> bool:
        import datetime
        return datetime.datetime.now().weekday() >= 5

    def _calculate_trade_amount(self, total_krw: float) -> int:
        """1회 매매 금액을 계산한다."""
        amount = int(total_krw * self.max_trade_ratio)
        if self._is_weekend():
            amount = int(amount * (1 - self.weekend_reduction))
        # MAX_TRADE_AMOUNT 안전장치
        import os
        max_amount = int(os.getenv("MAX_TRADE_AMOUNT", "100000"))
        return min(amount, max_amount)
