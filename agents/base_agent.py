"""
전략 에이전트 추상 기본 클래스

모든 전략 에이전트(보수적/보통/공격적)는 이 클래스를 상속하고,
자기만의 임계값과 규칙으로 매매 판단을 내린다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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
            # 극공포 추가 보너스: FGI ≤ 20이면 "공포 속 반등" 가산
            # (역사적으로 극공포 구간은 최고의 매수 기회)
            if fgi <= 20:
                pts += 5
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

    # ── 포지션 과다 분할 매도 ──
    overweight_profit_threshold: float = 5.0   # 포지션 과다 시 분할 매도 트리거 수익률(%)
    overweight_sell_ratio: float = 1 / 3       # 분할 매도 비율 (보유량의 1/3)

    def evaluate_sell(
        self,
        profit_pct: float,
        current_fgi: int,
        current_rsi: float,
        buy_score: dict,
        ai_signal_score: int,
        drop_context: dict | None = None,
        btc_position_ratio: float = 0.0,
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

        # ── 포지션 과다 분할 매도 (일반 목표 수익보다 우선) ──
        max_position = float(os.getenv("MAX_POSITION_RATIO", "0.5"))
        if btc_position_ratio > max_position and profit_pct >= self.overweight_profit_threshold:
            return {
                "action": "sell_partial",
                "reason": (
                    f"포지션 과다 분할 매도: BTC 비중 {btc_position_ratio:.0%} > "
                    f"{max_position:.0%} 한도, 수익률 +{profit_pct:.1f}% >= "
                    f"+{self.overweight_profit_threshold}% → 보유량 1/3 매도"
                ),
                "type": "overweight_rebalance",
                "sell_ratio": self.overweight_sell_ratio,
            }

        # 목표 수익 달성
        if profit_pct >= self.target_profit_pct:
            # AI 시그널이 강세면 매도 유예 (1회만)
            if ai_signal_score > 20:
                # 이전 사이클에서 이미 유예했는지 확인 (무한 유예 방지)
                state_file = Path(__file__).resolve().parent.parent / "data" / "agent_state.json"
                already_deferred = False
                try:
                    with open(state_file, encoding="utf-8") as _sf:
                        _st = json.load(_sf)
                        already_deferred = _st.get("deferred_target_profit", False)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if already_deferred:
                    # 이미 1회 유예함 → 매도 실행, 플래그 초기화
                    try:
                        with open(state_file, encoding="utf-8") as _sf:
                            _st = json.load(_sf)
                        _st["deferred_target_profit"] = False
                        with open(state_file, "w", encoding="utf-8") as _sf:
                            json.dump(_st, _sf, ensure_ascii=False, indent=2)
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    return {
                        "action": "sell",
                        "reason": f"목표 수익 {profit_pct:.1f}% 달성, AI 시그널 강세({ai_signal_score})이나 이미 1회 유예 완료 → 매도",
                        "type": "target_profit",
                    }
                # 첫 유예 → 플래그 설정
                try:
                    with open(state_file, encoding="utf-8") as _sf:
                        _st = json.load(_sf)
                    _st["deferred_target_profit"] = True
                    with open(state_file, "w", encoding="utf-8") as _sf:
                        json.dump(_st, _sf, ensure_ascii=False, indent=2)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
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

        # 강제 손절 -- 어떤 상황에서도 무조건 매도
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

    def save_buy_score_detail(
        self,
        decision: Decision,
        market_data: dict,
    ) -> str | None:
        """매수 점수 상세 내역을 buy_score_detail 테이블에 저장한다.

        DB 저장 실패가 매매 로직에 영향을 주지 않도록 전체를 try/except로 감싼다.
        """
        try:
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
            if not supabase_url or not supabase_key:
                return

            bs = decision.buy_score or {}
            fgi_obj = bs.get("fgi", {}) if isinstance(bs.get("fgi"), dict) else {}
            rsi_obj = bs.get("rsi", {}) if isinstance(bs.get("rsi"), dict) else {}
            sma_obj = bs.get("sma", {}) if isinstance(bs.get("sma"), dict) else {}
            news_obj = bs.get("news", {}) if isinstance(bs.get("news"), dict) else {}
            ext_obj = bs.get("external", {}) if isinstance(bs.get("external"), dict) else {}

            indicators = market_data.get("indicators", {})
            ticker = market_data.get("ticker", {})

            # SMA position description
            sma_val = sma_obj.get("value", 0)
            sma_position = "above" if sma_val >= 0 else f"below_{abs(sma_val):.1f}%"

            # News sentiment from market_data
            news_data = market_data.get("news", {})
            news_sentiment_str = news_data.get("overall_sentiment", "neutral")
            news_sentiment_val = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(
                news_sentiment_str, 0.0
            )

            # External signal summary
            ext_signal = decision.external_signal or {}
            fusion = ext_signal.get("fusion", {})
            ext_signal_text = fusion.get("signal", "unknown") if fusion else "unknown"

            # Trade params
            tp = decision.trade_params or {}
            buy_amount = tp.get("amount") if decision.decision == "buy" else None
            sell_pct = None
            if decision.decision == "sell" and tp.get("volume"):
                sell_pct = 100.0  # full sell

            # ADX regime
            adx_regime = indicators.get("adx_regime", None)

            # Price change as market trend
            price_change = ticker.get("signed_change_rate", 0) * 100
            if price_change > 2:
                market_trend = "bullish"
            elif price_change < -2:
                market_trend = "bearish"
            else:
                market_trend = "sideways"

            # cycle_id 생성
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from scripts.cycle_id import get_or_create_cycle_id
                _cycle_id = get_or_create_cycle_id("agent")
            except Exception:
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _cycle_id = _dt.now(_tz(_td(hours=9))).strftime("%Y%m%d-%H%M") + "-agent"

            total_score = bs.get("total", 0)

            row = {
                "cycle_id": _cycle_id,
                "agent_type": self.name,
                "threshold": self.buy_score_threshold,
                "fgi_score": fgi_obj.get("score", 0),
                "fgi_value": fgi_obj.get("value"),
                "rsi_score": rsi_obj.get("score", 0),
                "rsi_value": rsi_obj.get("value"),
                "sma_score": sma_obj.get("score", 0),
                "sma_position": sma_position,
                "news_score": news_obj.get("score", 0),
                "news_sentiment": news_sentiment_val,
                "external_bonus": ext_obj.get("score", 0),
                "external_signal": ext_signal_text,
                "total_score": total_score,
                "action": decision.decision,
                "buy_amount": buy_amount,
                "sell_pct": sell_pct,
                "confidence": round(decision.confidence, 2),
                "reason": decision.reason,
                "btc_price": ticker.get("trade_price"),
                "market_trend": market_trend,
                "adx_regime": adx_regime,
                # 니어미스 + AI 거부권 + 감독 오버라이드 추적
                "points_from_threshold": round(total_score - self.buy_score_threshold, 2),
                "is_near_miss": abs(total_score - self.buy_score_threshold) <= 5,
                "price_at_decision": int(market_data.get("current_price", 0) or ticker.get("trade_price", 0) or 0) or None,
                "was_ai_vetoed": getattr(decision, '_was_ai_vetoed', False),
                "ai_veto_reason": getattr(decision, '_ai_veto_reason', None),
                "orchestrator_override": getattr(decision, '_orchestrator_override', False),
                "override_reason": getattr(decision, '_override_reason', None),
                "original_action": getattr(decision, '_original_action', None),
            }

            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }

            r = requests.post(
                f"{supabase_url}/rest/v1/buy_score_detail",
                headers=headers,
                json=row,
                timeout=10,
            )
            if r.ok:
                # 저장된 레코드의 ID를 반환하여 decisions와 연결
                try:
                    resp_data = r.json()
                    if isinstance(resp_data, list) and resp_data:
                        return resp_data[0].get("id")
                    elif isinstance(resp_data, dict):
                        return resp_data.get("id")
                except Exception:
                    pass
            else:
                print(
                    f"[base_agent] buy_score_detail INSERT 실패 ({r.status_code}): {r.text[:300]}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[base_agent] buy_score_detail 저장 예외: {e}", file=sys.stderr)
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

    def _calculate_trade_amount(self, total_krw: float, external_bonus: int = 0) -> int:
        """1회 매매 금액을 계산한다."""
        amount = int(total_krw * self.max_trade_ratio)
        if self._is_weekend():
            if external_bonus >= 10:
                pass  # 주말이더라도 외부 점수가 높으면 축소 룰 무효화 (불장/호재)
            else:
                amount = int(amount * (1 - self.weekend_reduction))
        # MAX_TRADE_AMOUNT 안전장치
        import os
        max_amount = int(os.getenv("MAX_TRADE_AMOUNT", "100000"))
        return min(amount, max_amount)
