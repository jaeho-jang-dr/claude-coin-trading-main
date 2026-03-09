"""
Orchestrator — 감독 에이전트

시장 상황을 평가하고, 적절한 전략 에이전트를 선택/교체한다.
DB에 전환 이력을 저장하고, 과거 성과를 학습하여 판단을 개선한다.

설계 원칙:
  - 보수적↔공격적 직행 허용 (급변 시장 대응)
  - 진입 조건 탄력적 (다중 시그널 가중 평가)
  - 전환 쿨다운 상황별 차등 (긴급 상황은 즉시 전환)
  - DB 학습: 같은 상황에서 과거 전환이 좋았는지/나빴는지 참고
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agents.conservative import ConservativeAgent
from agents.moderate import ModerateAgent
from agents.aggressive import AggressiveAgent
from agents.base_agent import BaseStrategyAgent, Decision


PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_DIR / "data" / "agent_state.json"
AUTO_EMERGENCY_FILE = PROJECT_DIR / "data" / "auto_emergency.json"

AGENTS = {
    "conservative": ConservativeAgent,
    "moderate": ModerateAgent,
    "aggressive": AggressiveAgent,
}


def _load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "active_agent": "conservative",
            "last_switch_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "switch_history": [],
        }


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


class Orchestrator:
    """감독 에이전트: 시장 상황에 따라 전략 에이전트를 자율 교체한다."""

    # ── 시장 국면 분류 기준 ──
    PHASE_EXTREME_FEAR = "extreme_fear"   # FGI ≤ 20
    PHASE_FEAR = "fear"                   # FGI 21~35
    PHASE_NEUTRAL = "neutral"             # FGI 36~60
    PHASE_GREED = "greed"                 # FGI 61~80
    PHASE_EXTREME_GREED = "extreme_greed" # FGI > 80

    def __init__(self):
        self.state = _load_state()
        self._active_agent_name = self.state.get("active_agent", "conservative")
        self._switch_reason = ""
        self._learning_data = None  # DB에서 로드한 학습 데이터

    @property
    def active_agent(self) -> BaseStrategyAgent:
        cls = AGENTS.get(self._active_agent_name, ConservativeAgent)
        return cls()

    def run(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        past_decisions: list[dict] | None = None,
    ) -> dict:
        """
        전체 실행 흐름:
        1. DB에서 과거 전환 성과 학습
        2. 시장 국면 분류
        3. 전략 전환 필요 여부 평가
        4. 활성 에이전트에게 매매 판단 위임
        5. 전환 이력 DB 저장
        """
        # ── 긴급정지 확인 (사용자 수동 > 감독 자동) ──
        if os.getenv("EMERGENCY_STOP", "false").lower() == "true":
            return {
                "active_agent": f"🚨 긴급정지",
                "decision": {
                    "decision": "hold",
                    "reason": "사용자 EMERGENCY_STOP 활성화 — 모든 매매 차단",
                    "confidence": 1.0,
                    "buy_score": {},
                    "trade_params": {},
                    "external_signal_summary": {},
                    "agent_name": "🚨 사용자 긴급정지",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                },
                "switch": None,
                "market_state": {},
            }

        # 감독 자동 긴급정지 확인
        auto_em = self._check_auto_emergency_active()
        if auto_em:
            # 자동 해제 조건 확인
            can_lift = self._can_lift_auto_emergency(
                market_data, external_data, portfolio
            )
            if can_lift:
                self._deactivate_auto_emergency(
                    "시장 안정화 확인 — 자동 긴급정지 해제"
                )
            else:
                return {
                    "active_agent": f"🚨 감독 자동긴급정지",
                    "decision": {
                        "decision": "hold",
                        "reason": (
                            f"감독 자동 긴급정지 활성 중 "
                            f"(사유: {auto_em.get('reason', '?')}, "
                            f"발동: {auto_em.get('activated_at', '?')})"
                        ),
                        "confidence": 1.0,
                        "buy_score": {},
                        "trade_params": {},
                        "external_signal_summary": {},
                        "agent_name": "🚨 감독 자동긴급정지",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                    },
                    "switch": None,
                    "market_state": {},
                    "auto_emergency": auto_em,
                }

        # 학습 데이터 로드
        self._learning_data = self._load_learning_data()

        external_signal = external_data.get("external_signal", {})

        # 사용자 피드백 반영
        self._apply_feedback(external_data.get("sources", {}).get("user_feedback", []))

        # 성과 리뷰 반영
        self._performance = external_data.get("sources", {}).get("performance_review", {})

        # 시장 상태 종합 평가
        market_state = self._evaluate_market_state(
            market_data, external_data, portfolio, past_decisions or []
        )

        # 전략 전환 평가
        switch_result = self._evaluate_switch(market_state)

        if switch_result:
            self._do_switch(switch_result)
            # DB에 전환 기록
            self._record_switch_to_db(switch_result, market_state)

        # 하락 컨텍스트 구축 (매도 vs DCA 판단의 핵심)
        drop_context = self._build_drop_context(market_data, external_data, portfolio)

        # 자동 긴급정지 발동 여부 평가
        emergency_trigger = self._evaluate_auto_emergency(
            market_state, drop_context, portfolio
        )
        if emergency_trigger:
            self._activate_auto_emergency(emergency_trigger, market_state)
            # 보유 중이면 강제 전량 매도 지시
            btc = portfolio.get("btc", {})
            if btc.get("balance", 0) > 0:
                from agents.base_agent import Decision as Dec
                return {
                    "active_agent": "🚨 감독 자동긴급정지",
                    "switch": None,
                    "decision": Dec(
                        decision="sell",
                        confidence=0.95,
                        reason=f"[감독 자동긴급정지] {emergency_trigger['reason']} → 전량 매도",
                        buy_score={},
                        trade_params={
                            "side": "ask",
                            "market": "KRW-BTC",
                            "volume": btc.get("balance", 0),
                        },
                        external_signal=external_signal,
                        agent_name="🚨 감독 자동긴급정지",
                    ).to_dict(),
                    "market_state": market_state,
                    "drop_context": drop_context,
                    "auto_emergency": emergency_trigger,
                }
            else:
                return {
                    "active_agent": "🚨 감독 자동긴급정지",
                    "switch": None,
                    "decision": {
                        "decision": "hold",
                        "reason": f"[감독 자동긴급정지] {emergency_trigger['reason']}",
                        "confidence": 1.0,
                        "buy_score": {},
                        "trade_params": {},
                        "external_signal_summary": {},
                        "agent_name": "🚨 감독 자동긴급정지",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                    },
                    "market_state": market_state,
                    "drop_context": drop_context,
                    "auto_emergency": emergency_trigger,
                }

        # 활성 에이전트에게 위임 (drop_context 전달)
        agent = self.active_agent
        decision = agent.decide(market_data, external_signal, portfolio,
                                drop_context=drop_context)

        # 감독 오버라이드: 에이전트 결정을 최종 검증
        decision = self._override_decision(decision, drop_context, market_state)

        # DCA 이력 추적
        self._track_dca(decision)

        # 상태 업데이트
        if decision.decision in ("buy", "sell"):
            self.state["last_trade_time"] = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")
            _save_state(self.state)

        return {
            "active_agent": f"{agent.emoji} {agent.name}",
            "switch": switch_result,
            "decision": decision.to_dict(),
            "market_state": market_state,
            "drop_context": drop_context,
        }

    # ── 시장 상태 종합 평가 ────────────────────────────

    def _evaluate_market_state(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        past_decisions: list[dict],
    ) -> dict:
        """시장의 모든 지표를 하나의 상태 객체로 정리한다."""
        fgi = self._get_fgi(external_data)
        rsi = market_data.get("indicators", {}).get("rsi_14", 50)
        price_change_24h = market_data.get("ticker", {}).get("signed_change_rate", 0) * 100
        btc_ratio = portfolio.get("btc_ratio", 0)
        consecutive_losses = self._count_consecutive_losses(past_decisions)
        self.state["consecutive_losses"] = consecutive_losses

        fusion = external_data.get("external_signal", {}).get("fusion", {})
        fusion_signal = fusion.get("signal", "neutral")
        fusion_score = external_data.get("external_signal", {}).get("total_score", 0)

        bs = external_data.get("sources", {}).get("binance_sentiment", {})
        kimchi_pct = bs.get("kimchi_premium", {}).get("premium_pct", 0)
        ls_ratio = bs.get("top_trader_long_short", {}).get("current_ratio", 1.0)
        funding_rate = bs.get("funding_rate", {}).get("current_rate", 0)

        # 매크로 점수
        macro = external_data.get("sources", {}).get("macro", {}).get("analysis", {})
        macro_score = macro.get("macro_score", 0)
        macro_sentiment = macro.get("sentiment", "neutral")

        # ETH/BTC z-score
        eth_btc = external_data.get("sources", {}).get("eth_btc", {})
        eth_btc_z = eth_btc.get("eth_btc_z_score", 0)

        # 뉴스 감성
        news_sent = external_data.get("sources", {}).get("news_sentiment", {})
        news_sentiment = news_sent.get("overall_sentiment", "neutral")

        # 시장 국면 분류
        phase = self._classify_phase(fgi)

        # 위험도 점수 (0~100, 높을수록 위험)
        danger_score = self._calculate_danger_score(
            fgi=fgi, rsi=rsi, price_change_24h=price_change_24h,
            kimchi_pct=kimchi_pct, ls_ratio=ls_ratio, btc_ratio=btc_ratio,
            consecutive_losses=consecutive_losses,
            macro_score=macro_score,
            news_sentiment=news_sentiment,
        )

        # 기회 점수 (0~100, 높을수록 매수 기회)
        opportunity_score = self._calculate_opportunity_score(
            fgi=fgi, rsi=rsi, price_change_24h=price_change_24h,
            fusion_signal=fusion_signal, fusion_score=fusion_score,
            funding_rate=funding_rate, kimchi_pct=kimchi_pct,
            macro_score=macro_score,
            news_sentiment=news_sentiment,
        )

        return {
            "phase": phase,
            "fgi": fgi,
            "rsi": rsi,
            "price_change_24h": price_change_24h,
            "btc_ratio": btc_ratio,
            "consecutive_losses": consecutive_losses,
            "fusion_signal": fusion_signal,
            "fusion_score": fusion_score,
            "kimchi_pct": kimchi_pct,
            "ls_ratio": ls_ratio,
            "funding_rate": funding_rate,
            "macro_score": macro_score,
            "macro_sentiment": macro_sentiment,
            "eth_btc_z": eth_btc_z,
            "news_sentiment": news_sentiment,
            "danger_score": danger_score,
            "opportunity_score": opportunity_score,
        }

    def _classify_phase(self, fgi: int) -> str:
        if fgi <= 20:
            return self.PHASE_EXTREME_FEAR
        elif fgi <= 35:
            return self.PHASE_FEAR
        elif fgi <= 60:
            return self.PHASE_NEUTRAL
        elif fgi <= 80:
            return self.PHASE_GREED
        else:
            return self.PHASE_EXTREME_GREED

    def _calculate_danger_score(self, *, fgi, rsi, price_change_24h,
                                 kimchi_pct, ls_ratio, btc_ratio,
                                 consecutive_losses, macro_score=0,
                                 news_sentiment="neutral") -> int:
        """위험도 점수. 높을수록 보수적으로 전환해야 한다."""
        score = 0

        # 연속 손절 (최대 30점)
        score += min(consecutive_losses * 10, 30)

        # BTC 과다 보유 (최대 20점)
        if btc_ratio > 0.3:
            score += int((btc_ratio - 0.3) * 100)  # 0.3→0, 0.5→20, 0.7→40

        # 급락 (최대 25점)
        if price_change_24h < -3:
            score += min(int(abs(price_change_24h) * 5), 25)

        # 김치 프리미엄 과열 (최대 15점)
        if kimchi_pct > 3:
            score += min(int((kimchi_pct - 3) * 5), 15)

        # 롱 과밀 (최대 10점)
        if ls_ratio > 1.2:
            score += min(int((ls_ratio - 1.2) * 20), 10)

        # 매크로 약세 (최대 15점)
        if macro_score < -10:
            score += min(int(abs(macro_score) * 0.5), 15)

        # 뉴스 부정 (최대 10점)
        if news_sentiment in ("negative",):
            score += 10
        elif news_sentiment in ("slightly_negative",):
            score += 5

        return min(score, 100)

    def _calculate_opportunity_score(self, *, fgi, rsi, price_change_24h,
                                      fusion_signal, fusion_score,
                                      funding_rate, kimchi_pct,
                                      macro_score=0, news_sentiment="neutral") -> int:
        """기회 점수. 높을수록 공격적으로 전환할 근거가 있다."""
        score = 0

        # 극단적 공포 (최대 25점)
        if fgi <= 25:
            score += 25 - fgi  # FGI 0→25점, FGI 25→0점

        # RSI 과매도 (최대 20점)
        if rsi < 35:
            score += int((35 - rsi) * 1.3)

        # 반등 중 (최대 15점)
        if price_change_24h > 1:
            score += min(int(price_change_24h * 5), 15)

        # Data Fusion 강세 (최대 20점)
        if fusion_signal == "strong_buy":
            score += 20
        elif fusion_signal == "buy":
            score += 10
        elif fusion_score > 10:
            score += 5

        # 숏 과밀 (역발상 기회, 최대 10점)
        if funding_rate < -0.01:
            score += 10
        elif funding_rate < 0:
            score += 5

        # 김치 디스카운트 (최대 10점)
        if kimchi_pct < -1:
            score += min(int(abs(kimchi_pct) * 3), 10)

        # 매크로 강세 (최대 10점)
        if macro_score > 10:
            score += min(int(macro_score * 0.5), 10)

        # 뉴스 긍정 (최대 8점)
        if news_sentiment in ("positive",):
            score += 8
        elif news_sentiment in ("slightly_positive",):
            score += 4

        return min(score, 100)

    # ── 전략 전환 평가 ────────────────────────────────

    def _evaluate_switch(self, ms: dict) -> dict | None:
        """시장 상태를 보고 전략 전환이 필요한지 판단한다."""
        current = self._active_agent_name

        # 쿨다운 체크 (긴급 상황은 면제)
        is_emergency = ms["danger_score"] >= 70 or ms["price_change_24h"] < -7
        if not is_emergency and self._is_on_cooldown():
            return None

        # 데이터 수집 실패가 많으면 전환 금지
        # (market_state에는 없으므로 state에서 체크)

        # 학습 기반 조정: 과거에 이 상황에서 전환이 나빴으면 신중하게
        learning_penalty = self._get_learning_penalty(current, ms["phase"])

        # ── 점수 기반 전환 판단 ──
        danger = ms["danger_score"]
        opportunity = ms["opportunity_score"]

        # 학습 페널티 적용 (나쁜 전환 이력이 있으면 점수 조정)
        danger += learning_penalty.get("danger_adjust", 0)
        opportunity += learning_penalty.get("opportunity_adjust", 0)

        # 성과 리뷰 조정 (연패 → 위험도↑, 연승 → 기회↑)
        perf_adj = self._get_performance_adjustment()
        if perf_adj > 0:
            danger += perf_adj
        elif perf_adj < 0:
            opportunity += abs(perf_adj)

        # 사용자 피드백 바이어스 반영
        fb_bias = self.state.get("feedback_bias")
        if fb_bias == "conservative":
            danger += 10
        elif fb_bias == "aggressive":
            opportunity += 10

        target = self._decide_target(current, ms, danger, opportunity)

        if target and target != current:
            return {
                "from": current,
                "to": target,
                "reason": self._switch_reason,
                "danger_score": ms["danger_score"],
                "opportunity_score": ms["opportunity_score"],
                "market_phase": ms["phase"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            }

        return None

    def _decide_target(self, current: str, ms: dict,
                        danger: int, opportunity: int) -> str | None:
        """위험도/기회 점수와 시장 국면으로 최적 에이전트를 결정한다."""
        self._switch_reason = ""

        # ══════════════════════════════════════════
        # 긴급 보수적 전환 (danger 기반)
        # ══════════════════════════════════════════

        # 위험도 70+ → 무조건 보수적 (공격적→보수적 직행 허용)
        if danger >= 70:
            if current != "conservative":
                self._switch_reason = (
                    f"위험도 {danger}점 — 긴급 보수 전환 "
                    f"(FGI {ms['fgi']}, 24h {ms['price_change_24h']:+.1f}%, "
                    f"연속손절 {ms['consecutive_losses']}회)"
                )
                return "conservative"

        # 위험도 45~69 + 현재 공격적 → 보통으로 하향
        if 45 <= danger < 70 and current == "aggressive":
            self._switch_reason = (
                f"위험도 {danger}점 상승 — 공격적→보통 하향 "
                f"(김치P {ms['kimchi_pct']:.1f}%, 롱숏 {ms['ls_ratio']:.2f})"
            )
            return "moderate"

        # 위험도 50+ + 현재 보통 → 보수적
        if danger >= 50 and current == "moderate":
            self._switch_reason = (
                f"위험도 {danger}점 — 보통→보수적 하향 "
                f"(FGI {ms['fgi']}, 연속손절 {ms['consecutive_losses']}회)"
            )
            return "conservative"

        # ══════════════════════════════════════════
        # 공격적 전환 (opportunity 기반)
        # ══════════════════════════════════════════

        # FOMO 방지: 급락 중(-5% 이상)에는 공격적 전환 금지
        # 단, 급락 후 반등(price_change > 0이고 FGI 극공포)은 허용
        fomo_block = (ms["price_change_24h"] < -5 and
                      not (ms["fgi"] <= 20 and ms["price_change_24h"] > -8))

        if not fomo_block:
            # 기회 점수 60+ → 공격적 직행 (보수적→공격적도 가능)
            if opportunity >= 60 and danger < 30:
                if current != "aggressive":
                    self._switch_reason = (
                        f"기회 점수 {opportunity}점 — 강한 매수 기회 "
                        f"(FGI {ms['fgi']}, RSI {ms['rsi']:.0f}, "
                        f"Fusion {ms['fusion_signal']}) → 공격적 직행"
                    )
                    return "aggressive"

            # 기회 점수 40~59 → 보통에서 공격적, 또는 보수적에서 보통
            if 40 <= opportunity < 60 and danger < 35:
                if current == "moderate":
                    self._switch_reason = (
                        f"기회 점수 {opportunity}점 — 매수 기회 증가 "
                        f"(FGI {ms['fgi']}, Fusion {ms['fusion_signal']}) → 공격적"
                    )
                    return "aggressive"
                elif current == "conservative":
                    self._switch_reason = (
                        f"기회 점수 {opportunity}점 — 시장 기회 감지 "
                        f"(FGI {ms['fgi']}, RSI {ms['rsi']:.0f}) → 보통"
                    )
                    return "moderate"

            # 기회 점수 25~39 → 보수적에서 보통
            if 25 <= opportunity < 40 and danger < 30:
                if current == "conservative":
                    self._switch_reason = (
                        f"기회 점수 {opportunity}점 — 시장 안정화 "
                        f"(FGI {ms['fgi']}, RSI {ms['rsi']:.0f}) → 보통 복귀"
                    )
                    return "moderate"

        # ══════════════════════════════════════════
        # 횡보장 정상화 (중립 구간)
        # ══════════════════════════════════════════

        # 횡보 안정: danger/opportunity 모두 낮으면 보통이 적절
        if danger < 25 and opportunity < 25:
            if ms["phase"] == self.PHASE_NEUTRAL:
                if current == "aggressive":
                    self._switch_reason = (
                        f"횡보 중립 (FGI {ms['fgi']}, RSI {ms['rsi']:.0f}) "
                        f"— 모멘텀 부재 → 보통 하향"
                    )
                    return "moderate"
                elif current == "conservative" and ms["consecutive_losses"] == 0:
                    self._switch_reason = (
                        f"횡보 안정 (FGI {ms['fgi']}, 위험도 {danger}점) "
                        f"— 리스크 해소 → 보통 복귀"
                    )
                    return "moderate"

        return None

    def _is_on_cooldown(self) -> bool:
        """전환 쿨다운 확인. 기본 2시간, 같은 날 3회 이상 전환 시 4시간."""
        from datetime import datetime, timedelta, timezone

        # 당일 전환 횟수 체크
        today_switches = 0
        for sw in self.state.get("switch_history", []):
            try:
                sw_time = sw.get("timestamp", "")
                if sw_time[:10] == time.strftime("%Y-%m-%d"):
                    today_switches += 1
            except (ValueError, TypeError):
                pass

        # 잦은 전환이면 쿨다운 강화
        cooldown_hours = 2 if today_switches < 3 else 4

        # 마지막 전환 시간 체크
        last_switch = self.state.get("last_switch_time")
        if last_switch:
            try:
                kst = timezone(timedelta(hours=9))
                switch_dt = datetime.fromisoformat(last_switch)
                now = datetime.now(kst)
                if now - switch_dt < timedelta(hours=cooldown_hours):
                    return True
            except (ValueError, TypeError):
                pass

        return False

    def _do_switch(self, switch_info: dict) -> None:
        self._active_agent_name = switch_info["to"]
        self.state["active_agent"] = switch_info["to"]
        self.state["last_switch_time"] = switch_info["timestamp"]
        self.state["switch_history"].append(switch_info)
        self.state["switch_history"] = self.state["switch_history"][-30:]
        _save_state(self.state)

    # ── DB 학습 ────────────────────────────────────

    def _load_learning_data(self) -> dict | None:
        """Supabase에서 과거 전환 성과를 조회한다."""
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return None

        try:
            import requests
            # 전환 성과 요약 조회
            resp = requests.get(
                f"{url}/rest/v1/agent_switch_performance",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                },
                timeout=5,
            )
            if resp.status_code == 200:
                rows = resp.json()
                return {(r["from_agent"], r["to_agent"]): r for r in rows}

            # 뷰가 아직 없으면 (테이블만 있는 경우) 직접 조회
            resp2 = requests.get(
                f"{url}/rest/v1/agent_switches",
                params={
                    "select": "from_agent,to_agent,outcome,profit_after_24h",
                    "outcome": "not.is.null",
                    "order": "created_at.desc",
                    "limit": "50",
                },
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                },
                timeout=5,
            )
            if resp2.status_code == 200:
                rows = resp2.json()
                return self._aggregate_learning(rows)
        except Exception:
            pass

        return None

    def _aggregate_learning(self, rows: list[dict]) -> dict:
        """raw 전환 이력을 집계한다."""
        agg: dict = {}
        for r in rows:
            key = (r["from_agent"], r["to_agent"])
            if key not in agg:
                agg[key] = {"good": 0, "bad": 0, "neutral": 0, "total": 0, "profits": []}
            agg[key][r.get("outcome", "neutral")] += 1
            agg[key]["total"] += 1
            if r.get("profit_after_24h") is not None:
                agg[key]["profits"].append(float(r["profit_after_24h"]))

        result = {}
        for key, data in agg.items():
            total_evaluated = data["good"] + data["bad"] + data["neutral"]
            result[key] = {
                "from_agent": key[0],
                "to_agent": key[1],
                "total_switches": data["total"],
                "good_count": data["good"],
                "bad_count": data["bad"],
                "success_rate_pct": (
                    round(data["good"] / total_evaluated * 100, 1)
                    if total_evaluated > 0 else None
                ),
                "avg_profit_24h": (
                    round(sum(data["profits"]) / len(data["profits"]), 2)
                    if data["profits"] else None
                ),
            }
        return result

    def _get_learning_penalty(self, current: str, phase: str) -> dict:
        """과거 학습 데이터를 기반으로 전환 점수 조정값을 반환한다."""
        if not self._learning_data:
            return {"danger_adjust": 0, "opportunity_adjust": 0}

        adjust = {"danger_adjust": 0, "opportunity_adjust": 0}

        # 현재→공격적 전환 이력이 나쁘면 기회 점수 깎기
        key_to_agg = (current, "aggressive")
        if key_to_agg in self._learning_data:
            data = self._learning_data[key_to_agg]
            rate = data.get("success_rate_pct")
            if rate is not None and rate < 40 and data.get("total_switches", 0) >= 3:
                adjust["opportunity_adjust"] = -10  # 공격적 전환 억제

        # 현재→보수적 전환 이력이 나쁘면 위험 점수 깎기
        key_to_con = (current, "conservative")
        if key_to_con in self._learning_data:
            data = self._learning_data[key_to_con]
            rate = data.get("success_rate_pct")
            if rate is not None and rate < 40 and data.get("total_switches", 0) >= 3:
                adjust["danger_adjust"] = -10  # 보수적 전환 억제

        return adjust

    def _record_switch_to_db(self, switch_info: dict, market_state: dict) -> None:
        """전환 이력을 Supabase에 기록한다."""
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return

        try:
            import requests
            row = {
                "from_agent": switch_info["from"],
                "to_agent": switch_info["to"],
                "reason": switch_info["reason"],
                "fgi_at_switch": market_state.get("fgi"),
                "rsi_at_switch": market_state.get("rsi"),
                "price_change_24h": market_state.get("price_change_24h"),
                "kimchi_premium": market_state.get("kimchi_pct"),
                "fusion_signal": market_state.get("fusion_signal"),
                "consecutive_losses": market_state.get("consecutive_losses", 0),
            }
            requests.post(
                f"{url}/rest/v1/agent_switches",
                json=row,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                timeout=5,
            )
        except Exception:
            pass  # DB 기록 실패는 매매에 영향 없음

    # ── 피드백 & 성과 ────────────────────────────

    def _apply_feedback(self, feedback_list: list) -> None:
        """사용자 피드백을 전략 전환 판단에 반영한다."""
        if not feedback_list:
            return
        for fb in feedback_list:
            content = (fb.get("content", "") or "").lower()
            # 사용자가 전략 전환 관련 피드백을 남겼으면 반영
            if "보수" in content or "conservative" in content or "안전" in content:
                self.state["feedback_bias"] = "conservative"
            elif "공격" in content or "aggressive" in content:
                self.state["feedback_bias"] = "aggressive"
            elif "보통" in content or "moderate" in content:
                self.state["feedback_bias"] = "moderate"

    def _get_performance_adjustment(self) -> int:
        """성과가 나쁘면 위험도를 높이고, 좋으면 기회 점수를 높인다."""
        perf = getattr(self, "_performance", {})
        if not perf or not perf.get("available"):
            return 0

        win_rate = perf.get("win_rate_pct", 50)
        streak_type = perf.get("recent_streak_type", "none")
        streak = perf.get("recent_streak", 0)

        # 연패 중이면 위험도 가산
        if streak_type == "loss" and streak >= 3:
            return 15  # danger에 +15
        elif streak_type == "loss" and streak >= 2:
            return 8
        # 연승 중이면 기회 가산 (음수 = opportunity에 가산)
        elif streak_type == "win" and streak >= 3 and win_rate >= 60:
            return -10
        return 0

    # ── 유틸 ────────────────────────────────────

    def _count_consecutive_losses(self, past_decisions: list[dict]) -> int:
        count = 0
        for d in past_decisions:
            pl = d.get("profit_loss")
            decision = d.get("decision", "")
            if decision in ("buy", "sell", "매수", "매도") and pl is not None and float(pl) < 0:
                count += 1
            else:
                break
        return count

    def _get_fgi(self, external_data: dict) -> int:
        fg = external_data.get("sources", {}).get("fear_greed", {})
        current = fg.get("current", {})
        return int(current.get("value", 50))

    # ── 하락 컨텍스트 (매도 vs DCA 판단 핵심) ────────────

    def _build_drop_context(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
    ) -> dict:
        """
        감독이 하락장에서 매도/물타기를 판단하기 위한 종합 컨텍스트를 구축한다.

        5가지 요소를 분석:
          1) 하락 속도 (4h vs 24h 비교)
          2) 하락 추세 지속 여부 (연속 음봉, 거래량 동반)
          3) DCA 이력 (이미 물타기 했는지)
          4) 외부 약세 시그널 겹침 수
          5) 캐스케이딩 위험 종합 점수 (0~100)
        """
        # ── 1) 하락 속도: 4시간봉 기반 단기 가격 변동 ──
        candles_4h = market_data.get("candles_4h", [])
        price_change_4h = 0.0
        price_change_12h = 0.0
        if len(candles_4h) >= 2:
            latest = candles_4h[-1].get("trade_price", 0)
            prev_1 = candles_4h[-2].get("trade_price", 0)
            if prev_1 > 0:
                price_change_4h = (latest - prev_1) / prev_1 * 100
        if len(candles_4h) >= 4:
            latest = candles_4h[-1].get("trade_price", 0)
            prev_3 = candles_4h[-4].get("trade_price", 0)
            if prev_3 > 0:
                price_change_12h = (latest - prev_3) / prev_3 * 100

        price_change_24h = market_data.get("change_rate_24h", 0) * 100
        # fallback: ticker 구조도 지원
        if abs(price_change_24h) < 0.001:
            price_change_24h = (
                market_data.get("ticker", {}).get("signed_change_rate", 0) * 100
            )

        # ── 2) 하락 추세 지속 여부: 연속 음봉 + 거래량 분석 ──
        consecutive_red = 0
        volume_ratio = 1.0
        if len(candles_4h) >= 6:
            # 최근 6개 캔들에서 연속 음봉 수
            for c in reversed(candles_4h[-6:]):
                op = c.get("opening_price", 0)
                cl = c.get("trade_price", 0)
                if cl < op:
                    consecutive_red += 1
                else:
                    break

            # 최근 1개 캔들 거래량 vs 이전 5개 평균
            recent_vol = candles_4h[-1].get("candle_acc_trade_volume", 0)
            past_vols = [
                c.get("candle_acc_trade_volume", 0) for c in candles_4h[-6:-1]
            ]
            avg_vol = sum(past_vols) / max(len(past_vols), 1)
            if avg_vol > 0:
                volume_ratio = recent_vol / avg_vol

        # 하락 추세 판정: 3봉 이상 연속 음봉 또는 12h -2% 이상 하락
        trend_falling = consecutive_red >= 3 or price_change_12h < -2.0

        # ── 3) DCA 이력 확인 ──
        dca_history = self.state.get("dca_history", {})
        btc_dca = dca_history.get("KRW-BTC", {})
        dca_already_done = btc_dca.get("dca_count", 0) > 0

        # ── 4) 외부 약세 시그널 겹침 수 ──
        ext_sources = external_data.get("sources", {})
        external_bearish_count = 0
        external_bearish_details: list[str] = []

        # 고래 방향: 거래소 입금(매도 징후)
        whale_data = ext_sources.get("whale_tracker", {})
        whale_dir = whale_data.get("whale_score", {}).get("direction", "neutral")
        if whale_dir == "exchange_deposit":
            external_bearish_count += 1
            external_bearish_details.append("고래 거래소 입금(매도 징후)")

        # 바이낸스: 펀딩비/롱숏/김치프리미엄
        binance = ext_sources.get("binance_sentiment", {})
        funding_rate = binance.get("funding_rate", {}).get("current_rate", 0)
        if isinstance(funding_rate, (int, float)) and funding_rate > 0.001:
            external_bearish_count += 1
            external_bearish_details.append(f"극단 양수 펀딩({funding_rate*100:.3f}%)")

        ls_ratio = binance.get("top_trader_long_short", {}).get("current_ratio", 1.0)
        if isinstance(ls_ratio, (int, float)) and ls_ratio > 1.5:
            external_bearish_count += 1
            external_bearish_details.append(f"롱 과밀({ls_ratio:.2f})")

        kimchi_pct = binance.get("kimchi_premium", {}).get("premium_pct", 0)
        if isinstance(kimchi_pct, (int, float)) and kimchi_pct > 5.0:
            external_bearish_count += 1
            external_bearish_details.append(f"극단 김치P({kimchi_pct:.1f}%)")

        # 뉴스 감성
        news_sent = ext_sources.get("news_sentiment", {})
        if news_sent.get("overall_sentiment") == "negative":
            external_bearish_count += 1
            external_bearish_details.append("뉴스 부정적")

        # 매크로 약세
        macro = ext_sources.get("macro", {}).get("analysis", {})
        macro_score = macro.get("macro_score", 0)
        if isinstance(macro_score, (int, float)) and macro_score < -15:
            external_bearish_count += 1
            external_bearish_details.append(f"매크로 약세({macro_score})")

        # ── 5) 캐스케이딩 위험 종합 점수 (0~100) ──
        cascade_risk = 0

        # 급락 속도 (최대 30점)
        if price_change_4h < -3:
            cascade_risk += 30
        elif price_change_4h < -1.5:
            cascade_risk += 15
        elif price_change_4h < -0.5:
            cascade_risk += 5

        # 하락 가속: 4h 하락이 24h 하락의 50%+ → 가속 중
        if price_change_24h < -1 and price_change_4h < 0:
            accel = abs(price_change_4h) / max(abs(price_change_24h), 0.1)
            if accel > 0.5:
                cascade_risk += 20
            elif accel > 0.3:
                cascade_risk += 10

        # 거래량 급증 + 하락 동반 (최대 20점)
        if volume_ratio > 2.0 and price_change_4h < -1:
            cascade_risk += 20
        elif volume_ratio > 1.5 and price_change_4h < -0.5:
            cascade_risk += 10

        # 연속 음봉 (최대 15점)
        if consecutive_red >= 5:
            cascade_risk += 15
        elif consecutive_red >= 3:
            cascade_risk += 8

        # 외부 약세 시그널 (최대 25점, 개당 6~7점)
        cascade_risk += min(external_bearish_count * 7, 25)

        cascade_risk = min(cascade_risk, 100)

        return {
            "price_change_4h": round(price_change_4h, 2),
            "price_change_12h": round(price_change_12h, 2),
            "price_change_24h": round(price_change_24h, 2),
            "consecutive_red_candles": consecutive_red,
            "volume_ratio": round(volume_ratio, 2),
            "trend_falling": trend_falling,
            "dca_already_done": dca_already_done,
            "dca_history": btc_dca,
            "external_bearish_count": external_bearish_count,
            "external_bearish_details": external_bearish_details,
            "cascade_risk": cascade_risk,
            "whale_direction": whale_dir,
            "funding_rate": funding_rate,
        }

    def _override_decision(
        self,
        decision: "Decision",
        drop_context: dict,
        market_state: dict,
    ) -> "Decision":
        """
        감독의 최종 검증: 에이전트 결정을 오버라이드할 수 있다.

        오버라이드 케이스:
          - 에이전트가 DCA 결정 + 캐스케이딩 위험 극심 → 매도로 전환
          - 에이전트가 매수 결정 + 급락 중 → 관망으로 전환
        """
        from agents.base_agent import Decision as Dec

        cascade = drop_context.get("cascade_risk", 0)
        is_dca = decision.trade_params.get("is_dca", False)
        trend_falling = drop_context.get("trend_falling", False)

        # ① DCA인데 캐스케이딩 극심 → 매도로 전환
        if decision.decision == "buy" and is_dca and cascade >= 70:
            btc_holding = {}  # 매도 볼륨은 에이전트가 이미 계산했을 수 있음
            return Dec(
                decision="sell",
                confidence=0.85,
                reason=(
                    f"[감독 오버라이드] 에이전트가 DCA를 결정했으나 "
                    f"캐스케이딩 위험 {cascade}점 → 매도 전환 "
                    f"(4h {drop_context['price_change_4h']:+.1f}%, "
                    f"약세지표 {drop_context['external_bearish_count']}개)"
                ),
                buy_score=decision.buy_score,
                trade_params=decision.trade_params,  # 실행 스크립트에서 재처리
                external_signal=decision.external_signal,
                agent_name=f"🎯 감독 오버라이드 ({decision.agent_name})",
            )

        # ② 신규 매수인데 급락 진행 중 → 관망으로 전환
        if (decision.decision == "buy"
                and not is_dca
                and drop_context.get("price_change_4h", 0) < -3
                and trend_falling):
            return Dec(
                decision="hold",
                confidence=0.75,
                reason=(
                    f"[감독 오버라이드] 매수 시그널이나 "
                    f"급락 진행 중 (4h {drop_context['price_change_4h']:+.1f}%, "
                    f"연속 음봉 {drop_context['consecutive_red_candles']}개) → 관망 전환"
                ),
                buy_score=decision.buy_score,
                trade_params={},
                external_signal=decision.external_signal,
                agent_name=f"🎯 감독 오버라이드 ({decision.agent_name})",
            )

        return decision

    def _track_dca(self, decision: "Decision") -> None:
        """DCA 이력을 추적한다. 매도 시 이력 초기화."""
        dca_history = self.state.setdefault("dca_history", {})
        market = decision.trade_params.get("market", "KRW-BTC")

        if decision.decision == "buy" and decision.trade_params.get("is_dca"):
            # DCA 실행 기록
            if market not in dca_history:
                dca_history[market] = {"dca_count": 0, "dca_total_amount": 0}
            dca_history[market]["dca_count"] += 1
            dca_history[market]["dca_total_amount"] += decision.trade_params.get("amount", 0)
            dca_history[market]["last_dca_time"] = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")

        elif decision.decision == "sell":
            # 매도 시 해당 마켓의 DCA 이력 초기화
            if market in dca_history:
                del dca_history[market]

        _save_state(self.state)

    # ── 자동 긴급정지 시스템 ────────────────────────

    def _evaluate_auto_emergency(
        self,
        market_state: dict,
        drop_context: dict,
        portfolio: dict,
    ) -> dict | None:
        """
        감독이 자동 긴급정지를 발동할지 평가한다.

        발동 조건 (하나라도 충족 시):
          ① 플래시 크래시: 4h -10% 이상
          ② 캐스케이딩 극단 + 위험도 극단: cascade_risk ≥ 90 AND danger ≥ 80
          ③ 외부 전면 약세: 약세 지표 5개+ 동시 겹침
          ④ 포트폴리오 위기: 연속 손절 5회+ AND 총 손실 -15% 이상
        """
        reasons: list[str] = []

        # ① 플래시 크래시
        pc4h = drop_context.get("price_change_4h", 0)
        if pc4h < -10:
            reasons.append(f"플래시 크래시 (4h {pc4h:+.1f}%)")

        # ② 캐스케이딩 + 위험도 동시 극단
        cascade = drop_context.get("cascade_risk", 0)
        danger = market_state.get("danger_score", 0)
        if cascade >= 90 and danger >= 80:
            reasons.append(
                f"캐스케이딩 {cascade}점 + 위험도 {danger}점 동시 극단"
            )

        # ③ 외부 전면 약세
        ext_bearish = drop_context.get("external_bearish_count", 0)
        if ext_bearish >= 5:
            reasons.append(
                f"외부 약세 {ext_bearish}개 동시 겹침 "
                f"({', '.join(drop_context.get('external_bearish_details', []))})"
            )

        # ④ 포트폴리오 위기
        consec_losses = market_state.get("consecutive_losses", 0)
        btc = portfolio.get("btc", {})
        profit_pct = btc.get("profit_pct", 0) if isinstance(btc, dict) else 0
        if consec_losses >= 5 and profit_pct < -15:
            reasons.append(
                f"연속 손절 {consec_losses}회 + 포트폴리오 {profit_pct:.1f}% 하락"
            )

        if not reasons:
            return None

        return {
            "type": "auto_emergency",
            "reason": " / ".join(reasons),
            "conditions": {
                "price_change_4h": pc4h,
                "cascade_risk": cascade,
                "danger_score": danger,
                "external_bearish": ext_bearish,
                "consecutive_losses": consec_losses,
                "portfolio_profit_pct": profit_pct,
            },
            "activated_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        }

    def _activate_auto_emergency(
        self, trigger: dict, market_state: dict
    ) -> None:
        """자동 긴급정지를 발동한다. 플래그 파일 생성 + 텔레그램 알림."""
        AUTO_EMERGENCY_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "active": True,
            "reason": trigger["reason"],
            "conditions": trigger["conditions"],
            "activated_at": trigger["activated_at"],
            "activated_by": "orchestrator",
            "market_state_snapshot": {
                "fgi": market_state.get("fgi"),
                "rsi": market_state.get("rsi"),
                "price_change_24h": market_state.get("price_change_24h"),
                "danger_score": market_state.get("danger_score"),
                "opportunity_score": market_state.get("opportunity_score"),
            },
        }

        with open(AUTO_EMERGENCY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 텔레그램 알림 (실패해도 계속 진행)
        self._notify_emergency("activate", trigger["reason"])

        print(
            f"[EMERGENCY] 🚨 자동 긴급정지 발동: {trigger['reason']}",
            file=__import__("sys").stderr,
        )

    def _deactivate_auto_emergency(self, reason: str) -> None:
        """자동 긴급정지를 해제한다."""
        try:
            AUTO_EMERGENCY_FILE.unlink(missing_ok=True)
        except Exception:
            pass

        self._notify_emergency("deactivate", reason)

        print(
            f"[EMERGENCY] ✅ 자동 긴급정지 해제: {reason}",
            file=__import__("sys").stderr,
        )

    def _check_auto_emergency_active(self) -> dict | None:
        """자동 긴급정지 플래그 파일이 존재하는지 확인한다."""
        try:
            if AUTO_EMERGENCY_FILE.exists():
                with open(AUTO_EMERGENCY_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("active"):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _can_lift_auto_emergency(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
    ) -> bool:
        """
        자동 긴급정지 해제 조건:
          - 발동 후 최소 12시간 경과
          - 현재 cascade_risk < 30
          - 현재 danger_score < 40
          - 4h 가격 변동이 -3% 이내 (급락 종료)
        """
        from datetime import datetime, timedelta, timezone as tz

        auto_em = self._check_auto_emergency_active()
        if not auto_em:
            return False

        # 최소 12시간 경과 확인
        activated_at = auto_em.get("activated_at", "")
        try:
            kst = tz(timedelta(hours=9))
            act_time = datetime.fromisoformat(activated_at)
            now = datetime.now(kst)
            if now - act_time < timedelta(hours=12):
                return False
        except (ValueError, TypeError):
            return False  # 시간 파싱 실패 → 해제하지 않음

        # 현재 시장 상태 간이 평가
        # (full market_state는 아직 계산 전이므로 간이 지표 사용)
        candles = market_data.get("candles_4h", [])
        pc4h = 0.0
        if len(candles) >= 2:
            latest = candles[-1].get("trade_price", 0)
            prev = candles[-2].get("trade_price", 0)
            if prev > 0:
                pc4h = (latest - prev) / prev * 100

        if pc4h < -3:
            return False  # 아직 급락 중

        # FGI 확인
        fgi = self._get_fgi(external_data)
        if fgi <= 15:
            return False  # 아직 극단적 공포

        return True

    def _notify_emergency(self, action: str, reason: str) -> None:
        """긴급정지 발동/해제 시 텔레그램 알림."""
        try:
            import subprocess
            emoji = "🚨" if action == "activate" else "✅"
            title = "자동 긴급정지 발동" if action == "activate" else "자동 긴급정지 해제"
            subprocess.run(
                [
                    "python3", "scripts/notify_telegram.py",
                    "error", f"{emoji} {title}", reason,
                ],
                capture_output=True,
                timeout=10,
                cwd=str(PROJECT_DIR),
            )
        except Exception:
            pass  # 알림 실패는 무시

    def get_status(self) -> dict:
        agent = self.active_agent
        auto_em = self._check_auto_emergency_active()
        return {
            "active_agent": f"{agent.emoji} {agent.name}",
            "description": agent.description,
            "state": self.state,
            "learning": self._learning_data,
            "dca_history": self.state.get("dca_history", {}),
            "auto_emergency": auto_em,
            "user_emergency_stop": os.getenv("EMERGENCY_STOP", "false").lower() == "true",
        }
