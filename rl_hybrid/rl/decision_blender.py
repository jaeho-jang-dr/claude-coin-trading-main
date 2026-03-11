"""DecisionBlender — LLM 분석 + RL 추론 + 에이전트 규칙 융합

세 가지 소스의 매매 신호를 가중 결합하여 최종 결정을 도출한다.

소스 및 기본 가중치:
  - Agent (규칙 기반): 40% — 검증된 점수제 시스템
  - RL (강화학습):     35% — 데이터 기반 최적 행동
  - LLM (Gemini):      25% — 맥락 해석 + 비정형 시그널

가중치는 각 소스의 라이브 성과에 따라 동적으로 조정된다.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.decision_blender")


@dataclass
class BlendedDecision:
    """융합 매매 결정"""
    decision: str               # "buy" | "sell" | "hold"
    confidence: float           # 0.0 ~ 1.0
    reason: str
    action_value: float         # 연속값 [-1, 1]

    # 개별 소스 결과
    agent_decision: str = ""
    agent_confidence: float = 0.0
    rl_action: float = 0.0
    rl_value: float = 0.0
    llm_recommendation: str = ""
    llm_confidence: float = 0.0

    # 가중치
    weights_used: dict = field(default_factory=dict)

    # 매매 파라미터
    trade_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "action_value": self.action_value,
            "sources": {
                "agent": {"decision": self.agent_decision, "confidence": self.agent_confidence},
                "rl": {"action": self.rl_action, "value": self.rl_value},
                "llm": {"recommendation": self.llm_recommendation, "confidence": self.llm_confidence},
            },
            "weights": self.weights_used,
            "trade_params": self.trade_params,
        }


class DecisionBlender:
    """세 소스 매매 신호 융합기"""

    # 행동 매핑: 이산 → 연속값
    ACTION_MAP = {
        "strong_buy": 0.9,
        "cautious_buy": 0.5,
        "buy": 0.6,
        "hold": 0.0,
        "cautious_sell": -0.5,
        "sell": -0.6,
        "strong_sell": -0.9,
    }

    def __init__(
        self,
        agent_weight: float = 0.40,
        rl_weight: float = 0.35,
        llm_weight: float = 0.25,
        decision_threshold: float = 0.25,
    ):
        """
        Args:
            agent_weight: 에이전트(규칙) 가중치
            rl_weight: RL 모델 가중치
            llm_weight: LLM(Gemini) 가중치
            decision_threshold: 매수/매도 결정 임계값 (action_value 절대값)
        """
        self.base_weights = {
            "agent": agent_weight,
            "rl": rl_weight,
            "llm": llm_weight,
        }
        self.decision_threshold = decision_threshold

        # 동적 가중치 조정용 성과 추적
        self.performance_history: dict[str, list[float]] = {
            "agent": [],
            "rl": [],
            "llm": [],
        }

    def blend(
        self,
        agent_result: dict = None,
        rl_prediction: dict = None,
        llm_analysis: dict = None,
        portfolio: dict = None,
        market_state: dict = None,
    ) -> BlendedDecision:
        """세 소스를 융합하여 최종 결정 도출

        Args:
            agent_result: Orchestrator.run() 출력의 "decision" 부분
            rl_prediction: DistributedTrainer.predict() 결과
            llm_analysis: Gemini 분석 결과

        Returns:
            BlendedDecision
        """
        weights = self._get_dynamic_weights(agent_result, rl_prediction, llm_analysis)

        # 각 소스를 연속값으로 변환
        agent_action = self._agent_to_continuous(agent_result)
        rl_action = self._rl_to_continuous(rl_prediction)
        llm_action = self._llm_to_continuous(llm_analysis)

        # 가중 평균
        blended_action = (
            weights["agent"] * agent_action["value"]
            + weights["rl"] * rl_action["value"]
            + weights["llm"] * llm_action["value"]
        )

        # 만장일치 부스트: 세 소스가 같은 방향이면 신뢰도 강화
        directions = [
            np.sign(agent_action["value"]),
            np.sign(rl_action["value"]),
            np.sign(llm_action["value"]),
        ]
        agreement = sum(1 for d in directions if d == np.sign(blended_action)) / 3
        if agreement >= 0.9:
            blended_action *= 1.2
            blended_action = np.clip(blended_action, -1, 1)

        # 최종 이산 결정
        decision, confidence = self._continuous_to_decision(blended_action, agreement)

        # 매매 파라미터 결정
        trade_params = self._build_trade_params(
            decision, blended_action, portfolio, market_state
        )

        # 근거 문자열 생성
        reason = self._build_reason(
            decision, agent_result, rl_prediction, llm_analysis,
            weights, agreement
        )

        return BlendedDecision(
            decision=decision,
            confidence=confidence,
            reason=reason,
            action_value=float(blended_action),
            agent_decision=agent_action["decision"],
            agent_confidence=agent_action["confidence"],
            rl_action=rl_action["value"],
            rl_value=rl_action.get("state_value", 0),
            llm_recommendation=llm_action["decision"],
            llm_confidence=llm_action["confidence"],
            weights_used=weights,
            trade_params=trade_params,
        )

    def _agent_to_continuous(self, result: dict = None) -> dict:
        """에이전트 결과 → 연속값"""
        if not result:
            return {"value": 0.0, "decision": "hold", "confidence": 0.0}

        decision = result.get("decision", "hold")
        confidence = float(result.get("confidence", 0.5))

        value_map = {"buy": 0.6, "sell": -0.6, "hold": 0.0}
        base_value = value_map.get(decision, 0.0)

        # buy_score 기반 강도 조절
        buy_score = result.get("buy_score", {})
        total = buy_score.get("total", 50)
        threshold = buy_score.get("threshold", 70)
        if decision == "buy" and threshold > 0:
            intensity = min(total / threshold, 1.5)
            base_value *= intensity

        return {
            "value": float(base_value * confidence),
            "decision": decision,
            "confidence": confidence,
        }

    def _rl_to_continuous(self, prediction: dict = None) -> dict:
        """RL 추론 → 연속값"""
        if not prediction:
            return {"value": 0.0, "decision": "hold", "confidence": 0.0, "state_value": 0.0}

        action = float(prediction.get("action", 0.0))
        value = float(prediction.get("value", 0.0))

        if action > 0.3:
            decision = "buy"
        elif action < -0.3:
            decision = "sell"
        else:
            decision = "hold"

        confidence = min(abs(action), 1.0)

        return {
            "value": action,
            "decision": decision,
            "confidence": confidence,
            "state_value": value,
        }

    def _llm_to_continuous(self, analysis: dict = None) -> dict:
        """LLM 분석 → 연속값"""
        if not analysis:
            return {"value": 0.0, "decision": "hold", "confidence": 0.0}

        rec = analysis.get("recommended_action", "hold")
        confidence = float(analysis.get("confidence", 0.5))

        value = self.ACTION_MAP.get(rec, 0.0) * confidence

        return {
            "value": value,
            "decision": rec,
            "confidence": confidence,
        }

    def _continuous_to_decision(
        self, action: float, agreement: float
    ) -> tuple[str, float]:
        """연속값 → 이산 결정 + 신뢰도"""
        abs_action = abs(action)
        confidence = min(abs_action * agreement, 1.0)

        if action > self.decision_threshold:
            return "buy", confidence
        elif action < -self.decision_threshold:
            return "sell", confidence
        else:
            return "hold", max(1.0 - abs_action * 2, 0.3)

    def _get_dynamic_weights(
        self,
        agent_result: dict = None,
        rl_prediction: dict = None,
        llm_analysis: dict = None,
    ) -> dict:
        """소스 가용성 + 과거 성과 기반 동적 가중치"""
        weights = self.base_weights.copy()

        # 소스 비가용 시 가중치 재분배
        available = {
            "agent": agent_result is not None,
            "rl": rl_prediction is not None,
            "llm": llm_analysis is not None,
        }

        active_sources = [k for k, v in available.items() if v]
        if not active_sources:
            return {"agent": 0.5, "rl": 0.25, "llm": 0.25}

        # 비가용 소스 가중치를 가용 소스에 분배
        total_inactive_weight = sum(weights[k] for k in weights if k not in active_sources)
        if total_inactive_weight > 0:
            for k in weights:
                if k not in active_sources:
                    weights[k] = 0
                else:
                    share = total_inactive_weight / len(active_sources)
                    weights[k] += share

        # 성과 기반 미세 조정
        for source in active_sources:
            history = self.performance_history.get(source, [])
            if len(history) >= 5:
                recent_win_rate = sum(1 for r in history[-10:] if r > 0) / len(history[-10:])
                # 승률에 비례하여 ±10% 조정
                adjustment = (recent_win_rate - 0.5) * 0.1
                weights[source] = max(0.1, weights[source] + adjustment)

        # 정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _build_trade_params(
        self,
        decision: str,
        action_value: float,
        portfolio: dict = None,
        market_state: dict = None,
    ) -> dict:
        """매매 파라미터 결정"""
        import os
        max_amount = int(os.getenv("MAX_TRADE_AMOUNT", "100000"))
        max_ratio = float(os.getenv("MAX_POSITION_RATIO", "0.5"))

        if decision == "hold":
            return {"side": "none", "amount": 0}

        # 행동 강도에 비례한 금액
        intensity = min(abs(action_value), 1.0)
        amount = int(max_amount * intensity * 0.8)  # 최대의 80%

        if decision == "buy":
            return {
                "side": "bid",
                "market": "KRW-BTC",
                "amount": amount,
                "is_dca": intensity < 0.5,
            }
        else:
            # 매도: 보유량의 비율로
            sell_ratio = min(intensity * 0.5, 1.0)
            return {
                "side": "ask",
                "market": "KRW-BTC",
                "sell_ratio": sell_ratio,
            }

    def _build_reason(
        self,
        decision: str,
        agent_result: dict,
        rl_prediction: dict,
        llm_analysis: dict,
        weights: dict,
        agreement: float,
    ) -> str:
        """결정 근거 문자열 생성"""
        parts = [f"[Blended {decision.upper()}]"]

        if agent_result:
            parts.append(
                f"Agent({weights.get('agent', 0):.0%})={agent_result.get('decision', '?')}"
            )
        if rl_prediction:
            parts.append(
                f"RL({weights.get('rl', 0):.0%})={rl_prediction.get('action', 0):.2f}"
            )
        if llm_analysis:
            parts.append(
                f"LLM({weights.get('llm', 0):.0%})={llm_analysis.get('recommended_action', '?')}"
            )

        parts.append(f"합의={agreement:.0%}")
        return " | ".join(parts)

    def record_outcome(self, source: str, reward: float):
        """각 소스의 결과 기록 (동적 가중치 조정용)"""
        if source in self.performance_history:
            self.performance_history[source].append(reward)
            # 최근 100건만 유지
            if len(self.performance_history[source]) > 100:
                self.performance_history[source] = self.performance_history[source][-100:]
