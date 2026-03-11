#!/usr/bin/env python3
"""
SB3 PPO v3 모델 예측기

run_agents.py 파이프라인에서 호출하여 RL 시그널을 생성한다.
42차원 관측벡터 → SB3 PPO predict → action [-1, 1] 반환.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# SB3 lazy import (없으면 graceful degradation)
_SB3_AVAILABLE = False
_PPO = None


def _ensure_sb3():
    global _SB3_AVAILABLE, _PPO
    if _SB3_AVAILABLE:
        return True
    try:
        from stable_baselines3 import PPO
        _PPO = PPO
        _SB3_AVAILABLE = True
        return True
    except ImportError:
        return False


def _find_best_model() -> Path | None:
    """sb3_registry.json에서 현재 활성 모델 경로를 찾는다."""
    registry_path = PROJECT_DIR / "data" / "rl_models" / "sb3_registry.json"
    models_dir = PROJECT_DIR / "data" / "rl_models"

    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text(encoding="utf-8"))
            current = reg.get("current", "")
            if current:
                # versions/ 디렉토리에서 찾기
                version_model = models_dir / "versions" / current / "model.zip"
                if version_model.exists():
                    return version_model
        except Exception:
            pass

    # fallback: best 디렉토리
    best = models_dir / "best" / "best_model.zip"
    if best.exists():
        return best

    # fallback: v7_final
    v7 = models_dir / "v7_final.zip"
    if v7.exists():
        return v7

    return None


# 모델 캐시 (프로세스 내 재사용)
_cached_model = None
_cached_model_path = None


def load_model():
    """SB3 PPO 모델 로드 (캐시)"""
    global _cached_model, _cached_model_path

    if not _ensure_sb3():
        return None

    model_path = _find_best_model()
    if model_path is None:
        return None

    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model

    try:
        model = _PPO.load(str(model_path), device="cpu")
        _cached_model = model
        _cached_model_path = model_path
        return model
    except Exception:
        return None


def predict(
    market_data: dict,
    external_data: dict,
    portfolio: dict,
    agent_state: dict | None = None,
) -> dict | None:
    """SB3 RL 예측 수행

    Returns:
        {
            "action": float,          # [-1, 1] 연속값
            "interpretation": str,    # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
            "confidence": float,      # [0, 1]
            "model_path": str,
        }
        또는 None (모델 미사용)
    """
    model = load_model()
    if model is None:
        return None

    try:
        from rl_hybrid.rl.state_encoder import StateEncoder

        encoder = StateEncoder()
        obs = encoder.encode(
            market_data=market_data,
            external_data=external_data,
            portfolio=portfolio,
            agent_state=agent_state or {},
        )

        # SB3 predict (deterministic)
        action_array, _states = model.predict(obs, deterministic=True)
        action = float(action_array[0]) if hasattr(action_array, '__len__') else float(action_array)
        action = np.clip(action, -1.0, 1.0)

        return {
            "action": round(action, 4),
            "interpretation": _interpret_action(action),
            "confidence": round(min(abs(action), 1.0), 3),
            "model_path": str(_cached_model_path),
        }
    except Exception as e:
        return {"error": str(e)}


def _interpret_action(action: float) -> str:
    """연속 action [-1, 1]을 이산 해석으로 변환"""
    if action > 0.5:
        return "strong_buy"
    elif action > 0.2:
        return "buy"
    elif action > -0.2:
        return "hold"
    elif action > -0.5:
        return "sell"
    else:
        return "strong_sell"


def blend_with_agent(
    agent_decision: dict,
    rl_prediction: dict,
    agent_weight: float = 0.6,
    rl_weight: float = 0.4,
) -> dict:
    """에이전트 결정과 RL 예측을 블렌딩

    Args:
        agent_decision: orchestrator.run()['decision'] (Decision.to_dict())
        rl_prediction: predict() 반환값
        agent_weight: 에이전트 가중치 (기본 0.6)
        rl_weight: RL 가중치 (기본 0.4)

    Returns:
        수정된 agent_decision dict (원본 보존, rl_blend 필드 추가)
    """
    if not rl_prediction or "error" in rl_prediction:
        agent_decision["rl_blend"] = {"used": False, "reason": "RL 모델 미사용"}
        return agent_decision

    rl_action = rl_prediction["action"]
    rl_interp = rl_prediction["interpretation"]
    agent_action = agent_decision.get("decision", "hold")

    # 에이전트 결정을 연속값으로 변환
    AGENT_ACTION_MAP = {
        "buy": 0.6,
        "sell": -0.6,
        "hold": 0.0,
    }
    agent_value = AGENT_ACTION_MAP.get(agent_action, 0.0)

    # 가중 블렌딩
    blended_value = agent_value * agent_weight + rl_action * rl_weight

    # 블렌딩 결과 → 이산 결정
    blended_decision = _interpret_action(blended_value)
    # strong_buy/buy → buy, strong_sell/sell → sell
    final_decision = (
        "buy" if blended_decision in ("strong_buy", "buy") else
        "sell" if blended_decision in ("strong_sell", "sell") else
        "hold"
    )

    # 신뢰도 조정
    original_confidence = agent_decision.get("confidence", 0.5)
    agreement = (agent_action == final_decision)

    if agreement:
        # 에이전트와 RL이 동의 → 신뢰도 상향
        adjusted_confidence = min(1.0, original_confidence * 1.15)
    else:
        # 의견 불일치 → 신뢰도 하향
        adjusted_confidence = original_confidence * 0.8

    # 매매 금액 조정 (RL이 hold인데 에이전트가 buy → 축소)
    trade_params = agent_decision.get("trade_params", {})
    amount_modifier = 1.0

    if agent_action == "buy" and rl_interp in ("hold", "sell", "strong_sell"):
        amount_modifier = 0.5  # 50% 축소
    elif agent_action == "buy" and rl_interp in ("strong_buy", "buy"):
        amount_modifier = 1.0  # 유지
    elif agent_action == "sell" and rl_interp in ("hold", "buy", "strong_buy"):
        amount_modifier = 0.7  # 70% 축소

    if "amount" in trade_params and amount_modifier < 1.0:
        trade_params["amount"] = int(trade_params["amount"] * amount_modifier)
    if "volume" in trade_params and amount_modifier < 1.0:
        trade_params["volume"] = round(float(trade_params["volume"]) * amount_modifier, 8)

    # 결과에 RL 블렌드 정보 추가
    blend_info = {
        "used": True,
        "rl_action": rl_action,
        "rl_interpretation": rl_interp,
        "agent_value": agent_value,
        "blended_value": round(blended_value, 4),
        "blended_decision": blended_decision,
        "agreement": agreement,
        "amount_modifier": amount_modifier,
        "weights": {"agent": agent_weight, "rl": rl_weight},
        "model_path": rl_prediction.get("model_path", ""),
    }

    # 결정 변경 여부
    decision_changed = (final_decision != agent_action)
    if decision_changed:
        # RL이 에이전트 결정을 오버라이드 → 보수적으로: hold로 전환만 허용
        # (RL이 buy/sell을 새로 제안하는 것은 위험 → 에이전트 원래 결정 유지)
        if final_decision == "hold" and agent_action in ("buy", "sell"):
            agent_decision["decision"] = "hold"
            agent_decision["reason"] += f" | RL 시그널({rl_interp}) 불일치로 관망 전환"
            agent_decision["trade_params"] = {}
            blend_info["override"] = f"{agent_action} → hold"
        else:
            # RL이 매매 제안하지만 에이전트가 hold → 에이전트 우선
            blend_info["override"] = None
            blend_info["note"] = f"RL={rl_interp} vs Agent={agent_action}, 에이전트 우선"
    else:
        blend_info["override"] = None

    agent_decision["confidence"] = round(adjusted_confidence, 3)
    agent_decision["rl_blend"] = blend_info

    return agent_decision


if __name__ == "__main__":
    # 독립 실행 테스트
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")

    model = load_model()
    if model:
        print(f"모델 로드 성공: {_cached_model_path}")
        # 더미 데이터로 테스트
        dummy_obs = np.random.rand(42).astype(np.float32)
        action, _ = model.predict(dummy_obs, deterministic=True)
        a = float(action[0]) if hasattr(action, '__len__') else float(action)
        print(f"더미 predict: action={a:.4f}, interp={_interpret_action(a)}")
    else:
        print("모델 로드 실패 (SB3 미설치 또는 모델 파일 없음)")
