#!/usr/bin/env python3
"""
사용자 피드백 루프 스크립트 (User Feedback Loop)

오케스트레이터의 전략 선택 및 매매 성향에 영향을 주기 위해
사용자가 직접 피드백을 제공합니다. (예: 시장이 좋다고 느끼면 양수, 나쁘면 음수)

사용법:
  python scripts/feedback.py [score]
  - score: -1.0 ~ 1.0 (양수: 공격적, 음수: 보수적)
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_DIR / "data" / "orchestrator_state.json"
KST = timezone(timedelta(hours=9))

def apply_feedback(score_str: str):
    try:
        score = float(score_str)
    except ValueError:
        print("피드백 점수는 숫자여야 합니다. (예: 0.5, -0.2)")
        sys.exit(1)
        
    if not (-1.0 <= score <= 1.0):
        print("피드백 점수는 -1.0과 1.0 사이여야 합니다.")
        sys.exit(1)
        
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    state = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            pass
            
    # 피드백 반영 (최대 -1.0 ~ 1.0 사이로 누적)
    current_bias = state.get("feedback_bias", 0.0)
    new_bias = max(-1.0, min(1.0, current_bias + score))
    
    state["feedback_bias"] = new_bias
    state["last_feedback_time"] = datetime.now(KST).isoformat()
    state["last_feedback_score"] = score
    
    temp_file = STATE_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    temp_file.replace(STATE_FILE)
    
    print(f"✅ 사용자 피드백이 적용되었습니다. (입력: {score})")
    print(f"📊 현재 누적 피드백 Bias: {current_bias:.2f} -> {new_bias:.2f} (오케스트레이터 판단에 영향을 줍니다)")
    print(f"⏰ 반영 시간: {state['last_feedback_time']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts/feedback.py [점수(-1.0~1.0)]")
        print("점수가 양수면 공격적 투자를 선호, 음수면 보수적 투자를 선호하게 됩니다.")
        sys.exit(1)
        
    apply_feedback(sys.argv[1])
