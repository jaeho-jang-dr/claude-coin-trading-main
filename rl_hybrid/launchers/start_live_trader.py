"""라이브 트레이더 실행

사용법:
    python start_live_trader.py                    # 단일 사이클
    python start_live_trader.py --loop             # 30분 간격 반복
    python start_live_trader.py --loop --interval 3600  # 1시간 간격
    python start_live_trader.py --dry-run          # DRY_RUN 강제
    python start_live_trader.py --llm              # Gemini 분석 포함
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.rl.live_trader import main

if __name__ == "__main__":
    main()
