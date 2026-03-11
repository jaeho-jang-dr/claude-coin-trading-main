"""Main Brain 노드 단독 실행"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rl_hybrid.nodes.main_brain import MainBrainNode

if __name__ == "__main__":
    MainBrainNode().start()
