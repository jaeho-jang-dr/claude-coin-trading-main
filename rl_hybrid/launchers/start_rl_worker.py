"""RL Worker 노드 단독 실행

사용법:
    python start_rl_worker.py                          # 기본 설정
    python start_rl_worker.py --id rl_worker_1         # 워커 ID 지정
    python start_rl_worker.py --days 365 --steps 4096  # 데이터/롤아웃 조정
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", default="rl_worker_0")
parser.add_argument("--days", type=int, default=180)
parser.add_argument("--steps", type=int, default=2048)
parser.add_argument("--interval", type=int, default=60)
args = parser.parse_args()

from rl_hybrid.nodes.rl_worker import RLWorkerNode

node = RLWorkerNode(
    worker_id=args.id,
    data_days=args.days,
    rollout_steps=args.steps,
    rollout_interval=args.interval,
)
node.start()
