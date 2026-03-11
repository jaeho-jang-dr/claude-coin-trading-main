"""RL Worker 노드 — 로컬 환경에서 롤아웃 수집 + 경험 전송

역할:
  - 로컬 BitcoinTradingEnv에서 롤아웃(trajectory) 수집
  - 수집된 trajectory를 Main Brain으로 전송
  - Main Brain에서 업데이트된 가중치를 수신하여 로컬 정책 갱신
  - PUB 채널 구독하여 최신 시세로 라이브 환경 보강
"""

import json
import logging
import os
import sys
import time
import threading
from typing import Optional

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.nodes.base_node import BaseNode
from rl_hybrid.config import config
from rl_hybrid.protocol import (
    ZMQMessage, MsgType, Action, make_request, make_heartbeat,
)
from rl_hybrid.rl.data_loader import HistoricalDataLoader
from rl_hybrid.rl.environment import BitcoinTradingEnv
from rl_hybrid.rl.trajectory import TrajectoryBuffer, Transition

logger = logging.getLogger("node.rl_worker")

# PyTorch lazy import
try:
    import torch
    from rl_hybrid.rl.distributed_trainer import ActorCritic
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RLWorkerNode(BaseNode):
    """RL Worker — DEALER + SUB, 로컬 환경 롤아웃"""

    def __init__(
        self,
        worker_id: str = "rl_worker_0",
        data_days: int = 180,
        rollout_steps: int = 2048,
        rollout_interval: int = 60,  # 롤아웃 간 대기(초)
    ):
        super().__init__(worker_id)
        self.worker_id = worker_id
        self.data_days = data_days
        self.rollout_steps = rollout_steps
        self.rollout_interval = rollout_interval

        self.dealer: Optional[zmq.Socket] = None
        self.sub: Optional[zmq.Socket] = None

        # RL 구성요소
        self.env: Optional[BitcoinTradingEnv] = None
        self.model: Optional[ActorCritic] = None
        self.buffer = TrajectoryBuffer(max_size=rollout_steps * 2)

        # 상태
        self.weights_version = 0
        self.rollouts_completed = 0
        self.latest_market_data: dict = {}

    def _setup_sockets(self):
        # DEALER → Main Brain ROUTER
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.dealer.connect(self.zmq_config.router_addr)
        self.logger.info(f"DEALER 연결: {self.zmq_config.router_addr}")

        # SUB → Main Brain PUB
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.zmq_config.pub_addr)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.logger.info(f"SUB 연결: {self.zmq_config.pub_addr}")

        # 환경 + 모델 초기화
        self._init_environment()
        self._init_model()

        # 초기 가중치 요청
        self._request_weights()

    def _init_environment(self):
        """히스토리컬 데이터로 환경 생성"""
        self.logger.info(f"환경 초기화: {self.data_days}일 데이터 로딩...")
        loader = HistoricalDataLoader()
        raw = loader.load_candles(days=self.data_days, interval="1h")
        candles = loader.compute_indicators(raw)

        self.env = BitcoinTradingEnv(
            candles=candles,
            initial_balance=10_000_000,
        )
        self.logger.info(f"환경 준비 완료: {len(candles)}봉")

    def _init_model(self):
        """로컬 정책 네트워크 초기화"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch 미설치 — RL Worker 비활성화")
            return

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.model = ActorCritic(obs_dim, action_dim)
        self.logger.info(f"로컬 모델 초기화: obs={obs_dim}, action={action_dim}")

    def _send_heartbeat(self):
        """Main Brain에 하트비트 전송"""
        msg = make_heartbeat(
            self.worker_id,
            status="alive",
            uptime=self.uptime,
            rollouts=self.rollouts_completed,
            buffer_size=len(self.buffer.transitions),
            weights_version=self.weights_version,
        )
        try:
            self.dealer.send(b"" + msg.serialize())
        except Exception:
            pass

    def _main_loop(self):
        """메인 루프: 롤아웃 수집 → 전송 → 가중치 수신"""
        poller = zmq.Poller()
        poller.register(self.dealer, zmq.POLLIN)
        poller.register(self.sub, zmq.POLLIN)

        self.logger.info("RL Worker 메인 루프 시작")
        last_rollout = 0

        while self._running:
            # 1. 메시지 수신 (100ms)
            events = dict(poller.poll(100))

            if self.dealer in events:
                self._handle_dealer_message()

            if self.sub in events:
                self._handle_sub_message()

            # 2. 주기적 롤아웃 수집
            now = time.time()
            if now - last_rollout >= self.rollout_interval and self.model:
                self._run_rollout()
                last_rollout = now

                # 버퍼가 충분히 차면 전송
                if self.buffer.is_ready(self.rollout_steps):
                    self._submit_trajectory()

    def _handle_dealer_message(self):
        """Main Brain 메시지 처리"""
        try:
            frames = self.dealer.recv_multipart(zmq.NOBLOCK)
            data = frames[-1] if frames else None
            if not data:
                return
            msg = ZMQMessage.deserialize(data)
        except Exception as e:
            self.logger.error(f"메시지 수신 에러: {e}")
            return

        if msg.action == Action.WEIGHTS_UPDATE.value:
            self._apply_weights(msg.payload)
        elif msg.action == Action.START_ROLLOUT.value:
            self.logger.info("롤아웃 시작 명령 수신")
            self._run_rollout()
            if self.buffer.is_ready(self.rollout_steps):
                self._submit_trajectory()
        elif msg.action == Action.TRAJECTORY_ACK.value:
            self.logger.info(
                f"Trajectory ACK: update=#{msg.payload.get('update_count', '?')}"
            )

    def _handle_sub_message(self):
        """PUB 채널 시장 데이터 캐시"""
        try:
            data = self.sub.recv(zmq.NOBLOCK)
            msg = ZMQMessage.deserialize(data)
            if msg.action == Action.MARKET_UPDATE.value:
                self.latest_market_data = msg.payload
        except Exception:
            pass

    def _run_rollout(self):
        """로컬 환경에서 롤아웃 수집

        rollout_steps만큼 환경을 진행하면서 transition을 버퍼에 저장한다.
        """
        if not self.model or not self.env:
            return

        obs, info = self.env.reset()
        episode_reward = 0.0
        steps = 0

        self.model.eval()

        with torch.no_grad():
            for _ in range(self.rollout_steps):
                # 행동 추론
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, log_prob, value = self.model.get_action(obs_tensor)

                action_np = action.squeeze().numpy()
                action_clipped = np.clip(action_np, -1.0, 1.0)

                # 환경 스텝
                next_obs, reward, terminated, truncated, step_info = self.env.step(
                    np.array([action_clipped])
                )
                done = terminated or truncated

                # 버퍼에 저장
                self.buffer.add(Transition(
                    obs=obs.copy(),
                    action=np.array([action_clipped]),
                    reward=reward,
                    next_obs=next_obs.copy(),
                    done=done,
                    value=float(value.numpy()),
                    log_prob=float(log_prob.numpy()),
                    info={"price": step_info.get("price", 0)},
                ))

                episode_reward += reward
                obs = next_obs
                steps += 1

                if done:
                    obs, info = self.env.reset()
                    episode_reward = 0.0

        self.rollouts_completed += 1
        stats = self.buffer.stats()
        self.logger.info(
            f"롤아웃 #{self.rollouts_completed} 완료: "
            f"{steps}스텝, buffer={stats['buffer_size']}, "
            f"episodes={stats['episodes_completed']}"
        )

    def _submit_trajectory(self):
        """수집된 trajectory를 Main Brain으로 전송"""
        try:
            trajectory_data = self.buffer.serialize()
            msg = make_request(
                Action.SUBMIT_TRAJECTORY,
                self.worker_id,
                {
                    "trajectory": trajectory_data,
                    "worker_id": self.worker_id,
                    "weights_version": self.weights_version,
                    "rollouts_completed": self.rollouts_completed,
                },
            )
            self.dealer.send(b"" + msg.serialize())
            self.buffer.clear()
            self.logger.info(
                f"Trajectory 전송: {trajectory_data['n_steps']}스텝, "
                f"avg_reward={trajectory_data['avg_reward']:.4f}"
            )
        except Exception as e:
            self.logger.error(f"Trajectory 전송 실패: {e}")

    def _request_weights(self):
        """Main Brain에 최신 가중치 요청"""
        msg = make_request(
            Action.REQUEST_WEIGHTS,
            self.worker_id,
            {"current_version": self.weights_version},
        )
        try:
            self.dealer.send(b"" + msg.serialize())
            self.logger.info("가중치 요청 전송")
        except Exception as e:
            self.logger.error(f"가중치 요청 실패: {e}")

    def _apply_weights(self, payload: dict):
        """Main Brain에서 수신한 가중치 적용"""
        if not self.model:
            return

        try:
            weights = payload.get("weights", {})
            version = payload.get("version", 0)

            if version <= self.weights_version:
                self.logger.debug(f"이미 최신 가중치 (v{version})")
                return

            state_dict = {}
            for name, values in weights.items():
                state_dict[name] = torch.FloatTensor(np.array(values))

            self.model.load_state_dict(state_dict)
            self.weights_version = version

            self.logger.info(
                f"가중치 적용 완료: v{version}, "
                f"{len(weights)}개 파라미터"
            )
        except Exception as e:
            self.logger.error(f"가중치 적용 실패: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="rl_worker_0", help="워커 ID")
    parser.add_argument("--days", type=int, default=180, help="데이터 기간")
    parser.add_argument("--steps", type=int, default=2048, help="롤아웃 스텝")
    parser.add_argument("--interval", type=int, default=60, help="롤아웃 간격(초)")
    args = parser.parse_args()

    node = RLWorkerNode(
        worker_id=args.id,
        data_days=args.days,
        rollout_steps=args.steps,
        rollout_interval=args.interval,
    )
    node.start()
