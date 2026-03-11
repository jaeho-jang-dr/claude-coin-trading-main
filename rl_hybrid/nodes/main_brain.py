"""메인 브레인 노드 — 분산 시스템의 중앙 오케스트레이터 + 글로벌 RL 트레이너

역할:
  - ROUTER 소켓으로 워커들의 요청/응답 수신
  - PUB 소켓으로 시장 데이터 + 가중치 브로드캐스트
  - Phase 1: 데이터 수집 → LLM Worker 분석 → 에이전트 판단 → Trading Worker 매매
  - Phase 3: RL Worker trajectory 수신 → 글로벌 PPO 업데이트 → 가중치 배포
  - 워커 건강 상태 모니터링 (하트비트)
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
    ZMQMessage, MsgType, Action,
    make_request, make_broadcast, make_heartbeat,
)

logger = logging.getLogger("node.main_brain")

# 분산 트레이너 lazy import
try:
    from rl_hybrid.rl.distributed_trainer import DistributedTrainer, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False


class MainBrainNode(BaseNode):
    """메인 브레인 — ROUTER + PUB + 글로벌 RL 트레이너"""

    def __init__(self, enable_rl: bool = True, obs_dim: int = 42):
        super().__init__("main_brain")
        self.router: Optional[zmq.Socket] = None
        self.pub: Optional[zmq.Socket] = None

        # 워커 상태 추적
        self.worker_health: dict[str, dict] = {}
        self.pending_requests: dict[str, dict] = {}

        # Phase 3: 분산 RL 트레이너
        self.enable_rl = enable_rl and TORCH_AVAILABLE
        self.trainer: Optional[DistributedTrainer] = None
        self.weights_version = 0
        self._rl_update_lock = threading.Lock()

        # LLM 분석 캐시 (RL 관측 보강용)
        self.latest_llm_analysis: dict = {}
        self.latest_llm_embedding: Optional[list[float]] = None

        if self.enable_rl:
            self.trainer = DistributedTrainer(obs_dim=obs_dim)
            self.trainer.load_model()  # 기존 모델 있으면 로드
            self.logger.info("분산 RL 트레이너 활성화")
        else:
            self.logger.info("RL 트레이너 비활성화 (PyTorch 미설치 또는 비활성)")

    def _setup_sockets(self):
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.bind(self.zmq_config.router_addr)
        self.logger.info(f"ROUTER 바인딩: {self.zmq_config.router_addr}")

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(self.zmq_config.pub_addr)
        self.logger.info(f"PUB 바인딩: {self.zmq_config.pub_addr}")

    def _main_loop(self):
        """메인 이벤트 루프"""
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)

        last_cycle = 0
        last_rl_check = 0
        cycle_interval = 60 * 30       # 분석 사이클: 30분
        rl_check_interval = 10          # RL 업데이트 체크: 10초

        self.logger.info("메인 루프 시작 — 워커 연결 대기 중...")

        while self._running:
            events = dict(poller.poll(100))
            if self.router in events:
                self._handle_router_message()

            now = time.time()

            # 분석 사이클
            if now - last_cycle >= cycle_interval:
                if self._has_worker("llm_worker") and self._has_worker("trading_worker"):
                    self._run_analysis_cycle()
                    last_cycle = now
                elif now - last_cycle > cycle_interval + 10:
                    last_cycle = now

            # RL 업데이트 체크
            if self.enable_rl and now - last_rl_check >= rl_check_interval:
                self._check_rl_update()
                last_rl_check = now

            self._check_worker_health()

    def _handle_router_message(self):
        """ROUTER 소켓 메시지 처리"""
        identity, msg = self.recv_msg_with_identity(self.router)
        if not msg:
            return

        self.worker_health[msg.sender] = {
            "last_seen": time.time(),
            "status": "alive",
            "identity": identity,
        }

        if msg.msg_type == MsgType.HEARTBEAT.value:
            self.logger.debug(f"하트비트: {msg.sender}")
            return

        if msg.msg_type == MsgType.RESPONSE.value:
            self._handle_response(msg)
            return

        if msg.msg_type == MsgType.REQUEST.value:
            self._handle_request(msg, identity)
            return

        self.logger.warning(f"알 수 없는 메시지: type={msg.msg_type}, action={msg.action}")

    def _handle_request(self, msg: ZMQMessage, identity: bytes):
        """워커 요청 처리 (Phase 3: RL Worker 요청 포함)"""

        if msg.action == Action.SUBMIT_TRAJECTORY.value:
            self._handle_trajectory_submission(msg, identity)

        elif msg.action == Action.REQUEST_WEIGHTS.value:
            self._handle_weights_request(msg, identity)

        else:
            self.logger.warning(f"알 수 없는 요청: action={msg.action}")

    def _handle_response(self, msg: ZMQMessage):
        """워커 응답 처리"""
        req_id = msg.request_id
        if req_id in self.pending_requests:
            pending = self.pending_requests.pop(req_id)
            elapsed = time.time() - pending["time"]
            self.logger.info(
                f"응답 수신: action={msg.action}, from={msg.sender}, "
                f"elapsed={elapsed:.1f}s"
            )
            if pending.get("callback"):
                pending["callback"](msg)
        else:
            self.logger.debug(f"미등록 응답: req_id={req_id}")

    # === Phase 3: 분산 RL 학습 ===

    def _handle_trajectory_submission(self, msg: ZMQMessage, identity: bytes):
        """RL Worker의 trajectory 수신 처리"""
        if not self.trainer:
            return

        payload = msg.payload
        trajectory = payload.get("trajectory", {})
        worker_id = payload.get("worker_id", msg.sender)

        self.trainer.receive_trajectory(trajectory, worker_id)

        # ACK 응답
        response = msg.reply({
            "received": True,
            "update_count": self.trainer.update_count,
            "weights_version": self.weights_version,
        })
        response.sender = self.node_name
        self.send_msg(self.router, response, identity=identity)

    def _handle_weights_request(self, msg: ZMQMessage, identity: bytes):
        """RL Worker의 가중치 요청 처리"""
        if not self.trainer:
            response = msg.reply({}, error="RL 트레이너 비활성화")
            response.sender = self.node_name
            self.send_msg(self.router, response, identity=identity)
            return

        weights = self.trainer.get_weights_serializable()
        response = msg.reply({
            "weights": weights,
            "version": self.weights_version,
            "update_count": self.trainer.update_count,
        })
        response.sender = self.node_name
        self.send_msg(self.router, response, identity=identity)
        self.logger.info(f"가중치 전송: v{self.weights_version} → {msg.sender}")

    def _check_rl_update(self):
        """trajectory가 충분히 모였으면 글로벌 업데이트 실행"""
        if not self.trainer or not self.trainer.should_update():
            return

        with self._rl_update_lock:
            stats = self.trainer.update()

        if stats:
            self.weights_version += 1
            self._broadcast_weights()
            self.logger.info(
                f"=== 글로벌 PPO 업데이트 #{stats['update_count']} ===\n"
                f"  policy_loss={stats['policy_loss']:.4f}, "
                f"  value_loss={stats['value_loss']:.4f}, "
                f"  entropy={stats['entropy']:.4f}, "
                f"  total_steps={stats['total_steps']}"
            )

    def _broadcast_weights(self):
        """모든 RL Worker에게 새 가중치 브로드캐스트"""
        if not self.trainer:
            return

        weights = self.trainer.get_weights_serializable()
        now = time.time()

        for name, info in self.worker_health.items():
            if name.startswith("rl_worker") and info["status"] == "alive":
                msg = ZMQMessage(
                    msg_type=MsgType.RESPONSE.value,
                    action=Action.WEIGHTS_UPDATE.value,
                    sender=self.node_name,
                    payload={
                        "weights": weights,
                        "version": self.weights_version,
                        "update_count": self.trainer.update_count,
                    },
                )
                try:
                    self.send_msg(self.router, msg, identity=info["identity"])
                    self.logger.debug(f"가중치 배포: v{self.weights_version} → {name}")
                except Exception as e:
                    self.logger.error(f"가중치 배포 실패 ({name}): {e}")

    # === RL 정책 추론 (매매 판단 보조) ===

    def get_rl_prediction(self, obs: np.ndarray) -> Optional[dict]:
        """RL 모델로 현재 관측에 대한 행동 추론

        기존 에이전트 결정을 보강하는 추가 신호로 활용.
        """
        if not self.trainer:
            return None

        try:
            action, log_prob, value = self.trainer.predict(obs, deterministic=True)
            return {
                "action": action,
                "value": value,
                "log_prob": log_prob,
                "interpretation": self._interpret_rl_action(action),
            }
        except Exception as e:
            self.logger.debug(f"RL 추론 실패: {e}")
            return None

    @staticmethod
    def _interpret_rl_action(action: float) -> str:
        """RL 행동값을 해석"""
        if action > 0.3:
            return "strong_buy" if action > 0.7 else "cautious_buy"
        elif action < -0.3:
            return "strong_sell" if action < -0.7 else "cautious_sell"
        return "hold"

    # === Phase 1 기존 로직 ===

    def send_to_worker(self, worker_name, action, payload, callback=None):
        """특정 워커에게 요청 전송"""
        worker = self.worker_health.get(worker_name)
        if not worker or not worker.get("identity"):
            self.logger.warning(f"워커 미연결: {worker_name}")
            return None

        msg = make_request(action, self.node_name, payload)
        self.send_msg(self.router, msg, identity=worker["identity"])

        self.pending_requests[msg.request_id] = {
            "action": action.value,
            "time": time.time(),
            "callback": callback,
        }
        return msg.request_id

    def broadcast_market_update(self, market_data: dict):
        """PUB 소켓으로 시장 데이터 브로드캐스트"""
        msg = make_broadcast(Action.MARKET_UPDATE, self.node_name, market_data)
        self.pub.send(msg.serialize())

    def _run_analysis_cycle(self):
        """분석 사이클 — LLM 분석 + RL 추론 통합"""
        self.logger.info("=== 분석 사이클 시작 ===")

        try:
            market_data = self._collect_market_data()
            if not market_data:
                self.logger.error("시장 데이터 수집 실패")
                return

            self.broadcast_market_update(market_data)

            # LLM Worker에 분석 요청
            self.send_to_worker(
                "llm_worker",
                Action.ANALYZE_MARKET,
                {
                    "market_data": market_data,
                    "external_data": {},
                    "context": {"cycle_id": f"cycle_{int(time.time())}"},
                },
                callback=self._on_analysis_complete,
            )

        except Exception as e:
            self.logger.error(f"분석 사이클 에러: {e}", exc_info=True)

    def _on_analysis_complete(self, msg: ZMQMessage):
        """Gemini 분석 완료 → RL 추론과 결합"""
        if msg.error:
            self.logger.error(f"분석 실패: {msg.error}")
            return

        analysis = msg.payload
        self.latest_llm_analysis = analysis
        self.logger.info(
            f"Gemini 분석 수신: regime={analysis.get('market_regime')}, "
            f"action={analysis.get('recommended_action')}"
        )

        # Phase 3: RL 정책 추론 결합
        if self.trainer:
            from rl_hybrid.rl.state_encoder import StateEncoder
            encoder = StateEncoder()
            # 간이 관측 생성 (실 운영에서는 전체 데이터 사용)
            obs = np.zeros(encoder.obs_dim, dtype=np.float32)
            rl_pred = self.get_rl_prediction(obs)
            if rl_pred:
                self.logger.info(
                    f"RL 추론: action={rl_pred['action']:.3f}, "
                    f"interp={rl_pred['interpretation']}, "
                    f"value={rl_pred['value']:.4f}"
                )
                # TODO: LLM 분석 + RL 추론 + 에이전트 결정을 종합하여 최종 판단
                # TODO: Trading Worker에 매매 요청

    def _collect_market_data(self):
        """기존 collect_market_data.py 호출"""
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/collect_market_data.py"],
                capture_output=True, text=True, timeout=30,
                cwd=config.project_root,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            self.logger.error(f"시장 데이터 수집 에러: {result.stderr[:500]}")
            return None
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패: {e}")
            return None

    def _has_worker(self, name: str) -> bool:
        """워커 연결 상태 확인"""
        timeout = self.zmq_config.heartbeat_interval * self.zmq_config.max_missed_heartbeats
        w = self.worker_health.get(name)
        return w is not None and time.time() - w["last_seen"] < timeout

    def _check_worker_health(self):
        """워커 건강 점검"""
        now = time.time()
        timeout = self.zmq_config.heartbeat_interval * self.zmq_config.max_missed_heartbeats
        for name, info in list(self.worker_health.items()):
            if now - info["last_seen"] > timeout and info["status"] == "alive":
                self.logger.warning(f"워커 타임아웃: {name}")
                info["status"] = "dead"

    def get_system_status(self) -> dict:
        """전체 시스템 상태 조회"""
        status = {
            "uptime": self.uptime,
            "workers": {},
            "rl": None,
        }

        for name, info in self.worker_health.items():
            status["workers"][name] = {
                "status": info["status"],
                "last_seen_ago": time.time() - info["last_seen"],
            }

        if self.trainer:
            status["rl"] = {
                **self.trainer.get_stats(),
                "weights_version": self.weights_version,
            }

        return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rl", action="store_true", help="RL 트레이너 비활성화")
    parser.add_argument("--obs-dim", type=int, default=42, help="관측 차원")
    args = parser.parse_args()

    node = MainBrainNode(enable_rl=not args.no_rl, obs_dim=args.obs_dim)
    node.start()
