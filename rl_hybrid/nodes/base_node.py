"""추상 기반 노드 — ZMQ 소켓 관리, 하트비트, 로깅 공통 로직

모든 노드(Main Brain, LLM Worker, Trading Worker)는 이 클래스를 상속한다.
"""

import abc
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import zmq

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.config import config, ZMQConfig
from rl_hybrid.protocol import ZMQMessage, make_heartbeat


class BaseNode(abc.ABC):
    """분산 노드 추상 기반 클래스"""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.zmq_config: ZMQConfig = config.zmq
        self.ctx = zmq.Context()
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._start_time = time.time()

        # 로깅 설정
        log_dir = os.path.join(config.project_root, config.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"node.{node_name}")
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{node_name}.log"), encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        self.logger.addHandler(fh)

        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        self.logger.addHandler(ch)

        # 시그널 핸들러 (graceful shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.logger.info(f"종료 시그널 수신 (sig={signum}), 정리 중...")
        self.stop()

    @abc.abstractmethod
    def _setup_sockets(self):
        """노드별 ZMQ 소켓 설정 (서브클래스에서 구현)"""
        pass

    @abc.abstractmethod
    def _main_loop(self):
        """노드별 메인 루프 (서브클래스에서 구현)"""
        pass

    def start(self):
        """노드 시작"""
        self.logger.info(f"=== {self.node_name} 노드 시작 ===")
        self._running = True
        self._setup_sockets()
        self._start_heartbeat()

        try:
            self._main_loop()
        except Exception as e:
            self.logger.error(f"메인 루프 에러: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """노드 정지 및 리소스 정리"""
        if not self._running:
            return
        self._running = False
        self.logger.info(f"=== {self.node_name} 노드 종료 ===")

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

        self.ctx.term()

    def send_msg(self, socket: zmq.Socket, msg: ZMQMessage, identity: bytes = None):
        """메시지 전송 (ROUTER인 경우 identity 프레임 포함)"""
        msg.sender = self.node_name
        data = msg.serialize()
        if identity:
            socket.send_multipart([identity, b"", data])
        else:
            socket.send(data)

    def recv_msg(self, socket: zmq.Socket, timeout_ms: int = None) -> Optional[ZMQMessage]:
        """메시지 수신 (타임아웃 지원)"""
        if timeout_ms:
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            events = dict(poller.poll(timeout_ms))
            if socket not in events:
                return None

        try:
            data = socket.recv(zmq.NOBLOCK)
            return ZMQMessage.deserialize(data)
        except zmq.Again:
            return None

    def recv_msg_with_identity(self, socket: zmq.Socket, timeout_ms: int = None):
        """ROUTER 소켓에서 identity 포함 수신 → (identity, ZMQMessage)"""
        if timeout_ms:
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            events = dict(poller.poll(timeout_ms))
            if socket not in events:
                return None, None

        try:
            frames = socket.recv_multipart(zmq.NOBLOCK)
            if len(frames) >= 3:
                identity = frames[0]
                msg = ZMQMessage.deserialize(frames[2])
                return identity, msg
            elif len(frames) == 2:
                identity = frames[0]
                msg = ZMQMessage.deserialize(frames[1])
                return identity, msg
        except zmq.Again:
            pass
        return None, None

    def _start_heartbeat(self):
        """백그라운드 하트비트 전송 스레드 시작"""
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        """주기적 하트비트 전송"""
        while self._running:
            try:
                self._send_heartbeat()
            except Exception as e:
                self.logger.debug(f"하트비트 전송 실패: {e}")
            time.sleep(self.zmq_config.heartbeat_interval)

    def _send_heartbeat(self):
        """서브클래스에서 오버라이드 가능한 하트비트 전송"""
        pass

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time
