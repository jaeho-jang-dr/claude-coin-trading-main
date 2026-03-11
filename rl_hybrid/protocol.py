"""ZeroMQ 메시지 프로토콜 — 노드 간 통신 규격 정의

모든 노드 간 메시지는 ZMQMessage 포맷으로 직렬화(msgpack)하여 교환한다.
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import msgpack


class MsgType(str, Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    HEARTBEAT = "HEARTBEAT"
    BROADCAST = "BROADCAST"


class Action(str, Enum):
    # Brain → LLM Worker
    ANALYZE_MARKET = "analyze_market"
    RAG_QUERY = "rag_query"
    EMBED_AND_STORE = "embed_and_store"

    # LLM Worker → Brain
    ANALYSIS_RESULT = "analysis_result"
    RAG_RESULT = "rag_result"
    EMBED_RESULT = "embed_result"

    # Brain → Trading Worker
    EXECUTE_TRADE = "execute_trade"
    GET_PORTFOLIO = "get_portfolio"
    SAFETY_CHECK = "safety_check"

    # Trading Worker → Brain
    TRADE_RESULT = "trade_result"
    PORTFOLIO_STATE = "portfolio_state"
    SAFETY_RESULT = "safety_result"

    # Broadcast (PUB/SUB)
    MARKET_UPDATE = "market_update"

    # === Phase 3: 분산 RL 학습 ===
    # RL Worker → Brain
    SUBMIT_TRAJECTORY = "submit_trajectory"
    REQUEST_WEIGHTS = "request_weights"

    # Brain → RL Worker
    TRAJECTORY_ACK = "trajectory_ack"
    WEIGHTS_UPDATE = "weights_update"
    START_ROLLOUT = "start_rollout"

    # Brain → LLM Worker (RL용)
    REQUEST_EMBEDDING = "request_embedding"
    EMBEDDING_RESULT = "embedding_result"

    # Health
    HEARTBEAT = "heartbeat"


@dataclass
class ZMQMessage:
    """노드 간 교환되는 메시지 봉투"""
    msg_type: str
    action: str
    sender: str
    payload: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def serialize(self) -> bytes:
        """msgpack 직렬화"""
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "ZMQMessage":
        """msgpack 역직렬화"""
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)

    def reply(self, payload: dict, error: str = None) -> "ZMQMessage":
        """요청에 대한 응답 메시지 생성"""
        # action 매핑: request → response
        response_map = {
            Action.ANALYZE_MARKET.value: Action.ANALYSIS_RESULT.value,
            Action.RAG_QUERY.value: Action.RAG_RESULT.value,
            Action.EMBED_AND_STORE.value: Action.EMBED_RESULT.value,
            Action.EXECUTE_TRADE.value: Action.TRADE_RESULT.value,
            Action.GET_PORTFOLIO.value: Action.PORTFOLIO_STATE.value,
            Action.SAFETY_CHECK.value: Action.SAFETY_RESULT.value,
            # Phase 3: RL
            Action.SUBMIT_TRAJECTORY.value: Action.TRAJECTORY_ACK.value,
            Action.REQUEST_WEIGHTS.value: Action.WEIGHTS_UPDATE.value,
            Action.REQUEST_EMBEDDING.value: Action.EMBEDDING_RESULT.value,
            Action.START_ROLLOUT.value: Action.TRAJECTORY_ACK.value,
        }
        return ZMQMessage(
            msg_type=MsgType.RESPONSE.value,
            action=response_map.get(self.action, self.action),
            sender="",  # 응답 노드가 채움
            payload=payload,
            request_id=self.request_id,
            error=error,
        )


def make_heartbeat(sender: str, status: str = "alive", **extra) -> ZMQMessage:
    """하트비트 메시지 생성"""
    return ZMQMessage(
        msg_type=MsgType.HEARTBEAT.value,
        action=Action.HEARTBEAT.value,
        sender=sender,
        payload={"status": status, **extra},
    )


def make_request(action: Action, sender: str, payload: dict) -> ZMQMessage:
    """요청 메시지 생성 헬퍼"""
    return ZMQMessage(
        msg_type=MsgType.REQUEST.value,
        action=action.value,
        sender=sender,
        payload=payload,
    )


def make_broadcast(action: Action, sender: str, payload: dict) -> ZMQMessage:
    """브로드캐스트 메시지 생성"""
    return ZMQMessage(
        msg_type=MsgType.BROADCAST.value,
        action=action.value,
        sender=sender,
        payload=payload,
    )
