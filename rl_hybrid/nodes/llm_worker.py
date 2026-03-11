"""LLM/RAG 워커 노드 — Gemini 분석 + pgvector RAG 전담

역할:
  - Main Brain의 analyze_market 요청 → Gemini API 분석 → 결과 반환
  - rag_query 요청 → pgvector 유사 검색 → 결과 반환
  - embed_and_store 요청 → 임베딩 생성 + 저장
  - PUB 채널 구독하여 시장 데이터 캐시
"""

import json
import logging
import os
import sys
import time
from typing import Optional

import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.nodes.base_node import BaseNode
from rl_hybrid.config import config
from rl_hybrid.protocol import (
    ZMQMessage, MsgType, Action, make_heartbeat,
)
from rl_hybrid.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger("node.llm_worker")


class LLMWorkerNode(BaseNode):
    """LLM/RAG 워커 — DEALER + SUB"""

    def __init__(self):
        super().__init__("llm_worker")
        self.dealer: Optional[zmq.Socket] = None
        self.sub: Optional[zmq.Socket] = None
        self.pipeline: Optional[RAGPipeline] = None
        self.latest_market_data: dict = {}

    def _setup_sockets(self):
        # DEALER → Main Brain ROUTER
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, self.node_name)
        self.dealer.connect(self.zmq_config.router_addr)
        self.logger.info(f"DEALER 연결: {self.zmq_config.router_addr}")

        # SUB → Main Brain PUB (시장 데이터 수신)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.zmq_config.pub_addr)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # 모든 토픽 구독
        self.logger.info(f"SUB 연결: {self.zmq_config.pub_addr}")

        # RAG 파이프라인 초기화
        try:
            self.pipeline = RAGPipeline()
            self.logger.info("RAG 파이프라인 초기화 완료")
        except Exception as e:
            self.logger.error(f"RAG 파이프라인 초기화 실패: {e}")

    def _send_heartbeat(self):
        """Main Brain에 하트비트 전송"""
        msg = make_heartbeat(self.node_name, status="alive", uptime=self.uptime)
        try:
            self.dealer.send(b"" + msg.serialize())
        except Exception:
            pass

    def _main_loop(self):
        """메시지 수신 루프"""
        poller = zmq.Poller()
        poller.register(self.dealer, zmq.POLLIN)
        poller.register(self.sub, zmq.POLLIN)

        self.logger.info("LLM Worker 메인 루프 시작")

        while self._running:
            events = dict(poller.poll(100))

            # DEALER: Main Brain 요청 처리
            if self.dealer in events:
                self._handle_dealer_message()

            # SUB: 시장 데이터 캐시
            if self.sub in events:
                self._handle_sub_message()

    def _handle_dealer_message(self):
        """Main Brain 요청 처리"""
        try:
            frames = self.dealer.recv_multipart(zmq.NOBLOCK)
            # DEALER는 빈 프레임 제거 후 데이터 프레임만
            data = frames[-1] if frames else None
            if not data:
                return

            msg = ZMQMessage.deserialize(data)
        except Exception as e:
            self.logger.error(f"메시지 수신 에러: {e}")
            return

        self.logger.info(f"요청 수신: action={msg.action}, req_id={msg.request_id}")

        handlers = {
            Action.ANALYZE_MARKET.value: self._handle_analyze,
            Action.RAG_QUERY.value: self._handle_rag_query,
            Action.EMBED_AND_STORE.value: self._handle_embed_store,
        }

        handler = handlers.get(msg.action)
        if handler:
            response = handler(msg)
            if response:
                response.sender = self.node_name
                self.dealer.send(b"" + response.serialize())
        else:
            self.logger.warning(f"알 수 없는 액션: {msg.action}")

    def _handle_sub_message(self):
        """PUB 채널 시장 데이터 캐시"""
        try:
            data = self.sub.recv(zmq.NOBLOCK)
            msg = ZMQMessage.deserialize(data)
            if msg.action == Action.MARKET_UPDATE.value:
                self.latest_market_data = msg.payload
                self.logger.debug("시장 데이터 캐시 갱신")
        except Exception:
            pass

    def _handle_analyze(self, msg: ZMQMessage) -> Optional[ZMQMessage]:
        """시장 분석 요청 처리"""
        if not self.pipeline:
            return msg.reply({}, error="RAG 파이프라인 미초기화")

        try:
            market_data = msg.payload.get("market_data", {})
            external_data = msg.payload.get("external_data", {})
            context = msg.payload.get("context", {})
            cycle_id = context.get("cycle_id", f"cycle_{int(time.time())}")

            # 외부 데이터가 비어있으면 자체 수집
            if not external_data:
                external_data = self._collect_external_data()

            analysis = self.pipeline.analyze_and_store(
                cycle_id=cycle_id,
                market_data=market_data,
                external_data=external_data,
            )

            if analysis:
                return msg.reply(analysis)
            else:
                return msg.reply({}, error="Gemini 분석 실패")

        except Exception as e:
            self.logger.error(f"분석 처리 에러: {e}", exc_info=True)
            return msg.reply({}, error=str(e))

    def _handle_rag_query(self, msg: ZMQMessage) -> Optional[ZMQMessage]:
        """RAG 유사 검색 요청 처리"""
        if not self.pipeline:
            return msg.reply({"results": []}, error="RAG 파이프라인 미초기화")

        try:
            query_text = msg.payload.get("query_text", "")
            top_k = msg.payload.get("top_k", 5)

            results = self.pipeline.query_similar(query_text, top_k)

            # 직렬화를 위해 embedding 필드 제거
            for r in results:
                r.pop("embedding", None)

            return msg.reply({"results": results})

        except Exception as e:
            return msg.reply({"results": []}, error=str(e))

    def _handle_embed_store(self, msg: ZMQMessage) -> Optional[ZMQMessage]:
        """임베딩 생성 + 저장 요청 처리"""
        if not self.pipeline:
            return msg.reply({}, error="RAG 파이프라인 미초기화")

        try:
            cycle_id = msg.payload.get("cycle_id", "")
            analysis = msg.payload.get("analysis", {})
            market_state = msg.payload.get("market_state", {})
            decision_id = msg.payload.get("decision_id")

            embed_text = self.pipeline.gemini.build_embedding_text(analysis, market_state)
            embedding = self.pipeline.gemini.generate_embedding(embed_text)

            if embedding:
                record_id = self.pipeline.store.store_analysis(
                    cycle_id=cycle_id,
                    analysis_text=embed_text,
                    analysis_json=analysis,
                    embedding=embedding,
                    decision_id=decision_id,
                )
                return msg.reply({"embedding_id": record_id, "success": True})

            return msg.reply({"success": False}, error="임베딩 생성 실패")

        except Exception as e:
            return msg.reply({"success": False}, error=str(e))

    def _collect_external_data(self) -> dict:
        """외부 데이터 자체 수집 (FGI, 뉴스 등)"""
        external = {}
        try:
            import subprocess
            # FGI 수집
            result = subprocess.run(
                [sys.executable, "scripts/collect_fear_greed.py"],
                capture_output=True, text=True, timeout=15,
                cwd=config.project_root,
            )
            if result.returncode == 0:
                external["fgi"] = json.loads(result.stdout)
        except Exception as e:
            self.logger.debug(f"외부 데이터 수집 에러: {e}")

        return external


if __name__ == "__main__":
    node = LLMWorkerNode()
    node.start()
