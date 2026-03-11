"""RAG 파이프라인 — 수집 → Gemini 분석 → 임베딩 → 저장 → 검색

전체 RAG 사이클을 조율하는 오케스트레이터.
LLM Worker 노드에서 호출된다.
"""

import json
import logging
import time
from typing import Optional

from rl_hybrid.rag.gemini_client import GeminiClient
from rl_hybrid.rag.embedding_store import EmbeddingStore

logger = logging.getLogger("rag.pipeline")


class RAGPipeline:
    """RAG 전체 파이프라인"""

    def __init__(self):
        self.gemini = GeminiClient()
        self.store = EmbeddingStore()

    def analyze_and_store(
        self,
        cycle_id: str,
        market_data: dict,
        external_data: dict,
        decision_id: str = None,
    ) -> Optional[dict]:
        """시장 분석 → 임베딩 → 저장 전체 파이프라인

        1. 과거 유사 분석을 RAG 검색 (선행 임베딩 기반)
        2. Gemini에게 시장 분석 요청 (RAG 컨텍스트 포함)
        3. 분석 결과를 임베딩으로 변환
        4. pgvector에 저장

        Returns:
            Gemini 분석 결과 dict + embedding_id, 실패 시 None
        """
        start = time.time()

        # Step 1: RAG 컨텍스트 — 최근 분석으로 fallback
        rag_context = self._build_rag_context(market_data)

        # Step 2: Gemini 분석
        analysis = self.gemini.analyze_market(
            market_data=market_data,
            external_data=external_data,
            rag_context=rag_context,
        )
        if not analysis:
            return None

        # Step 3: 임베딩 생성
        cp = market_data.get("current_price", "N/A")
        btc_price = cp.get("trade_price", "N/A") if isinstance(cp, dict) else cp
        market_snapshot = {
            "btc_price": btc_price,
            "rsi": market_data.get("indicators", {}).get("rsi_14", "N/A"),
            "fgi": external_data.get("fgi", {}).get("value", "N/A") if isinstance(external_data.get("fgi"), dict) else "N/A",
        }
        embed_text = self.gemini.build_embedding_text(analysis, market_snapshot)
        embedding = self.gemini.generate_embedding(embed_text)

        if not embedding:
            logger.warning("임베딩 생성 실패 — 분석 결과만 반환")
            return analysis

        # Step 4: pgvector 저장
        record_id = self.store.store_analysis(
            cycle_id=cycle_id,
            analysis_text=embed_text,
            analysis_json=analysis,
            embedding=embedding,
            decision_id=decision_id,
        )

        analysis["_embedding_id"] = record_id
        analysis["_pipeline_time"] = round(time.time() - start, 2)

        logger.info(
            f"RAG 파이프라인 완료: {analysis['_pipeline_time']}초, "
            f"regime={analysis['market_regime']}"
        )
        return analysis

    def query_similar(
        self, query_text: str, top_k: int = 5
    ) -> list[dict]:
        """텍스트 기반 유사 분석 검색

        1. 쿼리 텍스트를 임베딩으로 변환
        2. pgvector 코사인 유사도 검색
        """
        embedding = self.gemini.generate_embedding(query_text)
        if not embedding:
            return []

        return self.store.search_similar(
            query_embedding=embedding,
            top_k=top_k,
        )

    def _build_rag_context(self, market_data: dict) -> str:
        """현재 시장 상태와 유사한 과거 분석을 RAG 컨텍스트로 구성"""
        # 현재 상태 요약 텍스트 → 임베딩 → 유사 검색
        cp = market_data.get("current_price", "N/A")
        btc_price = cp.get("trade_price", cp) if isinstance(cp, dict) else cp
        change = cp.get("signed_change_rate", "N/A") if isinstance(cp, dict) else market_data.get("change_rate_24h", "N/A")
        current_summary = (
            f"BTC {btc_price}원 "
            f"RSI {market_data.get('indicators', {}).get('rsi_14', 'N/A')} "
            f"24h변동 {change}"
        )

        embedding = self.gemini.generate_embedding(current_summary)
        if not embedding:
            # fallback: 최근 분석 텍스트 사용
            recent = self.store.get_recent_analyses(limit=3)
            if recent:
                return "\n---\n".join(r.get("analysis_text", "") for r in recent)
            return "과거 분석 데이터 없음"

        similar = self.store.search_similar(query_embedding=embedding, top_k=3)
        if not similar:
            return "과거 유사 분석 없음"

        context_parts = []
        for s in similar:
            context_parts.append(
                f"[유사도 {s['similarity']:.2f}] {s.get('analysis_text', '')}"
            )
        return "\n---\n".join(context_parts)
