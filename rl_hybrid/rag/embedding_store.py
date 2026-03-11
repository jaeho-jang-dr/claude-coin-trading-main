"""pgvector 임베딩 저장소 — Supabase REST API 기반

rag_analysis_vectors 테이블에 Gemini 분석 임베딩을 저장하고,
RPC 함수로 코사인 유사도 검색을 수행한다.

DB 직접 연결(psycopg2) 대신 Supabase REST API를 사용하여
pooler 연결 문제를 우회한다.
"""

import json
import logging
from typing import Optional

import requests

from rl_hybrid.config import config

logger = logging.getLogger("rag.embedding_store")


class EmbeddingStore:
    """Supabase REST API 기반 임베딩 CRUD"""

    def __init__(self):
        self.base_url = config.supabase.url
        self.api_key = config.supabase.service_role_key
        if not self.base_url or not self.api_key:
            raise ValueError("SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY 환경변수가 필요합니다")

        self.headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def store_analysis(
        self,
        cycle_id: str,
        analysis_text: str,
        analysis_json: dict,
        embedding: list[float],
        market_regime: str = None,
        decision_id: str = None,
    ) -> Optional[str]:
        """분석 결과 + 임베딩 저장

        Returns:
            생성된 레코드 UUID, 실패 시 None
        """
        try:
            payload = {
                "cycle_id": cycle_id,
                "analysis_text": analysis_text,
                "analysis_json": analysis_json,
                "market_regime": market_regime or analysis_json.get("market_regime"),
                "embedding": embedding,
            }
            if decision_id:
                payload["decision_id"] = decision_id

            resp = requests.post(
                f"{self.base_url}/rest/v1/rag_analysis_vectors",
                headers=self.headers,
                json=payload,
                timeout=15,
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                record_id = data[0]["id"] if isinstance(data, list) and data else None
                logger.info(f"임베딩 저장 완료: id={record_id}, cycle={cycle_id}")
                return record_id
            else:
                logger.error(f"임베딩 저장 실패: {resp.status_code} {resp.text[:300]}")
                return None

        except Exception as e:
            logger.error(f"임베딩 저장 실패: {e}")
            return None

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = None,
        threshold: float = None,
    ) -> list[dict]:
        """RPC 함수를 통한 코사인 유사도 검색

        Returns:
            [{"id", "cycle_id", "analysis_text", "analysis_json",
              "market_regime", "similarity", "created_at"}, ...]
        """
        top_k = top_k or config.rag.top_k
        threshold = threshold or config.rag.similarity_threshold

        try:
            # match_similar_analyses RPC 호출
            resp = requests.post(
                f"{self.base_url}/rest/v1/rpc/match_similar_analyses",
                headers=self.headers,
                json={
                    "query_embedding": query_embedding,
                    "match_limit": top_k,
                    "min_similarity": threshold,
                },
                timeout=15,
            )

            if resp.status_code == 200:
                results = resp.json()
                logger.info(f"유사 분석 검색: {len(results)}건 (threshold={threshold})")
                return results
            else:
                logger.error(f"유사 분석 검색 실패: {resp.status_code} {resp.text[:300]}")
                return []

        except Exception as e:
            logger.error(f"유사 분석 검색 실패: {e}")
            return []

    def get_recent_analyses(self, limit: int = 10) -> list[dict]:
        """최근 분석 조회 (RAG 컨텍스트 구성용)"""
        try:
            resp = requests.get(
                f"{self.base_url}/rest/v1/rag_analysis_vectors",
                headers=self.headers,
                params={
                    "select": "cycle_id,analysis_text,analysis_json,market_regime,created_at",
                    "order": "created_at.desc",
                    "limit": limit,
                },
                timeout=10,
            )

            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"최근 분석 조회 실패: {resp.status_code}")
                return []

        except Exception as e:
            logger.error(f"최근 분석 조회 실패: {e}")
            return []
