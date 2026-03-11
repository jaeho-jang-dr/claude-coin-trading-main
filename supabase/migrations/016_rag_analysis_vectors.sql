-- ======================================================
-- RAG 분석 벡터 테이블 — Gemini 임베딩 저장소
-- Phase 1: LLM-RL 하이브리드 분산 시스템
-- ======================================================

-- pgvector 확장 활성화 (이미 있으면 무시)
CREATE EXTENSION IF NOT EXISTS vector;

-- Gemini 분석 결과 + 768차원 임베딩 저장 테이블
CREATE TABLE IF NOT EXISTS rag_analysis_vectors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    cycle_id TEXT,
    decision_id UUID REFERENCES decisions(id) ON DELETE SET NULL,
    analysis_text TEXT NOT NULL,
    analysis_json JSONB,
    market_regime TEXT,
    embedding vector(3072),
    gemini_model TEXT DEFAULT 'gemini-2.5-pro',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 벡터 인덱스 생략: pgvector의 HNSW/IVFFlat 모두 2000d 제한
-- 3072d gemini-embedding-001에는 순차 스캔 사용 (수천 건 이하에서는 충분)
-- 대규모 데이터 시 차원 축소(PCA) 후 인덱스 적용 권장

-- 시간순 조회 인덱스
CREATE INDEX IF NOT EXISTS idx_rag_analysis_created
    ON rag_analysis_vectors (created_at DESC);

-- cycle_id 조회 인덱스
CREATE INDEX IF NOT EXISTS idx_rag_analysis_cycle
    ON rag_analysis_vectors (cycle_id);

-- market_regime 필터 인덱스
CREATE INDEX IF NOT EXISTS idx_rag_analysis_regime
    ON rag_analysis_vectors (market_regime);

-- ======================================================
-- 유사 분석 검색 RPC 함수
-- 사용: SELECT * FROM match_similar_analyses(query_vec, 5);
-- ======================================================
CREATE OR REPLACE FUNCTION match_similar_analyses(
    query_embedding vector(3072),
    match_limit int DEFAULT 5,
    min_similarity float DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    cycle_id TEXT,
    analysis_text TEXT,
    analysis_json JSONB,
    market_regime TEXT,
    similarity float,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.cycle_id,
        r.analysis_text,
        r.analysis_json,
        r.market_regime,
        (1 - (r.embedding <=> query_embedding))::float AS similarity,
        r.created_at
    FROM rag_analysis_vectors r
    WHERE r.embedding IS NOT NULL
      AND (1 - (r.embedding <=> query_embedding)) >= min_similarity
    ORDER BY r.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$;

-- ======================================================
-- 코멘트
-- ======================================================
COMMENT ON TABLE rag_analysis_vectors IS
    'Gemini 시장 분석 결과 + gemini-embedding-001 벡터 저장소 (3072d)';
COMMENT ON FUNCTION match_similar_analyses IS
    '코사인 유사도 기반 과거 유사 분석 검색 (RAG retrieval)';
