-- RAG: pgvector 기반 유사 결정 검색
-- 매매 결정 시 시장 상태를 임베딩하여 저장, 유사도 검색으로 과거 경험 리콜

-- 1. pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. decisions 테이블에 임베딩 컬럼 추가
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS state_embedding vector(1536);
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS embedding_text TEXT;

-- 3. HNSW 인덱스 (코사인 유사도용)
CREATE INDEX IF NOT EXISTS idx_decisions_embedding ON decisions
USING hnsw (state_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. 유사도 검색 RPC 함수
CREATE OR REPLACE FUNCTION match_similar_decisions(
  query_embedding vector(1536),
  match_limit int DEFAULT 3
)
RETURNS TABLE (
  id uuid,
  decision text,
  reason text,
  confidence decimal,
  current_price bigint,
  trade_amount bigint,
  profit_loss decimal,
  fear_greed_value integer,
  rsi_value decimal,
  created_at timestamptz,
  cycle_id text,
  source text,
  outcome_1h_pct decimal,
  outcome_4h_pct decimal,
  outcome_24h_pct decimal,
  was_correct_24h boolean,
  embedding_text text,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.decision,
    d.reason,
    d.confidence,
    d.current_price,
    d.trade_amount,
    d.profit_loss,
    d.fear_greed_value,
    d.rsi_value,
    d.created_at,
    d.cycle_id,
    d.source,
    d.outcome_1h_pct,
    d.outcome_4h_pct,
    d.outcome_24h_pct,
    d.was_correct_24h,
    d.embedding_text,
    1 - (d.state_embedding <=> query_embedding) AS similarity
  FROM decisions d
  WHERE d.state_embedding IS NOT NULL
  ORDER BY d.state_embedding <=> query_embedding
  LIMIT match_limit;
END;
$$;
