-- 015: decisions ↔ external_signal_log ↔ buy_score_detail 직접 연결
-- 과거 어느 시점이든 매매 판단 + 외부 정보 + 점수 상세를 한 번에 조회 가능

-- 1. decisions 테이블에 FK 컬럼 추가
ALTER TABLE decisions
  ADD COLUMN IF NOT EXISTS external_signal_id UUID REFERENCES external_signal_log(id),
  ADD COLUMN IF NOT EXISTS buy_score_id UUID REFERENCES buy_score_detail(id);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_decisions_ext_signal ON decisions(external_signal_id);
CREATE INDEX IF NOT EXISTS idx_decisions_buy_score ON decisions(buy_score_id);

-- 2. 통합 조회 뷰: 매매 판단 + 외부 정보 + 점수 상세 + 시장 컨텍스트
CREATE OR REPLACE VIEW v_decision_full AS
SELECT
  -- 매매 판단 기본
  d.id AS decision_id,
  d.created_at,
  d.cycle_id,
  d.source,
  d.decision,
  d.confidence,
  d.reason,
  d.current_price,
  d.profit_loss,
  d.trade_amount,

  -- 외부 정보 (external_signal_log)
  e.id AS ext_signal_id,
  e.fgi_value AS ext_fgi,
  e.fgi_classification AS ext_fgi_class,
  e.fusion_score,
  e.fusion_signal,
  e.long_short_ratio,
  e.funding_rate,
  e.open_interest_change,
  e.whale_score AS ext_whale_score,
  e.whale_flow,
  e.large_tx_count,
  e.news_sentiment AS ext_news_sentiment,
  e.news_score AS ext_news_score,
  e.news_count,
  e.macro_score,
  e.sp500_change,
  e.dxy_change,
  e.gold_change,
  e.kimchi_premium_pct,
  e.binance_score,
  e.eth_btc_ratio,
  e.eth_btc_trend,
  e.coingecko_score,
  e.volume_anomaly,

  -- 매수 점수 상세 (buy_score_detail)
  b.id AS buy_score_id,
  b.agent_type,
  b.threshold AS buy_threshold,
  b.total_score AS buy_total_score,
  b.fgi_score,
  b.rsi_score,
  b.sma_score,
  b.news_score AS buy_news_score,
  b.external_bonus,
  b.is_near_miss,
  b.was_ai_vetoed,
  b.points_from_threshold,

  -- 시장 컨텍스트 (market_context_log)
  m.btc_price AS ctx_btc_price,
  m.btc_24h_change,
  m.rsi_14 AS ctx_rsi,
  m.sma_20 AS ctx_sma20,
  m.adx_value,
  m.adx_regime,
  m.bb_position,
  m.fgi_value AS ctx_fgi,
  m.funding_rate AS ctx_funding_rate,
  m.long_short_ratio AS ctx_ls_ratio,
  m.kimchi_premium,
  m.whale_flow AS ctx_whale_flow,
  m.active_agent,
  m.danger_score,
  m.opportunity_score,
  m.krw_balance,
  m.btc_balance,
  m.position_ratio

FROM decisions d
LEFT JOIN external_signal_log e
  ON e.id = d.external_signal_id
LEFT JOIN buy_score_detail b
  ON b.id = d.buy_score_id
LEFT JOIN market_context_log m
  ON m.decision_id = d.id;

-- 3. cycle_id 기반 fallback 뷰 (FK 없는 과거 데이터용)
CREATE OR REPLACE VIEW v_decision_full_by_cycle AS
SELECT
  d.id AS decision_id,
  d.created_at,
  d.cycle_id,
  d.source,
  d.decision,
  d.confidence,
  d.reason,
  d.current_price,
  d.profit_loss,

  -- 외부 정보 (cycle_id 매칭)
  e.fgi_value AS ext_fgi,
  e.fusion_score,
  e.fusion_signal,
  e.whale_flow,
  e.news_sentiment AS ext_news_sentiment,
  e.macro_score,
  e.kimchi_premium_pct,

  -- 매수 점수 (cycle_id 매칭)
  b.agent_type,
  b.total_score AS buy_total_score,
  b.threshold AS buy_threshold,
  b.fgi_score,
  b.rsi_score,
  b.sma_score,
  b.news_score AS buy_news_score,
  b.external_bonus,

  -- 시장 컨텍스트
  m.danger_score,
  m.opportunity_score,
  m.active_agent

FROM decisions d
LEFT JOIN LATERAL (
  SELECT * FROM external_signal_log el
  WHERE el.cycle_id = d.cycle_id
  ORDER BY el.recorded_at DESC
  LIMIT 1
) e ON true
LEFT JOIN LATERAL (
  SELECT * FROM buy_score_detail bl
  WHERE bl.cycle_id = d.cycle_id
  ORDER BY bl.created_at DESC
  LIMIT 1
) b ON true
LEFT JOIN LATERAL (
  SELECT * FROM market_context_log ml
  WHERE ml.cycle_id = d.cycle_id
  ORDER BY ml.recorded_at DESC
  LIMIT 1
) m ON true;
