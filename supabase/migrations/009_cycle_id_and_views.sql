-- 통합 cycle_id: 모든 거래 관련 테이블을 하나의 사이클로 연결
-- 형식: YYYYMMDD-HHmm-{source} (예: 20260310-1400-agent)

-- 1. 기존 테이블에 cycle_id 추가
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'unknown';
ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE external_signal_log ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE market_context_log ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE agent_switches ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE scalp_trades ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE scalp_trade_log ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE scalp_sessions ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE ai_signal_log ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE whale_detections ADD COLUMN IF NOT EXISTS cycle_id TEXT;
ALTER TABLE news_sentiment_log ADD COLUMN IF NOT EXISTS cycle_id TEXT;

-- 2. cycle_id 인덱스 (모든 테이블 통일)
CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON decisions(cycle_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_cycle ON portfolio_snapshots(cycle_id);
CREATE INDEX IF NOT EXISTS idx_market_data_cycle ON market_data(cycle_id);
CREATE INDEX IF NOT EXISTS idx_exec_logs_cycle ON execution_logs(cycle_id);
CREATE INDEX IF NOT EXISTS idx_ext_signal_cycle ON external_signal_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_buy_score_cycle ON buy_score_detail(cycle_id);
CREATE INDEX IF NOT EXISTS idx_mkt_ctx_cycle ON market_context_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_agent_switch_cycle ON agent_switches(cycle_id);
CREATE INDEX IF NOT EXISTS idx_scalp_trades_cycle ON scalp_trades(cycle_id);
CREATE INDEX IF NOT EXISTS idx_scalp_trade_log_cycle ON scalp_trade_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_scalp_sessions_cycle ON scalp_sessions(cycle_id);
CREATE INDEX IF NOT EXISTS idx_ai_signal_cycle ON ai_signal_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_whale_cycle ON whale_detections(cycle_id);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_cycle ON news_sentiment_log(cycle_id);

-- 3. source 인덱스 (decisions)
CREATE INDEX IF NOT EXISTS idx_decisions_source ON decisions(source);

-- 4. 복합 인덱스: 날짜+소스 (빠른 일별 조회)
CREATE INDEX IF NOT EXISTS idx_decisions_date_source ON decisions(((created_at AT TIME ZONE 'Asia/Seoul')::date), source);

-- ============================================================
-- 리콜 뷰: 한 번의 쿼리로 전체 컨텍스트 조회
-- ============================================================

-- V1: 단타 거래 전체 맥락 (decisions + 점수 + 시장 + 외부시그널)
CREATE OR REPLACE VIEW v_trade_recall AS
SELECT
  d.id AS decision_id,
  d.cycle_id,
  d.source,
  d.created_at,
  d.decision AS action,
  d.confidence,
  d.current_price AS price,
  d.trade_amount,
  d.reason,
  d.profit_loss,
  d.executed,
  -- 매수 점수 상세
  bs.total_score AS buy_total,
  bs.fgi_score AS buy_fgi,
  bs.rsi_score AS buy_rsi,
  bs.sma_score AS buy_sma,
  bs.news_score AS buy_news,
  bs.external_bonus AS buy_ext,
  bs.agent_type,
  bs.threshold AS buy_threshold,
  -- 시장 컨텍스트
  mc.rsi_14,
  mc.sma_20,
  mc.fgi_value,
  mc.fgi_class,
  mc.funding_rate,
  mc.long_short_ratio,
  mc.kimchi_premium,
  mc.whale_flow,
  mc.macro_sentiment,
  mc.eth_btc_trend,
  mc.active_agent,
  mc.danger_score,
  mc.opportunity_score,
  mc.position_ratio,
  -- 외부 시그널
  es.fusion_score AS ext_fusion,
  es.fusion_signal AS ext_signal,
  es.binance_score AS ext_binance,
  es.whale_score AS ext_whale,
  es.macro_score AS ext_macro,
  es.coingecko_score AS ext_coingecko,
  es.eth_btc_score AS ext_ethbtc
FROM decisions d
LEFT JOIN buy_score_detail bs ON bs.decision_id = d.id
LEFT JOIN market_context_log mc ON mc.decision_id = d.id
LEFT JOIN LATERAL (
  SELECT * FROM external_signal_log e
  WHERE e.recorded_at BETWEEN d.created_at - INTERVAL '5 minutes' AND d.created_at + INTERVAL '5 minutes'
  ORDER BY ABS(EXTRACT(EPOCH FROM (e.recorded_at - d.created_at)))
  LIMIT 1
) es ON true
ORDER BY d.created_at DESC;

-- V2: 초단타 거래 전체 맥락
CREATE OR REPLACE VIEW v_scalp_recall AS
SELECT
  t.id,
  t.cycle_id,
  t.session_date,
  t.trade_no,
  t.strategy,
  t.entry_time,
  t.exit_time,
  t.entry_price,
  t.exit_price,
  t.amount_krw,
  t.pnl_pct,
  t.pnl_krw,
  t.exit_reason,
  t.signal_reason,
  t.confidence,
  t.market_trend,
  t.news_sentiment,
  t.news_score,
  t.fgi_value,
  t.rsi_value,
  t.sma20_vs_price,
  t.was_good_trade,
  t.lesson,
  t.dry_run
FROM scalp_trade_log t
ORDER BY t.entry_time DESC;

-- V3: 일별 요약 (토큰 최소화)
CREATE OR REPLACE VIEW v_daily_summary AS
SELECT
  trade_date,
  total_decisions,
  buys,
  sells,
  holds,
  avg_confidence,
  total_pnl,
  avg_buy_score,
  (SELECT COUNT(*) FROM scalp_trade_log s WHERE s.session_date = sub.trade_date) AS scalp_trades,
  (SELECT SUM(s.pnl_krw) FROM scalp_trade_log s WHERE s.session_date = sub.trade_date) AS scalp_pnl,
  avg_fgi,
  avg_rsi
FROM (
  SELECT
    d.created_at::date AS trade_date,
    COUNT(*) AS total_decisions,
    COUNT(*) FILTER (WHERE d.decision = '매수') AS buys,
    COUNT(*) FILTER (WHERE d.decision = '매도') AS sells,
    COUNT(*) FILTER (WHERE d.decision = '관망') AS holds,
    ROUND(AVG(d.confidence)::numeric, 2) AS avg_confidence,
    SUM(d.profit_loss) AS total_pnl,
    ROUND(AVG(bs.total_score)::numeric, 1) AS avg_buy_score,
    ROUND(AVG(mc.fgi_value)::numeric, 0) AS avg_fgi,
    ROUND(AVG(mc.rsi_14)::numeric, 1) AS avg_rsi
  FROM decisions d
  LEFT JOIN buy_score_detail bs ON bs.decision_id = d.id
  LEFT JOIN market_context_log mc ON mc.decision_id = d.id
  GROUP BY d.created_at::date
) sub
ORDER BY trade_date DESC;

-- V4: cycle 단위 조회 (cycle_id로 모든 것 한번에)
CREATE OR REPLACE VIEW v_cycle_detail AS
SELECT
  d.cycle_id,
  d.source,
  d.created_at,
  d.decision AS action,
  d.current_price AS price,
  d.trade_amount,
  d.confidence,
  d.profit_loss,
  SUBSTRING(d.reason, 1, 100) AS reason_short,
  bs.total_score,
  bs.agent_type,
  mc.fgi_value,
  mc.rsi_14,
  mc.danger_score,
  mc.opportunity_score,
  es.fusion_signal AS ext_signal,
  es.fusion_score AS ext_score
FROM decisions d
LEFT JOIN buy_score_detail bs ON bs.decision_id = d.id
LEFT JOIN market_context_log mc ON mc.decision_id = d.id
LEFT JOIN LATERAL (
  SELECT * FROM external_signal_log e
  WHERE e.cycle_id = d.cycle_id
  LIMIT 1
) es ON true
WHERE d.cycle_id IS NOT NULL
ORDER BY d.created_at DESC;
