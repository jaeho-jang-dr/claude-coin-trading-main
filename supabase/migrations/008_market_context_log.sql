-- 008_market_context_log.sql
-- 매매 결정 시점의 전체 시장 컨텍스트를 기록하는 테이블
-- "왜 이 시점에 매수/매도했는가?"를 사후 분석하기 위한 풀 스냅샷

CREATE TABLE IF NOT EXISTS market_context_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  decision_id UUID,
  recorded_at TIMESTAMPTZ DEFAULT NOW(),
  -- Price
  btc_price BIGINT,
  btc_24h_change DECIMAL(6,2),
  btc_volume DECIMAL(15,2),
  -- Technical indicators
  rsi_14 DECIMAL(5,2),
  sma_20 BIGINT,
  sma_vs_price TEXT,
  macd_value DECIMAL(10,2),
  macd_signal DECIMAL(10,2),
  macd_histogram DECIMAL(10,2),
  bb_upper BIGINT,
  bb_lower BIGINT,
  bb_position TEXT,
  adx_value DECIMAL(5,2),
  adx_regime TEXT,
  -- Sentiment
  fgi_value INTEGER,
  fgi_class TEXT,
  news_sentiment DECIMAL(4,2),
  news_positive INTEGER,
  news_negative INTEGER,
  news_neutral INTEGER,
  -- Binance Derivatives
  funding_rate DECIMAL(8,6),
  long_short_ratio DECIMAL(5,3),
  open_interest DECIMAL(15,2),
  oi_change_pct DECIMAL(6,2),
  kimchi_premium DECIMAL(5,2),
  -- Whale / On-chain
  whale_flow TEXT,
  large_tx_count INTEGER,
  exchange_net_flow DECIMAL(10,2),
  -- Macro
  sp500_trend TEXT,
  dxy_trend TEXT,
  gold_trend TEXT,
  macro_sentiment TEXT,
  -- ETH/BTC
  eth_btc_ratio DECIMAL(8,6),
  eth_btc_trend TEXT,
  -- Portfolio state at decision time
  krw_balance BIGINT,
  btc_balance DECIMAL(10,8),
  btc_avg_price BIGINT,
  total_asset_value BIGINT,
  position_ratio DECIMAL(4,2),
  -- Agent state
  active_agent TEXT,
  danger_score INTEGER,
  opportunity_score INTEGER,
  -- Full data as JSON backup
  raw_data JSONB
);

CREATE INDEX idx_mkt_ctx_recorded ON market_context_log(recorded_at DESC);
CREATE INDEX idx_mkt_ctx_decision ON market_context_log(decision_id);
