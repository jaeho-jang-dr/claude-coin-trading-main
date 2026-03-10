CREATE TABLE IF NOT EXISTS external_signal_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  recorded_at TIMESTAMPTZ DEFAULT NOW(),
  -- Binance Futures
  binance_score INTEGER,
  long_short_ratio DECIMAL(5,3),
  funding_rate DECIMAL(8,6),
  open_interest_change DECIMAL(6,2),
  -- Whale / On-chain
  whale_score INTEGER,
  whale_flow TEXT,
  large_tx_count INTEGER,
  exchange_inflow DECIMAL(10,2),
  exchange_outflow DECIMAL(10,2),
  -- Macro
  macro_score INTEGER,
  sp500_change DECIMAL(5,2),
  dxy_change DECIMAL(5,2),
  gold_change DECIMAL(5,2),
  us10y_change DECIMAL(5,2),
  -- Market sentiment
  fgi_value INTEGER,
  fgi_classification TEXT,
  news_sentiment DECIMAL(4,2),
  news_score INTEGER,
  news_count INTEGER,
  -- ETH/BTC
  eth_btc_ratio DECIMAL(8,6),
  eth_btc_score INTEGER,
  eth_btc_trend TEXT,
  -- CoinGecko
  coingecko_score INTEGER,
  volume_anomaly BOOLEAN,
  volume_change_pct DECIMAL(6,2),
  -- Kimchi Premium
  kimchi_premium_pct DECIMAL(5,2),
  -- Fusion result
  fusion_score INTEGER,
  fusion_signal TEXT,
  fusion_details JSONB,
  -- Context
  source TEXT DEFAULT 'agent',
  session_type TEXT
);

CREATE INDEX idx_ext_signal_recorded ON external_signal_log(recorded_at DESC);
CREATE INDEX idx_ext_signal_fusion ON external_signal_log(fusion_signal);
