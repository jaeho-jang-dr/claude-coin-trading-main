CREATE TABLE IF NOT EXISTS buy_score_detail (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  decision_id UUID,
  recorded_at TIMESTAMPTZ DEFAULT NOW(),
  -- Agent info
  agent_type TEXT NOT NULL,
  threshold INTEGER NOT NULL,
  -- Individual scores
  fgi_score DECIMAL(5,2),
  fgi_value INTEGER,
  rsi_score DECIMAL(5,2),
  rsi_value DECIMAL(5,2),
  sma_score DECIMAL(5,2),
  sma_position TEXT,
  news_score DECIMAL(5,2),
  news_sentiment DECIMAL(4,2),
  external_bonus DECIMAL(5,2),
  external_signal TEXT,
  -- Total
  total_score DECIMAL(5,2) NOT NULL,
  -- Decision
  action TEXT NOT NULL,
  buy_amount INTEGER,
  sell_pct DECIMAL(5,2),
  confidence DECIMAL(3,2),
  reason TEXT,
  -- Market snapshot
  btc_price BIGINT,
  market_trend TEXT,
  adx_regime TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_buy_score_recorded ON buy_score_detail(recorded_at DESC);
CREATE INDEX idx_buy_score_agent ON buy_score_detail(agent_type);
CREATE INDEX idx_buy_score_action ON buy_score_detail(action);
