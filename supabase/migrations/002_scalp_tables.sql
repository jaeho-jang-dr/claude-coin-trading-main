-- 초단타(scalp) 시스템 테이블
-- 실시간 봇의 매매기록, 고래감지, 뉴스감성, 전략알림을 저장

-- 1. 초단타 매매 기록
CREATE TABLE scalp_trades (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  strategy TEXT NOT NULL CHECK (strategy IN ('news', 'spike', 'whale')),
  side TEXT NOT NULL CHECK (side IN ('bid', 'ask')),
  entry_price BIGINT NOT NULL,
  exit_price BIGINT,
  amount_krw INTEGER NOT NULL,
  btc_qty DECIMAL(18,8) NOT NULL,
  entry_time TIMESTAMPTZ NOT NULL,
  exit_time TIMESTAMPTZ,
  exit_reason TEXT,
  pnl_pct DECIMAL(6,3),
  pnl_krw INTEGER,
  confidence DECIMAL(3,2),
  signal_reason TEXT,
  dry_run BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scalp_trades_entry ON scalp_trades(entry_time DESC);
CREATE INDEX idx_scalp_trades_strategy ON scalp_trades(strategy);
CREATE INDEX idx_scalp_trades_pnl ON scalp_trades(pnl_pct);

-- 2. 고래 감지 기록
CREATE TABLE whale_detections (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  side TEXT NOT NULL CHECK (side IN ('BID', 'ASK')),
  volume DECIMAL(18,8) NOT NULL,
  price BIGINT NOT NULL,
  krw_amount BIGINT NOT NULL,
  detected_at TIMESTAMPTZ NOT NULL,
  -- 감지 시점의 컨텍스트
  whale_buy_count INTEGER,   -- 최근 윈도우 내 고래 매수 건수
  whale_sell_count INTEGER,  -- 최근 윈도우 내 고래 매도 건수
  buy_ratio DECIMAL(5,2),    -- 매수 금액 비율
  triggered_trade BOOLEAN DEFAULT FALSE,  -- 이 고래로 인해 매매 발생했는지
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_whale_detected_at ON whale_detections(detected_at DESC);
CREATE INDEX idx_whale_side ON whale_detections(side);
CREATE INDEX idx_whale_amount ON whale_detections(krw_amount DESC);

-- 3. 뉴스 감성 이력
CREATE TABLE news_sentiment_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  sentiment TEXT NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
  score DECIMAL(4,2) NOT NULL,        -- -1.0 ~ +1.0
  prev_sentiment TEXT,                 -- 이전 감성 (급변 추적)
  source TEXT DEFAULT 'rss',           -- rss / tavily
  headline_count INTEGER,              -- 분석한 헤드라인 수
  positive_count INTEGER,
  negative_count INTEGER,
  urgent_headlines JSONB,              -- 긴급 뉴스 목록
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_news_sentiment_created ON news_sentiment_log(created_at DESC);
CREATE INDEX idx_news_sentiment_score ON news_sentiment_log(score);

-- 4. 전략 변곡점 알림 기록
CREATE TABLE strategy_alerts (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  alert_type TEXT NOT NULL CHECK (alert_type IN (
    'rsi_oversold', 'rsi_extreme_oversold', 'rsi_overbought',
    'whale_buy_reversal', 'whale_sell_pressure',
    'price_spike', 'price_crash',
    'support_break', 'resistance_break',
    'news_extreme',
    'emergency_stop'
  )),
  message TEXT NOT NULL,
  price BIGINT,
  indicator_value DECIMAL(8,2),       -- RSI값, 변동%, 감성점수 등
  sound_played BOOLEAN DEFAULT FALSE,
  telegram_sent BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_strategy_alerts_created ON strategy_alerts(created_at DESC);
CREATE INDEX idx_strategy_alerts_type ON strategy_alerts(alert_type);

-- 5. 초단타 세션 요약
CREATE TABLE scalp_sessions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  start_time TIMESTAMPTZ NOT NULL,
  end_time TIMESTAMPTZ,
  duration_min INTEGER,
  mode TEXT NOT NULL CHECK (mode IN ('dry_run', 'live')),
  total_trades INTEGER DEFAULT 0,
  wins INTEGER DEFAULT 0,
  losses INTEGER DEFAULT 0,
  win_rate DECIMAL(5,2),
  total_pnl_krw INTEGER DEFAULT 0,
  budget INTEGER NOT NULL,
  -- 세션 중 시장 상태
  start_price BIGINT,
  end_price BIGINT,
  price_change_pct DECIMAL(6,3),
  whale_count INTEGER DEFAULT 0,
  news_sentiment_avg DECIMAL(4,2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scalp_sessions_start ON scalp_sessions(start_time DESC);

-- 6. AI 복합 시그널 이력
CREATE TABLE ai_signal_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  composite_score INTEGER NOT NULL,     -- -85 ~ +85
  interpretation TEXT NOT NULL,          -- strong_buy, weak_buy, neutral, weak_sell, strong_sell
  orderbook_imbalance INTEGER,
  trade_pressure INTEGER,
  whale_direction INTEGER,
  tf_divergence INTEGER,
  volatility_regime INTEGER,
  volume_anomaly INTEGER,
  price BIGINT,
  details JSONB,                         -- 상세 데이터
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ai_signal_created ON ai_signal_log(created_at DESC);
CREATE INDEX idx_ai_signal_score ON ai_signal_log(composite_score);
