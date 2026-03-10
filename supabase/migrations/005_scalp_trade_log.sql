-- 초단타 거래 로그 (모든 거래를 최소한의 텍스트라도 반드시 기록)
-- 기존 scalp_trades는 진입/청산 별도 row → 이 테이블은 완결된 거래 1건 = 1 row

CREATE TABLE IF NOT EXISTS scalp_trade_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_date DATE NOT NULL,
  trade_no INTEGER NOT NULL,                -- 세션 내 거래 번호
  strategy TEXT NOT NULL,                    -- news / spike / whale
  entry_time TIMESTAMPTZ NOT NULL,
  exit_time TIMESTAMPTZ,
  entry_price BIGINT NOT NULL,
  exit_price BIGINT,
  amount_krw INTEGER NOT NULL,
  pnl_pct DECIMAL(6,3),
  pnl_krw INTEGER,
  exit_reason TEXT,
  signal_reason TEXT,                        -- 진입 사유
  confidence DECIMAL(3,2),
  -- 진입 시점 시장 컨텍스트
  market_trend TEXT,                         -- uptrend / downtrend / sideways
  news_sentiment TEXT,                       -- positive / negative / neutral
  news_score DECIMAL(4,2),
  fgi_value INTEGER,
  rsi_value DECIMAL(5,2),
  sma20_vs_price TEXT,                       -- above / below
  -- 사후 분석
  was_good_trade BOOLEAN,                    -- 사후 판정 (true=좋은 거래)
  lesson TEXT,                               -- 교훈/메모
  dry_run BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scalp_trade_log_date ON scalp_trade_log(session_date DESC);
CREATE INDEX idx_scalp_trade_log_strategy ON scalp_trade_log(strategy);
CREATE INDEX idx_scalp_trade_log_pnl ON scalp_trade_log(pnl_krw);
