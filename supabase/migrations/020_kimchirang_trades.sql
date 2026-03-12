-- Kimchirang 차익거래 기록 테이블
CREATE TABLE IF NOT EXISTS kimchirang_trades (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  action TEXT NOT NULL CHECK (action IN ('enter', 'exit', 'stop_loss')),
  kp_at_execution DECIMAL(8,4) NOT NULL,
  entry_kp DECIMAL(8,4),
  exit_kp DECIMAL(8,4),
  pnl_pct DECIMAL(8,4),

  -- Upbit 레그
  upbit_order_id TEXT,
  upbit_side TEXT,
  upbit_price BIGINT,
  upbit_qty DECIMAL(18,8),

  -- Binance 레그
  binance_order_id TEXT,
  binance_side TEXT,
  binance_price DECIMAL(18,4),
  binance_qty DECIMAL(18,8),

  -- 메타
  fx_rate DECIMAL(10,2),
  funding_rate DECIMAL(10,6),
  spread_cost DECIMAL(8,4),
  latency_ms INTEGER,
  dry_run BOOLEAN DEFAULT TRUE,
  both_success BOOLEAN,
  hold_duration_min DECIMAL(10,1),

  -- KP 통계 스냅샷
  kp_stats JSONB,

  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kimchirang_trades_created ON kimchirang_trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_kimchirang_trades_action ON kimchirang_trades(action);
