-- Kimchirang 확장 테이블: KP 히스토리 + RL 모델 성과 + 세션 기록

-- 1. KP 히스토리 (1분 간격 스냅샷, 패턴 분석용)
CREATE TABLE IF NOT EXISTS kimchirang_kp_history (
  id BIGSERIAL PRIMARY KEY,
  mid_kp DECIMAL(8,4) NOT NULL,
  entry_kp DECIMAL(8,4),
  exit_kp DECIMAL(8,4),
  kp_ma_1m DECIMAL(8,4),
  kp_ma_5m DECIMAL(8,4),
  kp_z_score DECIMAL(8,4),
  kp_velocity DECIMAL(8,4),
  spread_cost DECIMAL(8,4),
  funding_rate DECIMAL(10,6),
  upbit_bid BIGINT,
  upbit_ask BIGINT,
  binance_bid DECIMAL(18,4),
  binance_ask DECIMAL(18,4),
  fx_rate DECIMAL(10,2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kr_kp_history_created ON kimchirang_kp_history(created_at DESC);

-- 2. RL 모델 성과 기록
CREATE TABLE IF NOT EXISTS kimchirang_rl_models (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  training_days INTEGER,
  training_steps INTEGER,
  eval_pnl_pct DECIMAL(8,2),
  eval_trades INTEGER,
  eval_reward DECIMAL(10,2),
  kp_data_range TEXT,           -- "2024-01-01 ~ 2026-03-12"
  kp_min DECIMAL(8,4),
  kp_max DECIMAL(8,4),
  kp_mean DECIMAL(8,4),
  hyperparams JSONB,            -- lr, ent_coef, etc.
  notes TEXT,
  deployed BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 세션 기록 (봇 시작/종료, 거래 집계)
CREATE TABLE IF NOT EXISTS kimchirang_sessions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  start_time TIMESTAMPTZ NOT NULL,
  end_time TIMESTAMPTZ,
  mode TEXT DEFAULT 'dry_run',  -- dry_run / live
  rl_model_version TEXT,
  total_trades INTEGER DEFAULT 0,
  total_pnl_pct DECIMAL(8,4) DEFAULT 0,
  best_trade_pnl DECIMAL(8,4),
  worst_trade_pnl DECIMAL(8,4),
  avg_hold_min DECIMAL(10,1),
  avg_kp_entry DECIMAL(8,4),
  avg_kp_exit DECIMAL(8,4),
  errors INTEGER DEFAULT 0,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. kimchirang_trades에 세션/RL 관련 컬럼 추가
ALTER TABLE kimchirang_trades ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES kimchirang_sessions(id);
ALTER TABLE kimchirang_trades ADD COLUMN IF NOT EXISTS rl_action INTEGER;        -- 0=Hold, 1=Enter, 2=Exit
ALTER TABLE kimchirang_trades ADD COLUMN IF NOT EXISTS rl_confidence DECIMAL(5,4);
ALTER TABLE kimchirang_trades ADD COLUMN IF NOT EXISTS rule_action INTEGER;      -- 규칙 기반 결정
ALTER TABLE kimchirang_trades ADD COLUMN IF NOT EXISTS kp_ma_5m DECIMAL(8,4);
ALTER TABLE kimchirang_trades ADD COLUMN IF NOT EXISTS kp_z_score DECIMAL(8,4);

CREATE INDEX IF NOT EXISTS idx_kr_trades_session ON kimchirang_trades(session_id);
