-- 초단타 6시간 정산 + 훈련 일지 테이블
-- 2026-03-14

-- 정산 리포트 (6시간 간격)
CREATE TABLE IF NOT EXISTS scalp_settlements (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    epoch INT NOT NULL,
    elapsed_hours NUMERIC(6,2),
    period_trades INT DEFAULT 0,
    period_wins INT DEFAULT 0,
    period_losses INT DEFAULT 0,
    period_win_rate NUMERIC(5,2),
    period_pnl_krw INT DEFAULT 0,
    estimated_fee_krw INT DEFAULT 0,
    fee_covered BOOLEAN DEFAULT FALSE,
    cumulative_trades INT DEFAULT 0,
    cumulative_pnl_krw INT DEFAULT 0,
    strategy_stats JSONB,
    market_trend TEXT,
    rsi_value NUMERIC(5,2),
    fgi_value INT,
    current_price BIGINT,
    dry_run BOOLEAN DEFAULT TRUE,
    cycle_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 훈련 일지 (6시간 간격, 개선점 자동 도출)
CREATE TABLE IF NOT EXISTS scalp_training_journal (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    epoch INT NOT NULL,
    elapsed_hours NUMERIC(6,2),
    period_summary TEXT,
    cumulative_summary TEXT,
    fee_analysis TEXT,
    strategy_breakdown JSONB,
    market_context TEXT,
    improvement_notes JSONB,
    dry_run BOOLEAN DEFAULT TRUE,
    cycle_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_scalp_settlements_cycle ON scalp_settlements(cycle_id);
CREATE INDEX IF NOT EXISTS idx_scalp_settlements_created ON scalp_settlements(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scalp_journal_cycle ON scalp_training_journal(cycle_id);
CREATE INDEX IF NOT EXISTS idx_scalp_journal_created ON scalp_training_journal(created_at DESC);
