-- 시그널 시도 기록: 생성/차단/무신호 모두 기록
-- "필터가 막은 거래가 성공했을까?" 분석 가능

CREATE TABLE IF NOT EXISTS signal_attempt_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  recorded_at TIMESTAMPTZ DEFAULT NOW(),
  cycle_id TEXT,
  -- 시그널 정보
  strategy TEXT NOT NULL,
  signal_type TEXT NOT NULL CHECK (signal_type IN ('generated', 'blocked', 'no_signal')),
  action TEXT,
  confidence DECIMAL(3,2),
  suggested_amount INTEGER,
  signal_reason TEXT,
  -- 차단 정보 (signal_type='blocked'일 때)
  block_filter TEXT,
  block_reason TEXT,
  -- 시장 상태
  btc_price BIGINT,
  market_trend TEXT,
  rsi_value DECIMAL(5,2),
  fgi_value INTEGER,
  news_score DECIMAL(4,2),
  -- 사후 추적
  price_1h_after BIGINT,
  outcome_1h_pct DECIMAL(6,3),
  would_have_won BOOLEAN,
  aftermath_updated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_signal_attempt_recorded ON signal_attempt_log(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_signal_attempt_type ON signal_attempt_log(signal_type);
CREATE INDEX IF NOT EXISTS idx_signal_attempt_strategy ON signal_attempt_log(strategy);
CREATE INDEX IF NOT EXISTS idx_signal_attempt_blocked ON signal_attempt_log(recorded_at DESC) WHERE signal_type = 'blocked';
CREATE INDEX IF NOT EXISTS idx_signal_attempt_no_aftermath ON signal_attempt_log(recorded_at DESC) WHERE price_1h_after IS NULL AND signal_type != 'no_signal';

-- 필터 효과 분석 뷰
CREATE OR REPLACE VIEW v_filter_effectiveness AS
SELECT
  block_filter,
  COUNT(*) AS total_blocked,
  COUNT(*) FILTER (WHERE would_have_won = true) AS would_have_won_count,
  COUNT(*) FILTER (WHERE would_have_won = false) AS would_have_lost_count,
  ROUND(
    COUNT(*) FILTER (WHERE would_have_won = false)::numeric /
    NULLIF(COUNT(*) FILTER (WHERE would_have_won IS NOT NULL), 0) * 100, 1
  ) AS filter_save_rate,
  ROUND(AVG(outcome_1h_pct)::numeric, 2) AS avg_outcome_if_traded
FROM signal_attempt_log
WHERE signal_type = 'blocked' AND aftermath_updated_at IS NOT NULL
GROUP BY block_filter
ORDER BY total_blocked DESC;

-- 전략별 시그널 발화율 뷰
CREATE OR REPLACE VIEW v_signal_fire_rate AS
SELECT
  strategy,
  COUNT(*) AS total_checks,
  COUNT(*) FILTER (WHERE signal_type = 'generated') AS signals_generated,
  COUNT(*) FILTER (WHERE signal_type = 'blocked') AS signals_blocked,
  COUNT(*) FILTER (WHERE signal_type = 'no_signal') AS no_signals,
  ROUND(
    COUNT(*) FILTER (WHERE signal_type != 'no_signal')::numeric / NULLIF(COUNT(*), 0) * 100, 2
  ) AS fire_rate_pct,
  recorded_at::date AS check_date
FROM signal_attempt_log
GROUP BY strategy, recorded_at::date
ORDER BY check_date DESC, strategy;
