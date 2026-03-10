-- 결정 사후 추적: 모든 결정 후 1h/4h/24h 가격 변동 기록
-- "관망했는데 가격이 올랐나?", "매수했는데 4시간 후 어떻게 됐나?"

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS price_1h_after BIGINT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS price_4h_after BIGINT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS price_24h_after BIGINT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS outcome_1h_pct DECIMAL(6,3);
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS outcome_4h_pct DECIMAL(6,3);
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS outcome_24h_pct DECIMAL(6,3);
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS was_correct_1h BOOLEAN;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS was_correct_4h BOOLEAN;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS was_correct_24h BOOLEAN;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS aftermath_updated_at TIMESTAMPTZ;

-- 초단타도 동일 (scalp_trade_log)
ALTER TABLE scalp_trade_log ADD COLUMN IF NOT EXISTS price_1h_after BIGINT;
ALTER TABLE scalp_trade_log ADD COLUMN IF NOT EXISTS outcome_1h_pct DECIMAL(6,3);
ALTER TABLE scalp_trade_log ADD COLUMN IF NOT EXISTS was_good_entry BOOLEAN;
ALTER TABLE scalp_trade_log ADD COLUMN IF NOT EXISTS aftermath_updated_at TIMESTAMPTZ;

-- 인덱스: 아직 aftermath 안 채워진 행 빠르게 찾기
CREATE INDEX IF NOT EXISTS idx_decisions_no_1h ON decisions(created_at DESC) WHERE price_1h_after IS NULL;
CREATE INDEX IF NOT EXISTS idx_decisions_no_4h ON decisions(created_at DESC) WHERE price_4h_after IS NULL;
CREATE INDEX IF NOT EXISTS idx_decisions_no_24h ON decisions(created_at DESC) WHERE price_24h_after IS NULL;
CREATE INDEX IF NOT EXISTS idx_scalp_no_aftermath ON scalp_trade_log(entry_time DESC) WHERE price_1h_after IS NULL;

-- 회고 뷰: 결정 정확도 분석
CREATE OR REPLACE VIEW v_decision_accuracy AS
SELECT
  decision,
  source,
  COUNT(*) AS total,
  -- 1시간 후
  ROUND(AVG(outcome_1h_pct)::numeric, 2) AS avg_1h_pct,
  COUNT(*) FILTER (WHERE was_correct_1h = true) AS correct_1h,
  ROUND(COUNT(*) FILTER (WHERE was_correct_1h = true)::numeric / NULLIF(COUNT(*) FILTER (WHERE was_correct_1h IS NOT NULL), 0) * 100, 1) AS accuracy_1h,
  -- 4시간 후
  ROUND(AVG(outcome_4h_pct)::numeric, 2) AS avg_4h_pct,
  COUNT(*) FILTER (WHERE was_correct_4h = true) AS correct_4h,
  ROUND(COUNT(*) FILTER (WHERE was_correct_4h = true)::numeric / NULLIF(COUNT(*) FILTER (WHERE was_correct_4h IS NOT NULL), 0) * 100, 1) AS accuracy_4h,
  -- 24시간 후
  ROUND(AVG(outcome_24h_pct)::numeric, 2) AS avg_24h_pct,
  COUNT(*) FILTER (WHERE was_correct_24h = true) AS correct_24h,
  ROUND(COUNT(*) FILTER (WHERE was_correct_24h = true)::numeric / NULLIF(COUNT(*) FILTER (WHERE was_correct_24h IS NOT NULL), 0) * 100, 1) AS accuracy_24h
FROM decisions
WHERE aftermath_updated_at IS NOT NULL
GROUP BY decision, source;

-- 놓친 기회 뷰: 관망했는데 가격이 크게 오른 경우
CREATE OR REPLACE VIEW v_missed_opportunities AS
SELECT
  id,
  cycle_id,
  created_at,
  current_price,
  outcome_1h_pct,
  outcome_4h_pct,
  outcome_24h_pct,
  reason,
  confidence,
  fear_greed_value,
  rsi_value
FROM decisions
WHERE decision = '관망'
  AND outcome_24h_pct > 1.5  -- 24시간 후 1.5% 이상 상승
  AND aftermath_updated_at IS NOT NULL
ORDER BY outcome_24h_pct DESC;

-- 잘못된 거래 뷰: 매수했는데 가격이 떨어진 경우
CREATE OR REPLACE VIEW v_bad_trades AS
SELECT
  id,
  cycle_id,
  created_at,
  current_price,
  trade_amount,
  outcome_1h_pct,
  outcome_4h_pct,
  outcome_24h_pct,
  reason,
  confidence,
  fear_greed_value,
  rsi_value
FROM decisions
WHERE decision = '매수'
  AND outcome_4h_pct < -0.5  -- 4시간 후 0.5% 이상 하락
  AND aftermath_updated_at IS NOT NULL
ORDER BY outcome_4h_pct ASC;
