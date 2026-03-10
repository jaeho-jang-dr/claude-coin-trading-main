-- 니어미스 + AI 거부권 추적
-- "점수 68인데 임계 70이라 안 샀다 → 다음날 5% 올랐다" 분석 가능

-- buy_score_detail 테이블 확장
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS points_from_threshold DECIMAL(5,2);
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS is_near_miss BOOLEAN DEFAULT FALSE;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS was_ai_vetoed BOOLEAN DEFAULT FALSE;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS ai_veto_reason TEXT;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS original_action TEXT;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS orchestrator_override BOOLEAN DEFAULT FALSE;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS override_reason TEXT;
-- 사후 추적
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS price_at_decision BIGINT;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS price_1h_after BIGINT;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS price_4h_after BIGINT;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS outcome_1h_pct DECIMAL(6,3);
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS outcome_4h_pct DECIMAL(6,3);
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS would_have_profited BOOLEAN;
ALTER TABLE buy_score_detail ADD COLUMN IF NOT EXISTS aftermath_updated_at TIMESTAMPTZ;

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_buy_score_near_miss ON buy_score_detail(is_near_miss) WHERE is_near_miss = true;
CREATE INDEX IF NOT EXISTS idx_buy_score_vetoed ON buy_score_detail(was_ai_vetoed) WHERE was_ai_vetoed = true;
CREATE INDEX IF NOT EXISTS idx_buy_score_no_aftermath ON buy_score_detail(recorded_at DESC) WHERE price_1h_after IS NULL;

-- 니어미스 분석 뷰
CREATE OR REPLACE VIEW v_near_miss_analysis AS
SELECT
  recorded_at,
  agent_type,
  total_score,
  threshold,
  points_from_threshold,
  was_ai_vetoed,
  ai_veto_reason,
  action,
  price_at_decision,
  outcome_1h_pct,
  outcome_4h_pct,
  would_have_profited,
  CASE
    WHEN would_have_profited = true AND action = 'hold' THEN '놓친 기회'
    WHEN would_have_profited = false AND action = 'hold' THEN '올바른 관망'
    WHEN would_have_profited = true AND action IN ('buy', '매수') THEN '좋은 매수'
    WHEN would_have_profited = false AND action IN ('buy', '매수') THEN '나쁜 매수'
    ELSE '미평가'
  END AS evaluation
FROM buy_score_detail
WHERE is_near_miss = true OR was_ai_vetoed = true
ORDER BY recorded_at DESC;

-- AI 거부권 효과 분석 뷰
CREATE OR REPLACE VIEW v_ai_veto_effectiveness AS
SELECT
  ai_veto_reason,
  COUNT(*) AS total_vetoes,
  COUNT(*) FILTER (WHERE would_have_profited = false) AS saved_from_loss,
  COUNT(*) FILTER (WHERE would_have_profited = true) AS missed_profit,
  ROUND(
    COUNT(*) FILTER (WHERE would_have_profited = false)::numeric /
    NULLIF(COUNT(*) FILTER (WHERE would_have_profited IS NOT NULL), 0) * 100, 1
  ) AS veto_accuracy_pct,
  ROUND(AVG(outcome_4h_pct)::numeric, 2) AS avg_outcome_if_bought
FROM buy_score_detail
WHERE was_ai_vetoed = true AND aftermath_updated_at IS NOT NULL
GROUP BY ai_veto_reason
ORDER BY total_vetoes DESC;
