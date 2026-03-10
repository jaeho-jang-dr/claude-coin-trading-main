-- 실행 실패 상세 추적 + 자가진단 뷰

-- execution_logs 테이블 확장
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS execution_started_at TIMESTAMPTZ;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS execution_completed_at TIMESTAMPTZ;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS success BOOLEAN;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS error_phase TEXT;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS error_code TEXT;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;
ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS phases_completed JSONB;

-- decisions 테이블에 실행 추적 필드 추가
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS execution_attempted BOOLEAN;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS execution_started_at TIMESTAMPTZ;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS execution_completed_at TIMESTAMPTZ;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS execution_latency_ms INTEGER;

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_exec_logs_success ON execution_logs(success) WHERE success = false;
CREATE INDEX IF NOT EXISTS idx_exec_logs_error ON execution_logs(error_phase);
CREATE INDEX IF NOT EXISTS idx_decisions_exec_attempted ON decisions(execution_attempted);

-- 실행 실패 분석 뷰
CREATE OR REPLACE VIEW v_execution_failures AS
SELECT
  el.created_at,
  el.cycle_id,
  el.execution_mode,
  el.error_phase,
  el.error_code,
  el.error_message,
  el.retry_count,
  el.duration_ms,
  d.decision,
  d.current_price,
  d.trade_amount
FROM execution_logs el
LEFT JOIN decisions d ON d.id = el.decision_id
WHERE el.success = false
ORDER BY el.created_at DESC;

-- 시스템 건강도 뷰 (최근 24시간)
CREATE OR REPLACE VIEW v_system_health AS
SELECT
  execution_mode,
  COUNT(*) AS total_runs,
  COUNT(*) FILTER (WHERE success = true) AS success_count,
  COUNT(*) FILTER (WHERE success = false) AS failure_count,
  ROUND(
    COUNT(*) FILTER (WHERE success = true)::numeric / NULLIF(COUNT(*), 0) * 100, 1
  ) AS success_rate,
  ROUND(AVG(duration_ms)::numeric, 0) AS avg_duration_ms,
  MAX(duration_ms) AS max_duration_ms,
  MAX(created_at) AS last_run
FROM execution_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY execution_mode;
