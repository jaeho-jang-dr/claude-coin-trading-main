-- Lifeline 자가치유 시스템 — 컴포넌트별 헬스체크 및 자동 복구 이력
-- component: upbit_api, supabase, rl_model, disk, memory, process 등

CREATE TABLE IF NOT EXISTS system_health_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    component           TEXT NOT NULL,
    severity            TEXT NOT NULL DEFAULT 'INFO'
                        CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    issue_summary       TEXT NOT NULL,
    raw_traceback       TEXT,
    ai_diagnosis        TEXT,
    healing_action      TEXT,
    resolution_status   TEXT DEFAULT 'DETECTED'
                        CHECK (resolution_status IN (
                            'DETECTED', 'DIAGNOSING', 'HEALING',
                            'RESOLVED', 'FAILED', 'PENDING'
                        )),
    healing_duration_ms INTEGER,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_health_logs_timestamp ON system_health_logs(timestamp DESC);
CREATE INDEX idx_health_logs_component ON system_health_logs(component);
CREATE INDEX idx_health_logs_severity ON system_health_logs(severity);
CREATE INDEX idx_health_logs_resolution ON system_health_logs(resolution_status);

-- 최근 24시간 컴포넌트별 요약 뷰
CREATE OR REPLACE VIEW v_system_health_summary AS
SELECT
    component,
    COUNT(*)                                              AS total_checks,
    COUNT(*) FILTER (WHERE severity = 'ERROR')            AS error_count,
    COUNT(*) FILTER (WHERE severity = 'CRITICAL')         AS critical_count,
    ROUND(
        COUNT(*) FILTER (WHERE resolution_status = 'RESOLVED')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE resolution_status != 'DETECTED' OR severity != 'INFO'), 0)
        * 100, 1
    )                                                     AS resolution_rate_pct
FROM system_health_logs
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY component
ORDER BY critical_count DESC, error_count DESC;

-- 최근 50건 인시던트 (INFO 제외)
CREATE OR REPLACE VIEW v_recent_incidents AS
SELECT
    id, timestamp, component, severity,
    issue_summary, ai_diagnosis, healing_action,
    resolution_status, healing_duration_ms
FROM system_health_logs
WHERE severity != 'INFO'
ORDER BY timestamp DESC
LIMIT 50;

COMMENT ON TABLE system_health_logs IS 'Lifeline 자가치유 시스템 — 컴포넌트별 헬스체크, AI 진단, 자동 복구 이력';
