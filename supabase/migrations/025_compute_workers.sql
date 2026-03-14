-- ============================================================
-- 분산 컴퓨팅 워커 관리 시스템
-- 워커 등록 + 티어 권한 + 하트비트 + 작업 통계
-- ============================================================

-- ────────────────────────────────────────────────
-- 1. 워커 레지스트리
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS compute_workers (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  worker_id TEXT NOT NULL UNIQUE,           -- 'mac-mini', 'win-pc1', 'community-01' 등
  worker_name TEXT,                          -- 표시 이름 (예: '아들 컴', '커뮤니티 김철수')
  tier TEXT NOT NULL DEFAULT 'collaborator' CHECK (tier IN (
    'owner', 'coworker', 'collaborator', 'viewer'
  )),
  -- 머신 정보
  platform TEXT,                             -- 'Darwin', 'Windows', 'Linux'
  architecture TEXT,                         -- 'arm64', 'x86_64'
  ram_gb INTEGER,                            -- RAM (GB)
  cpu_info TEXT,                             -- CPU 모델
  gpu_info TEXT,                             -- GPU 모델 (있으면)
  python_version TEXT,
  -- 상태
  status TEXT NOT NULL DEFAULT 'offline' CHECK (status IN (
    'online', 'busy', 'offline', 'error', 'suspended'
  )),
  is_main_brain BOOLEAN DEFAULT FALSE,       -- 현재 주 컴 여부
  -- 하트비트
  last_heartbeat TIMESTAMPTZ,
  heartbeat_interval_sec INTEGER DEFAULT 60, -- 하트비트 간격
  -- 능력
  can_gpu BOOLEAN DEFAULT FALSE,             -- GPU 학습 가능 여부
  max_concurrent_tasks INTEGER DEFAULT 1,    -- 동시 작업 수
  allowed_task_types TEXT[] DEFAULT ARRAY['train_lgbm', 'train_xgboost', 'backtest', 'analyze'],
  -- 통계
  tasks_completed INTEGER DEFAULT 0,
  tasks_failed INTEGER DEFAULT 0,
  total_training_hours DECIMAL(8,2) DEFAULT 0,
  avg_task_duration_sec INTEGER,
  -- 메타
  registered_at TIMESTAMPTZ DEFAULT NOW(),
  last_online_at TIMESTAMPTZ,
  notes TEXT
);

-- 주 컴은 1대만 (트리거)
CREATE OR REPLACE FUNCTION enforce_single_main_brain()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.is_main_brain = TRUE THEN
    UPDATE compute_workers
    SET is_main_brain = FALSE
    WHERE id != NEW.id AND is_main_brain = TRUE;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_single_main_brain ON compute_workers;
CREATE TRIGGER trg_single_main_brain
  BEFORE INSERT OR UPDATE ON compute_workers
  FOR EACH ROW EXECUTE FUNCTION enforce_single_main_brain();

CREATE INDEX IF NOT EXISTS idx_workers_status ON compute_workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_tier ON compute_workers(tier);

-- ────────────────────────────────────────────────
-- 2. 워커 하트비트 로그 (온라인 이력)
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS worker_heartbeats (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  worker_id TEXT NOT NULL REFERENCES compute_workers(worker_id),
  recorded_at TIMESTAMPTZ DEFAULT NOW(),
  status TEXT NOT NULL,
  -- 시스템 리소스
  cpu_usage_pct DECIMAL(5,2),
  ram_usage_pct DECIMAL(5,2),
  disk_free_gb DECIMAL(8,2),
  -- 현재 작업
  current_task_id UUID,
  current_task_type TEXT,
  task_progress_pct DECIMAL(5,2)
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_worker ON worker_heartbeats(worker_id, recorded_at DESC);

-- 오래된 하트비트 자동 정리 (7일 이상)
-- cron으로 주기적 실행: DELETE FROM worker_heartbeats WHERE recorded_at < NOW() - INTERVAL '7 days';

-- ────────────────────────────────────────────────
-- 3. 작업 할당 규칙 뷰
-- ────────────────────────────────────────────────

-- 현재 온라인 워커 + 가용 슬롯
CREATE OR REPLACE VIEW v_available_workers AS
SELECT
  w.worker_id,
  w.worker_name,
  w.tier,
  w.platform,
  w.ram_gb,
  w.can_gpu,
  w.max_concurrent_tasks,
  w.allowed_task_types,
  w.tasks_completed,
  w.avg_task_duration_sec,
  -- 현재 실행 중인 작업 수
  COALESCE(running.count, 0) AS current_tasks,
  -- 가용 슬롯
  w.max_concurrent_tasks - COALESCE(running.count, 0) AS available_slots,
  -- 마지막 하트비트
  w.last_heartbeat,
  EXTRACT(EPOCH FROM (NOW() - w.last_heartbeat)) AS heartbeat_age_sec
FROM compute_workers w
LEFT JOIN (
  SELECT assigned_worker, COUNT(*) AS count
  FROM scalp_training_tasks
  WHERE status = 'running'
  GROUP BY assigned_worker
) running ON running.assigned_worker = w.worker_id
WHERE w.status IN ('online', 'busy')
  AND w.last_heartbeat > NOW() - INTERVAL '5 minutes'
ORDER BY
  -- 가용 슬롯 많은 순 → 완료 작업 많은 순 (경험치)
  (w.max_concurrent_tasks - COALESCE(running.count, 0)) DESC,
  w.tasks_completed DESC;

-- 워커별 성과 요약
CREATE OR REPLACE VIEW v_worker_stats AS
SELECT
  w.worker_id,
  w.worker_name,
  w.tier,
  w.platform,
  w.ram_gb,
  w.status,
  w.tasks_completed,
  w.tasks_failed,
  CASE WHEN (w.tasks_completed + w.tasks_failed) > 0
    THEN ROUND(w.tasks_completed::numeric / (w.tasks_completed + w.tasks_failed) * 100, 1)
    ELSE NULL
  END AS success_rate,
  w.total_training_hours,
  w.avg_task_duration_sec,
  w.is_main_brain,
  w.registered_at,
  w.last_online_at,
  -- 최근 24시간 완료 작업 수
  COALESCE(recent.count, 0) AS tasks_24h
FROM compute_workers w
LEFT JOIN (
  SELECT assigned_worker, COUNT(*) AS count
  FROM scalp_training_tasks
  WHERE status = 'completed'
    AND completed_at > NOW() - INTERVAL '24 hours'
  GROUP BY assigned_worker
) recent ON recent.assigned_worker = w.worker_id
ORDER BY w.tier, w.worker_id;
