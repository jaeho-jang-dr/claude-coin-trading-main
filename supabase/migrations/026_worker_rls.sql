-- 026_worker_rls.sql
-- Worker RLS: per-worker API token + tier-based access control
-- Owner: SERVICE_ROLE_KEY (bypasses RLS)
-- Coworker/Collaborator: ANON_KEY + x-worker-token header (RLS enforced)
-- 2026-03-15

-- ════════════════════════════════════════════════
-- A. Worker Token + Helper Functions
-- ════════════════════════════════════════════════

-- A1. api_token column
ALTER TABLE compute_workers
  ADD COLUMN IF NOT EXISTS api_token UUID DEFAULT gen_random_uuid() UNIQUE;

UPDATE compute_workers SET api_token = gen_random_uuid() WHERE api_token IS NULL;
ALTER TABLE compute_workers ALTER COLUMN api_token SET NOT NULL;
CREATE INDEX IF NOT EXISTS idx_workers_api_token ON compute_workers(api_token);

-- A2. Helper: extract worker tier from x-worker-token header
CREATE OR REPLACE FUNCTION get_worker_tier()
RETURNS TEXT
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
  SELECT tier FROM compute_workers
  WHERE api_token = (
    current_setting('request.headers', true)::json->>'x-worker-token'
  )::uuid
  AND status != 'suspended'
  LIMIT 1;
$$;

-- A3. Helper: extract worker_id from token
CREATE OR REPLACE FUNCTION get_worker_id()
RETURNS TEXT
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
  SELECT worker_id FROM compute_workers
  WHERE api_token = (
    current_setting('request.headers', true)::json->>'x-worker-token'
  )::uuid
  AND status != 'suspended'
  LIMIT 1;
$$;

-- A4. Helper: check if request uses service_role (owner)
CREATE OR REPLACE FUNCTION is_service_role()
RETURNS BOOLEAN
LANGUAGE sql STABLE
AS $$
  SELECT current_setting('request.jwt.claim.role', true) = 'service_role';
$$;


-- ════════════════════════════════════════════════
-- B. RLS Policies
-- ════════════════════════════════════════════════

-- ────────────────────────────────────────────────
-- B1. compute_workers: workers can read all, update own row only
-- ────────────────────────────────────────────────
ALTER TABLE compute_workers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "srv_full" ON compute_workers
  FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON compute_workers
  FOR SELECT USING (get_worker_tier() IS NOT NULL);
CREATE POLICY "worker_update_own" ON compute_workers
  FOR UPDATE USING (worker_id = get_worker_id())
  WITH CHECK (worker_id = get_worker_id());

-- ────────────────────────────────────────────────
-- B2. worker_heartbeats: insert/read own only
-- ────────────────────────────────────────────────
ALTER TABLE worker_heartbeats ENABLE ROW LEVEL SECURITY;

CREATE POLICY "srv_full" ON worker_heartbeats
  FOR ALL USING (is_service_role());
CREATE POLICY "worker_read_own" ON worker_heartbeats
  FOR SELECT USING (worker_id = get_worker_id());
CREATE POLICY "worker_insert_own" ON worker_heartbeats
  FOR INSERT WITH CHECK (worker_id = get_worker_id());

-- ────────────────────────────────────────────────
-- B3. scalp_training_tasks: coworker/collaborator claim+update assigned
-- ────────────────────────────────────────────────
ALTER TABLE scalp_training_tasks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "srv_full" ON scalp_training_tasks
  FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_training_tasks
  FOR SELECT USING (get_worker_tier() IN ('coworker', 'collaborator'));
CREATE POLICY "worker_claim_update" ON scalp_training_tasks
  FOR UPDATE USING (
    get_worker_tier() IN ('coworker', 'collaborator')
    AND (status = 'pending' OR (assigned_worker = get_worker_id() AND status = 'running'))
  );
CREATE POLICY "worker_insert_result" ON scalp_training_tasks
  FOR INSERT WITH CHECK (
    get_worker_tier() IN ('coworker', 'collaborator')
    AND status IN ('completed', 'failed')
  );

-- ────────────────────────────────────────────────
-- B4. scalp_model_versions: collaborator cannot deploy (is_active=true)
-- ────────────────────────────────────────────────
ALTER TABLE scalp_model_versions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "srv_full" ON scalp_model_versions
  FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_model_versions
  FOR SELECT USING (get_worker_tier() IS NOT NULL);
CREATE POLICY "worker_insert" ON scalp_model_versions
  FOR INSERT WITH CHECK (
    get_worker_tier() IN ('coworker', 'collaborator')
    AND (get_worker_tier() = 'coworker' OR is_active = FALSE)
  );
CREATE POLICY "coworker_update" ON scalp_model_versions
  FOR UPDATE USING (get_worker_tier() = 'coworker');

-- ────────────────────────────────────────────────
-- B5. Trading/Portfolio (read-only for all workers)
-- ────────────────────────────────────────────────

-- decisions
ALTER TABLE decisions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON decisions FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON decisions FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

-- portfolio_snapshots
ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON portfolio_snapshots FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON portfolio_snapshots FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

-- ────────────────────────────────────────────────
-- B6. Strategy/Feedback (coworker can insert feedback)
-- ────────────────────────────────────────────────

-- feedback
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON feedback FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON feedback FOR SELECT
  USING (get_worker_tier() IS NOT NULL);
CREATE POLICY "coworker_insert" ON feedback FOR INSERT
  WITH CHECK (get_worker_tier() = 'coworker');

-- strategy_history
ALTER TABLE strategy_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON strategy_history FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON strategy_history FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

-- ────────────────────────────────────────────────
-- B7. Agent system (read-only)
-- ────────────────────────────────────────────────

ALTER TABLE agent_switches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON agent_switches FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON agent_switches FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

-- ────────────────────────────────────────────────
-- B8. Scalp data (read-only for training data access)
-- ────────────────────────────────────────────────

ALTER TABLE scalp_trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON scalp_trades FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_trades FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

ALTER TABLE scalp_trade_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON scalp_trade_log FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_trade_log FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

ALTER TABLE scalp_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON scalp_sessions FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_sessions FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

ALTER TABLE scalp_settlements ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON scalp_settlements FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_settlements FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

ALTER TABLE scalp_training_journal ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON scalp_training_journal FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_training_journal FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

ALTER TABLE scalp_market_snapshot ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON scalp_market_snapshot FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON scalp_market_snapshot FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

-- ────────────────────────────────────────────────
-- B9. Market/Signal data (read-only, training data)
-- ────────────────────────────────────────────────

ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON market_data FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON market_data FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

ALTER TABLE signal_attempt_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON signal_attempt_log FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON signal_attempt_log FOR SELECT
  USING (get_worker_tier() IS NOT NULL);

-- ────────────────────────────────────────────────
-- B10. Execution/Admin (owner + coworker only)
-- ────────────────────────────────────────────────

ALTER TABLE execution_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON execution_logs FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON execution_logs FOR SELECT
  USING (get_worker_tier() IN ('coworker'));

ALTER TABLE app_changelog ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON app_changelog FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON app_changelog FOR SELECT
  USING (get_worker_tier() IN ('coworker'));

-- ────────────────────────────────────────────────
-- B11. RL Training logs (read + insert for workers)
-- ────────────────────────────────────────────────

ALTER TABLE rl_training_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "service_role_all" ON rl_training_log;
CREATE POLICY "srv_full" ON rl_training_log FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON rl_training_log FOR SELECT
  USING (get_worker_tier() IS NOT NULL);
CREATE POLICY "worker_insert" ON rl_training_log FOR INSERT
  WITH CHECK (get_worker_tier() IN ('coworker', 'collaborator'));

-- ────────────────────────────────────────────────
-- B12. Kimchirang tables (owner only via service_role)
-- ────────────────────────────────────────────────

ALTER TABLE kimchirang_trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON kimchirang_trades FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON kimchirang_trades FOR SELECT
  USING (get_worker_tier() IN ('coworker'));

ALTER TABLE kimchirang_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON kimchirang_sessions FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON kimchirang_sessions FOR SELECT
  USING (get_worker_tier() IN ('coworker'));

ALTER TABLE kimchirang_kp_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "srv_full" ON kimchirang_kp_history FOR ALL USING (is_service_role());
CREATE POLICY "worker_read" ON kimchirang_kp_history FOR SELECT
  USING (get_worker_tier() IS NOT NULL);
