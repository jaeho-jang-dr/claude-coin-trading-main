-- ============================================================
-- 초단타 ML 학습 시스템 (v5)
-- 4개 테이블 + signal_attempt_log 확장 + 분석 뷰
-- ============================================================

-- ────────────────────────────────────────────────
-- 1. 시장 스냅샷 (5분 간격, 봇이 자동 기록)
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scalp_market_snapshot (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  recorded_at TIMESTAMPTZ DEFAULT NOW(),
  cycle_id TEXT,
  -- 가격
  btc_price BIGINT NOT NULL,
  -- 기술지표
  rsi_1h DECIMAL(5,2),
  sma20 BIGINT,
  market_trend TEXT,                    -- uptrend / downtrend / sideways
  -- 감성
  fgi INTEGER,
  news_sentiment TEXT,
  news_score DECIMAL(4,2),
  -- 고래
  whale_buy_count INTEGER DEFAULT 0,
  whale_sell_count INTEGER DEFAULT 0,
  whale_buy_krw BIGINT DEFAULT 0,
  whale_sell_krw BIGINT DEFAULT 0,
  whale_buy_ratio DECIMAL(5,3),        -- 0.000 ~ 1.000
  -- 체결 강도
  trade_buy_volume DECIMAL(18,8),      -- 최근 60초 매수 체결량
  trade_sell_volume DECIMAL(18,8),     -- 최근 60초 매도 체결량
  trade_pressure_ratio DECIMAL(5,3),   -- 매수/(매수+매도)
  -- 가격 모멘텀 (price_history 버퍼에서 계산)
  momentum_1m_pct DECIMAL(6,3),        -- 최근 1분 변동률
  momentum_5m_pct DECIMAL(6,3),        -- 최근 5분 변동률
  momentum_15m_pct DECIMAL(6,3),       -- 최근 15분 변동률
  -- 변동성
  volatility_5m DECIMAL(6,3),          -- 5분 가격 표준편차/평균 * 100
  -- 포지션 상태
  open_positions INTEGER DEFAULT 0,
  daily_trade_count INTEGER DEFAULT 0,
  daily_pnl_krw INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_scalp_snapshot_recorded ON scalp_market_snapshot(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_scalp_snapshot_trend ON scalp_market_snapshot(market_trend, recorded_at DESC);

-- ────────────────────────────────────────────────
-- 2. signal_attempt_log 확장: 다중 시간대 사후 추적
-- ────────────────────────────────────────────────
ALTER TABLE signal_attempt_log
  ADD COLUMN IF NOT EXISTS price_1m_after BIGINT,
  ADD COLUMN IF NOT EXISTS price_5m_after BIGINT,
  ADD COLUMN IF NOT EXISTS price_15m_after BIGINT,
  ADD COLUMN IF NOT EXISTS price_30m_after BIGINT,
  ADD COLUMN IF NOT EXISTS outcome_1m_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS outcome_5m_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS outcome_15m_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS outcome_30m_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS would_have_won_5m BOOLEAN,
  ADD COLUMN IF NOT EXISTS would_have_won_15m BOOLEAN,
  ADD COLUMN IF NOT EXISTS would_have_won_30m BOOLEAN,
  ADD COLUMN IF NOT EXISTS best_price_30m BIGINT,
  ADD COLUMN IF NOT EXISTS worst_price_30m BIGINT,
  ADD COLUMN IF NOT EXISTS best_exit_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS worst_drawdown_pct DECIMAL(6,3),
  -- 진입 시점 추가 컨텍스트
  ADD COLUMN IF NOT EXISTS momentum_1m_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS whale_buy_ratio DECIMAL(5,3),
  ADD COLUMN IF NOT EXISTS trade_pressure_ratio DECIMAL(5,3),
  ADD COLUMN IF NOT EXISTS volatility_5m DECIMAL(6,3);

-- 사후 추적 미완료 인덱스 (시간대별)
CREATE INDEX IF NOT EXISTS idx_signal_no_1m ON signal_attempt_log(recorded_at DESC)
  WHERE price_1m_after IS NULL AND signal_type != 'no_signal';
CREATE INDEX IF NOT EXISTS idx_signal_no_5m ON signal_attempt_log(recorded_at DESC)
  WHERE price_5m_after IS NULL AND signal_type != 'no_signal';
CREATE INDEX IF NOT EXISTS idx_signal_no_15m ON signal_attempt_log(recorded_at DESC)
  WHERE price_15m_after IS NULL AND signal_type != 'no_signal';
CREATE INDEX IF NOT EXISTS idx_signal_no_30m ON signal_attempt_log(recorded_at DESC)
  WHERE price_30m_after IS NULL AND signal_type != 'no_signal';

-- ────────────────────────────────────────────────
-- 3. 학습 작업 큐 (Windows 워커가 폴링)
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scalp_training_tasks (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  task_type TEXT NOT NULL CHECK (task_type IN (
    'backtest', 'train_lgbm', 'train_xgboost', 'train_pytorch',
    'evaluate', 'parameter_sweep', 'analyze'
  )),
  params JSONB NOT NULL DEFAULT '{}',    -- 하이퍼파라미터, 날짜 범위 등
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
    'pending', 'running', 'completed', 'failed', 'cancelled'
  )),
  priority INTEGER DEFAULT 5,            -- 1=최고, 10=최저
  assigned_worker TEXT,                  -- 워커 호스트명
  result JSONB,                          -- 학습 결과 메트릭
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_training_tasks_status ON scalp_training_tasks(status, priority);
CREATE INDEX IF NOT EXISTS idx_training_tasks_worker ON scalp_training_tasks(assigned_worker, status);

-- ────────────────────────────────────────────────
-- 4. 모델 버전 레지스트리
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scalp_model_versions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  version_tag TEXT NOT NULL UNIQUE,       -- 예: 'lgbm_v001', 'xgb_v002'
  model_type TEXT NOT NULL CHECK (model_type IN (
    'lightgbm', 'xgboost', 'pytorch', 'rule_based'
  )),
  -- 모델 저장
  model_config JSONB,                     -- 하이퍼파라미터
  features_used JSONB,                    -- 사용된 피처 목록
  model_blob BYTEA,                       -- 직렬화된 모델 (작은 모델용)
  model_path TEXT,                        -- 파일 경로 (큰 모델용)
  -- 성능 메트릭
  train_accuracy DECIMAL(5,3),
  test_accuracy DECIMAL(5,3),
  train_samples INTEGER,
  test_samples INTEGER,
  precision_score DECIMAL(5,3),
  recall_score DECIMAL(5,3),
  f1_score DECIMAL(5,3),
  -- 백테스트 결과
  backtest_return_pct DECIMAL(8,3),
  backtest_sharpe DECIMAL(6,3),
  backtest_max_drawdown DECIMAL(6,3),
  backtest_win_rate DECIMAL(5,2),
  backtest_trades INTEGER,
  -- 라이브 성과 (적용 후 추적)
  live_trades INTEGER DEFAULT 0,
  live_win_rate DECIMAL(5,2),
  live_return_pct DECIMAL(8,3),
  -- 상태
  is_active BOOLEAN DEFAULT FALSE,
  training_task_id UUID REFERENCES scalp_training_tasks(id),
  promoted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_versions_active ON scalp_model_versions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_model_versions_type ON scalp_model_versions(model_type, created_at DESC);

-- 활성 모델은 타입당 1개만 (트리거)
CREATE OR REPLACE FUNCTION enforce_single_active_model()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.is_active = TRUE THEN
    UPDATE scalp_model_versions
    SET is_active = FALSE, promoted_at = NULL
    WHERE model_type = NEW.model_type AND id != NEW.id AND is_active = TRUE;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_single_active_model ON scalp_model_versions;
CREATE TRIGGER trg_single_active_model
  BEFORE INSERT OR UPDATE ON scalp_model_versions
  FOR EACH ROW EXECUTE FUNCTION enforce_single_active_model();

-- ────────────────────────────────────────────────
-- 5. scalp_trade_log 확장: ML 기록
-- ────────────────────────────────────────────────
ALTER TABLE scalp_trade_log
  ADD COLUMN IF NOT EXISTS cycle_id TEXT,
  ADD COLUMN IF NOT EXISTS model_version TEXT,      -- 적용된 ML 모델
  ADD COLUMN IF NOT EXISTS model_confidence DECIMAL(5,3),
  ADD COLUMN IF NOT EXISTS momentum_1m_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS whale_buy_ratio DECIMAL(5,3),
  ADD COLUMN IF NOT EXISTS trade_pressure DECIMAL(5,3),
  ADD COLUMN IF NOT EXISTS volatility_5m DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS trailing_stop_hit BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS highest_pnl_pct DECIMAL(6,3),
  ADD COLUMN IF NOT EXISTS grace_period_saved BOOLEAN DEFAULT FALSE;

-- ────────────────────────────────────────────────
-- 6. 분석 뷰
-- ────────────────────────────────────────────────

-- 시그널 품질 (다중 시간대)
CREATE OR REPLACE VIEW v_signal_quality AS
SELECT
  strategy,
  signal_type,
  COUNT(*) AS total,
  -- 1분 후
  ROUND(AVG(outcome_1m_pct)::numeric, 3) AS avg_1m_pct,
  COUNT(*) FILTER (WHERE outcome_1m_pct > 0.15) AS win_1m,
  -- 5분 후
  ROUND(AVG(outcome_5m_pct)::numeric, 3) AS avg_5m_pct,
  COUNT(*) FILTER (WHERE would_have_won_5m = true) AS win_5m,
  -- 15분 후
  ROUND(AVG(outcome_15m_pct)::numeric, 3) AS avg_15m_pct,
  COUNT(*) FILTER (WHERE would_have_won_15m = true) AS win_15m,
  -- 30분 후
  ROUND(AVG(outcome_30m_pct)::numeric, 3) AS avg_30m_pct,
  COUNT(*) FILTER (WHERE would_have_won_30m = true) AS win_30m,
  -- 최적 청산
  ROUND(AVG(best_exit_pct)::numeric, 3) AS avg_best_exit,
  ROUND(AVG(worst_drawdown_pct)::numeric, 3) AS avg_worst_dd
FROM signal_attempt_log
WHERE signal_type != 'no_signal'
  AND outcome_1m_pct IS NOT NULL
GROUP BY strategy, signal_type
ORDER BY strategy, signal_type;

-- 시간대별 승률 (시간대 패턴 발견)
CREATE OR REPLACE VIEW v_scalp_hourly_performance AS
SELECT
  EXTRACT(HOUR FROM entry_time AT TIME ZONE 'Asia/Seoul') AS hour_kst,
  strategy,
  COUNT(*) AS trades,
  SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
  ROUND(
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0) * 100, 1
  ) AS win_rate,
  ROUND(AVG(pnl_pct)::numeric, 3) AS avg_pnl_pct,
  SUM(pnl_krw) AS total_pnl_krw
FROM scalp_trade_log
GROUP BY hour_kst, strategy
ORDER BY hour_kst, strategy;

-- 시장 상태별 승률
CREATE OR REPLACE VIEW v_scalp_trend_performance AS
SELECT
  market_trend,
  strategy,
  COUNT(*) AS trades,
  SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
  ROUND(
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0) * 100, 1
  ) AS win_rate,
  ROUND(AVG(pnl_pct)::numeric, 3) AS avg_pnl_pct,
  SUM(pnl_krw) AS total_pnl_krw
FROM scalp_trade_log
WHERE market_trend IS NOT NULL
GROUP BY market_trend, strategy
ORDER BY market_trend, strategy;

-- 필터 효과 (다중 시간대)
CREATE OR REPLACE VIEW v_filter_effectiveness_v2 AS
SELECT
  block_filter,
  COUNT(*) AS total_blocked,
  -- 1분 후
  COUNT(*) FILTER (WHERE outcome_1m_pct > 0.15) AS would_win_1m,
  COUNT(*) FILTER (WHERE outcome_1m_pct < -0.15) AS would_lose_1m,
  -- 5분 후
  COUNT(*) FILTER (WHERE would_have_won_5m = true) AS would_win_5m,
  COUNT(*) FILTER (WHERE would_have_won_5m = false) AS would_lose_5m,
  -- 15분 후
  COUNT(*) FILTER (WHERE would_have_won_15m = true) AS would_win_15m,
  -- 필터 보호율 (5분 기준)
  ROUND(
    COUNT(*) FILTER (WHERE would_have_won_5m = false)::numeric /
    NULLIF(COUNT(*) FILTER (WHERE would_have_won_5m IS NOT NULL), 0) * 100, 1
  ) AS filter_save_rate_5m,
  ROUND(AVG(best_exit_pct)::numeric, 3) AS avg_best_exit_missed
FROM signal_attempt_log
WHERE signal_type = 'blocked'
GROUP BY block_filter
ORDER BY total_blocked DESC;

-- 모델 성능 비교
CREATE OR REPLACE VIEW v_model_comparison AS
SELECT
  version_tag,
  model_type,
  test_accuracy,
  f1_score,
  backtest_win_rate,
  backtest_return_pct,
  backtest_sharpe,
  live_trades,
  live_win_rate,
  live_return_pct,
  is_active,
  created_at
FROM scalp_model_versions
ORDER BY created_at DESC;
