-- RL 학습/추론/모델 버전 전면 DB 추적
-- 모든 RL 활동이 반드시 DB에 기록되어야 한다

-- ============================================================
-- 1. rl_training_cycles: 훈련 사이클 기록
--    continuous_learner, weekly_retrain, monthly_training 등
-- ============================================================
CREATE TABLE IF NOT EXISTS rl_training_cycles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 식별
    cycle_type      TEXT NOT NULL,              -- 'continuous' | 'weekly' | 'monthly' | 'manual'
    algorithm       TEXT NOT NULL,              -- 'ppo' | 'sac' | 'td3' | 'cql' | 'bcq' | 'dt' | 'multi_agent' | 'self_tuning'
    module          TEXT NOT NULL,              -- 'continuous_learner' | 'weekly_retrain' | 'offline_rl' | 'decision_transformer' ...
    -- 훈련 설정
    training_steps  INT,
    training_epochs INT,
    data_days       INT,
    data_count      INT,                        -- 사용된 데이터 건수
    interval        TEXT DEFAULT '4h',           -- 캔들 간격
    obs_dim         INT DEFAULT 42,
    morl_enabled    BOOLEAN DEFAULT FALSE,
    -- 훈련 결과 메트릭
    avg_return_pct  FLOAT,
    avg_sharpe      FLOAT,
    avg_mdd         FLOAT,
    avg_trades      FLOAT,
    policy_loss     FLOAT,
    value_loss      FLOAT,
    entropy         FLOAT,
    -- Offline RL 전용
    direction_accuracy FLOAT,                   -- CQL/BCQ 방향 정확도
    q_loss          FLOAT,
    cql_penalty     FLOAT,
    -- Decision Transformer 전용
    best_eval_loss  FLOAT,
    n_sequences     INT,
    context_length  INT,
    -- 결과
    model_version   TEXT,                        -- 등록된 모델 버전 (v001 등)
    model_path      TEXT,
    status          TEXT NOT NULL DEFAULT 'completed', -- 'running' | 'completed' | 'failed' | 'rolled_back'
    error_message   TEXT,
    -- 비교
    baseline_sharpe FLOAT,                       -- 훈련 전 베이스라인
    improved        BOOLEAN,                     -- 성능 향상 여부
    -- 시간
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    elapsed_seconds FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 핵심 쿼리용 인덱스
CREATE INDEX idx_tc_cycle_type ON rl_training_cycles(cycle_type);
CREATE INDEX idx_tc_algorithm ON rl_training_cycles(algorithm);
CREATE INDEX idx_tc_module ON rl_training_cycles(module);
CREATE INDEX idx_tc_status ON rl_training_cycles(status);
CREATE INDEX idx_tc_created_at ON rl_training_cycles(created_at DESC);
CREATE INDEX idx_tc_algorithm_created ON rl_training_cycles(algorithm, created_at DESC);

COMMENT ON TABLE rl_training_cycles IS '모든 RL 훈련 사이클 기록 — 알고리즘/모듈/날짜별 쿼리 가능';

-- ============================================================
-- 2. rl_model_predictions: 추론 시 개별 모델 예측 기록
--    run_agents Phase 2.5 앙상블 추론 결과
-- ============================================================
CREATE TABLE IF NOT EXISTS rl_model_predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 연결
    decision_id     UUID REFERENCES decisions(id) ON DELETE SET NULL,
    cycle_id        TEXT,                        -- run_agents 실행 cycle_id
    -- 앙상블 전체 결과
    ensemble_action FLOAT,                       -- 앙상블 평균 action (-1~1)
    ensemble_direction TEXT,                     -- 'buy' | 'sell' | 'hold'
    num_models      INT,                         -- 참여 모델 수
    -- 개별 모델 예측
    sb3_action      FLOAT,                       -- SB3 PPO 예측값
    sb3_version     TEXT,                        -- SB3 모델 버전
    dt_action       FLOAT,                       -- Decision Transformer 예측값
    dt_version      TEXT,
    multi_agent_action FLOAT,                    -- Multi-Agent 합의값
    multi_agent_direction TEXT,                   -- 'buy' | 'sell' | 'hold'
    multi_agent_scalp_action FLOAT,              -- Scalping agent 개별
    multi_agent_swing_action FLOAT,              -- Swing agent 개별
    offline_action  FLOAT,                       -- Offline RL (CQL) 예측값
    offline_version TEXT,
    -- 시장 상태 스냅샷
    btc_price       FLOAT,
    rsi_14          FLOAT,
    fgi             INT,
    danger_score    FLOAT,
    opportunity_score FLOAT,
    -- 사후 평가 (4h/24h 후 업데이트)
    price_after_4h  FLOAT,
    price_after_24h FLOAT,
    return_after_4h FLOAT,                       -- (price_4h - btc_price) / btc_price * 100
    return_after_24h FLOAT,
    prediction_quality TEXT,                     -- 'correct' | 'wrong' | 'neutral' (방향 일치 여부)
    -- 시간
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_mp_decision_id ON rl_model_predictions(decision_id);
CREATE INDEX idx_mp_cycle_id ON rl_model_predictions(cycle_id);
CREATE INDEX idx_mp_ensemble_dir ON rl_model_predictions(ensemble_direction);
CREATE INDEX idx_mp_predicted_at ON rl_model_predictions(predicted_at DESC);
CREATE INDEX idx_mp_quality ON rl_model_predictions(prediction_quality);

COMMENT ON TABLE rl_model_predictions IS 'RL 앙상블 추론 기록 — 개별 모델 예측값 + 사후 평가';

-- ============================================================
-- 3. rl_model_versions: 모델 버전 레지스트리 (DB 복제본)
--    registry.json의 DB 미러링
-- ============================================================
CREATE TABLE IF NOT EXISTS rl_model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id      TEXT NOT NULL UNIQUE,         -- v001, v002, ...
    algorithm       TEXT NOT NULL,                -- ppo, cql, dt, multi_agent
    model_path      TEXT,
    -- 평가 메트릭
    sharpe_ratio    FLOAT,
    total_return_pct FLOAT,
    max_drawdown    FLOAT,
    eval_episodes   INT,
    -- 훈련 설정
    training_steps  INT,
    training_days   INT,
    training_config JSONB,                        -- 전체 훈련 설정 (유연한 구조)
    -- 상태
    is_active       BOOLEAN DEFAULT FALSE,        -- 현재 운영 모델 여부
    is_best         BOOLEAN DEFAULT FALSE,        -- 역대 best 여부
    -- 라이브 성과
    live_trades     INT DEFAULT 0,
    live_win_rate   FLOAT,
    live_avg_return FLOAT,
    live_start_date TIMESTAMPTZ,
    -- 관리
    notes           TEXT,
    promoted_from   TEXT,                         -- 어떤 훈련에서 승격됐는지
    rollback_count  INT DEFAULT 0,
    retired_at      TIMESTAMPTZ,                  -- 퇴역 시각
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_mv_version ON rl_model_versions(version_id);
CREATE INDEX idx_mv_algorithm ON rl_model_versions(algorithm);
CREATE INDEX idx_mv_active ON rl_model_versions(is_active);
CREATE INDEX idx_mv_sharpe ON rl_model_versions(sharpe_ratio DESC NULLS LAST);
CREATE INDEX idx_mv_created ON rl_model_versions(created_at DESC);

COMMENT ON TABLE rl_model_versions IS '모델 버전 레지스트리 — registry.json의 DB 미러링, 라이브 성과 추적';

-- ============================================================
-- 4. rl_parameter_tuning: Self-Tuning 파라미터 변경 이력
-- ============================================================
CREATE TABLE IF NOT EXISTS rl_parameter_tuning (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 변경 내용
    parameter_name  TEXT NOT NULL,                -- 'learning_rate', 'clip_range', 'risk_weight' 등
    old_value       FLOAT,
    new_value       FLOAT,
    change_reason   TEXT,                         -- 'auto_tuning' | 'rollback' | 'manual'
    -- 변경 전후 성과
    before_sharpe   FLOAT,
    after_sharpe    FLOAT,
    before_return   FLOAT,
    after_return    FLOAT,
    -- 승인/롤백
    approved        BOOLEAN,                     -- 텔레그램 승인 여부
    rolled_back     BOOLEAN DEFAULT FALSE,
    rollback_reason TEXT,
    -- 시간
    tuned_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pt_param ON rl_parameter_tuning(parameter_name);
CREATE INDEX idx_pt_tuned_at ON rl_parameter_tuning(tuned_at DESC);
CREATE INDEX idx_pt_reason ON rl_parameter_tuning(change_reason);

COMMENT ON TABLE rl_parameter_tuning IS 'Self-Tuning 파라미터 변경 이력 — 자동/수동 조정 추적';

-- ============================================================
-- 5. 사후 평가 자동화 뷰
-- ============================================================
CREATE OR REPLACE VIEW v_rl_model_performance AS
SELECT
    mv.version_id,
    mv.algorithm,
    mv.sharpe_ratio AS eval_sharpe,
    mv.total_return_pct AS eval_return,
    mv.is_active,
    mv.live_trades,
    mv.live_win_rate,
    mv.live_avg_return,
    COUNT(mp.id) AS prediction_count,
    AVG(mp.return_after_4h) AS avg_return_4h,
    AVG(mp.return_after_24h) AS avg_return_24h,
    COUNT(CASE WHEN mp.prediction_quality = 'correct' THEN 1 END)::FLOAT
        / NULLIF(COUNT(CASE WHEN mp.prediction_quality IS NOT NULL THEN 1 END), 0) AS accuracy
FROM rl_model_versions mv
LEFT JOIN rl_model_predictions mp ON (
    mp.sb3_version = mv.version_id
    OR mp.dt_version = mv.version_id
    OR mp.offline_version = mv.version_id
)
GROUP BY mv.id, mv.version_id, mv.algorithm, mv.sharpe_ratio,
         mv.total_return_pct, mv.is_active, mv.live_trades,
         mv.live_win_rate, mv.live_avg_return
ORDER BY mv.created_at DESC;

CREATE OR REPLACE VIEW v_rl_training_summary AS
SELECT
    algorithm,
    module,
    cycle_type,
    COUNT(*) AS total_cycles,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) AS completed,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) AS failed,
    COUNT(CASE WHEN improved = TRUE THEN 1 END) AS improved_count,
    AVG(avg_sharpe) AS avg_sharpe,
    AVG(avg_return_pct) AS avg_return,
    AVG(elapsed_seconds) AS avg_duration_sec,
    MAX(created_at) AS last_training
FROM rl_training_cycles
GROUP BY algorithm, module, cycle_type
ORDER BY last_training DESC;

COMMENT ON VIEW v_rl_model_performance IS '모델별 평가+라이브 성과+예측 정확도 종합';
COMMENT ON VIEW v_rl_training_summary IS '알고리즘/모듈별 훈련 통계 요약';
