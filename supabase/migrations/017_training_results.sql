-- 분산 RL 훈련 결과 수집 테이블
-- 여러 Trainer가 각자 훈련 후 결과를 업로드하면 Admin이 비교/선발

CREATE TABLE IF NOT EXISTS rl_training_results (
    id              BIGSERIAL PRIMARY KEY,
    trainer_id      TEXT NOT NULL,                -- 훈련자 식별자 (e.g. "trainer-mac-A")
    algorithm       TEXT NOT NULL DEFAULT 'ppo',  -- ppo, sac, td3
    -- 훈련 설정
    training_days   INT NOT NULL DEFAULT 180,
    training_steps  INT NOT NULL DEFAULT 100000,
    interval        TEXT NOT NULL DEFAULT '1h',
    edge_cases      BOOLEAN DEFAULT FALSE,
    synthetic_ratio FLOAT DEFAULT 0.0,
    -- 평가 결과 (순수 실제 데이터 기준)
    avg_return_pct  FLOAT NOT NULL,
    avg_sharpe      FLOAT,
    avg_mdd         FLOAT,
    avg_trades      FLOAT,
    -- 모델 메타
    observation_dim INT DEFAULT 42,
    model_hash      TEXT,                         -- 모델 파일 SHA256
    model_url       TEXT,                         -- 모델 다운로드 URL (Supabase Storage 등)
    -- 비교 기준
    best_return_pct FLOAT,                        -- 업로드 시점 best 모델 수익률
    improvement     FLOAT,                        -- avg_return - best_return
    -- 관리
    status          TEXT DEFAULT 'submitted',     -- submitted, reviewed, promoted, rejected
    review_note     TEXT,                         -- Admin 리뷰 코멘트
    promoted_at     TIMESTAMPTZ,                  -- best로 승격된 시각
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_training_results_trainer ON rl_training_results(trainer_id);
CREATE INDEX idx_training_results_status ON rl_training_results(status);
CREATE INDEX idx_training_results_return ON rl_training_results(avg_return_pct DESC);

COMMENT ON TABLE rl_training_results IS '분산 RL 훈련 결과 — 여러 Trainer가 업로드, Admin이 선발';
