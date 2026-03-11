-- RL 학습 결과 로그 테이블
-- training_scheduler.py 의 tier1/tier2/tier3 학습 결과를 저장하여 추후 쿼리 가능하게 한다.

CREATE TABLE IF NOT EXISTS rl_training_log (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 학습 설정
    tier            TEXT NOT NULL,           -- tier1, tier2, tier3
    tier_name       TEXT,                    -- Quick Incremental, Daily Training, Weekly Full Retrain
    steps           INTEGER NOT NULL,        -- 학습 스텝 수
    data_days       INTEGER NOT NULL,        -- 학습 데이터 일수
    lr_ratio        REAL NOT NULL,           -- 학습률 비율
    candles_count   INTEGER,                 -- 사용된 캔들 수

    -- 베이스라인 (학습 전)
    baseline_return_pct  REAL,
    baseline_sharpe      REAL,
    baseline_mdd         REAL,
    baseline_trades      REAL,

    -- 학습 후 성과
    new_return_pct       REAL,
    new_sharpe           REAL,
    new_mdd              REAL,
    new_trades           REAL,

    -- 결과
    improved        BOOLEAN NOT NULL,        -- 성과 개선 여부
    version_id      TEXT,                    -- 모델 버전 ID (개선 시)
    elapsed_sec     REAL,                    -- 학습 소요 시간 (초)
    rollback        BOOLEAN DEFAULT FALSE,   -- 롤백 여부

    -- 추가 메타데이터
    notes           TEXT                     -- 비고 (롤백 사유 등)
);

-- 인덱스: tier별, 날짜별 조회 최적화
CREATE INDEX idx_rl_training_tier ON rl_training_log(tier);
CREATE INDEX idx_rl_training_created ON rl_training_log(created_at DESC);

-- RLS 활성화
ALTER TABLE rl_training_log ENABLE ROW LEVEL SECURITY;

-- service_role 전체 접근 정책
CREATE POLICY "service_role_all" ON rl_training_log
    FOR ALL USING (true) WITH CHECK (true);

COMMENT ON TABLE rl_training_log IS 'RL (SB3 PPO) 학습 결과 로그. tier1/2/3 학습마다 베이스라인 대비 성과를 기록한다.';
