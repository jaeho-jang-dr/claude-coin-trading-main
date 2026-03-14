-- 앱 변경 이력 테이블
-- 코드/전략/설정 수정 시 자동 기록하여 앱 진화 과정을 추적한다.

CREATE TABLE IF NOT EXISTS app_changelog (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  version TEXT NOT NULL,                -- 시맨틱 버전 (예: 1.3.0)
  severity TEXT NOT NULL CHECK (severity IN ('critical', 'major', 'minor', 'patch')),
  category TEXT NOT NULL CHECK (category IN ('bugfix', 'feature', 'improvement', 'refactor', 'config', 'strategy')),
  summary TEXT NOT NULL,                -- 변경 요약 (1줄)
  details JSONB,                        -- 상세 내용 (파일목록, 변경사항 등)
  files_modified TEXT[],                -- 수정된 파일 경로 배열
  verified BOOLEAN DEFAULT FALSE,       -- 테스트/검증 완료 여부
  verification_result TEXT,             -- 검증 결과 요약
  changed_by TEXT DEFAULT 'claude_session',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_app_changelog_version ON app_changelog(version);
CREATE INDEX idx_app_changelog_created_at ON app_changelog(created_at DESC);
CREATE INDEX idx_app_changelog_severity ON app_changelog(severity);
