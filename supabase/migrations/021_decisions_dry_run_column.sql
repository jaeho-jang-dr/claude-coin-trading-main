-- decisions 테이블에 dry_run 컬럼 추가
-- execute_trade.py의 _record_trade_to_db에서 dry_run 필드를 기록하기 위해 필요

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS dry_run boolean DEFAULT false;

COMMENT ON COLUMN decisions.dry_run IS '드라이런 여부 (true: 시뮬레이션, false: 실제 매매)';
