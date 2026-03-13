-- kimchirang_trades.action CHECK 제약 조건에 'error' 추가
-- db.py record_error()가 action='error'로 기록하므로 허용 필요

ALTER TABLE kimchirang_trades DROP CONSTRAINT IF EXISTS kimchirang_trades_action_check;
ALTER TABLE kimchirang_trades ADD CONSTRAINT kimchirang_trades_action_check
  CHECK (action IN ('enter', 'exit', 'stop_loss', 'error'));
