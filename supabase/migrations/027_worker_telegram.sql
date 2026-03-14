-- 워커 연락처: 텔레그램 + 이메일 (가입 시 필수)
ALTER TABLE compute_workers ADD COLUMN IF NOT EXISTS telegram_chat_id TEXT;
ALTER TABLE compute_workers ADD COLUMN IF NOT EXISTS email TEXT;

COMMENT ON COLUMN compute_workers.telegram_chat_id IS '워커의 텔레그램 chat_id (숫자) — 직접 메시지 발송용';
COMMENT ON COLUMN compute_workers.email IS '워커 이메일 — 연락 및 관리용';
