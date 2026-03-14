-- 텔레그램 메시지 수신/발신 기록
-- 워커↔관리자 양방향 메시징

CREATE TABLE IF NOT EXISTS telegram_messages (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  chat_id TEXT NOT NULL,                -- 텔레그램 chat_id
  worker_id TEXT,                        -- compute_workers.worker_id (매핑된 경우)
  worker_name TEXT,                      -- 표시 이름
  direction TEXT NOT NULL CHECK (direction IN ('incoming', 'outgoing')),
  message TEXT NOT NULL,
  telegram_message_id BIGINT,            -- 텔레그램 메시지 ID
  telegram_username TEXT,                -- @username
  telegram_first_name TEXT,              -- 텔레그램 이름
  is_read BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tg_msg_chat ON telegram_messages(chat_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tg_msg_unread ON telegram_messages(is_read, direction)
  WHERE NOT is_read AND direction = 'incoming';

-- 7일 보관 정책: telegram_listener.py가 1시간마다 자동 정리
-- 수동 정리: DELETE FROM telegram_messages WHERE created_at < NOW() - INTERVAL '7 days';
COMMENT ON TABLE telegram_messages IS '텔레그램 송수신 기록 — 7일 보관 후 자동 삭제';
