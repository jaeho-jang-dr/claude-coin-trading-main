-- 텔레그램 연락처 통합 관리
-- 워커(owner/coworker/collaborator) + 친구(friend) 모두 포함
-- 앱 내에서 모든 등록자 간 텔레그램 송수신 가능

CREATE TABLE IF NOT EXISTS telegram_contacts (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  chat_id TEXT NOT NULL UNIQUE,          -- 텔레그램 chat_id (숫자)
  name TEXT NOT NULL,                     -- 표시 이름
  role TEXT NOT NULL DEFAULT 'friend' CHECK (role IN (
    'owner', 'coworker', 'collaborator', 'viewer', 'friend'
  )),
  email TEXT,
  worker_id TEXT,                         -- compute_workers 연동 (워커인 경우)
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_tg_contacts_role ON telegram_contacts(role);
CREATE INDEX IF NOT EXISTS idx_tg_contacts_chat ON telegram_contacts(chat_id);
