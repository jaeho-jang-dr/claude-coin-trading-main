-- 텔레그램 테이블 RLS 정책
-- telegram_contacts: 모든 워커가 읽기 가능
-- telegram_messages: 자기 메시지만 읽기, 발신 가능

-- ── telegram_contacts ──
CREATE POLICY contacts_service_all ON telegram_contacts
  FOR ALL USING (is_service_role());

CREATE POLICY contacts_worker_read ON telegram_contacts
  FOR SELECT USING (
    get_worker_tier() IN ('owner','coworker','collaborator','viewer')
    AND is_active = true
  );

-- ── telegram_messages ──
CREATE POLICY messages_service_all ON telegram_messages
  FOR ALL USING (is_service_role());

CREATE POLICY messages_worker_read ON telegram_messages
  FOR SELECT USING (
    get_worker_tier() IN ('owner','coworker','collaborator','viewer')
    AND chat_id = (
      SELECT telegram_chat_id FROM compute_workers
      WHERE worker_id = get_worker_id()
    )
  );

CREATE POLICY messages_worker_insert ON telegram_messages
  FOR INSERT WITH CHECK (
    get_worker_tier() IN ('owner','coworker','collaborator')
  );
