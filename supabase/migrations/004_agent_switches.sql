-- 에이전트 전략 전환 이력 + 학습 데이터
-- Orchestrator가 자율적으로 전략을 전환하고, 결과를 추적하여 학습한다.

-- 1. 전환 이력 (모든 전환 기록)
CREATE TABLE agent_switches (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  from_agent TEXT NOT NULL,
  to_agent TEXT NOT NULL,
  reason TEXT NOT NULL,

  -- 전환 시점 시장 상태 (학습용 컨텍스트)
  fgi_at_switch INTEGER,
  rsi_at_switch DECIMAL(5,2),
  price_at_switch BIGINT,
  price_change_24h DECIMAL(6,2),
  kimchi_premium DECIMAL(5,2),
  fusion_signal TEXT,            -- strong_buy / buy / neutral / sell / strong_sell
  consecutive_losses INTEGER DEFAULT 0,

  -- 전환 후 성과 평가 (나중에 업데이트)
  price_after_4h BIGINT,         -- 전환 4시간 후 가격
  price_after_24h BIGINT,        -- 전환 24시간 후 가격
  profit_after_4h DECIMAL(6,2),  -- 4시간 후 수익률 %
  profit_after_24h DECIMAL(6,2), -- 24시간 후 수익률 %
  outcome TEXT,                  -- 'good' | 'neutral' | 'bad' (사후 평가)
  outcome_reason TEXT,           -- 평가 근거

  evaluated_at TIMESTAMPTZ,      -- 성과 평가 시점
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_switches_created ON agent_switches(created_at DESC);
CREATE INDEX idx_agent_switches_outcome ON agent_switches(outcome);
CREATE INDEX idx_agent_switches_agents ON agent_switches(from_agent, to_agent);

-- 2. 전환 성과 요약 뷰 (학습 참조용)
CREATE OR REPLACE VIEW agent_switch_performance AS
SELECT
  from_agent,
  to_agent,
  COUNT(*) as total_switches,
  COUNT(*) FILTER (WHERE outcome = 'good') as good_count,
  COUNT(*) FILTER (WHERE outcome = 'bad') as bad_count,
  COUNT(*) FILTER (WHERE outcome = 'neutral') as neutral_count,
  ROUND(
    COUNT(*) FILTER (WHERE outcome = 'good')::DECIMAL / NULLIF(COUNT(*) FILTER (WHERE outcome IS NOT NULL), 0) * 100,
    1
  ) as success_rate_pct,
  ROUND(AVG(profit_after_24h)::DECIMAL, 2) as avg_profit_24h,
  MAX(created_at) as last_switch_at
FROM agent_switches
WHERE evaluated_at IS NOT NULL
GROUP BY from_agent, to_agent;

-- 3. 시장 상황별 전환 성과 뷰
CREATE OR REPLACE VIEW agent_switch_by_market AS
SELECT
  from_agent,
  to_agent,
  CASE
    WHEN fgi_at_switch <= 25 THEN 'extreme_fear'
    WHEN fgi_at_switch <= 40 THEN 'fear'
    WHEN fgi_at_switch <= 60 THEN 'neutral'
    WHEN fgi_at_switch <= 75 THEN 'greed'
    ELSE 'extreme_greed'
  END as market_phase,
  COUNT(*) as switches,
  ROUND(AVG(profit_after_24h)::DECIMAL, 2) as avg_profit_24h,
  COUNT(*) FILTER (WHERE outcome = 'good') as good_count,
  COUNT(*) FILTER (WHERE outcome = 'bad') as bad_count
FROM agent_switches
WHERE evaluated_at IS NOT NULL
GROUP BY from_agent, to_agent, market_phase;
