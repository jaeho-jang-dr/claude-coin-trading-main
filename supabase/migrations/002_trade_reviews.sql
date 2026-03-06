-- 매매 회고 테이블 + 성과 분석 뷰
-- decisions 테이블의 매수/매도 기록을 기반으로 사후 평가를 기록한다.

-- 1. 매매 회고 테이블
CREATE TABLE trade_reviews (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  decision_id UUID NOT NULL REFERENCES decisions(id),
  -- 매매 시점 스냅샷
  entry_price BIGINT NOT NULL,                -- 진입 가격
  exit_price BIGINT,                          -- 청산 가격 (미청산이면 NULL)
  entry_at TIMESTAMPTZ NOT NULL,              -- 진입 시각
  exit_at TIMESTAMPTZ,                        -- 청산 시각
  -- 성과
  profit_loss_krw BIGINT DEFAULT 0,           -- 손익 (KRW)
  profit_loss_pct DECIMAL(10,4) DEFAULT 0,    -- 손익률 (%)
  holding_hours DECIMAL(10,2),                -- 보유 시간
  -- 결정 당시 지표
  entry_rsi DECIMAL(5,2),
  entry_fgi INTEGER,
  entry_sma20_deviation DECIMAL(10,4),        -- SMA20 대비 이탈률 (%)
  entry_news_sentiment TEXT,
  -- 사후 분석
  max_profit_pct DECIMAL(10,4),               -- 보유 기간 중 최대 수익률
  max_drawdown_pct DECIMAL(10,4),             -- 보유 기간 중 최대 낙폭
  exit_reason TEXT,                           -- 청산 사유 (목표수익/손절/과매수/수동)
  was_correct BOOLEAN,                        -- 사후 판단: 올바른 결정이었는가
  lesson TEXT,                                -- 회고 코멘트
  -- 메타
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trade_reviews_decision ON trade_reviews(decision_id);
CREATE INDEX idx_trade_reviews_status ON trade_reviews(status);
CREATE INDEX idx_trade_reviews_entry_at ON trade_reviews(entry_at DESC);
CREATE INDEX idx_trade_reviews_profit ON trade_reviews(profit_loss_pct);

-- 2. 매매 이력 종합 뷰 (decisions + trade_reviews 조인)
CREATE OR REPLACE VIEW v_trade_history AS
SELECT
  d.id AS decision_id,
  d.market,
  d.decision,
  d.confidence,
  d.reason,
  d.current_price,
  d.fear_greed_value,
  d.rsi_value,
  d.sma20_price,
  d.trade_amount,
  d.executed,
  d.created_at AS decided_at,
  -- 회고 데이터
  tr.id AS review_id,
  tr.entry_price,
  tr.exit_price,
  tr.profit_loss_krw,
  tr.profit_loss_pct,
  tr.holding_hours,
  tr.max_profit_pct,
  tr.max_drawdown_pct,
  tr.exit_reason,
  tr.was_correct,
  tr.lesson,
  tr.status AS review_status
FROM decisions d
LEFT JOIN trade_reviews tr ON tr.decision_id = d.id
WHERE d.decision IN ('매수', '매도')
ORDER BY d.created_at DESC;

-- 3. 성과 요약 뷰 (기간별 승률, 평균 수익률, 총 손익)
CREATE OR REPLACE VIEW v_performance_summary AS
SELECT
  COUNT(*) AS total_trades,
  COUNT(*) FILTER (WHERE tr.status = 'closed') AS closed_trades,
  COUNT(*) FILTER (WHERE tr.status = 'open') AS open_trades,
  COUNT(*) FILTER (WHERE tr.profit_loss_pct > 0 AND tr.status = 'closed') AS winning_trades,
  COUNT(*) FILTER (WHERE tr.profit_loss_pct <= 0 AND tr.status = 'closed') AS losing_trades,
  ROUND(
    COUNT(*) FILTER (WHERE tr.profit_loss_pct > 0 AND tr.status = 'closed')::DECIMAL
    / NULLIF(COUNT(*) FILTER (WHERE tr.status = 'closed'), 0) * 100, 2
  ) AS win_rate_pct,
  ROUND(AVG(tr.profit_loss_pct) FILTER (WHERE tr.status = 'closed'), 4) AS avg_profit_loss_pct,
  COALESCE(SUM(tr.profit_loss_krw) FILTER (WHERE tr.status = 'closed'), 0) AS total_profit_loss_krw,
  ROUND(AVG(tr.holding_hours) FILTER (WHERE tr.status = 'closed'), 2) AS avg_holding_hours,
  ROUND(AVG(tr.max_drawdown_pct) FILTER (WHERE tr.status = 'closed'), 4) AS avg_max_drawdown_pct,
  COUNT(*) FILTER (WHERE tr.was_correct = true) AS correct_decisions,
  COUNT(*) FILTER (WHERE tr.was_correct = false) AS wrong_decisions
FROM trade_reviews tr;

-- 4. 월별 성과 뷰
CREATE OR REPLACE VIEW v_monthly_performance AS
SELECT
  TO_CHAR(tr.entry_at, 'YYYY-MM') AS month,
  COUNT(*) AS trades,
  COUNT(*) FILTER (WHERE tr.profit_loss_pct > 0) AS wins,
  COUNT(*) FILTER (WHERE tr.profit_loss_pct <= 0) AS losses,
  ROUND(
    COUNT(*) FILTER (WHERE tr.profit_loss_pct > 0)::DECIMAL
    / NULLIF(COUNT(*), 0) * 100, 2
  ) AS win_rate_pct,
  COALESCE(SUM(tr.profit_loss_krw), 0) AS total_pnl_krw,
  ROUND(AVG(tr.profit_loss_pct), 4) AS avg_pnl_pct,
  MAX(tr.profit_loss_pct) AS best_trade_pct,
  MIN(tr.profit_loss_pct) AS worst_trade_pct
FROM trade_reviews tr
WHERE tr.status = 'closed'
GROUP BY TO_CHAR(tr.entry_at, 'YYYY-MM')
ORDER BY month DESC;

-- 5. updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_trade_reviews_updated_at
  BEFORE UPDATE ON trade_reviews
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();
