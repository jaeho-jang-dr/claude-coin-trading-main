-- 매매 회고 뷰 + 트리거 (002에서 누락된 부분)

-- 1. 매매 이력 종합 뷰
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

-- 2. 성과 요약 뷰
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

-- 3. 월별 성과 뷰
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

-- 4. updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_trade_reviews_updated_at ON trade_reviews;
CREATE TRIGGER tr_trade_reviews_updated_at
  BEFORE UPDATE ON trade_reviews
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();
