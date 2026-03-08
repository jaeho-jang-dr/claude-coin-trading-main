# 데이터베이스 스키마

> Supabase (PostgreSQL) — 12개 테이블 + 3개 뷰

---

## 테이블 관계도

```
decisions ──1:1──▶ trade_reviews    (사후 평가)
decisions ◀──FK── execution_logs   (실행 로그)
decisions ◀─────── feedback         (피드백 → 다음 실행에 반영)

scalp_sessions ─── scalp_trades    (세션별 매매)
whale_detections ── strategy_alerts (고래 → 알림)
```

---

## 핵심 테이블

### `decisions` — 매매 결정

AI의 모든 매매 결정을 기록한다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID | PK |
| market | TEXT | `KRW-BTC` |
| decision | TEXT | `매수` / `매도` / `관망` |
| confidence | DECIMAL(3,2) | 신뢰도 (0.00~0.99) |
| reason | TEXT | 결정 근거 |
| current_price | BIGINT | 결정 시점 가격 |
| fear_greed_value | INTEGER | FGI 값 (0~100) |
| rsi_value | DECIMAL(5,2) | RSI 값 |
| sma20_price | BIGINT | SMA20 가격 |
| trade_amount | BIGINT | 매매 금액 (KRW) |
| executed | BOOLEAN | 실제 체결 여부 |
| execution_result | JSONB | 체결 상세 (주문ID, 체결가 등) |
| profit_loss | DECIMAL(10,2) | 사후 수익률 (%) — 자동 업데이트 |
| market_data_snapshot | TEXT | 전체 시장 데이터 스냅샷 |
| created_at | TIMESTAMPTZ | 결정 시각 |

**인덱스:** `created_at DESC`, `market`, `decision`

---

### `portfolio_snapshots` — 포트폴리오

매 실행마다 포트폴리오 상태를 스냅샷한다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| total_krw | BIGINT | KRW 잔고 |
| total_crypto_value | BIGINT | 암호화폐 평가액 |
| total_value | BIGINT | 총 평가액 |
| holdings | JSONB | 보유 코인 상세 |
| daily_return | DECIMAL(10,4) | 일간 수익률 |
| cumulative_return | DECIMAL(10,4) | 누적 수익률 |

---

### `feedback` — 사용자 피드백

사용자가 전략/행동 변경을 요청하면 저장하고, 다음 실행에 반영한다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| type | TEXT | `parameter_change` / `behavior_change` / `one_time` / `general` |
| content | TEXT | 피드백 내용 |
| applied | BOOLEAN | 반영 여부 |
| applied_at | TIMESTAMPTZ | 반영 시각 |
| expires_at | TIMESTAMPTZ | 만료 시각 (일시적 피드백용) |

**피드백 루프:**
```
사용자 피드백 → feedback 테이블 (applied=false)
    ↓
다음 cron 실행 시 프롬프트에 주입
    ↓
Claude가 피드백 반영하여 결정
    ↓
save_decision.py가 applied=true 처리
```

---

## 초단타 테이블

### `scalp_trades` — 초단타 매매

| 컬럼 | 타입 | 설명 |
|------|------|------|
| strategy | TEXT | `news` / `spike` / `whale` |
| side | TEXT | `bid` / `ask` |
| entry_price | BIGINT | 진입가 |
| exit_price | BIGINT | 청산가 |
| amount_krw | INTEGER | 매매 금액 |
| btc_qty | DECIMAL(18,8) | BTC 수량 |
| pnl_pct | DECIMAL(6,3) | 수익률 (%) |
| pnl_krw | INTEGER | 손익 (KRW) |
| exit_reason | TEXT | 청산 사유 (익절/손절/시간만료) |
| confidence | DECIMAL(3,2) | 시그널 신뢰도 |
| dry_run | BOOLEAN | DRY_RUN 여부 |

---

### `whale_detections` — 고래 감지

| 컬럼 | 타입 | 설명 |
|------|------|------|
| side | TEXT | `BID` / `ASK` |
| volume | DECIMAL(18,8) | 거래량 (BTC) |
| krw_amount | BIGINT | 거래 금액 (KRW) |
| whale_buy_count | INTEGER | 감지 윈도우 내 매수 고래 수 |
| whale_sell_count | INTEGER | 감지 윈도우 내 매도 고래 수 |
| triggered_trade | BOOLEAN | 이 고래로 매매 발생 여부 |

---

### `strategy_alerts` — 전략 변곡점 알림

| alert_type | 설명 |
|------------|------|
| `rsi_oversold` | RSI 30 이하 |
| `rsi_extreme_oversold` | RSI 25 이하 |
| `rsi_overbought` | RSI 75 이상 |
| `whale_buy_reversal` | 고래 매수 전환 |
| `whale_sell_pressure` | 고래 매도 집중 |
| `price_spike` | 10분 내 1.5%+ 급등 |
| `price_crash` | 10분 내 1.5%+ 급락 |
| `support_break` | 심리적 지지선 이탈 |
| `resistance_break` | 전략 전환 제안 |
| `news_extreme` | 뉴스 감성 극단 |
| `emergency_stop` | 긴급 정지 |

---

### `scalp_sessions` — 세션 요약

| 컬럼 | 타입 | 설명 |
|------|------|------|
| mode | TEXT | `dry_run` / `live` |
| total_trades | INTEGER | 총 매매 횟수 |
| wins / losses | INTEGER | 승/패 |
| win_rate | DECIMAL(5,2) | 승률 (%) |
| total_pnl_krw | INTEGER | 총 손익 (KRW) |
| whale_count | INTEGER | 감지된 고래 수 |

---

### `ai_signal_log` — AI 복합 시그널

| 컬럼 | 타입 | 설명 |
|------|------|------|
| composite_score | INTEGER | 종합 점수 (-85 ~ +85) |
| interpretation | TEXT | `strong_buy` ~ `strong_sell` |
| orderbook_imbalance | INTEGER | 호가 불균형 점수 |
| trade_pressure | INTEGER | 체결 압력 점수 |
| whale_direction | INTEGER | 고래 방향 점수 |
| tf_divergence | INTEGER | 다중 TF 정렬 점수 |
| volatility_regime | INTEGER | 변동성 레짐 점수 |
| volume_anomaly | INTEGER | 거래량 이상 점수 |

---

## 매매 회고

### `trade_reviews` — 사후 평가

| 컬럼 | 타입 | 설명 |
|------|------|------|
| decision_id | UUID FK | decisions 참조 |
| entry_price / exit_price | BIGINT | 진입/청산 가격 |
| profit_loss_pct | DECIMAL(10,4) | 손익률 (%) |
| holding_hours | DECIMAL(10,2) | 보유 시간 |
| max_profit_pct | DECIMAL(10,4) | 보유 중 최대 수익률 |
| max_drawdown_pct | DECIMAL(10,4) | 보유 중 최대 낙폭 |
| exit_reason | TEXT | 청산 사유 |
| was_correct | BOOLEAN | 올바른 결정이었는가 |
| lesson | TEXT | 회고 코멘트 |
| status | TEXT | `open` / `closed` / `cancelled` |

---

## 분석 뷰

### `v_trade_history`

decisions + trade_reviews를 조인한 매매 이력 종합 뷰.

### `v_performance_summary`

| 항목 | 설명 |
|------|------|
| total_trades | 총 매매 수 |
| win_rate_pct | 승률 (%) |
| avg_profit_loss_pct | 평균 수익률 |
| total_profit_loss_krw | 총 손익 (KRW) |
| avg_holding_hours | 평균 보유 시간 |
| correct_decisions | 올바른 결정 수 |

### `v_monthly_performance`

월별 성과: 매매 수, 승률, 총 손익, 최고/최저 거래.
