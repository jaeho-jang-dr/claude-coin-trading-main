# 스크립트 상세 문서

> `scripts/` 디렉토리의 모든 스크립트를 역할별로 정리한다.

---

## 데이터 수집

### `collect_market_data.py`

Upbit API에서 BTC/KRW 시장 데이터를 수집하고 기술지표를 계산한다.

**수집 데이터:**
- 현재가 (ticker)
- 일봉 30일 + 4시간봉 42개
- 호가 (orderbook)
- 최근 100건 체결

**계산 지표:**
| 지표 | 설명 |
|------|------|
| RSI(14) | 과매수/과매도 판단 |
| SMA(20, 50, 200) | 이동평균선 |
| EMA(10) | 지수이동평균 |
| MACD | 추세 전환 감지 |
| 볼린저밴드 | 변동성 판단 |
| 스토캐스틱 | 모멘텀 |
| ADX | 추세 강도 (trending/ranging/transitioning) |
| ATR | 변동성 기반 포지션 사이징 |

**ETH/BTC 비율 분석:**
- ETH와 BTC의 30일 가격 비율 z-score 계산
- 극단적 저평가/고평가 시그널 제공

**API 안전장치:**
- 429 (Rate Limit) 시 exponential backoff 재시도 (최대 3회)
- 토큰 절약: raw candle 배열 대신 `daily_summary_5d` 요약만 전송

```bash
python3 scripts/collect_market_data.py    # JSON 출력
```

---

### `collect_fear_greed.py`

Alternative.me API에서 Crypto Fear & Greed Index를 수집한다.

- 현재값 (0~100)
- 분류 (Extreme Fear / Fear / Neutral / Greed / Extreme Greed)
- 7일 추이

```bash
python3 scripts/collect_fear_greed.py     # JSON 출력
```

---

### `collect_news.py`

Tavily Search API로 BTC 관련 뉴스를 수집한다.

**5개 카테고리:**
1. BTC 핵심 뉴스
2. 지정학/경제
3. 알트코인 동향
4. 경제 지표
5. 규제/정책

**API 사용량 관리:**
- 월 999회 제한 → 주중 30/일, 주말 36/일
- `data/tavily_usage.json`에 사용량 추적
- 우선순위: BTC > 지정학 > 알트 > 경제 > 규제

```bash
python3 scripts/collect_news.py           # JSON 출력
```

---

### `collect_ai_signal.py`

Upbit 실시간 데이터로 6가지 AI 복합 시그널을 계산한다.

| 시그널 | 배점 | 설명 |
|--------|------|------|
| 호가 불균형 | ±15 | 매수/매도 호가 비율 |
| 체결 압력 (CVD) | ±15 | 순매수/순매도 비율 |
| 고래 방향 | ±20 | 대량 체결 방향 |
| 다중 타임프레임 | ±15 | 1h/4h/1d 추세 정렬 |
| 변동성 레짐 | ±10 | 급등/급락/안정 판단 |
| 거래량 이상 | ±10 | 평균 대비 거래량 급증 |

**출력:** 복합 점수 (-85 ~ +85) → `strong_buy` / `weak_buy` / `neutral` / `weak_sell` / `strong_sell`

```bash
python3 scripts/collect_ai_signal.py      # JSON 출력
```

---

### `collect_onchain_data.py`

Binance Futures + mempool.space에서 온체인 데이터를 수집한다.

| 데이터 | 소스 | 시그널 |
|--------|------|--------|
| 펀딩레이트 | Binance | 과열_롱/과열_숏/중립 |
| 롱/숏 비율 | Binance | 극단적_롱/극단적_숏/균형 |
| 오픈 인터레스트 | Binance | 급증/급감/안정 |
| 네트워크 수수료 | mempool.space | 매우_활발/활발/한산/보통 |

**종합 시그널:** 역발상 로직 (롱 과열 → bearish, 숏 과열 → bullish)

```bash
python3 scripts/collect_onchain_data.py   # JSON 출력
```

---

### `capture_chart.py`

Playwright headless Chromium으로 Upbit BTC/KRW 차트를 스크린샷한다.

- `data/charts/` 디렉토리에 PNG 저장
- Claude가 차트 이미지를 직접 분석

```bash
python3 scripts/capture_chart.py          # 스크린샷 저장
```

---

### `get_portfolio.py`

Upbit API로 현재 포트폴리오를 조회한다.

- KRW 잔고
- 보유 코인별: 수량, 평균매수가, 현재가, 평가금액, 수익률
- 총 평가금액

```bash
python3 scripts/get_portfolio.py          # JSON 출력
```

---

## 매매 실행

### `execute_trade.py`

Upbit API로 시장가 매수/매도를 실행한다.

**안전장치 (매매 전 반드시 확인):**
1. `EMERGENCY_STOP` 체크
2. `DRY_RUN` 체크
3. `MAX_TRADE_AMOUNT` 초과 여부
4. 일일 매매 횟수 (`MAX_DAILY_TRADES`)
5. 최소 간격 (`MIN_TRADE_INTERVAL_HOURS`)
6. 락파일로 동시 실행 방지 (stale lock 2분 타임아웃 + PID 확인)

**에러 처리:**
- Non-JSON 응답 (점검 시 HTML 반환) → 안전 처리
- PID=0 os.kill 방어

```bash
python3 scripts/execute_trade.py bid 100000    # 10만원 매수
python3 scripts/execute_trade.py ask 0.001     # 0.001 BTC 매도
```

---

### `short_term_trader.py`

실시간 WebSocket 기반 초단타 트레이딩 봇.

**3가지 전략:**
| 전략 | 트리거 | 설명 |
|------|--------|------|
| 뉴스 반응 | RSS 감성 급변 | 긍정/부정 뉴스 감지 → 선제 매매 |
| 급등/급락 리바운드 | 가격 급변 | 급변동 후 되돌림 포착 |
| 고래 추종 | 대량 체결 | 3천만원+ 체결 방향 추종 |

**안전장치:**
- DRY_RUN 강제 (기본)
- 포지션별 자동 손절/익절/시간제한
- 연속 5회 에러 → 긴급 정지
- 인증 에러 → 즉시 정지

**모니터링:**
- 5분마다 상태 리포트
- 10분마다 전략 변곡점 알림 (RSI, 고래, 급변동, 지지선)

```bash
python3 scripts/short_term_trader.py --dry-run   # 시뮬레이션
python3 scripts/short_term_trader.py --status     # 상태 확인
```

---

## 알림

### `notify_telegram.py`

Telegram Bot API로 알림을 전송한다.

- MarkdownV2 포맷
- 차트 이미지 첨부 지원
- 알림 유형: 매매 결과, 분석 리포트, 에러, 상태

```bash
python3 scripts/notify_telegram.py "메시지 내용"
```

---

### `dashboard.py`

Flask 웹 대시보드.

- 포트폴리오 현황
- 최근 매매 결정
- 시장 데이터
- 긴급 정지 토글
- QR 코드 (로컬 + 원격)

**접속:**
- 로컬: `http://localhost:5555`
- 원격: `https://dashboard.wwwmoksu.com` (Cloudflare Tunnel)

---

## 자동화

### `cron_run.sh`

cron 실행 래퍼. 전체 파이프라인을 안전하게 실행한다.

**실행 순서:**
1. `.env` 로드 → `.venv` 활성화
2. `EMERGENCY_STOP` 확인
3. `run_analysis.sh` 실행 (데이터 수집 + 프롬프트 조립)
4. `claude -p --dangerously-skip-permissions` 실행
5. 응답을 `save_decision.py`로 파이프 → Supabase 저장
6. 텔레그램 알림 전송
7. 에러 시 텔레그램으로 에러 알림

**로그:**
- `logs/executions/{TIMESTAMP}.log`
- `logs/claude_responses/{TIMESTAMP}.txt`

---

### `run_analysis.sh`

데이터 수집 → 프롬프트 조립 오케스트레이터.

**수집 순서:**
```
collect_market_data.py  → 시장 데이터
collect_fear_greed.py   → 공포탐욕지수
collect_news.py         → 뉴스
capture_chart.py        → 차트 스크린샷
get_portfolio.py        → 포트폴리오
collect_ai_signal.py    → AI 복합 시그널
collect_onchain_data.py → 온체인 데이터
```

**프롬프트에 포함되는 것:**
- `strategy.md` (매매 전략)
- 수집된 모든 데이터
- Supabase에서 과거 10건 결정 + profit_loss
- 미반영 피드백
- 초단타 성과 (크로스 학습)
- 고래 동향

---

### `startup.sh`

macOS 부팅 시 자동 실행.

1. tmux `blockchain` 세션 생성
2. Claude Code Remote Control 시작
3. 대시보드 서버 시작
4. QR 페이지 생성

---

### `watchdog_remote.sh`

Claude Code Remote Control 감시 + 킵얼라이브.

**설계 철학:** 치료보다 예방

**복구 우선순위:**
1. active → 5분마다 킵얼라이브 (10분 타임아웃 예방)
2. reconnecting → disconnect → /rc 재연결
3. 재연결 실패 → Claude 완전 재시작 (최후 수단)
4. Claude 없음 → 새로 시작
5. tmux 없음 → startup.sh

---

### `save_decision.py`

Claude의 응답을 파싱하여 Supabase에 저장한다.

**JSON 파싱 5단계 fallback:**
1. ` ```json ` 코드펜스
2. ` ``` ` 코드펜스
3. 전체 텍스트 JSON
4. 가장 큰 `{ }` 블록 (depth 기반, 문자열 내 중괄호 무시)
5. 불완전 JSON 복구 (닫는 괄호 추가)

**추가 기능:**
- 과거 결정의 profit_loss 자동 업데이트 (일괄 PATCH)
- 포트폴리오 스냅샷 저장
- 사용된 피드백 applied=true 처리
