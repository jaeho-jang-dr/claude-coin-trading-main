# 아키텍처

> 코드가 아닌 자연어로 전략을 정의하고, AI가 데이터를 해석하여 판단하는 구조.

---

## 설계 철학

```
전통적 봇:  if RSI < 30 and FGI < 30: buy()     ← 코드로 하드코딩
이 시스템:  strategy.md → Claude가 읽고 판단      ← 자연어 전략
```

**핵심 원칙:**
- Python 스크립트는 **데이터 수집과 API 호출만** 담당
- 매매 판단은 **Claude가 자율적으로** 수행
- 전략 변경 = `strategy.md` 수정 (코드 변경 없음)

---

## 시스템 전체 흐름

```
┌─────────────────────────────────────────────────┐
│                    cron (4시간)                   │
│                  cron_run.sh                      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│             run_analysis.sh                      │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ Upbit    │ │ Alt.me   │ │ Tavily   │        │
│  │ 시세/호가 │ │ FGI      │ │ 뉴스     │        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘        │
│       │            │            │               │
│  ┌────┴─────┐ ┌────┴─────┐ ┌───┴──────┐       │
│  │ Binance  │ │ mempool  │ │ Supabase │       │
│  │ 선물/OI  │ │ 수수료   │ │ 과거결정  │       │
│  └────┬─────┘ └────┬─────┘ └───┬──────┘       │
│       │            │            │               │
│       └────────────┼────────────┘               │
│                    ▼                             │
│          strategy.md + 모든 데이터               │
│              → 프롬프트 조립                     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           claude -p (비대화형)                    │
│                                                  │
│   데이터 해석 → 점수 계산 → 매수/매도/관망 결정   │
│                                                  │
│   ┌─────────────┐     ┌─────────────┐           │
│   │execute_trade│     │notify_tg    │           │
│   │ (매매 실행) │     │ (알림 전송) │           │
│   └──────┬──────┘     └──────┬──────┘           │
└──────────┼───────────────────┼──────────────────┘
           │                   │
           ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│   Upbit 거래소    │  │   Telegram Bot   │
│   (주문 체결)     │  │   (결과 보고)     │
└──────────────────┘  └──────────────────┘
           │
           ▼
┌──────────────────┐
│   save_decision  │
│   → Supabase DB  │
└──────────────────┘
```

---

## 실행 모드

### 1. 자동 실행 (cron)

```
LaunchAgent (4시간 간격)
    → cron_run.sh
        → .env 로드 + EMERGENCY_STOP 확인
        → run_analysis.sh (7개 데이터 수집 + 프롬프트 조립)
        → claude -p --dangerously-skip-permissions
        → save_decision.py (응답 파싱 → Supabase)
        → notify_telegram.py (결과 보고)
        → 에러 시 텔레그램 에러 알림
```

### 2. 초단타 봇 (실시간)

```
LaunchAgent (매일 00:10, 랜덤 6시간)
    → short_term_trader.py
        → WebSocket 연결 (Upbit 실시간 체결)
        → 3전략 병렬 감시 (뉴스/급변동/고래)
        → 시그널 발생 → DRY_RUN 매매
        → Supabase 기록 + 텔레그램 알림
```

### 3. 대화형 (Claude 세션)

```
사용자 → claude (대화형)
    → 전략 수정 / 피드백 제출 / 수동 분석
    → 변경사항 → strategy.md / feedback 테이블
```

### 4. 대시보드 (상시)

```
LaunchAgent (부팅 시 + 상시)
    → dashboard.py (Flask, :5555)
    → Cloudflare Tunnel → dashboard.wwwmoksu.com
```

---

## 데이터 파이프라인 상세

### 수집 단계 (run_analysis.sh)

```
[1] collect_market_data.py  ─── Upbit REST API
    현재가, 일봉 30일, 4시간봉 42개, 호가, 체결
    → RSI, SMA, EMA, MACD, 볼린저, 스토캐스틱, ADX, ATR

[2] collect_fear_greed.py   ─── Alternative.me API
    현재 FGI + 7일 추이

[3] collect_news.py         ─── Tavily Search API
    5카테고리: BTC핵심 / 지정학 / 알트 / 경제 / 규제
    월 999회 제한 → 우선순위 기반 할당

[4] capture_chart.py        ─── Playwright headless
    Upbit BTC/KRW 차트 스크린샷 → PNG

[5] get_portfolio.py        ─── Upbit REST API (JWT)
    잔고, 보유 코인, 수익률

[6] collect_ai_signal.py    ─── Upbit REST API
    6요소 복합 시그널 (-85 ~ +85)

[7] collect_onchain_data.py ─── Binance + mempool.space
    펀딩레이트, 롱숏비율, OI, 네트워크 수수료
```

### 프롬프트 조립

```
strategy.md                    ← 전략 (자연어)
+ 7개 수집 데이터              ← 현재 시장 상태
+ Supabase 과거 결정 10건      ← 이전 판단 + 성과
+ 미반영 피드백                ← 사용자 요청
+ 초단타 성과 + 고래 동향      ← 크로스 학습
+ ETH/BTC 비율                ← 시장 구조
────────────────────────────
= 프롬프트 → claude -p
```

---

## 피드백 루프

```
[실행 1] 매수 결정 (confidence 0.75)
    │
    ├── Supabase에 결정 기록
    │
    ▼
[실행 2] 4시간 후 재실행
    │
    ├── 실행 1의 결정 + 현재가로 profit_loss 계산
    ├── "이전 매수가 105M → 현재 107M, +1.9% 수익"
    ├── "이전 결정의 정확도를 분석하고 반복 실수 방지"
    │
    ▼
[실행 3] 사용자 피드백 반영
    │
    ├── feedback 테이블: "매수 비율을 5%로 줄여줘"
    ├── applied=false → 프롬프트에 주입
    ├── Claude가 반영 → applied=true 처리
    │
    ▼
[실행 N] 점점 더 정확한 판단
```

---

## 안전장치 계층

```
Layer 1: 환경변수 (.env)
├── EMERGENCY_STOP     ← 모든 매매 즉시 중지
├── DRY_RUN            ← 분석만, 실매매 안 함
├── MAX_TRADE_AMOUNT   ← 1회 상한
├── MAX_DAILY_TRADES   ← 일일 상한
├── MAX_POSITION_RATIO ← 자산 비율 상한
└── MIN_TRADE_INTERVAL ← 최소 간격

Layer 2: execute_trade.py
├── EMERGENCY_STOP 체크
├── DRY_RUN 체크
├── MAX_TRADE_AMOUNT 초과 거부
├── 일일 매매 횟수 확인
├── 최소 간격 확인
└── 락파일 동시 실행 방지

Layer 3: strategy.md
├── 점수제 매수 (임계점 미달 시 매수 안 함)
├── 하이브리드 손절 (바닥 판단 후 분기)
├── AI 복합 시그널 보조 필터
└── 전략 전환 금지 조건
```

---

## 프로세스 관리 (LaunchAgent)

| plist | 프로세스 | 실행 시점 |
|-------|---------|----------|
| `com.claude.blockchain-startup` | startup.sh | 부팅 시 |
| `com.claude.coin-trading` | cron_run.sh | 0/4/8/12/16/20시 |
| `com.claude.short-term-trading` | short_term_trader.py | 매일 00:10 |
| `com.claude.watchdog-remote` | watchdog_remote.sh | 3분 간격 |
| `com.claude.cloudflared-tunnel` | cloudflared tunnel run | 부팅 시, 상시 |

---

## 외부 서비스 의존성

```
┌─────────────────────────────────────────┐
│              이 시스템                    │
├─────────────────────────────────────────┤
│                                         │
│  Upbit REST API ◀──── 시세, 매매, 잔고  │
│  Upbit WebSocket ◀─── 실시간 체결       │
│  Alternative.me ◀──── FGI              │
│  Tavily API ◀──────── 뉴스 (월 999회)   │
│  Binance Futures ◀─── 펀딩/롱숏/OI     │
│  mempool.space ◀───── 네트워크 수수료   │
│  Supabase ◀────────── DB 저장/조회      │
│  Telegram Bot ◀────── 알림 전송         │
│  Cloudflare Tunnel ◀─ 대시보드 원격접속 │
│  Claude Code ◀─────── AI 판단 엔진      │
│                                         │
└─────────────────────────────────────────┘
```

| 서비스 | 인증 | 제한 |
|--------|------|------|
| Upbit | JWT (Access + Secret) | 초당 10회 |
| Alternative.me | 없음 (무료) | — |
| Tavily | API Key | 월 999회 |
| Binance | 없음 (공개 API) | — |
| mempool.space | 없음 | — |
| Supabase | Service Role Key | — |
| Telegram | Bot Token | — |
| Cloudflare | Tunnel credentials | — |
| Claude | CLI 인증 | — |

---

## 디렉토리 구조

```
blockchain/
├── strategy.md          ← 전략 (이것만 바꾸면 전략 변경)
├── CLAUDE.md            ← 프로젝트 지침
├── .env                 ← API 키 + 안전장치 (git 제외)
│
├── scripts/
│   ├── 📊 수집 (7개)     collect_*.py, capture_chart.py, get_portfolio.py
│   ├── 💰 실행 (2개)     execute_trade.py, short_term_trader.py
│   ├── 📱 알림 (2개)     notify_telegram.py, dashboard.py
│   └── 🔄 자동화 (5개)   cron_run.sh, run_analysis.sh, startup.sh,
│                         watchdog_remote.sh, save_decision.py
│
├── supabase/migrations/  ← DB 스키마 (12 테이블 + 3 뷰)
├── tests/                ← 613 테스트, 97% 커버리지
├── data/                 ← 차트 이미지, 스냅샷
├── logs/                 ← 실행 로그, Claude 응답
└── docs/                 ← 상세 문서
```
