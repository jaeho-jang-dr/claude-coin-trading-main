# Claude Code Crypto Trading Bot

> **코드가 아닌 자연어로 전략을 쓰면, Claude가 읽고 판단하고 매매한다.**

### Based on [dantelabs](https://github.com/dantelabs)

이 프로젝트는 [dantelabs](https://github.com/dantelabs)님의 원본 앱을 **macOS 환경에 맞게 포팅**하면서 대폭 확장한 버전입니다.

**주요 변경점:**

| 항목 | 원본 (dantelabs) | 이 버전 (확장) |
|------|-----------------|---------------|
| 플랫폼 | Linux/범용 | **macOS** (LaunchAgent 자동화) |
| 단타 전략 | 보수적 1종 | **보수적 + 보통 + 공격적** 3단계 |
| 단타 실행 | 하루 3회 (8시간 간격) | **하루 6회** (4시간 간격) |
| 초단타 | 없음 | **하루 6시간** 실시간 봇 추가 (뉴스/급변동/고래 3전략) |
| 대시보드 | — | Flask + Cloudflare Tunnel 원격 접속 |
| DB | — | Supabase 12 테이블 + 3 뷰 |
| 테스트 | — | 613 tests, 97% coverage |

```
             strategy.md (자연어 전략)
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 시장 데이터     과거 성과      사용자 피드백
    │               │               │
    └───────────────┼───────────────┘
                    ▼
            ┌──────────────┐
            │  Claude -p   │  ← AI가 분석 + 판단
            │ 매수/매도/관망│
            └──────┬───────┘
                   ▼
         매매 → DB기록 → 텔레그램 알림
```

---

## 구조 한눈에 보기

```
blockchain/
├── strategy.md               ← 매매 전략 (이것만 바꾸면 전략이 바뀐다)
├── CLAUDE.md                 ← Claude 프로젝트 지침
├── .env                      ← API 키 + 안전장치
│
├── scripts/
│   ├── 📊 데이터 수집 (7개)    → docs/scripts.md#데이터-수집
│   ├── 💰 매매 실행 (2개)      → docs/scripts.md#매매-실행
│   ├── 📱 알림/대시보드 (2개)  → docs/scripts.md#알림
│   └── 🔄 자동화 (4개)        → docs/scripts.md#자동화
│
├── supabase/                 ← DB 스키마 (12 테이블 + 3 뷰)
├── tests/                    ← 613 테스트, 97% 커버리지
└── docs/                     ← 상세 문서
```

> **상세 문서:**
> [스크립트](docs/scripts.md) · [DB 스키마](docs/database.md) · [설정 가이드](docs/setup.md) · [전략 작성법](docs/strategy-guide.md) · [아키텍처](docs/architecture.md)

---

## 핵심 3줄 요약

1. **`strategy.md`에 전략을 자연어로 쓴다** — "RSI 25 이하 + 공포탐욕 30 이하이면 매수"
2. **cron이 4시간마다 실행** — 데이터 수집 → Claude 분석 → 매매 → DB 기록 → 텔레그램 알림
3. **피드백 루프** — 과거 결정의 성과를 다음 분석에 반영하여 계속 개선

---

## 실행 모드

| 모드 | 명령어 | 빈도 | 설명 |
|------|--------|------|------|
| **자동 매매** | `bash scripts/setup_cron.sh install` | 4시간 간격 | 데이터 수집 → Claude 분석 → 매매 |
| **초단타 봇** | `python3 scripts/short_term_trader.py` | 실시간 | 뉴스/급변동/고래 3전략 WebSocket |
| **대화형** | `claude` | 수시 | 전략 수정, 피드백, 수동 분석 |
| **대시보드** | `https://dashboard.wwwmoksu.com` | 상시 | 포트폴리오/매매내역/긴급정지 |

---

## 데이터 파이프라인

```
  Upbit        Alternative.me    Tavily       Binance      mempool.space
  시세/호가       공포탐욕지수      뉴스        펀딩레이트       수수료
  RSI/MACD       7일 추이       5카테고리     롱숏/OI        네트워크
    │               │              │            │              │
    └───────────────┼──────────────┼────────────┼──────────────┘
                    ▼
            run_analysis.sh
         (프롬프트 조립 → claude -p)
```

> 상세: [데이터 수집 스크립트](docs/scripts.md#데이터-수집)

---

## 안전장치

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| **`DRY_RUN`** | **`true`** | **분석만 수행, 실제 매매 안 함** |
| `EMERGENCY_STOP` | `false` | 모든 매매 즉시 중지 |
| `MAX_TRADE_AMOUNT` | 100,000원 | 1회 매매 상한 |
| `MAX_DAILY_TRADES` | 6회 | 일일 매매 상한 |
| `MIN_TRADE_INTERVAL_HOURS` | 4시간 | 매매 간 최소 간격 |
| `MAX_POSITION_RATIO` | 50% | 총 자산 대비 투자 비율 |

> **반드시 `DRY_RUN=true`로 충분히 테스트 후 실전 전환하세요.**

---

## 빠른 시작

```bash
# 1. 클론 & 환경 설정
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
cd claude-coin-trading-main
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. API 키 설정
cp .env.example .env     # .env 파일에 API 키 입력

# 3. 테스트 (DRY_RUN=true 상태)
python3 scripts/collect_market_data.py    # 시장 데이터 확인
python3 scripts/get_portfolio.py          # 포트폴리오 확인

# 4. cron 자동 매매 등록
bash scripts/setup_cron.sh install        # 4h/8h/12h/24h 선택
```

> 상세: [설정 가이드](docs/setup.md)

---

## 스크립트 한눈에 보기

| 구분 | 스크립트 | 한줄 설명 |
|------|---------|----------|
| 📊 | `collect_market_data.py` | Upbit 시세 + RSI, MACD, 볼린저밴드 |
| 📊 | `collect_fear_greed.py` | 공포탐욕지수 + 7일 추이 |
| 📊 | `collect_news.py` | BTC 뉴스 5카테고리 (Tavily) |
| 📊 | `collect_ai_signal.py` | 호가불균형/고래/변동성 등 6가지 |
| 📊 | `collect_onchain_data.py` | 펀딩레이트, 롱숏비율, OI |
| 📊 | `capture_chart.py` | 차트 스크린샷 (Playwright) |
| 📊 | `get_portfolio.py` | 잔고, 보유 코인, 수익률 |
| 💰 | `execute_trade.py` | Upbit 시장가 매매 (안전장치 내장) |
| 💰 | `short_term_trader.py` | 실시간 초단타 봇 (3전략) |
| 📱 | `notify_telegram.py` | 텔레그램 알림 |
| 📱 | `dashboard.py` | 웹 대시보드 (Flask) |
| 🔄 | `cron_run.sh` | cron 래퍼 (로깅 + 에러 알림) |
| 🔄 | `run_analysis.sh` | 데이터 수집 → 프롬프트 조립 |
| 🔄 | `startup.sh` | 부팅 시 자동 시작 |
| 🔄 | `watchdog_remote.sh` | Remote Control 감시 |

> 상세: [스크립트 문서](docs/scripts.md)

---

## 데이터베이스

```
┌─ 핵심 ────────────────────────┐  ┌─ 초단타 ───────────────────┐
│ decisions       매매 결정      │  │ scalp_trades    초단타 매매 │
│ portfolio_snapshots 포트폴리오 │  │ whale_detections 고래 감지  │
│ market_data     시장 데이터    │  │ news_sentiment_log 뉴스    │
│ feedback        사용자 피드백  │  │ strategy_alerts  전략 알림  │
│ execution_logs  실행 로그     │  │ scalp_sessions   세션 요약  │
│ strategy_history 전략 이력    │  │ ai_signal_log    AI 시그널  │
└───────────────────────────────┘  └────────────────────────────┘

┌─ 회고 ────────────────────────┐  ┌─ 뷰 ─────────────────────────┐
│ trade_reviews   매매 사후평가  │  │ v_trade_history     이력 종합 │
│                               │  │ v_performance_summary 성과   │
│                               │  │ v_monthly_performance 월별   │
└───────────────────────────────┘  └─────────────────────────────┘
```

> 상세: [DB 스키마 문서](docs/database.md)

---

## 24시간 스케줄

```
00:00 ─── 단타 매매 #1 ──────────────────────────
00:10 ─── 초단타 봇 (랜덤 6h, DRY_RUN) ─────────
04:00 ─── 단타 매매 #2 ──────────────────────────
08:00 ─── 단타 매매 #3 ──────────────────────────
12:00 ─── 단타 매매 #4 ──────────────────────────
16:00 ─── 단타 매매 #5 ──────────────────────────
20:00 ─── 단타 매매 #6 ──────────────────────────
 ∞    ─── Watchdog (3분) + Dashboard (상시) ─────
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| AI 엔진 | Claude Code (`claude -p` 비대화형) |
| 거래소 | Upbit REST API + WebSocket |
| DB | Supabase (PostgreSQL) |
| 알림 | Telegram Bot API |
| 차트 | Playwright (headless Chromium) |
| 대시보드 | Flask + Cloudflare Tunnel |
| 자동화 | macOS LaunchAgent |
| 테스트 | pytest — **613 tests, 97% coverage** |

---

## 교육 커리큘럼

빈 폴더에서 시작하여 Claude Code에 프롬프트를 하나씩 입력하면서 구축합니다.

| Step | 프롬프트 | 결과물 |
|------|---------|--------|
| 0 | "환경을 세팅해줘" | Python venv, .env |
| 1 | "시장 데이터를 수집해줘" | `collect_market_data.py` |
| 2 | "공포탐욕지수를 수집해줘" | `collect_fear_greed.py` |
| 3 | "BTC 뉴스를 수집해줘" | `collect_news.py` |
| 4 | "차트를 캡처해줘" | `capture_chart.py` |
| 5 | "포트폴리오를 조회해줘" | `get_portfolio.py` |
| 6 | "매매 전략을 작성해줘" | `strategy.md` |
| 7 | "매매 실행 스크립트를 만들어줘" | `execute_trade.py` |
| 8 | "텔레그램 알림을 만들어줘" | `notify_telegram.py` |
| 9 | "DB를 설계해줘" | Supabase 스키마 |
| 10 | "분석 파이프라인을 만들어줘" | `run_analysis.sh` |
| 11 | "시장을 분석해줘" | 첫 분석 실행 |
| 12 | "cron 자동화를 설정해줘" | `cron_run.sh` |

---

## 면책 조항

이 시스템은 **실제 자산을 거래**할 수 있습니다. 암호화폐 투자는 원금 손실의 위험이 있으며, 이 시스템 사용으로 발생하는 모든 손익에 대한 책임은 사용자 본인에게 있습니다.

---

<p align="center">Built with <a href="https://claude.ai/claude-code">Claude Code</a></p>
