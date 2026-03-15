# 🪙 Claude Coin Trading System

<p align="center">
  <img src="assets/logo.png" alt="Claude Coin Trading Bot Logo" width="280" />
</p>

<p align="center">
  <strong>코드 한 줄 안 짜고, 자연어로 전략을 쓰면 AI가 알아서 코인을 사고판다고? 네, 진짜입니다. 업비트에서 비트코인 스윙, 초단타 스캘핑, 바이낸스에서 김치프리미엄 차익거래까지 — 잠자는 동안 봇이 일하고, 나는 꿈에서 람보르기니를 탑니다. 돈 버는 것도 중요하지만, 낭만이야~ 🚗💨</strong>
</p>

<p align="center">
  <a href="#-목차">목차</a> •
  <a href="#1--설정-완벽-가이드-처음부터-끝까지">설정 가이드</a> •
  <a href="#2--비트코인-자동매매-봇-업비트">BTC 봇</a> •
  <a href="#3--김치랑-봇-kimchirang">김치랑 봇</a> •
  <a href="#4--분산-ml-훈련-시스템">분산 훈련</a> •
  <a href="#5--텔레그램-통합-통신">텔레그램</a>
</p>

---

## 📑 목차

1. [⚙️ 설정 완벽 가이드 (처음부터 끝까지)](#1--설정-완벽-가이드-처음부터-끝까지)
   - [1-1. 필수 프로그램 설치](#1-1-필수-프로그램-설치-git--python)
   - [1-2. 프로젝트 다운로드](#1-2-프로젝트-다운로드)
   - [1-3. 가상환경 및 패키지 설치](#1-3-가상환경-설정-및-패키지-설치)
   - [1-4. API 키 발급 가이드](#1-4-api-키-발급-가이드-6종)
   - [1-5. .env 파일 설정](#1-5-env-파일-설정)
   - [1-6. 데이터베이스 설정 (Supabase)](#1-6-데이터베이스-설정-supabase)
2. [🤖 비트코인 자동매매 봇 (업비트)](#2--비트코인-자동매매-봇-업비트)
   - [AI-RL 하이브리드 시스템 (V4)](#2-1-추천-ai-강화학습-하이브리드-시스템-v4-최신형)
   - [3계급 에이전트 전략](#2-2-3계급-에이전트-전략-시스템)
   - [초단타 스캘핑 봇](#2-3-초단타-스캘핑-봇)
3. [🌶️ 김치랑 봇 (Kimchirang)](#3--김치랑-봇-kimchirang)
   - [김치프리미엄 차익거래란?](#3-1-김치프리미엄-차익거래란)
   - [김치랑 실행 방법](#3-2-김치랑-실행-방법)
   - [RL 학습 (PPO + DQN)](#3-3-rl-학습-ppo--dqn-앙상블)
4. [🖥️ 분산 ML 훈련 시스템](#4--분산-ml-훈련-시스템)
   - [다중 컴퓨터 RL 훈련](#4-1-다중-컴퓨터-rl-훈련)
   - [워커 티어 시스템 (Owner / Coworker / Collaborator)](#4-2-워커-티어-시스템)
   - [워커 참여 방법](#4-3-워커-참여-방법)
   - [RLS 보안 모델](#4-4-rls-보안-모델)
5. [📱 텔레그램 통합 통신](#5--텔레그램-통합-통신)
   - [알림 시스템](#5-1-알림-시스템)
   - [멀티챗 터미널](#5-2-멀티챗-터미널-telegram_listenerpy)
   - [워커 관리 CLI](#5-3-워커-관리-cli-worker_adminpy)
6. [⏰ 자동화 스케줄링](#6--자동화-스케줄링)
7. [📊 대시보드](#7--대시보드)
8. [🏗️ 프로젝트 구조](#8--프로젝트-구조)
9. [🚫 면책 조항](#9--면책-조항)

---

## 시스템 진화 로드맵

```
📊 V1 (원작)  단순 지표 + AI 1회성 판단 → 매수/매도
      ↓
🤖 V2 (에이전트)  3계급 AI 스쿼드 + 감독관 자율 전략 전환 + DB 학습
      ↓
🧠 V3 (단타 + 스윙)  초단타 스캘핑 봇 추가 + 고래 추종 + 뉴스 반응
      ↓
🚀 V4 (RL 하이브리드)  강화학습(PPO+DQN) 자동 스케줄 학습
                        + 인공신경망이 스스로 최적 타이밍을 학습
                        + RAG 기반 과거 패턴 자동 진단
                        + 김치프리미엄 Delta-Neutral 차익거래 봇 (김치랑)
      ↓
🌐 V5 (현재 — 분산 훈련 + 팀 협업)
     ├─ 다중 컴퓨터 분산 RL 훈련 (Mac Mini / PC128 / PC36)
     ├─ Owner → Coworker → Collaborator 3계층 워커 시스템
     ├─ Supabase RLS 기반 권한 격리 (토큰 유출해도 안전)
     ├─ 텔레그램 앱 내 양방향 통신 + 멀티챗 터미널
     └─ HMM 시장 레짐 탐지 + 롤링 백테스트 자동 최적화
```

> 🧬 **V5 핵심 진화**: 혼자 돌리던 봇이 **팀 기반 분산 훈련 플랫폼**으로 진화. 여러 컴퓨터가 각자 다른 모델을 훈련하고, 결과를 Supabase에 공유하며, 텔레그램으로 실시간 소통. 보안은 DB 수준의 RLS(Row Level Security)로 — 워커 A는 워커 B의 데이터를 절대 볼 수 없습니다.

### 🎓 원작자 유튜브 & 운영 환경 안내

이 프로젝트의 원작자 **dantelabs**님의 유튜브 채널에서 전체 구축 과정을 영상으로 확인할 수 있습니다:

> 📺 **[Dante Labs YouTube](https://youtube.com/@dante-labs)** — 원작 튜토리얼 영상

| 환경 | 특징 | 원격 제어 | 추천 대상 |
|:----:|------|:---------:|----------|
| 🪟 **Windows (PowerShell)** | 가장 간단한 설정. 별도 설치 없이 바로 시작 | ❌ 원격 불필요 시 | 내 PC에서만 돌릴 분 |
| 🐧 **WSL (Ubuntu on Windows)** | 원작 유튜브 방식. Linux 명령어 + tmux + SSH 원격 가능 | ✅ SSH 원격 가능 | 24시간 서버 운영 + 원격 제어 원하는 분 |
| 🍎 **Mac / Linux** | 터미널 기반. SSH 원격이 기본 내장되어 가장 쉽게 원격 가능 | ✅ 원격 매우 쉬움 | Mac 사용자 / 리눅스 서버 운영자 |

> 💡 **결론**: 원격 제어가 필요 없으면 **Windows에서 그냥 PowerShell**로 충분합니다. 24시간 자동화 + 원격이 필요하면 **Mac** 또는 **WSL(Ubuntu)** 환경을 권장합니다.
>
> 🔑 **참고**: [Claude Code](https://claude.ai) **Pro 플랜 이상** 구독자는 운영체제에 상관없이 **모든 환경에서 원격 제어(Remote Control)**가 지원됩니다.

---

## 1. ⚙️ 설정 완벽 가이드 (처음부터 끝까지)

> 💰 **권장 시작 시드머니**
> - **업비트**: 50만원 (비트코인 자동매매용)
> - **바이낸스**: 300 USDT (김치랑 차익거래용)
>
> 소액으로 충분히 시작할 수 있습니다. 처음에는 반드시 `DRY_RUN=true` (모의매매)로 1~2주 테스트 후 실전 전환하세요!

### 1-1. 필수 프로그램 설치 (Git + Python)

#### 🪟 Windows 사용자

**① Git 설치**
1. [https://git-scm.com/downloads](https://git-scm.com/downloads) 접속
2. **"Download for Windows"** 버튼 클릭하여 설치 파일 다운로드
3. 다운받은 `.exe` 파일 실행 → **모든 옵션을 건드리지 말고** `Next` → ... → `Install` 클릭
4. 설치 확인: `git --version`

**② Python 설치**
1. [https://www.python.org/downloads/](https://www.python.org/downloads/) 접속
2. ⚠️ **반드시 맨 아래 `Add Python to PATH` 체크!** ⚠️
3. 설치 확인: `python --version`

#### 🍎 Mac 사용자

```bash
git --version          # 기본 내장. 안 되면: xcode-select --install
brew install python@3.12
python3 --version
```

---

### 1-2. 프로젝트 다운로드

```bash
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
cd claude-coin-trading-main
```

---

### 1-3. 가상환경 설정 및 패키지 설치

#### 🪟 Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

#### 🍎 Mac / Linux (터미널)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

---

### 1-4. API 키 발급 가이드 (6종)

| # | 서비스 | 용도 | 발급 경로 | 결과물 |
|:-:|--------|------|----------|--------|
| ① | **업비트** | 잔고 조회 + 코인 매매 | [upbit.com](https://upbit.com) → 마이페이지 → Open API 관리 | `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY` |
| ② | **바이낸스** | 김치랑 선물 숏 | [binance.com](https://www.binance.com) → API Management | `BINANCE_API_KEY`, `BINANCE_API_SECRET` |
| ③ | **수파베이스** | 매매 기록 DB | [supabase.com](https://supabase.com) → New Project → Settings → API | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` |
| ④ | **텔레그램** | 매매 알림 + 팀 통신 | `@BotFather` → `/newbot` + `@userinfobot` | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID` |
| ⑤ | **타빌리** | 뉴스 수집 | [tavily.com](https://tavily.com) → API Keys | `TAVILY_API_KEY` |
| ⑥ | **제미나이** | RAG 분석 | [aistudio.google.com](https://aistudio.google.com/app/apikey) | `GEMINI_API_KEY` |

> ⚠️ 업비트/바이낸스: **출금 권한은 절대 체크하지 마세요!**
> ⚠️ 수파베이스: `anon` 말고 **`service_role`** 키를 복사하세요.

---

### 1-5. .env 파일 설정

```bash
cp .env.example .env   # Mac
# Windows: .env.example 파일을 .env로 이름 변경
```

역할별 환경변수 템플릿이 준비되어 있습니다:

| 파일 | 대상 | 포함 키 |
|------|------|---------|
| `.env.owner` | 시스템 소유자 | 모든 API 키 + 거래소 + SERVICE_ROLE_KEY |
| `.env.coworker` | 공동 운영자 | ANON_KEY + WORKER_TOKEN + 선택적 TAVILY |
| `.env.collaborator` | 훈련 참여자 | ANON_KEY + WORKER_TOKEN (최소 권한) |

```ini
# ═══ 핵심 안전장치 ═══
DRY_RUN=true              # true: 모의매매 / false: 실매매
MAX_TRADE_AMOUNT=100000   # 1회 최대 매수 금액 (원)
MAX_DAILY_TRADES=3        # 하루 최대 매매 횟수
EMERGENCY_STOP=false      # true: 모든 매매 즉시 중지
```

> ⚠️ **최초 실행 시 반드시 `DRY_RUN=true`로 최소 2주 테스트!**

---

### 1-6. 데이터베이스 설정 (Supabase)

[Supabase 대시보드](https://supabase.com/dashboard) → SQL Editor에서 마이그레이션을 순서대로 실행:

| 순서 | 파일 | 내용 |
|:----:|------|------|
| 1 | `001_initial_schema.sql` | 기본 테이블 (decisions, portfolio 등) |
| 2 | `002_scalp_tables.sql` | 단타 매매 기록 |
| 3 | `004_agent_switches.sql` | 에이전트 전환 이력 + 학습 |
| 4 | `020_kimchirang_trades.sql` | 김치랑 거래 기록 |
| 5 | `024_scalp_ml_system.sql` | ML 학습 시스템 |
| 6 | `026_worker_rls.sql` | **워커 RLS 보안 정책** |
| 7 | `028_telegram_messages.sql` | 텔레그램 메시지 기록 |
| 8 | `029_telegram_contacts.sql` | 텔레그램 연락처 |

> 에러 나는 파일은 건너뛰어도 기본 동작에 문제 없습니다.

---

## 2. 🤖 비트코인 자동매매 봇 (업비트)

### 2-1. [추천] AI-강화학습 하이브리드 시스템 (V4 최신형)

Gemini 2.5 Pro의 **LLM 정성 분석** + PPO 모델의 **강화학습(RL) 정량 분석** + 3명의 **AI 스쿼드 규칙**이 결합된 **만장일치 의사결정 시스템**입니다.

```
🪟 Windows:  python rl_hybrid\launchers\start_all.py
🍎 Mac:      python3 rl_hybrid/launchers/start_all.py
```

### 2-2. 3계급 에이전트 전략 시스템

시장 상황에 따라 감독(Orchestrator)이 자동으로 세 전략을 전환합니다:

| 에이전트 | 스타일 | 매수 임계점 | 목표 수익 | 손절선 |
|:--------:|--------|:----------:|:---------:|:------:|
| 🛡️ **보수적** | 폭락장 저점 매수 | 60점 | +15% | -5% |
| ⚖️ **보통** | 조정장 균형 매매 | 50점 | +10% | -5% |
| 🔥 **공격적** | 달리는 말에 단타 | 40점 | +7% | -3% |

**점수 평가**: FGI, RSI(14), SMA(20), 뉴스 감성, Data Fusion, 바이낸스 롱숏비율, 김치프리미엄, 매크로 경제지표 등 **10가지 이상** 실시간 분석

**핵심 기능:**
- ⚔️ **포지션 과다 분할 매도**: BTC 비중 50% 초과 시 자동 매도
- 💧 **하이브리드 DCA**: 캐스케이드 위험도 실시간 계산, 진성 하락장엔 손절
- 🧠 **감독 DB 학습**: 과거 전략 전환 성과를 학습하여 판단 지속 개선
- 🚨 **자동 긴급정지**: 4시간 -10% 급락 시 전량 매도 + 매수 차단

### 2-3. 초단타 스캘핑 봇

1~2분 단위로 시장을 감시하며 소폭 변동을 먹고 빠지는 **속도전 봇**입니다.

```
🪟 Windows:  python scripts\short_term_trader.py --dry-run
🍎 Mac:      python3 scripts/short_term_trader.py --dry-run
```

**3가지 전략 동시 운영:**

| 전략 | 감지 방법 | 매매 기준 |
|------|----------|----------|
| 📰 **뉴스 반응** | RSS 실시간 감성 스캔 (3분 간격) | 강한 감성 ±0.4 이상 |
| 📈 **급등/급락** | 5분 내 0.8%+ 급변동 감지 | 모멘텀 +0.02% 확인 |
| 🐋 **고래 추종** | 5000만원+ 대량 체결 감지 | 같은 방향 75%+ 편향 |

**v5 안전장치**: 트레일링 스탑 (+0.15% 활성, -0.10% 거리), 모멘텀 확인 진입, 진입 보호 기간 180초, 하락 추세 전체 매수 차단, DRY_RUN 전용 일일 40회 한도

---

## 3. 🌶️ 김치랑 봇 (Kimchirang)

### 3-1. 김치프리미엄 차익거래란?

```
📈 김치프리미엄 3% 이상 → 진입!
   ├─ 업비트: BTC 현물 매수 (비싼 쪽)
   └─ 바이낸스: BTCUSDT 선물 숏 (헤지)

📉 김치프리미엄 0.5% 이하 → 청산!
   ├─ 업비트: BTC 현물 매도
   └─ 바이낸스: 선물 숏 커버

💰 수익 = 진입KP(3%) - 청산KP(0.5%) - 수수료(~0.18%) ≈ 2.3%
```

**핵심**: 양쪽 동시 포지션 → BTC 가격 방향 무관, 김프 수축에서만 수익 (Delta-Neutral)

### 3-2. 김치랑 실행 방법

```
🪟 Windows:  python -m kimchirang.main
🍎 Mac:      python3 -m kimchirang.main
```

### 3-3. RL 학습 (PPO + DQN 앙상블)

```bash
python -m kimchirang.train_simple --steps 300000 --days 730   # PPO
python -m kimchirang.train_dqn --steps 500000 --days 730      # DQN
```

| 모델 | 특징 | 장점 |
|------|------|------|
| **PPO** | 정책 그래디언트 | 안정적 학습, 연속적 정책 개선 |
| **DQN** | Q-value 직접 계산 | Experience Replay로 희귀 이벤트 반복 학습 |
| **앙상블** | 두 모델 합의 | 한쪽만 신호 → Hold (신중하게) |

---

## 4. 🖥️ 분산 ML 훈련 시스템

### 4-1. 다중 컴퓨터 RL 훈련

여러 컴퓨터가 **각자 다른 모델/전략을 동시에 훈련**하고, 결과를 Supabase DB에 공유합니다. 각 머신은 자신의 강점에 맞는 역할을 수행합니다.

```
┌─────────────────────────────────────────────────────┐
│                   Supabase (중앙 DB)                 │
│  scalp_training_tasks · scalp_model_versions         │
│  worker_heartbeats · compute_workers                 │
└──────────┬──────────────┬──────────────┬─────────────┘
           │              │              │
     ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
     │  Mac Mini  │ │   PC128   │ │   PC36    │
     │ (Apple Si) │ │ (GPU/CPU) │ │  (DRJAY)  │
     ├───────────┤ ├───────────┤ ├───────────┤
     │ 수집기 1m  │ │ DQN 200K  │ │ PPO 200K  │
     │ 백테스트   │ │ SAC 500K  │ │ 보상함수   │
     │ HMM 레짐   │ │ Arch 실험 │ │ 레짐별 PPO │
     │ LGB 필터   │ │ 1M 장훈련 │ │ 온라인학습 │
     └───────────┘ └───────────┘ └───────────┘
```

**실행:**

```bash
# 각 컴퓨터에서 자신의 역할로 실행
python -m scalp_ml.distributed_training --machine mac-mini
python -m scalp_ml.distributed_training --machine pc128
python -m scalp_ml.distributed_training --machine pc36
```

**머신별 역할:**

| 머신 | 역할 | 모델 | 기간 |
|------|------|------|------|
| **Mac Mini** | 데이터 수집 + 백테스트 + 레짐 탐지 | LightGBM, XGBoost, HMM | 7일 (3데몬 24/7) |
| **PC128** | 신경망 청산 최적화 (대규모) | DQN, SAC | 7일 (200K→1M 스텝) |
| **PC36** | 정책 최적화 + 보상함수 실험 | PPO | 7일 (200K→500K 스텝) |

**Mac Mini 3-데몬 구조:**

| 데몬 | 주기 | 산출물 |
|------|------|--------|
| 🔬 **수집기** | 1분 24/7 | 호가잔량 + 체결강도 + 1분봉 → parquet |
| 📊 **백테스터** | 6시간 | 8,640개 파라미터 조합 그리드서치 → `optimal_params.json` |
| 🎯 **레짐** | 3시간 | HMM 3/4-state 분류 + 레짐별 LightGBM 필터 → `regime_detector.pkl` |

**PC128 7-Phase:**

| Phase | 내용 | 스텝 |
|-------|------|------|
| DQN 기본 | [128,64] 네트워크 | 200K |
| 보상함수 실험 | v1~v3 비교 | 각 100K |
| SAC 연속행동 | [256,128] 네트워크 | 500K |
| 아키텍처 실험 | [64,32] ~ [256,256,128] | 각 100K |
| 장기 훈련 | 최적 모델 집중 | 1M |

**훈련 결과 공유:**
- 모든 phase 결과는 `scalp_training_tasks` 테이블에 자동 기록
- 모델 메트릭스(승률, PnL, Sharpe)는 `scalp_model_versions`에 등록
- 텔레그램으로 Phase별 완료/실패 알림

---

### 4-2. 워커 티어 시스템

분산 훈련에 참여하는 사람/머신마다 **역할(티어)**이 부여됩니다. DB 수준의 RLS(Row Level Security)로 권한이 격리됩니다.

```
👑 Owner (소유자)
├─ 모든 API 키 보유 (거래소 + SERVICE_ROLE_KEY)
├─ 실매매 실행 가능
├─ 모든 데이터 읽기/쓰기
├─ 워커 초대/정지/승격
└─ 텔레그램 전체 발송

🤝 Coworker (공동 운영자)
├─ ANON_KEY + 개인 WORKER_TOKEN
├─ 훈련 + 분석 가능
├─ 자신의 데이터만 읽기/쓰기 (RLS)
├─ 매매 실행 불가
└─ 선택적 TAVILY 뉴스 수집

🔧 Collaborator (훈련 참여자)
├─ ANON_KEY + 개인 WORKER_TOKEN
├─ 훈련만 가능 (분석/매매 불가)
├─ 자신의 데이터만 읽기/쓰기 (RLS)
└─ 최소 권한 원칙
```

**권한 매트릭스:**

| 기능 | Owner | Coworker | Collaborator |
|------|:-----:|:--------:|:------------:|
| 모델 훈련 | ✅ | ✅ | ✅ |
| 시그널 분석 | ✅ | ✅ | ❌ |
| 백테스트 | ✅ | ✅ | ❌ |
| 파라미터 스윕 | ✅ | ✅ | ❌ |
| 모델 배포 | ✅ | ❌ | ❌ |
| 실매매 실행 | ✅ | ❌ | ❌ |
| 워커 관리 | ✅ | ❌ | ❌ |
| 전체 데이터 조회 | ✅ | ❌ | ❌ |

---

### 4-3. 워커 참여 방법

#### 방법 A: 초대코드 (제로 지식 — 관리자가 토큰을 모름)

```bash
# 1. Owner가 초대코드 생성
python scripts/worker_admin.py invite

# 2. 초대코드를 참여자에게 전달 (이메일/텔레그램)
# 3. 참여자가 초대코드로 자가 등록 → 토큰 자동 발급
```

#### 방법 B: 직접 발송 (관리자가 토큰 생성 후 전달)

```bash
# Owner가 토큰 생성 + 즉시 텔레그램/이메일 발송
python scripts/worker_admin.py direct-send
```

#### 워커 실행

```bash
# .env.coworker 또는 .env.collaborator 복사 후 키 입력
cp .env.coworker .env

# ML 워커 데몬 시작
python -m scalp_ml.worker --worker-id "my-worker" --tier coworker
```

워커는 Supabase의 `scalp_training_tasks`를 폴링하며, 자신에게 할당된(또는 미할당 pending) 작업을 claim하여 실행합니다.

---

### 4-4. RLS 보안 모델

DB 수준에서 행(row) 단위 접근 제어를 적용합니다. **워커 토큰이 유출되어도** 다른 워커의 데이터에 접근할 수 없습니다.

```
┌─────────────────────────────────────────┐
│            Supabase PostgreSQL           │
│                                         │
│  Owner (SERVICE_ROLE_KEY)               │
│  → RLS 바이패스, 모든 테이블 풀 액세스   │
│                                         │
│  Worker (ANON_KEY + x-worker-token)     │
│  → RLS 적용:                            │
│    compute_workers: 자신 행만 수정       │
│    worker_heartbeats: 자신 것만 R/W     │
│    scalp_training_tasks:                │
│      - pending 작업 claim 가능           │
│      - 자신에게 할당된 작업만 update     │
│    scalp_model_versions: 읽기만          │
└─────────────────────────────────────────┘
```

**핵심 DB 함수:**
- `get_worker_tier(token)`: 토큰으로 티어 조회
- `get_worker_id(token)`: 토큰으로 워커 ID 조회
- `is_service_role()`: Owner 여부 판정

---

## 5. 📱 텔레그램 통합 통신

### 5-1. 알림 시스템

봇의 모든 매매 결과가 텔레그램으로 자동 발송됩니다:

| 알림 종류 | 내용 |
|----------|------|
| 📊 **매매 실행** | 매수/매도 결정, 금액, 근거, 포트폴리오 변동 |
| 🚨 **긴급 알림** | 연속 손절, 급락, 긴급정지 발동 |
| 📈 **일일 요약** | 거래 횟수, 수익률, 포트폴리오 현황 |
| 🌶️ **김치랑** | 진입/청산, KP 상태 5분 주기 보고 |
| 🖥️ **분산 훈련** | Phase별 완료/실패, 일일 보고, 최종 결과 |
| 👥 **워커 관리** | 초대코드 발송, 워커 등록/정지 알림 |

---

### 5-2. 멀티챗 터미널 (telegram_listener.py)

**한 화면에서 모든 워커/친구와 동시 양방향 통신**하는 텔레그램 터미널입니다.

```bash
python scripts/telegram_listener.py
```

```
┌──────────────────────────────────────┐
│  텔레그램 멀티챗 터미널               │
│                                      │
│  [09:30] 장세훈아들 > 훈련 완료됐어   │
│  [09:31] 나 > 결과 보내줘            │
│  [09:32] Worker-PC128 > Phase 3 완료  │
│                                      │
│  입력: 이름>메시지                     │
└──────────────────────────────────────┘
```

**입력 방식:**

| 형식 | 설명 | 예시 |
|------|------|------|
| `이름>메시지` | 특정 사람에게 전송 | `장세훈아들>훈련 어때?` |
| `이름:메시지` | 동일 (콜론도 가능) | `장세훈아들:결과 보내줘` |
| `/to 이름` | 대화 상대 고정 | `/to 장세훈아들` → 이후 메시지 자동 전송 |
| `/all 메시지` | 전체 공지 | `/all 오늘 9시 동기화합니다` |
| `/list` | 연락처 목록 | 등록된 모든 연락처 표시 |
| `/chat_id` | 내 Chat ID 확인 | 신규 등록 시 사용 |

**연락처 관리:**

`telegram_contacts` 테이블에서 관리합니다:

| 필드 | 설명 |
|------|------|
| `chat_id` | 텔레그램 숫자 ID |
| `name` | 표시 이름 |
| `role` | owner / coworker / collaborator / viewer / friend |
| `worker_id` | compute_workers 테이블 연동 (워커인 경우) |
| `is_active` | 활성 여부 |

---

### 5-3. 워커 관리 CLI (worker_admin.py)

Owner 전용 워커 관리 도구입니다.

```bash
python scripts/worker_admin.py <명령어>
```

| 명령어 | 설명 |
|--------|------|
| `list` | 전체 워커 목록 (상태, 티어, 마지막 heartbeat) |
| `invite` | 초대코드 생성 → 이메일/텔레그램 전달 |
| `direct-send` | 토큰 생성 + 즉시 발송 |
| `msg <이름> <메시지>` | 특정 워커에게 텔레그램 메시지 |
| `msg-all <메시지>` | 모든 워커에게 공지 |
| `reply <이름>` | 실시간 1:1 대화 |
| `chat` | 멀티챗 모드 진입 |
| `suspend <이름>` | 워커 정지 |
| `unsuspend <이름>` | 워커 정지 해제 |
| `promote <이름> <티어>` | 티어 변경 (owner/coworker/collaborator) |

---

## 6. ⏰ 자동화 스케줄링

#### 🪟 Windows (작업 스케줄러)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_all_schedules.ps1
```

#### 🍎 Mac / Linux (cron)

```bash
bash scripts/setup_cron.sh install    # 4시간 크론탭 등록
bash scripts/setup_cron.sh status     # 확인
bash scripts/setup_cron.sh uninstall  # 제거
```

---

## 7. 📊 대시보드

```
🪟 Windows:  python scripts\web_server.py
🍎 Mac:      python3 scripts/web_server.py

→ 브라우저에서 http://localhost:5000 접속
```

대시보드 기능: 실시간 포트폴리오, 매매 내역, 에이전트 전환 이력, 고래 감지, 전략별 성과

---

## 8. 🏗️ 프로젝트 구조

```
claude-coin-trading-main/
│
├── 📄 README.md              ← 이 파일
├── 📄 CLAUDE.md              ← AI 프로젝트 지침서
├── 📄 strategy.md            ← 자연어 매매 전략
├── 📄 .env                   ← API 키 (git 추적 제외)
├── 📄 .env.owner             ← Owner 환경변수 템플릿
├── 📄 .env.coworker          ← Coworker 환경변수 템플릿
├── 📄 .env.collaborator      ← Collaborator 환경변수 템플릿
│
├── 🤖 agents/                ← 비트코인 자동매매 에이전트
│   ├── orchestrator.py       ← 감독 에이전트 (전략 자율 전환 + DB 학습)
│   ├── conservative.py       ← 🛡️ 보수적 에이전트
│   ├── moderate.py           ← ⚖️ 보통 에이전트
│   ├── aggressive.py         ← 🔥 공격적 에이전트
│   ├── base_agent.py         ← 기본 클래스 (점수제 매수, 하이브리드 손절)
│   └── external_data.py      ← 외부 데이터 병렬 수집
│
├── 🌶️ kimchirang/            ← 김치프리미엄 차익거래 봇
│   ├── main.py               ← 메인 + RL 브릿지
│   ├── kp_engine.py          ← 실시간 KP 계산
│   ├── execution.py          ← Delta-Neutral 동시 주문
│   ├── train_simple.py       ← PPO 학습
│   └── train_dqn.py          ← DQN 학습
│
├── 🧠 scalp_ml/              ← ML 훈련 시스템 (분산 훈련)
│   ├── distributed_training.py ← 3대 분산 1주 훈련 (Mac Mini / PC128 / PC36)
│   ├── worker.py             ← ML 워커 데몬 (Supabase 폴링)
│   ├── scalp_exit_env.py     ← RL 청산 환경 (Gymnasium)
│   ├── train_exit_dqn.py     ← DQN 청산 훈련
│   ├── train_lgbm.py         ← LightGBM 시그널 분류기
│   ├── feature_engineer.py   ← 피처 엔지니어링 (21개 피처)
│   └── analyzer.py           ← 시그널 품질 분석
│
├── 📜 scripts/               ← 실행 스크립트 모음
│   ├── short_term_trader.py  ← 초단타 스캘핑 봇
│   ├── telegram_listener.py  ← 텔레그램 멀티챗 터미널
│   ├── worker_admin.py       ← 워커 관리 CLI (초대/정지/승격)
│   ├── collect_*.py          ← 데이터 수집 (시세/뉴스/FGI/차트)
│   ├── execute_trade.py      ← 매매 실행 (안전장치 내장)
│   └── run_agents.sh         ← 에이전트 모드 파이프라인
│
├── 🚀 rl_hybrid/             ← AI-RL 하이브리드 시스템
│   ├── launchers/            ← 통합 실행기
│   ├── nodes/                ← 분산 노드
│   ├── rl/                   ← 강화학습 모듈
│   └── rag/                  ← RAG (유사 패턴 검색)
│
├── 🗂️ supabase/              ← 데이터베이스
│   └── migrations/           ← SQL 마이그레이션 (30개 파일)
│       ├── 001~009           ← 기본 스키마
│       ├── 020~024           ← 김치랑 + ML 시스템
│       └── 026~030           ← 워커 RLS + 텔레그램 통신
│
├── 🌐 web/                   ← 웹 프론트엔드
├── 📊 data/                  ← 데이터 (차트/모델/스냅샷)
├── 📝 logs/                  ← 실행 로그
└── 📚 docs/                  ← 상세 문서
```

---

## 9. 🚫 면책 조항

이 프로그램은 오픈소스로 제공되는 **개인 연구 및 교육용** 파이썬 자동화 툴입니다.

시스템의 설계 결함, 서버 다운, 가격 오류, API 장애, 거래소 점검, 네트워크 지연 및 판단 미스로 발생하는 **모든 금전적 혹은 암호화폐 투자 손실의 법적·도의적 책임은 전적으로 다운로드하여 사용한 사용자 본인에게 있습니다.**

> ⚠️ **실제 자산을 투입하기 전에 반드시:**
> 1. `DRY_RUN=true` 모드로 **최소 2주 이상** 모의 테스트
> 2. 성능 검증 후 **1~5만원 소액**으로 실전 테스트
> 3. 안정성 확인 후 시드머니 점진적 확대
>
> **투자는 본인의 판단과 책임 하에 이루어져야 합니다.**

---

<p align="center">
  <sub>이 앱은 경제적 이익의 목적보다, <b>AI의 탐구</b>, <b>인공신경망과 강화학습의 본질 연구</b>,<br/>그리고 <b>소소한 창작의 즐거움</b>을 위해 만들어졌습니다.<br/>이를 사용하면서 발생한 경제적 손실에 대해 프로듀서 및 원작자는 책임지지 않습니다.</sub>
</p>

<p align="center">
  <sub>made by <a href="https://github.com/dandacompany"><b>dantelabs</b></a> (<a href="https://youtube.com/@dante-labs">YouTube</a>), reproduced & reprinted by <b>Dr. Jang Jaeho</b>.</sub>
</p>
