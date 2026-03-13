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
  <a href="#4--텔레그램-알림-설정">텔레그램</a>
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
4. [📱 텔레그램 알림 설정](#4--텔레그램-알림-설정)
5. [⏰ 자동화 스케줄링](#5--자동화-스케줄링)
6. [📊 대시보드](#6--대시보드)
7. [🏗️ 프로젝트 구조](#7--프로젝트-구조)
8. [🚫 면책 조항](#8--면책-조항)

---

## 1. ⚙️ 설정 완벽 가이드 (처음부터 끝까지)

> 💰 **권장 시작 시드머니**
> - **업비트**: 50만원 (비트코인 자동매매용)
> - **바이낸스**: 300 USDT (김치랑 차익거래용)
>
> 소액으로 충분히 시작할 수 있습니다. 처음에는 반드시 `DRY_RUN=true` (모의매매)로 1~2주 테스트 후 실전 전환하세요!

### 🎓 원작자 유튜브 & 운영 환경 안내

이 프로젝트의 원작자 **dantelabs**님의 유튜브 채널에서 전체 구축 과정을 영상으로 확인할 수 있습니다:

> 📺 **[Dante Labs YouTube](https://youtube.com/@dante-labs)** — 원작 튜토리얼 영상

원작은 숫자 지표와 AI의 단순 판단에 의존하는 매매 봇이었습니다. 본 프로젝트에서는 여기에서 한 걸음 더 나아가, **완전히 다른 차원의 시스템**으로 재탄생시켰습니다:

```
📊 V1 (원작)  단순 지표 + AI 1회성 판단 → 매수/매도
      ↓
🤖 V2 (에이전트)  3계급 AI 스쿼드 + 감독관 자율 전략 전환 + DB 학습
      ↓
🧠 V3 (단타 + 스윙)  초단타 스캘핑 봇 추가 + 고래 추종 + 뉴스 반응
      ↓
🚀 V4 (현재)  강화학습(PPO+DQN) 자동 스케줄 학습
              + 인공신경망이 스스로 최적 타이밍을 점점 정교하게 학습
              + RAG 기반 과거 패턴 자동 진단
              + 김치프리미엄 Delta-Neutral 차익거래 봇 (김치랑)
```

> 🧬 **핵심 진화**: 사람이 정해준 규칙대로만 움직이던 봇이, 이제는 **인공신경망이 스스로 시장 패턴을 학습**하고, **자체적으로 실패를 진단·개선**하며, **매매 판단의 정교함이 시간이 갈수록 진화**하는 자율 성장형 AI 트레이더가 되었습니다.

원작 유튜브에서는 24시간 자동 운영과 원격 제어를 위해 **Windows에서 WSL(Ubuntu)**을 사용합니다. 하지만 본 프로젝트는 **세 가지 환경** 모두에서 설정·운영이 가능합니다:

| 환경 | 특징 | 원격 제어 | 추천 대상 |
|:----:|------|:---------:|----------|
| 🪟 **Windows (PowerShell)** | 가장 간단한 설정. 별도 설치 없이 바로 시작 | ❌ 원격 불필요 시 | 내 PC에서만 돌릴 분 |
| 🐧 **WSL (Ubuntu on Windows)** | 원작 유튜브 방식. Linux 명령어 + tmux + SSH 원격 가능 | ✅ SSH 원격 가능 | 24시간 서버 운영 + 원격 제어 원하는 분 |
| 🍎 **Mac / Linux** | 터미널 기반. SSH 원격이 기본 내장되어 가장 쉽게 원격 가능 | ✅ 원격 매우 쉬움 | Mac 사용자 / 리눅스 서버 운영자 |

> 💡 **결론**: 원격 제어가 필요 없으면 **Windows에서 그냥 PowerShell**로 충분합니다. 24시간 자동화 + 원격이 필요하면 **Mac** 또는 **WSL(Ubuntu)** 환경을 권장합니다.
>
> 🔑 **참고**: [Claude Code](https://claude.ai) **Pro 플랜 이상** 구독자는 운영체제에 상관없이 **모든 환경에서 원격 제어(Remote Control)**가 지원됩니다. SSH 설정 없이도 Claude Code의 내장 원격 기능으로 어디서든 봇을 관리할 수 있습니다.

### 1-1. 필수 프로그램 설치 (Git + Python)

봇을 돌리려면 컴퓨터에 **Git**과 **Python** 두 가지가 설치되어 있어야 합니다.

#### 🪟 Windows 사용자

**① Git 설치**
1. [https://git-scm.com/downloads](https://git-scm.com/downloads) 접속
2. **"Download for Windows"** 버튼 클릭하여 설치 파일 다운로드
3. 다운받은 `.exe` 파일 실행 → **모든 옵션을 건드리지 말고** `Next` → `Next` → ... → `Install` 클릭
4. 설치 완료 후 **PowerShell** (또는 명령 프롬프트)을 열고 아래 명령어를 입력하여 확인:
   ```
   git --version
   ```
   `git version 2.xx.x` 같은 버전이 뜨면 성공! ✅

**② Python 설치**
1. [https://www.python.org/downloads/](https://www.python.org/downloads/) 접속
2. **"Download Python 3.1x.x"** 노란 버튼 클릭 (3.10 이상 필수!)
3. 설치 파일 실행 시 ⚠️ **반드시 맨 아래 `Add Python to PATH` 체크!** ⚠️ (이것을 안 하면 모든 게 안 됩니다)
4. `Install Now` 클릭하여 설치
5. 설치 확인:
   ```
   python --version
   ```
   `Python 3.1x.x`가 뜨면 성공! ✅

#### 🍎 Mac 사용자

**① Git 설치** — Mac에는 기본 내장되어 있습니다. 터미널을 열고 확인:
```bash
git --version
```
만약 안 된다면: `xcode-select --install` 실행

**② Python 설치**
```bash
# Homebrew로 설치 (Homebrew가 없으면 https://brew.sh 에서 먼저 설치)
brew install python@3.12
python3 --version
```

---

### 1-2. 프로젝트 다운로드

**PowerShell** (Windows) 또는 **터미널** (Mac)을 열고, 아래 명령어를 **한 줄씩** 복사하여 붙여넣고 엔터를 칩니다:

```bash
# 저장소 복제 (내 컴퓨터로 소스코드 다운로드)
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git

# 다운받은 폴더로 이동
cd claude-coin-trading-main
```

> 💡 **Tip**: 다운로드할 위치를 바꾸고 싶으면 `cd` 명령어로 원하는 폴더로 먼저 이동한 후 `git clone`을 실행하세요.

---

### 1-3. 가상환경 설정 및 패키지 설치

Python 가상환경은 이 프로젝트 전용 독립 공간입니다. 다른 프로그램과 충돌하지 않기 위해 반드시 만들어야 합니다.

#### 🪟 Windows (PowerShell)

```powershell
# 1. 가상환경 공간 만들기
python -m venv .venv

# 2. 가상환경 켜기 (진입)
.venv\Scripts\activate

# 3. 봇 작동에 필요한 패키지 다운로드
pip install -r requirements.txt

# 4. 브라우저 자동화 도구 설치 (차트 캡처용, 약 150MB)
playwright install chromium
```

> ⚠️ **PowerShell 보안 에러 발생 시**: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` 를 먼저 실행한 후 다시 시도하세요.

#### 🍎 Mac / Linux (터미널)

```bash
# 1. 가상환경 공간 만들기
python3 -m venv .venv

# 2. 가상환경 켜기 (진입)
source .venv/bin/activate

# 3. 봇 작동에 필요한 패키지 다운로드
pip install -r requirements.txt

# 4. 브라우저 자동화 도구 설치 (차트 캡처용, 약 150MB)
playwright install chromium
```

> 💡 **가상환경 상태 확인**: 터미널 프롬프트 앞에 `(.venv)` 가 보이면 가상환경 안에 있는 것입니다.
> - 끄기: `deactivate`
> - 다시 켜기: `.venv\Scripts\activate` (Windows) / `source .venv/bin/activate` (Mac)

---

### 1-4. API 키 발급 가이드 (6종)

프로그램을 실행하려면 아래 **6가지 API 키**가 필요합니다. 각 사이트에 가입하여 키를 발급받고, **메모장에 따로 적어두세요!**

#### ① 📈 업비트 (Upbit) API 키

| 항목 | 내용 |
|------|------|
| **용도** | 잔고 조회, 실시간 시세 조회 및 실제 코인 매수/매도 실행 |
| **가입** | [https://upbit.com](https://upbit.com) 회원가입 + 본인인증 |
| **발급 경로** | 로그인 → 마이페이지 → **Open API 관리** → `API Key 발급` 클릭 |
| **필수 권한** | ✅ 자산조회, ✅ 주문조회, ✅ 주문하기 |
| **주의** | 🚫 **출금하기 권한은 절대 체크하지 마세요!** |
| **발급 결과물** | `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY` |

> 💡 **보안 팁**: 특정 IP에서만 접속하도록 설정하면 더 안전합니다. 내 IP는 네이버에 "내 IP 확인" 검색하면 나옵니다.

#### ② 🔄 바이낸스 (Binance) API 키 — 김치랑 봇용

| 항목 | 내용 |
|------|------|
| **용도** | 김치프리미엄 차익거래 시 바이낸스 선물(Futures) 숏 포지션 실행 |
| **가입** | [https://www.binance.com](https://www.binance.com) 회원가입 + KYC 인증 |
| **발급 경로** | 로그인 → 우측 상단 프로필 아이콘 → **API Management** → `Create API` 클릭 |
| **필수 권한** | ✅ Enable Reading, ✅ Enable Futures |
| **주의** | 🚫 **Enable Withdrawals는 절대 체크하지 마세요!** |
| **발급 결과물** | `BINANCE_API_KEY`, `BINANCE_API_SECRET` |

> ⚠️ **중요**: 바이낸스는 **선물(Futures) 계좌가 별도**입니다. 발급 후 바이낸스 앱에서 `현물 → USDⓈ-M 선물`로 USDT를 이체해야 합니다. (최소 300 USDT 권장)

#### ③ 🗄️ 수파베이스 (Supabase) API 키

| 항목 | 내용 |
|------|------|
| **용도** | 투자 내역, 봇의 판단 근거, 수익률 등을 자동 저장하는 무료 클라우드 DB |
| **가입** | [https://supabase.com](https://supabase.com) GitHub 또는 이메일로 가입 |
| **발급 경로** | 가입 → `New Project` 클릭 → 프로젝트 이름 입력 (예: `coin-trading`) → 비밀번호 설정 → 리전 선택 (`Northeast Asia (Tokyo)` 권장) → `Create new project` 클릭 |
| **키 확인 경로** | 프로젝트 대시보드 → 좌측 메뉴 → ⚙️ `Settings` → `API` 탭 이동 |
| **발급 결과물** | `SUPABASE_URL` (Project URL), `SUPABASE_SERVICE_ROLE_KEY` (service_role secret 키 — `anon` 말고 **`service_role`** 입니다!) |

> ⚠️ **주의**: `anon public` 키가 아닌 **`service_role secret`** 키를 복사하세요! 스크롤을 내려야 보입니다.

#### ④ 📱 텔레그램 (Telegram) 봇 토큰

| 항목 | 내용 |
|------|------|
| **용도** | 매수/매도 체결 알림, 목표가 도달 시 내 폰으로 즉시 메시지, 봇 원격 제어 |
| **봇 생성** | 텔레그램 앱 검색창에 `@BotFather` 검색 → 대화 시작 → `/newbot` 입력 → 봇 이름 지정 (예: `MyTradingBot`) → 나오는 `HTTP API Token` 복사 |
| **내 ID 확인** | 텔레그램 검색창에 `@userinfobot` 검색 → 대화 시작 → 나오는 `Id:` 번호 (예: `123456789`) 복사 |
| **마지막 단계** | ⚠️ 생성한 봇(**@내봇이름**)에게 먼저 `/start` 메시지를 한번 보내세요! (안 하면 알림이 안 옵니다) |
| **발급 결과물** | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID` |

#### ⑤ 📰 타빌리 (Tavily) API 키

| 항목 | 내용 |
|------|------|
| **용도** | 글로벌 뉴스를 실시간 검색하여 비트코인 호재/악재 판별 (AI 웹 검색 엔진) |
| **가입** | [https://tavily.com](https://tavily.com) 가입 → 대시보드 → `API Keys` 메뉴 |
| **발급** | `Create API Key` 클릭 (무료 티어: 월 1,000건 검색 가능) |
| **발급 결과물** | `TAVILY_API_KEY` |

#### ⑥ 🤖 제미나이 (Gemini) API 키

| 항목 | 내용 |
|------|------|
| **용도** | V4의 핵심 — RAG (과거 유사 패턴 분석) 및 딥러닝 시장 맥락 분석 (Gemini 2.5 Pro) |
| **가입** | [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) 접속 → 구글 로그인 |
| **발급** | `Create API Key` 클릭 → 복사 (무료 사용 가능) |
| **발급 결과물** | `GEMINI_API_KEY` |

---

### 1-5. .env 파일 설정

이제 봇이 내 거래소 지갑과 데이터베이스를 사용할 수 있도록 API 키를 파일에 적어야 합니다.

#### 🪟 Windows

1. 다운받은 `claude-coin-trading-main` 폴더를 **파일 탐색기**로 열기
2. `.env.example` 파일을 찾아 **마우스 우클릭** → **이름 바꾸기** → `.env` 로 변경
   - 경고 메시지가 나오면 `예` 클릭
   - 파일이 안 보인다면: 파일 탐색기 상단 → `보기` → `숨긴 항목` 체크
3. `.env` 파일을 **우클릭** → **연결 프로그램** → **메모장**으로 열기

#### 🍎 Mac

```bash
# 파일 복사 + 편집기 열기
cp .env.example .env
nano .env
```

#### ✅ .env 파일 내용 입력

메모장에서 각 항목의 `your_xxx` 부분을 **위에서 발급받은 실제 키**로 교체하고 저장합니다:

```ini
# ═══════════════════════════════════════════════════
# 🏦 거래소 API (업비트 — 비트코인 자동매매용)
# ═══════════════════════════════════════════════════
UPBIT_ACCESS_KEY=여기에_업비트_액세스키_붙여넣기
UPBIT_SECRET_KEY=여기에_업비트_시크릿키_붙여넣기

# ═══════════════════════════════════════════════════
# 🔄 거래소 API (바이낸스 — 김치랑 차익거래용)
# ═══════════════════════════════════════════════════
BINANCE_API_KEY=여기에_바이낸스_API키_붙여넣기
BINANCE_API_SECRET=여기에_바이낸스_시크릿키_붙여넣기

# ═══════════════════════════════════════════════════
# 🗄️ 데이터베이스 (수파베이스)
# ═══════════════════════════════════════════════════
SUPABASE_URL=https://내프로젝트.supabase.co
SUPABASE_SERVICE_ROLE_KEY=여기에_service_role_키_붙여넣기

# ═══════════════════════════════════════════════════
# 📱 텔레그램 알림
# ═══════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN=여기에_봇토큰_붙여넣기
TELEGRAM_USER_ID=여기에_내_숫자ID_붙여넣기

# ═══════════════════════════════════════════════════
# 📰 뉴스 수집 + 🤖 AI 분석
# ═══════════════════════════════════════════════════
TAVILY_API_KEY=여기에_타빌리_키_붙여넣기
GEMINI_API_KEY=여기에_제미나이_키_붙여넣기

# ═══════════════════════════════════════════════════
# 🔥 투자 핵심 안전장치 (매우 중요!)
# ═══════════════════════════════════════════════════
DRY_RUN=true              # ✅ true: 실제 매매 안 함 (모의 연습)
                          # 🚨 false: 실제 내 돈으로 매매 시작!
MAX_TRADE_AMOUNT=100000   # 1회 최대 매수 금액 (원) -- 10만원
MAX_DAILY_TRADES=3        # 하루 최대 매매 횟수
MAX_POSITION_RATIO=0.5    # 전체 자산 중 최대 투자 비율 (0.5 = 50%)
EMERGENCY_STOP=false      # 🆘 true로 바꾸면 모든 매매 즉시 중지!

# ═══════════════════════════════════════════════════
# 📊 단타 전용 설정 (short_term_trader.py)
# ═══════════════════════════════════════════════════
SHORT_TERM_BUDGET=500000       # 단타 전용 자금 (50만원)
SHORT_TERM_MAX_TRADE=200000    # 1회 최대 (20만원)
SHORT_TERM_MAX_DAILY=10        # 일일 최대 10회
SHORT_TERM_STOP_LOSS=0.8       # 손절 0.8%
SHORT_TERM_TAKE_PROFIT=1.5     # 익절 1.5%
SHORT_TERM_MAX_HOLD_MIN=30     # 최대 보유 30분
```

> ⚠️ **최초 실행 시 반드시 `DRY_RUN=true` 상태로 실행하세요!** 앱이 데이터를 잘 불러오는지 확인한 후 `false`로 바꿔야 합니다. 최소 1~2주 모의 매매 후 소액(1~5만원)으로 테스트하는 것을 **간절히** 권장합니다.

---

### 1-6. 데이터베이스 설정 (Supabase)

봇의 모든 매매 기록, 분석 결과, 수익률을 저장하려면 Supabase에 데이터베이스 테이블을 만들어야 합니다.

#### Step 1: SQL Editor 열기
1. [Supabase 대시보드](https://supabase.com/dashboard) 접속 → 내 프로젝트 클릭
2. 좌측 메뉴에서 **`SQL Editor`** (</> 아이콘) 클릭

#### Step 2: 마이그레이션 SQL 실행 (순서대로!)

프로젝트의 `supabase/migrations/` 폴더에 있는 SQL 파일들을 **번호 순서대로** 실행합니다.

> **방법**: 각 `.sql` 파일을 메모장으로 열고 → **내용 전체 복사** → Supabase SQL Editor에 **붙여넣기** → **`Run`(실행)** 클릭

필수 마이그레이션 (최소한 이것들은 실행해야 합니다):

| 순서 | 파일명 | 내용 |
|:----:|--------|------|
| 1 | `001_initial_schema.sql` | 기본 테이블 (decisions, portfolio 등) |
| 2 | `002_scalp_tables.sql` | 단타 매매 기록 테이블 |
| 3 | `004_agent_switches.sql` | 에이전트 전환 이력 + 학습 테이블 |
| 4 | `005_scalp_trade_log.sql` | 단타 상세 기록 |
| 5 | `009_cycle_id_and_views.sql` | 사이클 관리 + 대시보드 뷰 |
| 6 | `016_rag_analysis_vectors.sql` | V4 RAG 분석용 (하이브리드 시스템) |
| 7 | `020_kimchirang_trades.sql` | 김치랑 봇 거래 기록 |
| 8 | `021_kimchirang_extended.sql` | 김치랑 확장 테이블 |

> 💡 나머지 마이그레이션 파일들도 순서대로 실행하면 더 상세한 기록과 분석이 가능합니다. 에러가 나는 파일은 건너뛰어도 기본 동작에 문제 없습니다.

---

## 2. 🤖 비트코인 자동매매 봇 (업비트)

업비트에서 BTC/KRW를 자동으로 매매하는 본체 봇입니다. 3명의 AI 에이전트가 시장을 분석하고, 감독관(Orchestrator)이 최적의 전략을 자율적으로 선택합니다.

### 2-1. [추천] AI-강화학습 하이브리드 시스템 (V4 최신형)

Gemini 2.5 Pro의 **LLM 정성 분석** + PPO 모델의 **강화학습(RL) 정량 분석** + 3명의 **AI 스쿼드 규칙**이 결합된 **만장일치 의사결정 시스템**입니다.

```
🪟 Windows:  python rl_hybrid\launchers\start_all.py
🍎 Mac:      python3 rl_hybrid/launchers/start_all.py
```

실행하면 텔레그램 연동, 실시간 데이터 분석, 매매 의사결정이 백그라운드 멀티 프로세스로 동시에 구동됩니다.

### 2-2. 3계급 에이전트 전략 시스템

시장 상황에 따라 감독(Orchestrator)이 자동으로 세 전략을 전환합니다:

| 에이전트 | 스타일 | 매수 임계점 | 목표 수익 | 손절선 | 특성 |
|:--------:|--------|:----------:|:---------:|:------:|------|
| 🛡️ **보수적** | 폭락장 저점 매수 | 60점 | +15% | -5% | 자산 보전 최우선. 발동 빈도 낮지만 안전 |
| ⚖️ **보통** | 조정장 균형 매매 | 50점 | +10% | -5% | 추세 전환점 노리기. 균형잡힌 수익/리스크 |
| 🔥 **공격적** | 달리는 말에 단타 | 40점 | +7% | -3% | 빈번한 매매, 빠른 익절/손절 |

**점수 평가 항목**: FGI(공포탐욕지수), RSI(14), SMA(20) 이탈도, 뉴스 감성, 외부 지표(Data Fusion), 바이낸스 롱숏비율, 김치프리미엄, 매크로 경제지표 등 **10가지 이상** 실시간 분석

**핵심 기능:**
- ⚔️ **포지션 과다 분할 매도**: BTC 비중이 50% 초과 시 +5% 수익에서 1/3 자동 매도
- 💧 **하이브리드 DCA(물타기)**: 캐스케이드(연쇄폭락) 위험도를 실시간 계산. 진성 하락장엔 즉시 손절, 박스권에서만 영리한 물타기
- 🧠 **감독 DB 학습**: 과거 같은 상황에서의 전략 전환 성과를 학습하여 판단을 지속 개선
- 🚨 **감독 자동 긴급정지**: 4시간 -10% 급락, 캐스케이딩 극단 등 위기 시 자동 전량 매도 + 매수 차단

### 2-3. 초단타 스캘핑 봇

1~2분 단위로 시장을 감시하며 소폭 변동을 먹고 빠지는 **속도전 봇**입니다. 스윙 전략과 **자금이 완전 분리**되어 서로 간섭하지 않습니다.

```
🪟 Windows:  python scripts\short_term_trader.py
🍎 Mac:      python3 scripts/short_term_trader.py

# 옵션
--dry-run    모의매매 모드
--live       실매매 모드 (DRY_RUN=false 필요)
--status     현재 상태 확인 후 종료
```

**3가지 전략 동시 운영:**

| 전략 | 감지 방법 | 매매 기준 |
|------|----------|----------|
| 📰 **뉴스 반응** | RSS 피드 실시간 감성 스캔 (2분 간격) | 강한 긍정/부정 감성 ±0.4 이상 |
| 📈 **급등/급락 리바운드** | 5분 내 0.5%+ 급변동 감지 | 되돌림 0.3%+ 확인 + 체결 방향 55%+ |
| 🐋 **고래 추종** | 2000만원+ 대량 체결 연속 감지 | 같은 방향 3건+ 연속 시 추종 |

**안전장치**: 자동 손절 1.2%, 자동 익절 0.5%, 최대 보유 15분, 하락 추세 매수 차단, 극공포(FGI≤5) 매수 차단, 일일 최대 20회

---

## 3. 🌶️ 김치랑 봇 (Kimchirang)

### 3-1. 김치프리미엄 차익거래란?

한국 업비트의 비트코인 가격이 해외 바이낸스보다 3~10% 비싸게 거래되는 현상(= **김치프리미엄**)을 이용한 차익거래입니다.

```
📈 김치프리미엄 3% 이상 → 진입!
   ├─ 업비트: BTC 현물 매수 (비싼 쪽)
   └─ 바이낸스: BTCUSDT 선물 숏 (헤지)

📉 김치프리미엄 0.5% 이하로 축소 → 청산!
   ├─ 업비트: BTC 현물 매도
   └─ 바이낸스: 선물 숏 커버 (매수)

💰 수익 = 진입KP(3%) - 청산KP(0.5%) - 수수료(~0.18%) ≈ 2.3% 수익!
```

**핵심 원리**: 양쪽 거래소에 동시 포지션을 잡으므로 **비트코인 가격이 오르든 내리든 상관없이** 김치프리미엄의 수축/확장에서만 수익이 발생합니다. (Delta-Neutral 전략)

### 3-2. 김치랑 실행 방법

```
🪟 Windows:  python -m kimchirang.main
🍎 Mac:      python3 -m kimchirang.main

# 실매매 모드
KR_DRY_RUN=false python -m kimchirang.main     # Mac/Linux
$env:KR_DRY_RUN="false"; python -m kimchirang.main  # Windows PowerShell
```

**김치랑 작동 흐름:**
1. 🔌 **WebSocket 연결**: 업비트 + 바이낸스 실시간 호가 동시 수신
2. 📊 **KP 엔진**: 1초마다 김치프리미엄 계산 (Entry KP / Exit KP / Z-Score / 속도 / 가속도)
3. 🤖 **RL 브릿지**: PPO + DQN 앙상블 강화학습으로 최적 진입/청산 타이밍 결정
4. ⚡ **동시 실행**: `asyncio.gather()`로 업비트+바이낸스 양쪽 동시 주문
5. 🛡️ **레그 리스크 방지**: 한쪽 주문 실패 시 다른 쪽 즉시 긴급 반대매매

**주요 파라미터 (config.py):**

| 파라미터 | 기본값 | 설명 |
|---------|:------:|------|
| `kp_entry_threshold` | 3.0% | 김프 3% 이상이면 진입 |
| `kp_exit_threshold` | 0.5% | 김프 0.5% 이하면 청산 |
| `kp_stop_loss` | 8.0% | 김프 8% 이상 확대 시 손절 |
| `trade_amount_krw` | 100,000원 | 1회 매매 금액 |
| `leverage` | 1x | 바이낸스 레버리지 (1x 권장) |
| `max_daily_trades` | 20 | 일일 최대 거래 횟수 |

### 3-3. RL 학습 (PPO + DQN 앙상블)

김치랑은 2개의 강화학습 모델을 앙상블로 운영합니다. 두 모델이 합의할 때만 행동하여 안정성을 높입니다.

```bash
# PPO 학습 (정책 기반)
python -m kimchirang.train_simple --steps 300000 --days 730

# DQN 학습 (가치 기반 — Experience Replay로 극단 이벤트 반복 학습)
python -m kimchirang.train_dqn --steps 500000 --days 730
```

| 모델 | 특징 | 장점 |
|------|------|------|
| **PPO** | 정책 그래디언트 기반 | 안정적 학습, 연속적 정책 개선 |
| **DQN** | Q-value 직접 계산 | Experience Replay로 희귀 이벤트(대청산, 10% 김프 등) 반복 학습 |
| **앙상블** | 두 모델 합의 시만 행동 | 한쪽만 신호 → Hold (신중하게) |

> 💡 RL 모델 없이도 **규칙 기반 모드**로 동작합니다. 학습은 선택사항입니다.

---

## 4. 📱 텔레그램 알림 설정

봇의 모든 매매 결과가 텔레그램으로 알림됩니다:

- 📊 **매매 실행**: 매수/매도 결정, 금액, 근거 요약, 포트폴리오 변동
- 🚨 **긴급 알림**: 연속 손절, 급락, 긴급정지 발동
- 📈 **일일 요약**: 당일 거래 횟수, 수익률, 포트폴리오 현황
- 🌶️ **김치랑**: 진입/청산 알림, KP 상태 5분 주기 보고

---

## 5. ⏰ 자동화 스케줄링

#### 🪟 Windows (작업 스케줄러)

1. 시작 메뉴에서 **"작업 스케줄러"** 검색 → 실행
2. 우측 **"작업 만들기"** 클릭
3. **일반** 탭: 이름 `CoinBot` 입력
4. **트리거** 탭: `새로 만들기` → 매일 → 4시간마다 반복 설정
5. **동작** 탭:
   - 프로그램: `.venv\Scripts\python.exe`의 **전체 경로** (예: `D:\Projects\claude-coin-trading-main\.venv\Scripts\python.exe`)
   - 인수 추가: `scripts\run_agents.py`
   - 시작 위치: `D:\Projects\claude-coin-trading-main`
6. **확인** 클릭

또는 PowerShell 스크립트로 자동 등록:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_all_schedules.ps1
```

#### 🍎 Mac / Linux (cron)

```bash
# 4시간 크론탭 자동 등록
bash scripts/setup_cron.sh install

# 등록된 크론탭 확인
bash scripts/setup_cron.sh status

# 크론탭 제거
bash scripts/setup_cron.sh uninstall
```

---

## 6. 📊 대시보드

웹 브라우저에서 봇의 실시간 상태와 수익률을 확인할 수 있습니다.

```
🪟 Windows:  python scripts\web_server.py
🍎 Mac:      python3 scripts/web_server.py

→ 브라우저에서 http://localhost:5000 접속
```

대시보드 기능:
- 📈 실시간 포트폴리오 현황 및 수익률 차트
- 📋 최근 매매 내역 및 판단 근거
- 🔄 에이전트 전환 이력
- 🐋 고래 감지 기록
- 📊 단타/스윙 전략별 성과 분석

---

## 7. 🏗️ 프로젝트 구조

```
claude-coin-trading-main/
│
├── 📄 README.md              ← 이 파일
├── 📄 CLAUDE.md              ← AI를 위한 프로젝트 지침서
├── 📄 strategy.md            ← 자연어 매매 전략 (LLM이 해석)
├── 📄 .env                   ← API 키 (git 추적 제외)
├── 📄 requirements.txt       ← Python 의존성 목록
│
├── 🤖 agents/                ← 비트코인 자동매매 에이전트
│   ├── orchestrator.py       ← 감독 에이전트 (전략 자율 전환)
│   ├── conservative.py       ← 🛡️ 보수적 에이전트
│   ├── moderate.py           ← ⚖️ 보통 에이전트
│   ├── aggressive.py         ← 🔥 공격적 에이전트
│   ├── base_agent.py         ← 기본 클래스 (점수제 매수, 하이브리드 손절)
│   └── external_data.py      ← 외부 데이터 병렬 수집
│
├── 🌶️ kimchirang/            ← 김치프리미엄 차익거래 봇
│   ├── main.py               ← 메인 진입점 + RL 브릿지
│   ├── config.py             ← 설정 (환경변수 + 파라미터)
│   ├── kp_engine.py          ← 실시간 KP 계산 + 통계 피처
│   ├── data_feeder.py        ← WebSocket 데이터 피드 (업비트+바이낸스)
│   ├── execution.py          ← Delta-Neutral 동시 주문 실행
│   ├── rl_env.py             ← Gymnasium RL 환경
│   ├── train_simple.py       ← PPO 학습 스크립트
│   └── train_dqn.py          ← DQN 학습 스크립트
│
├── 📜 scripts/               ← 실행 스크립트 모음 (67개)
│   ├── run_agents.py         ← 에이전트 모드 실행
│   ├── short_term_trader.py  ← 초단타 스캘핑 봇
│   ├── web_server.py         ← 웹 대시보드 서버
│   ├── collect_*.py          ← 데이터 수집 스크립트들
│   ├── train_*.py            ← RL 학습 스크립트들
│   └── setup_*.sh/ps1       ← 자동화 설정 스크립트들
│
├── 🧠 rl_hybrid/             ← AI-RL 하이브리드 분산 시스템
│   ├── launchers/            ← 통합 실행기
│   ├── nodes/                ← 분산 노드들
│   ├── rl/                   ← 강화학습 모듈
│   └── rag/                  ← RAG (유사 패턴 검색) 모듈
│
├── 🗂️ supabase/              ← 데이터베이스
│   └── migrations/           ← SQL 마이그레이션 (24개 파일)
│
├── 🌐 web/                   ← 웹 프론트엔드
├── 🔧 tools/                 ← 분석 도구
├── 📊 data/                  ← 데이터 저장소
├── 📝 logs/                  ← 실행 로그
├── 📚 docs/                  ← 상세 문서
└── 🎨 assets/                ← 이미지/로고
```

---

## 8. 🚫 면책 조항

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
