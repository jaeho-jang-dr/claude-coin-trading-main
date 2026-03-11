# Claude Code Crypto Trading Bot (V4)

> **코드가 아닌 자연어로 전략을 쓰면, AI(Claude)가 시장 지표를 분석하고 자동으로 매매하는 나만의 크립토 봇 시스템입니다.**

이 프로젝트는 [dantelabs](https://github.com/dantelabs)님의 초기 모델을 기반으로, 
**초단타 스캘핑 봇 개발**과 **다중 전략 에이전트(공격적/보통/보수적)**, 
그리고 **실시간 클라우드 대시보드 및 복합 DCA(물타기) 로직**을 추가하여 완전히 재설계한 **확장 버전**입니다.
기존의 리눅스 범용 버전을 Windows(윈도우)와 Mac(맥)에서 누구나 쉽게 돌릴 수 있도록 구성하였습니다.

---

## 📢 시작하기 전에 (필수 가입 및 준비물)
초보자분들도 바로 따라하실 수 있습니다. 프로그램을 실행하려면 아래 **5가지 API 키**가 반드시 필요합니다. 설치 전에 먼저 가입해서 키를 발급받아 메모장에 적어두세요!

### 1. 📈 업비트 (Upbit) API 키
- **용도**: 잔고 조회, 실시간 시세 조회 및 실제 코인 매수/매도 실행
- **방법**: [Upbit 로그인] -> 마이페이지 -> Open API 안내 -> **API Key 발급**
- **필수 권한**: 자산조회, 주문조회, 주문하기 *(출금하기 기능은 절대 체크하지 마세요!)*
- **발급 결과물**: `UPBIT_ACCESS_KEY` (액세스 키), `UPBIT_SECRET_KEY` (시크릿 키)
- *(참고: 특정 IP에서만 접속하게 설정해두면 더 안전합니다)*

### 2. 🗄️ 수파베이스 (Supabase) API 키
- **용도**: 나의 투자 내역, 봇의 매수/매도 판단 근거, 수익률 등을 평생 무료로 자동 저장하는 클라우드 데이터베이스
- **방법**: [Supabase.com](https://supabase.com/) 가입 -> **New Project** 생성 -> 설정(Settings) -> API 탭 이동
- **발급 결과물**: `SUPABASE_URL` (프로젝트 URL), `SUPABASE_SERVICE_ROLE_KEY` (service_role secret 키)

### 3. 📱 텔레그램 (Telegram) 봇 토큰
- **용도**: 매수/매도 체결 알림, 목표가 도달 시 내 폰으로 즉시 메시지 받기, 봇 원격 제어
- **방법 (봇 토큰)**: 텔레그램 검색창에 **`@BotFather`** 검색 -> `/newbot` 입력 -> 봇 이름(예: MyTradingBot) 지정 -> `HTTP API Token` 복사
- **방법 (내 ID 숫자)**: 내 고유 아이디를 확인하기 위해 **`@userinfobot`** 검색 -> `Id:` (예: 123456789) 화면 글씨 복사
- **발급 결과물**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID`

### 4. 📰 타빌리 (Tavily) API 키
- **용도**: 글로벌 뉴스 사이트를 실시간으로 검색해 비트코인 호재와 악재를 판별, 봇의 판단력을 높이는 AI 웹 검색 엔진
- **방법**: [Tavily.com](https://tavily.com/) 가입 -> **API Keys** 메뉴 클릭 생성 (무료 티어로 한 달에 1,000건까지 조회 가능)
- **발급 결과물**: `TAVILY_API_KEY`

### 5. 🤖 제미나이 (Gemini) API 키
- **용도**: V4의 핵심인 RAG (과거 유사 패턴 분석) 및 딥러닝 시장 맥락 분석 (Gemini 2.5 Pro 사용)
- **방법**: [Google AI Studio](https://aistudio.google.com/app/apikey) 접속 -> 구글 로그인 -> **Create API Key** 버튼 클릭
- **발급 결과물**: `GEMINI_API_KEY`

---

## 🚀 다운로드 및 설치 가이드 (Windows & Mac)
> **공통 준비물**: 내 컴퓨터에 **[Git](https://git-scm.com/downloads)**과 **[Python](https://www.python.org/downloads/)** (3.10 이상)이 설치되어 있어야 합니다.

### Step 1. 프로젝트 다운로드 (공통)
명령 프롬프트(CMD), PowerShell 또는 터미널을 열고 아래 명령어를 입력하여 봇 소스코드를 내 컴퓨터로 다운받습니다.

```bash
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
cd claude-coin-trading-main
```

### Step 2. 가상환경 설정 및 필수 프로그램 설치
> **안내**: 각자의 운영체제(OS)에 맞는 코드를 순서대로 복사해서 붙여넣고 엔터를 치세요!

| 🪟 Windows (명령 프롬프트 / PowerShell) | 🍎 Mac / Linux (터미널) |
|--------------------------------------|----------------------|
| `# 1. 가상환경 공간 만들기`<br>`python -m venv .venv` | `# 1. 가상환경 공간 만들기`<br>`python3 -m venv .venv` |
| `# 2. 가상환경 켜기 (진입)`<br>`.venv\Scripts\activate` | `# 2. 가상환경 켜기 (진입)`<br>`source .venv/bin/activate` |
| `# 3. 봇 작동에 필요한 패키지 다운로드`<br>`pip install -r requirements.txt` | `# 3. 봇 작동에 필요한 패키지 다운로드`<br>`pip install -r requirements.txt`|
| `# 4. 브라우저 자동화 도구(웹 차트 캡처용) 설치`<br>`playwright install chromium` | `# 4. 브라우저 자동화 도구(웹 차트 캡처용) 설치`<br>`playwright install chromium` |


### Step 3. API 키 설정 (내 지갑 연동)
이제 프로그램이 내 코인 지갑과 데이터베이스를 확인하도록, 위에서 가장 처음 준비한 **5개의 API 키**를 파일에 적어야 합니다.

#### 🪟 Windows 환경
1. 다운받은 폴더에 가면 `.env.example` 이라는 파일이 있습니다.
2. 파일에 대고 마우스 우클릭 -> **이름 바꾸기** -> `.env` 로 파일명을 바꿉니다. (끝의 확장자 다 떼고 점(.)으로 시작)
3. 해당 `.env` 파일을 더블클릭해서 **메모장**으로 엽니다.

#### 🍎 Mac 환경
터미널에서 명령어 한 줄을 치면 바로 파일이 복사되고 텍스트 편집기가 열립니다.
```bash
cp .env.example .env
nano .env  # 편집기 열기
```

#### ✅ [.env] 파일 내용 입력 예시
메모장을 열고 빈칸에 정확히 따옴표("") 안에 내 키를 복사해서 넣고 저장합니다.

```ini
UPBIT_ACCESS_KEY="내_업비트_엑세스_키"
UPBIT_SECRET_KEY="내_업비트_시크릿_키"
SUPABASE_URL="https://내프로젝트.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="내_수파베이스_역할_키"
TELEGRAM_BOT_TOKEN="내_텔레그램_토큰"
TELEGRAM_USER_ID="내_텔레그램_숫자_ID"
TAVILY_API_KEY="내_타빌리_키"
GEMINI_API_KEY="내_제미나이_키"

# 🔥 투자 심장부 기본 셋팅 (안전장치)
DRY_RUN=true            # ✅ true: 실제 매매 안함 (모의 연습), false: 🚨실제 내 돈으로 매매 시작!
MAX_TRADE_AMOUNT=100000 # 1회 최대 매수 금액 (원)
MAX_POSITION_RATIO=0.5  # 전체 자산 중 최대 코인 투자 비율 제한 (0.5 = 내 전 재산의 50%까지만 코인 구매 허용)
```
> ⚠️ **주의**: 처음 실행할 때는 **반드시 `DRY_RUN=true` 상태로 테스트**하여 앱이 내역을 잘 불러오는지 확인한 후 `false`로 바꾸세요.

---

## 🤖 봇 실행 방법 (실전 가이드)
이 봇은 다양한 전략 형태로 상황에 맞게 운영할 수 있습니다. 

### 1️⃣ [추천] AI-강화학습 하이브리드 시스템 (V4 최신형)
Gemini 2.5 Pro의 **LLM 정성 분석**과 PPO 모델의 **강화학습(RL) 정량 분석**, 그리고 3명의 기존 **AI 스쿼드 규칙**이 결합된 만장일치 의사결정 시스템입니다. 여러 봇을 통합 운영합니다.
- **실행 전에 해야 할 일!**: 먼저 Supabase 대시보드에서 `SQL Editor`를 열고 `supabase/migrations/016_rag_analysis_vectors.sql` 내용을 복사해 **RUN(실행)** 하여 데이터베이스를 최신형으로 업데이트해주세요!

**🪟 Windows**: `python rl_hybrid\launchers\start_all.py`  
**🍎 Mac**: `python3 rl_hybrid/launchers/start_all.py`

*(실행하면 텔레그램 연동, 실시간 데이터 분석, 의사 결정이 백그라운드 멀티 프로세스로 동시에 구동됩니다.)*

### 2️⃣ 정밀 스캘핑 초단타 봇 - 1~2분 단위 감시
이 봇은 단일 터미널에서 짧은 단위의 틱 변동성을 감시해 자잘한 반등 포인트나 하락을 먹고 빠지는 속도전용 봇입니다.

**🪟 Windows**: `python scripts\short_term_trader.py`  
**🍎 Mac**: `python3 scripts/short_term_trader.py`

### 3️⃣ (자동화 셋팅) 나는 컴을 꺼도 4시간마다 알아서 돌아가게 하고 싶다면?
* **🪟 Windows 사용자의 경우**:
  윈도우의 '작업 스케줄러(Task Scheduler)' 검색 후 실행 -> 우측 '작업 만들기' -> 트리거(4시간마다 반복) -> 동작 (프로그램 시작 -> `.venv/Scripts/python.exe`, 인수 `scripts/orchestrator.py`, 시작 위치 `봇 폴더 경로`) 식으로 설정해 주시면 됩니다.
  
* **🍎 Mac / Linux 사용자의 경우**:
  내장된 스크립트 하나면 바로 4시간 크론탭(자동화) 설정이 끝납니다.
  ```bash
  bash scripts/setup_cron.sh install
  ```

---

## 📈 주요 기능 및 로직 소개
- **세 계급 분할 에이전트**: 보수적(낙폭 과대 줍기) / 보통(추세 전환점 노리기) / 공격적(달리는 말에 단타) 3명의 AI 관리자가 각각 현재의 공포탐욕지수(FGI), RSI, 뉴스 극성을 종합 평가해 만장일치로 최고 결정된 전략으로 코인을 매수합니다.
- **포지션 과다 분할 익절 전략 (Overweight)**: 현재 봇이 투자한 비트코인 비율이 내 잔고의 50% 범위를 과도하게 넘어간 상태라면, 이익이 목표수익률에 도달하지 않은 아직 `+5%` 수준이라 하더라도 물량의 `1/3`을 안전하게 강제 분할 매도해 불확실성을 방어합니다.
- **물타기 (Hybrid DCA) 스마트 평가**: 기존 수많은 봇들이 -5%에 물리면 기계식으로 물을 탑니다. 하지만 이 봇은 하락의 '캐스케이드(연쇄폭락)' 위험성을 실시간 계산해, 악재 속 진성 하락장에는 절대 물타기를 금지하고 즉시 손절(칼치기) 런칭시키며, 박스권 휩소일 때만 영리하게 DCA(물타기)에 들어갑니다.
- **클라우드 데이터 무결성 보장**: 이 모든 판단 근거 프로세스와 수익률 통계는 전부 내 무료 Supabase DB에 기록되어 나만의 트레이딩 성과 페이지(대시보드) 구축을 돕습니다.

---

## 🚫 면책 조항
이 프로그램은 오픈소스로 제공되는 개인 연구용 파이썬 자동화 툴입니다. 시스템의 설계 결함, 서버 다운, 가격 오류 및 판단 미스로 발생하는 **모든 금전적 혹은 암호화폐 투자 손실의 법적·도의적 책임은 전적으로 다운로드하여 사용한 코인 사용자 본인에게 일임됩니다.** 
**실제 여러분의 피 같은 자산을 투입하기 전에는 반드시 `DRY_RUN=true` 모드로 1달 이상 충분히 모의 테스트하시고, 성능을 검증한 뒤 1만 원 소액으로만 먼저 검증하시길 간절히 권장합니다.**
