# Claude Code Crypto Trading Bot (V4)

> **코드가 아닌 자연어로 전략을 쓰면, AI(Claude)가 시장 지표를 분석하고 자동으로 매매하는 나만의 크립토 봇 시스템입니다.**

이 프로젝트는 초기 모델을 기반으로, **초단타 스캘핑 봇 개발**과 **다중 전략 에이전트(공격적/보통/보수적)**, 그리고 **실시간 클라우드 대시보드 및 복합 DCA(물타기) 로직**을 추가하여 완전히 재설계한 **확장 버전**입니다. 또한, 최신 업데이트를 통해 원격 접속 환경(SSH/Meshnet)과 LLM-RL 하이브리드 분산 시스템까지 통합되었습니다.

---

## ⚙️ 1. 설치 및 API 키 설정 (필수 가입 및 준비)

초보자분들도 바로 따라하실 수 있습니다. 프로그램을 실행하려면 환경 셋업과 **API 키** 발급이 가장 먼저 필요합니다. 아래 순서대로 정확히 설정해 주세요.

### Step 1. 필수 프로그램 설치 및 프로젝트 다운로드
내 컴퓨터에 **[Git](https://git-scm.com/downloads)**과 **[Python](https://www.python.org/downloads/)** (3.10 이상)이 설치되어 있어야 합니다.

명령 프롬프트(CMD), PowerShell 또는 Mac 터미널을 열고 아래 명령어를 입력하여 소스코드를 다운받습니다:
```bash
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
cd claude-coin-trading-main
```

### Step 2. 가상환경 설정 및 패키지 설치
OS에 맞게 아래 명령어를 순서대로 실행합니다.

| 🪟 Windows (CMD / PowerShell) | 🍎 Mac / Linux (터미널) |
|--------------------------------------|----------------------|
| `# 1. 가상환경 만들기`<br>`python -m venv .venv` | `# 1. 가상환경 만들기`<br>`python3 -m venv .venv` |
| `# 2. 가상환경 진입`<br>`.venv\Scripts\activate` | `# 2. 가상환경 진입`<br>`source .venv/bin/activate` |
| `# 3. 의존성 패키지 설치`<br>`pip install -r requirements.txt` | `# 3. 의존성 패키지 설치`<br>`pip install -r requirements.txt`|
| `# 4. 브라우저 자동화 도구 설치`<br>`playwright install chromium` | `# 4. 브라우저 자동화 도구 설치`<br>`playwright install chromium` |

### Step 3. 필수 API 키 발급
봇이 내 계좌를 보고 데이터베이스를 읽고 쓰려면 5가지 키가 필요합니다. 발급 후 메모장에 적어두세요.

1. **📈 업비트 (Upbit) API 키**
   - 용도: 잔고 조회 및 실제 코인 매수/매도 실행
   - 발급: [Upbit 로그인] -> 마이페이지 -> Open API -> **API Key 발급**
   - 권한: 자산조회, 주문조회, 주문하기 *(출금하기 절대 체크 X)*
2. **🗄️ 수파베이스 (Supabase) API 키 및 DB URL**
   - 용도: 투자 내역, 봇 판단 근거, 수익률 저장용 클라우드 DB
   - 발급: [Supabase](https://supabase.com/) 가입 -> 프로젝트 생성 -> Settings -> API
3. **📱 텔레그램 (Telegram) 봇 토큰**
   - 용도: 매매 체결 알림 및 봇 원격 제어
   - 발급: 텔레그램에서 `@BotFather` 검색 -> `/newbot` 입력 후 토큰 복사. 내 ID는 `@userinfobot`에서 확인.
4. **📰 타빌리 (Tavily) API 키**
   - 용도: 글로벌 뉴스 실시간 검색 (AI 웹 검색 엔진)
   - 발급: [Tavily.com](https://tavily.com/) 가입 -> API Keys 생성
5. **🤖 제미나이 (Gemini) API 키**
   - 용도: RAG (유사 패턴 분석) 및 LLM 판단 로직 속도 최적화
   - 발급: [Google AI Studio](https://aistudio.google.com/app/apikey) -> Create API Key

### Step 4. 환경 변수(`.env`) 파일 셋팅
다운받은 프로젝트 폴더의 `.env.example` 파일을 복사하여 `.env`로 이름을 바꾼 뒤 내용을 채워줍니다.
```bash
# Mac/Linux 터미널에서:
cp .env.example .env
nano .env (또는 원하는 에디터로 열기)
```

**[ .env 파일 주요 내용 예시 ]**
```ini
UPBIT_ACCESS_KEY="내_업비트_엑세스_키"
UPBIT_SECRET_KEY="내_업비트_시크릿_키"
TAVILY_API_KEY="tvly-내_타빌리_키"
SUPABASE_URL="https://내프로젝트.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="내_수파베이스_역할_키"
SUPABASE_DB_URL="postgresql://postgres.[your-project-ref]:[password]@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres"
TELEGRAM_BOT_TOKEN="내_텔레그램_토큰"
TELEGRAM_USER_ID="내_텔레그램_숫자_ID"
GEMINI_API_KEY="내_제미나이_키"

# 🔥 투자 심장부 기본 셋팅 (주의 깊게 설정)
DRY_RUN=true            # ✅ true: 실제 매매 안함 (모의 연습), false: 🚨실제 돈으로 매매 시작!
MAX_TRADE_AMOUNT=100000 # 1회 최대 매수 금액 (원)
MAX_POSITION_RATIO=0.5  # 전체 자산 중 최대 코인 투자 비율 (예: 0.5 = 내 전 재산의 50%까지만 코인 구매)
```
> ⚠️ **주의**: 첫 실행 시 **반드시 `DRY_RUN=true` 상태로 테스트**하여 앱이 내역을 잘 불러오는지 확인한 후 `false`로 바꾸세요.

---

## 🚀 2. 봇 실행 방법

설정이 완료되었으면 터미널/명령 프롬프트를 열고 원하는 방식의 봇을 실행할 수 있습니다. 

### 1️⃣ [추천] AI-강화학습 하이브리드 시스템 (V4 최신형)
Gemini 2.5 Pro의 **LLM 정성 분석**, PPO 모델의 **강화학습(RL) 정량 분석**, 그리고 3명의 **AI 스쿼드 규칙**이 결합된 만장일치 의사결정 시스템입니다. 
- **DB 필수 업데이트**: 먼저 Supabase의 `SQL Editor`에서 `supabase/migrations/016_rag_analysis_vectors.sql` 내용을 복사 후 실행(Run)해 주세요!

```bash
# 🪟 Windows: 
python rl_hybrid\launchers\start_all.py

# 🍎 Mac / Linux:
python3 rl_hybrid/launchers/start_all.py
```
*(실행 시 텔레그램 연동, 실시간 데이터 분석, 의사 결정이 백그라운드 멀티 프로세스로 구동됩니다.)*

### 2️⃣ 정밀 스캘핑 초단타 봇
짧은 1~2분 단위의 틱 변동성을 감시하여 작은 반등 포인트나 하락을 먹고 빠지는 속도전용 봇입니다. (비상시나 박스권 장세에 유리)
```bash
# 🪟 Windows: 
python scripts\short_term_trader.py

# 🍎 Mac / Linux:
python3 scripts/short_term_trader.py
```

### 3️⃣ 자동화 실행 설정 (알아서 4시간마다 돌게 하기)
컴퓨터/서버를 켜두고 자동 반복시킬 때의 설정법입니다.
- **🪟 Windows**: '작업 스케줄러' 실행 -> '작업 만들기' -> 4시간마다 트리거 생성 -> 동작으로 `.venv/Scripts/python.exe` 지정 후 인수 `scripts/orchestrator.py` 추가.
- **🍎 Mac / Linux**: 기본 셋팅 스크립트로 크론탭에 한 방에 등록합니다.
  ```bash
  bash scripts/setup_cron.sh install
  ```

---

## 🧠 3. 주요 기능 및 순차적 로직 설명

해당 봇은 단순히 지표 하나로 매수/매도하는 것이 아니라, 아래와 같은 순차적인 딥 다이브 분석 논리를 거쳐 진행됩니다.

1. **실시간 시장 감지**: 업비트로 시장의 OHLCV(초~일 단위 봉차트) 데이터를 가져오고, 브라우저를 띄워 실시간 차트를 시각적으로 캡처합니다.
2. **다중 뉴스 및 웹 스크랩핑**: Tavily API를 사용해 최근 비트코인 상승/하락에 영향을 미치는 주요 거시경제 뉴스와 암호화폐 소식들의 호재/악재 극성을 분석합니다.
3. **과거 패턴 RAG 비교 (Supabase)**: 지금까지 저장된 시장 데이터베이스 속에서 현재 맥락과 가장 비슷한 과거의 패턴을 검색(벡터 검색)해 냅니다.
4. **세 계급 분할 에이전트 다중 평가**:
   - **보수적 에이전트**: 낙폭 과대 등 극단적 공포에서의 진입점 파악
   - **보통 에이전트**: RSI 추세 등 보편적인 지표 추세 전환점 노리기
   - **공격적 에이전트**: 달리는 상승마에 올라타 짧은 단타 타점 모색  
   위 세 AI가 LLM과 RL(강화학습) 모델 결과를 기반으로 타겟 코인을 토론하여 **만장일치** 또는 **다수결 방식**으로 액션(매수/매도/관망)을 결정합니다.
5. **리스크 관리 로직 개입 (Overweight & DCA)**:
   - **Overweight 방어**: 코인 비중이 자산의 50%를 초과할 시 수익권(+5% 등) 목표에 조금 안 닿았더라도 일정 물량(1/3)을 안전하게 익절/손절하여 현금을 확보합니다.
   - **스마트 물타기(Hybrid DCA)**: 기존 기계식 물타기(e.g., -5%마다 무조건 매수)가 아닌 장세 하락의 '캐스케이드(연쇄 폭락)' 위험도를 평가합니다. 위험한 폭락장일 때는 칼같이 즉시 손절(런)하고, 단순히 지지선을 지루하게 터치하는 박스권일 때만 진득하게 저점 매수를 진행합니다.
6. **거래 실행 및 알림**: 최종 결정된 스텝을 텔레그램 메신저로 즉각 보고하며 거래를 완료합니다.

---

## 🌐 4. 원격 터미널 접속 관리 (팁)
외부에서 해당 봇 서버/Mac에 접속하여 상태를 확인하려면, 다음 네트워크망을 사용할 수 있도록 설정되어 있습니다 (설정된 환경 기준).
> **팁**: `NordVPN Meshnet`이 구동 중이라면 다른 기기에서 다음과 같이 터미널/SSH로 보안 접속하여 봇을 직접 제어할 수 있습니다.
```bash
# 예시 
ssh drj00@100.x.x.x (자신의 Meshnet IP)
```

---

## 🚫 면책 조항
이 프로그램은 오픈소스로 제공되는 개인 연구용 파이썬 자동화 툴입니다. 시스템의 설계 결함, 서버 다운, 가격 오류 및 판단 미스로 발생하는 **모든 금전적 혹은 암호화폐 투자 손실의 법적·도의적 책임은 전적으로 다운로드하여 사용한 코인 사용자 본인에게 일임됩니다.** 

**실제 투입 전에는 반드시 `.env` 파일의 `DRY_RUN=true` 모드로 1~2주 이상 충분히 모의 테스트하시고, 성능을 검증한 뒤 소액으로만 먼저 검증하시길 간절히 권장합니다.**
