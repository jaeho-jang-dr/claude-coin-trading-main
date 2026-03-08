# 설정 가이드

---

## 1. 환경 요구사항

- Python 3.9+
- macOS / Linux / Windows
- Claude Code CLI (`claude`)

---

## 2. 설치

```bash
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
cd claude-coin-trading-main

# Python 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# Playwright 브라우저 (차트 캡처용)
playwright install chromium
```

---

## 3. API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키를 입력한다:

### 필수

| 서비스 | 환경변수 | 발급처 |
|--------|---------|--------|
| **Upbit** | `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY` | [업비트 OpenAPI](https://upbit.com/mypage/open_api_management) |

### 권장

| 서비스 | 환경변수 | 발급처 | 용도 |
|--------|---------|--------|------|
| Tavily | `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | 뉴스 수집 |
| Supabase | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | [supabase.com](https://supabase.com) | DB 기록 |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID` | [@BotFather](https://t.me/BotFather) | 알림 |

### 안전장치 (기본값 그대로 사용 권장)

```env
DRY_RUN=true                    # 반드시 true로 시작
MAX_TRADE_AMOUNT=100000         # 1회 매매 상한 (KRW)
MAX_DAILY_TRADES=6              # 일일 매매 상한
MAX_POSITION_RATIO=0.5          # 총 자산 대비 최대 투자 비율
MIN_TRADE_INTERVAL_HOURS=4      # 최소 매매 간격
EMERGENCY_STOP=false            # 긴급 정지
```

---

## 4. Supabase 설정

1. [supabase.com](https://supabase.com)에서 프로젝트 생성
2. SQL Editor에서 마이그레이션 실행:

```bash
# 순서대로 실행
supabase/migrations/001_initial_schema.sql
supabase/migrations/002_scalp_tables.sql
supabase/migrations/002_trade_reviews.sql
supabase/migrations/003_trade_review_views.sql
```

---

## 5. 동작 확인

```bash
# 시장 데이터 수집 확인
python3 scripts/collect_market_data.py

# 포트폴리오 확인
python3 scripts/get_portfolio.py

# 공포탐욕지수 확인
python3 scripts/collect_fear_greed.py

# 전체 파이프라인 (분석만, 매매 안 함)
bash scripts/run_analysis.sh
```

---

## 6. cron 자동 매매

```bash
# 설치 (대화형 간격 선택)
bash scripts/setup_cron.sh install

# 상태 확인
bash scripts/setup_cron.sh status

# 해제
bash scripts/setup_cron.sh remove
```

---

## 7. 실전 전환 체크리스트

1. [ ] `DRY_RUN=true`로 최소 1주일 시뮬레이션 실행
2. [ ] Supabase에서 decisions 테이블의 결정 내역 검토
3. [ ] 텔레그램 알림 정상 수신 확인
4. [ ] `MAX_TRADE_AMOUNT`를 소액(5만원)으로 설정
5. [ ] `.env`에서 `DRY_RUN=false`로 변경
6. [ ] 소액으로 1~2일 실매매 테스트
7. [ ] 성과 확인 후 금액 점진적 증가

---

## 8. 대시보드 원격 접속 (선택)

Cloudflare Tunnel을 사용하면 어디서든 대시보드에 접속할 수 있다.

```bash
# cloudflared 설치
brew install cloudflared

# 터널 생성 (1회)
cloudflared tunnel create crypto-dashboard
cloudflared tunnel route dns crypto-dashboard dashboard.yourdomain.com

# config 작성
cat > ~/.cloudflared/config-dashboard.yml << EOF
tunnel: <TUNNEL_ID>
credentials-file: ~/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: dashboard.yourdomain.com
    service: http://localhost:5555
  - service: http_status:404
EOF

# 실행
cloudflared tunnel --config ~/.cloudflared/config-dashboard.yml run
```
