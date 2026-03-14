# 트러블슈팅 가이드

모든 컴퓨터(Mac Mini, Windows PC1, Windows PC2)에서 공통으로 참조하는 문제 해결 문서.
새로운 에러 해결 시 반드시 여기에 추가할 것.

---

## 1. Supabase DB 직접 연결 불가 (libpq dotted username 문제)

**심각도:** 🔴 치명적 (모든 컴퓨터에서 발생)
**최초 발견:** 2026-03-14
**상태:** 해결됨 (우회 방법)

### 증상

```
FATAL: password authentication failed for user "postgres"
```

- `psycopg2`, `psycopg(v3)`, `asyncpg`, `supabase CLI (db push)` **모두 동일 에러**
- 비밀번호가 맞는데도 인증 실패
- username이 `postgres.tvxgibvbpndorpcqluis`인데 서버에는 `postgres`로만 전달됨

### 원인

Supabase pooler는 `postgres.{project_ref}` 형식의 dotted username을 사용한다.
그런데 **libpq (PostgreSQL C 라이브러리)가 `.`을 구분자로 해석하여 앞부분만 전송**한다.
Python/Node/Go 드라이버 모두 내부적으로 libpq를 사용하거나 동일 파싱을 적용하므로,
어떤 언어/라이브러리를 사용해도 동일한 문제가 발생한다.

### ❌ 실패한 방법 (다시 시도하지 말 것!)

| # | 방법 | 결과 |
|---|------|------|
| 1 | `psycopg2.connect(url)` | username dot 잘림 |
| 2 | `psycopg2.connect(host=..., user=...)` | 동일 |
| 3 | `psycopg.connect()` (v3) | 동일 |
| 4 | `asyncpg.connect()` | 동일 (libpq 미사용인데도 동일 파싱) |
| 5 | `supabase db push -p 'password'` | 동일 (Go pgx 드라이버) |
| 6 | URL 인코딩 (`%2E`) | libpq가 디코딩 후 다시 split |
| 7 | `options='project=ref'` | "Tenant or user not found" |
| 8 | PGUSER 환경변수 | 동일 |
| 9 | 직접 연결 포트 5432 (`db.ref.supabase.co`) | DNS 실패 (MeshNet 환경) |

### ✅ 해결법: Supabase Management API

DB 연결을 우회하고 **HTTP REST API**로 SQL을 직접 실행한다.

#### Step 1: Access Token 확보

**macOS (Keychain에서 추출):**
```bash
# supabase CLI가 로그인되어 있어야 함 (supabase login)
RAW=$(security find-generic-password -s "Supabase CLI" -w)
TOKEN=$(echo "$RAW" | sed 's/^go-keyring-base64://' | base64 -d)
echo $TOKEN
# 결과: sbp_1e0c01c0f2b6d410f2e8a028163cc1c99cdbc84f
```

**Windows (Credential Manager에서 추출):**
```powershell
# supabase CLI가 로그인되어 있어야 함 (supabase login)
# Windows는 Credential Manager에 저장됨
# 또는 supabase 로그인 후 직접 토큰 확인:
supabase projects list  # 이 명령이 성공하면 토큰 유효
```

**토큰이 없으면:**
```bash
supabase login
# 브라우저에서 인증 후 토큰 자동 저장
```

#### Step 2: SQL 실행

```python
import requests

TOKEN = "sbp_1e0c01c0f2b6d410f2e8a028163cc1c99cdbc84f"
PROJECT_REF = "tvxgibvbpndorpcqluis"

url = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# SQL 파일 읽기
with open("supabase/migrations/024_scalp_ml_system.sql", "r") as f:
    sql = f.read()

resp = requests.post(url, headers=headers, json={"query": sql})
print(f"Status: {resp.status_code}")  # 201 = 성공
print(resp.json())
```

#### Step 3: 쿼리 결과 확인

```python
# SELECT 쿼리도 동일하게 실행 가능
resp = requests.post(url, headers=headers, json={
    "query": "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
})
for row in resp.json():
    print(row)
```

### 범용 헬퍼 스크립트

프로젝트에 포함된 `scripts/supabase_query.py` 사용:
```bash
# 마이그레이션 적용
python3 scripts/supabase_query.py --file supabase/migrations/024_scalp_ml_system.sql

# 직접 쿼리
python3 scripts/supabase_query.py --sql "SELECT count(*) FROM scalp_market_snapshot"
```

### 주의사항

- 연속 인증 실패 시 **circuit breaker** 발동 → 5~30분 대기 필요
- Management API는 **DDL (CREATE TABLE 등)도 실행 가능**
- 일반 데이터 CRUD는 Supabase REST API (`/rest/v1/`)를 사용 (이건 정상 작동)
- DB 비밀번호: `.env`의 `SUPABASE_DB_URL` 참조

---

## 2. Circuit Breaker (Supabase 인증 차단)

**심각도:** 🟡 주의
**최초 발견:** 2026-03-14

### 증상

```
FATAL: Circuit breaker open: Too many authentication errors
```

### 원인
Supabase pooler가 연속된 인증 실패를 감지하고 해당 IP/프로젝트를 일시 차단.

### 해결법
1. **5~30분 대기** (자동 해제)
2. 대기 중에는 `REST API` (`/rest/v1/`) 사용 — 이건 별도 경로라 차단 안 됨
3. 근본 원인(위 #1 참조) 해결 후 재시도

---

## 3. 초단타봇 이중 실행

**심각도:** 🟡 주의
**최초 발견:** 2026-03-14
**상태:** 해결됨

### 증상
- `short_term_trader.py`가 2개 프로세스로 동시 실행
- 같은 시그널에 2번 진입, 자금 초과 사용

### 원인
- `launchd` KeepAlive + `run_short_term_24h.sh`의 무한 루프가 겹침
- 봇 크래시 시 launchd가 재시작 + 쉘 스크립트도 재시작

### 해결법
`run_short_term_24h.sh`에 PID lock 추가 (이미 적용됨):
```bash
PIDFILE="data/.short_term_bot.pid"
if [ -f "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "이미 실행 중 (PID $OLD_PID) — 종료"
    exit 0
  fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
```

### 확인 방법
```bash
ps aux | grep short_term_trader | grep -v grep
# 1개만 보여야 정상
```

---

## 에러 추가 규칙

새로운 에러를 해결했으면 이 문서에 추가할 것:

1. **심각도** 표시 (🔴 치명적 / 🟡 주의 / 🟢 경미)
2. **증상** — 에러 메시지 그대로 복사
3. **원인** — 왜 발생하는지 (근본 원인)
4. **실패한 방법** — 다시 시도하지 말 것 목록
5. **해결법** — 복사해서 바로 실행 가능한 코드
6. **주의사항** — 부작용이나 제한
