# 분산 컴퓨팅 아키텍처

## 컴퓨터 구성

### 현재 (2026-03)
| 역할 | 머신 | 스펙 | 용도 |
|------|------|------|------|
| **주 컴 (Main Brain)** | Mac Mini | 128GB RAM | 라이브 봇 + 데이터 수집 + 오케스트레이터 |
| **워커 1** | Windows PC 1 | 32GB RAM | ML 학습 + 백테스트 |
| **워커 2** | Windows PC 2 | - | ML 학습 + 백테스트 |

### 예정 (1~2개월 내)
| 변경 | 내용 |
|------|------|
| 주 컴 교체 | Mac Mini → **Mac Studio 128GB** |
| Mac Mini | 워커로 전환 |
| Coworker PC 추가 | Co-worker 티어 |
| 커뮤니티 3~4대 | Collaborator 티어 |

---

## 워커 티어

| 티어 | 권한 | 데이터 접근 | 설명 |
|------|------|------------|------|
| **owner** | 전체 | 모든 테이블 R/W | 주 컴. 라이브 봇, 전략 결정, 모델 배포 |
| **coworker** | 학습+분석 | 학습 데이터 R, 결과 W | 공동작업자. 모델 학습, 백테스트, 분석 모두 가능 |
| **collaborator** | 학습만 | 학습 데이터 R, 결과 W | 커뮤니티. 할당된 학습 작업만 실행 |
| **viewer** | 읽기 | 결과/대시보드만 | 모니터링 전용 |

### 티어별 허용 작업

| 작업 | owner | coworker | collaborator | viewer |
|------|:-----:|:--------:|:------------:|:------:|
| 라이브 매매 | ✅ | ❌ | ❌ | ❌ |
| 모델 배포 (is_active) | ✅ | ❌ | ❌ | ❌ |
| 학습 (train_*) | ✅ | ✅ | ✅ | ❌ |
| 백테스트 | ✅ | ✅ | ✅ | ❌ |
| 분석 (analyze) | ✅ | ✅ | ❌ | ❌ |
| 파라미터 스윕 | ✅ | ✅ | ✅ | ❌ |
| 학습 데이터 조회 | ✅ | ✅ | ✅ | ❌ |
| 대시보드 조회 | ✅ | ✅ | ✅ | ✅ |
| 워커 등록/관리 | ✅ | ❌ | ❌ | ❌ |

---

## 워커 등록

각 컴퓨터는 처음 실행 시 자동 등록된다:

```bash
# 워커 시작 (자동 등록)
python3 -m scalp_ml.worker --worker-id "mac-mini" --tier owner

# Windows PC
python scalp_ml\worker.py --worker-id "win-pc1" --tier coworker

# 커뮤니티
python scalp_ml\worker.py --worker-id "community-01" --tier collaborator
```

---

## 주 컴 교체 절차 (Mac Mini → Mac Studio)

1. Mac Studio에 프로젝트 clone + `.env` 복사
2. `supabase login` 실행
3. 워커 등록: `--worker-id "mac-studio" --tier owner`
4. DB에서 기존 main brain 변경: `UPDATE compute_workers SET tier='coworker' WHERE worker_id='mac-mini'`
5. launchd plist 이전 (단타봇, 초단타봇, watchdog)
6. Mac Mini는 `--tier coworker`로 변경하여 학습 워커로 전환

---

## 커뮤니티 Collaborator 관리

### 슬롯 (최대 5명)

| 슬롯 | 상태 | GitHub | 이름 |
|------|------|--------|------|
| community-01 | 미배정 | - | - |
| community-02 | 미배정 | - | - |
| community-03 | 미배정 | - | - |
| community-04 | 미배정 | - | - |
| community-05 | 미배정 | - | - |

### Collaborator 온보딩 절차

1. **GitHub 초대**: `gh api repos/jaeho-jang-dr/claude-coin-trading-main/collaborators/USERNAME -X PUT -f permission=triage`
   - `triage` 권한: 코드 읽기 + 이슈/PR 작성 가능, 직접 push 불가
   - 코드 기여는 fork → PR → owner 리뷰 → merge
2. **워커 ID 배정**: `community-01` ~ `community-05` 중 빈 슬롯
3. **`.env.collaborator` 전달**: Supabase URL + 키만 포함 (거래소 키 없음)
4. **워커 실행 안내**: `docs/worker-setup.md` 참조

### Collaborator 배정 명령어

```bash
# 슬롯 배정 (이름/GitHub 업데이트)
python3 scripts/supabase_query.py --sql "
  UPDATE compute_workers
  SET worker_name = '홍길동', notes = 'GitHub: hongildong'
  WHERE worker_id = 'community-01'
"

# GitHub 초대 (triage 권한)
gh api repos/jaeho-jang-dr/claude-coin-trading-main/collaborators/hongildong -X PUT -f permission=triage

# 슬롯 회수 (탈퇴 시)
python3 scripts/supabase_query.py --sql "
  UPDATE compute_workers
  SET worker_name = '커뮤니티 #1 (미배정)', status = 'suspended', notes = '학습/백테스트만 가능. 분석/매매 불가'
  WHERE worker_id = 'community-01'
"
```

### Collaborator 규칙

- **할 수 있는 것**: ML 학습, 백테스트, 파라미터 스윕, 코드 PR 제출
- **할 수 없는 것**: 분석 조회, 모델 배포, 라이브 매매, 직접 코드 push
- **받는 것**: `.env.collaborator` (Supabase 접속 정보만)
- **안 받는 것**: 거래소 API 키, 텔레그램 봇 토큰, 대시보드 관리 권한
- **코드 기여**: fork → 기능 브랜치 → PR → owner 리뷰 후 merge
- **비활동 정리**: 30일 이상 하트비트 없으면 suspended 전환

### 왜 5명까지인가

- 학습 작업 큐는 FIFO — 워커가 많으면 작업이 빨리 소진되지만, 동시 실행 충돌 위험 증가
- Supabase 무료 플랜: 동시 연결 수 제한 있음
- 5명이면 owner 3대 + coworker 1대 + collaborator 5대 = **총 9대**, 충분한 연산력
- 그 이상 필요하면 Supabase Pro 플랜 + 워커 큐 고도화 후 확장

---

## 보안

### 키 분리 원칙

| 정보 | owner | coworker | collaborator |
|------|:-----:|:--------:|:------------:|
| Supabase URL + Key | ✅ | ✅ | ✅ |
| Upbit API Key | ✅ | ❌ | ❌ |
| Binance API Key | ✅ | ❌ | ❌ |
| Telegram Bot Token | ✅ | ❌ | ❌ |
| OpenAI API Key | ✅ | ✅ | ❌ |
| Supabase Access Token | ✅ | ✅ | ❌ |

### 코드 보안

- `.env` 파일은 각 머신 로컬에만 존재 (git 추적 제외)
- collaborator 코드 기여는 반드시 PR 리뷰 후 merge
- `.env`나 API 키가 포함된 커밋은 자동 거부 (`.gitignore` + pre-commit hook 권장)

### Supabase 접근

- 현재: 모든 티어가 동일한 `service_role` 키 사용 (워커 코드에서 티어 제한)
- 향후: collaborator용 `anon` 키 + RLS 정책으로 DB 레벨 격리 가능
