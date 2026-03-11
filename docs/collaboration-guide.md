# 다중 사용자 협업 가이드

## 티어 구분

| 티어 | 역할 | 할 수 있는 것 | 할 수 없는 것 |
|------|------|-------------|-------------|
| **Tier 1: Trainer** | RL 훈련만 | clone, 훈련, 결과 업로드 | 코드 수정, git push |
| **Tier 2: Developer** | 코드 + 훈련 | 브랜치 생성, PR 제출, 훈련 | main 직접 push |
| **Tier 3: Admin** | 전체 관리 | PR 승인, 모델 승격, 배포 | — |

---

## Tier 1: Trainer 가이드

### 초기 설정 (1회)

```bash
# 1. 저장소 클론
git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
cd claude-coin-trading-main

# 2. 환경 설정
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. 환경변수 (훈련에 필요한 것만)
cp .env.example .env
# .env 파일에서 SUPABASE_URL, SUPABASE_SERVICE_KEY만 설정
# (Admin에게 받은 값 입력)
```

### 훈련 실행

```bash
# 기본 훈련 (PPO, 180일, 100K 스텝)
python -m rl_hybrid.rl.trainer_submit

# 파라미터 변경
python -m rl_hybrid.rl.trainer_submit --algo sac --steps 200000

# 엣지 케이스 혼합 훈련
python -m rl_hybrid.rl.trainer_submit --edge-cases

# 더 긴 데이터로 훈련
python -m rl_hybrid.rl.trainer_submit --days 365 --steps 300000

# 이름 지정 (여러 머신 구분)
python -m rl_hybrid.rl.trainer_submit --trainer-id "my-macbook-pro"
```

### 훈련 실험 아이디어

자유롭게 파라미터를 조합해 보세요:

| 실험 | 명령어 |
|------|--------|
| 긴 기간 PPO | `--days 365 --steps 300000` |
| SAC 알고리즘 | `--algo sac --steps 150000` |
| 엣지 케이스 40% | `--edge-cases --synthetic-ratio 0.4` |
| 기존 best 파인튜닝 | `--model data/rl_models/best/best_model.zip --steps 30000` |
| 엣지+파인튜닝 | `--edge-cases --model data/rl_models/best/best_model.zip` |

### 주의사항

- **코드를 수정하지 마세요.** 파라미터만 변경합니다.
- 결과는 자동으로 DB에 업로드됩니다 (DB 미연결 시 로컬 JSON 저장).
- 모델 승격은 Admin만 할 수 있습니다.
- 최신 코드 받기: `git pull origin main`

---

## Tier 2: Developer 가이드

### 브랜치 워크플로우

```bash
# 1. 최신 main 받기
git checkout main
git pull origin main

# 2. 작업 브랜치 생성
git checkout -b feat/my-improvement

# 3. 코드 수정 + 커밋
git add <수정한 파일>
git commit -m "feat: 개선 내용 설명"

# 4. 푸시 + PR 생성
git push -u origin feat/my-improvement
gh pr create --title "feat: 개선 내용" --body "설명"
```

### PR 규칙

- main 직접 push 불가 (브랜치 보호 규칙)
- PR에 변경 이유와 테스트 결과 포함
- Admin 승인 후 병합
- 모델 차원(observation_dim) 변경은 반드시 성과 검증 데이터 첨부

### 코드 수정 가능 범위

| 가능 | 불가 |
|------|------|
| 새 시나리오 추가 (scenario_generator.py) | observation_dim 변경 |
| 리워드 함수 개선 (reward.py) | 안전장치 값 변경 (.env) |
| 새 알고리즘 추가 (policy.py) | DB 스키마 변경 |
| 버그 수정 | main 직접 push |

---

## Tier 3: Admin 가이드

### 제출 관리

```bash
# 제출 목록 확인
python -m rl_hybrid.rl.admin_review --list

# 리더보드
python -m rl_hybrid.rl.admin_review --leaderboard

# 모델 승격 (3번 제출을 best로)
python -m rl_hybrid.rl.admin_review --promote 3
```

### GitHub 설정

```bash
# 브랜치 보호 규칙 설정 (1회)
gh api repos/{owner}/{repo}/branches/main/protection -X PUT \
  -f "required_pull_request_reviews[required_approving_review_count]=1" \
  -f "enforce_admins=false" \
  -F "required_pull_request_reviews[dismiss_stale_reviews]=true"
```

### 사용자 초대

```bash
# Tier 1 (Trainer) — Read 권한
gh api repos/{owner}/{repo}/collaborators/{username} -X PUT -f permission=pull

# Tier 2 (Developer) — Write 권한
gh api repos/{owner}/{repo}/collaborators/{username} -X PUT -f permission=push
```

---

## 데이터 흐름

```
[Trainer A] ─훈련─→ trainer_submit.py ─→ ┐
[Trainer B] ─훈련─→ trainer_submit.py ─→ ├─→ Supabase DB (rl_training_results)
[Trainer C] ─훈련─→ trainer_submit.py ─→ ┘          │
                                                      ▼
                                              [Admin] admin_review.py
                                                      │
                                              --list / --leaderboard
                                                      │
                                              --promote → best 모델 교체
                                                      │
                                              git push → 전체 반영
```
