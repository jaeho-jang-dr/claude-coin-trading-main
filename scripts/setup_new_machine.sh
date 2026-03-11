#!/bin/bash
# === 새 컴퓨터 초기 셋업 스크립트 ===
#
# 사용법:
#   bash scripts/setup_new_machine.sh
#
# 사전 요구사항:
#   - Python 3.10+
#   - Node.js 18+
#   - Git
#   - Google Drive 동기화 (G:\내 드라이브\antigravity_env\)

set -e

echo "=== Claude Coin Trading 시스템 셋업 ==="
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[1/7] 프로젝트 루트: $PROJECT_ROOT"

# --- 2. .env 복원 ---
echo ""
echo "[2/7] .env 파일 복원..."
GDRIVE_ENV="G:/내 드라이브/antigravity_env/claude-coin-trading.env"
if [ -f ".env" ]; then
    echo "  .env 이미 존재 — 건너뜀"
elif [ -f "$GDRIVE_ENV" ]; then
    cp "$GDRIVE_ENV" .env
    echo "  Google Drive에서 복사 완료"
else
    cp .env.example .env
    echo "  [주의] .env.example에서 복사함 — API 키를 직접 입력하세요!"
fi

# --- 3. Python 가상환경 + 의존성 ---
echo ""
echo "[3/7] Python 가상환경 + 의존성 설치..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "  가상환경 생성 완료"
fi

# Windows: venv의 python을 직접 사용 (source activate가 bash에서 불안정)
if [ -f ".venv/Scripts/python.exe" ]; then
    VPYTHON=".venv/Scripts/python.exe"
    VPIP=".venv/Scripts/pip.exe"
else
    VPYTHON=".venv/bin/python"
    VPIP=".venv/bin/pip"
fi

"$VPYTHON" -m pip install --upgrade pip -q 2>/dev/null || true
"$VPYTHON" -m pip install -r requirements.txt -q
echo "  의존성 설치 완료"

# --- 4. RL 모델 복원 ---
echo ""
echo "[4/7] RL 모델 복원..."
GDRIVE_MODEL="G:/내 드라이브/antigravity_env/claude-coin-trading-models/best_model.zip"
MODEL_DIR="data/rl_models/best"

mkdir -p "$MODEL_DIR"
mkdir -p "data/rl_models"

if [ -f "$MODEL_DIR/best_model.zip" ]; then
    echo "  best_model.zip 이미 존재 — 건너뜀"
elif [ -f "$GDRIVE_MODEL" ]; then
    cp "$GDRIVE_MODEL" "$MODEL_DIR/"
    cp "$GDRIVE_MODEL" "data/rl_models/ppo_btc_latest.zip"
    echo "  Google Drive에서 모델 복사 완료 (1.3MB)"
else
    echo "  [주의] 모델 파일 없음 — 훈련 필요: python -m rl_hybrid.rl.train --interval 4h --steps 500000"
fi

# --- 5. Playwright 브라우저 ---
echo ""
echo "[5/7] Playwright 브라우저 설치..."
if "$VPYTHON" -c "from playwright.sync_api import sync_playwright" 2>/dev/null; then
    "$VPYTHON" -m playwright install chromium 2>/dev/null && echo "  Chromium 설치 완료" || echo "  [건너뜀] Chromium 설치 실패"
else
    echo "  [건너뜀] playwright 패키지 없음"
fi

# --- 6. MCP 서버 ---
echo ""
echo "[6/7] MCP 서버 설치..."
if command -v npm &>/dev/null; then
    npm install -g @jamesanz/bitcoin-mcp 2>/dev/null && echo "  bitcoin-mcp 설치 완료" || echo "  [건너뜀] npm 설치 실패"
else
    echo "  [건너뜀] npm 미설치"
fi

# --- 7. 디렉토리 생성 + 검증 ---
echo ""
echo "[7/7] 디렉토리 생성 + 검증..."
mkdir -p logs/nodes logs/executions logs/claude_responses
mkdir -p data/charts data/snapshots data/rl_models/best data/rl_models/backups

echo ""
echo "=== 검증 ==="
echo -n "  Python:     " && "$VPYTHON" --version
echo -n "  pip 패키지: " && "$VPYTHON" -m pip list 2>/dev/null | wc -l | tr -d ' '
echo "개"
echo -n "  .env:       " && ([ -f .env ] && echo "OK" || echo "MISSING")
echo -n "  RL 모델:    " && ([ -f "$MODEL_DIR/best_model.zip" ] && echo "OK ($(ls -lh $MODEL_DIR/best_model.zip | awk '{print $5}'))" || echo "MISSING")
echo -n "  SB3:        " && ("$VPYTHON" -c "import stable_baselines3; print('OK')" 2>/dev/null || echo "MISSING")
echo -n "  PyTorch:    " && ("$VPYTHON" -c "import torch; print('OK')" 2>/dev/null || echo "MISSING")
echo -n "  Playwright: " && ("$VPYTHON" -c "from playwright.sync_api import sync_playwright; print('OK')" 2>/dev/null || echo "MISSING")

echo ""
echo "=== 셋업 완료 ==="
echo ""
echo "다음 단계:"
echo "  1. E2E 테스트:  $VPYTHON tests/test_e2e_hybrid.py"
echo "  2. RL 훈련:     $VPYTHON -m rl_hybrid.rl.train --interval 4h --days 180 --steps 500000"
echo "  3. 시스템 시작:  $VPYTHON rl_hybrid/launchers/start_all.py"
echo "  4. 주간 재학습:  $VPYTHON scripts/setup_weekly_retrain.py"
echo ""
