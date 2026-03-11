#!/bin/bash
# Tier 1 (Trainer) 전용 초기 설정 스크립트
# 훈련에 필요한 최소 환경만 설정합니다.
#
# 사용법:
#   git clone https://github.com/jaeho-jang-dr/claude-coin-trading-main.git
#   cd claude-coin-trading-main
#   bash scripts/setup_trainer.sh

set -e

echo "========================================"
echo "  RL Trainer 환경 설정"
echo "========================================"

# Python 가상환경
if [ ! -d ".venv" ]; then
    echo "[1/4] Python 가상환경 생성..."
    python3 -m venv .venv
else
    echo "[1/4] 가상환경 이미 존재"
fi

# 활성화
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

# 의존성 설치
echo "[2/4] 의존성 설치..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# .env 설정
if [ ! -f ".env" ]; then
    echo "[3/4] .env 파일 생성..."
    cat > .env << 'ENVEOF'
# === Trainer 전용 설정 ===
# Admin에게 받은 값을 입력하세요

# Supabase (훈련 결과 업로드용)
SUPABASE_URL=
SUPABASE_SERVICE_KEY=

# 안전장치 (Trainer는 변경 불가)
DRY_RUN=true
EMERGENCY_STOP=false
ENVEOF
    echo "  → .env 파일이 생성되었습니다."
    echo "  → SUPABASE_URL과 SUPABASE_SERVICE_KEY를 입력하세요."
else
    echo "[3/4] .env 이미 존재"
fi

# Git hook: Trainer가 실수로 push 못하게 방지
echo "[4/4] Git pre-push hook 설정..."
mkdir -p .git/hooks
cat > .git/hooks/pre-push << 'HOOKEOF'
#!/bin/bash
# Trainer는 push 금지 (Admin/Developer만 가능)
TRAINER_MODE="${TRAINER_MODE:-true}"
if [ "$TRAINER_MODE" = "true" ]; then
    echo ""
    echo "⚠️  Trainer 모드에서는 push가 차단됩니다."
    echo "   코드 수정은 Tier 2 (Developer) 이상만 가능합니다."
    echo ""
    echo "   Developer로 전환하려면:"
    echo "   export TRAINER_MODE=false"
    echo ""
    exit 1
fi
HOOKEOF
chmod +x .git/hooks/pre-push

echo ""
echo "========================================"
echo "  설정 완료!"
echo "========================================"
echo ""
echo "훈련 실행:"
echo "  source .venv/bin/activate"
echo "  python -m rl_hybrid.rl.trainer_submit"
echo ""
echo "옵션 보기:"
echo "  python -m rl_hybrid.rl.trainer_submit --help"
echo ""
