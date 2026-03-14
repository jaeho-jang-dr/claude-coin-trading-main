#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 일일 유지보수 스크립트
#
# 매일 04:30 KST (cron_run.sh 직후) 실행
# 1. 시그널 사후 추적 (scalp_retrospective.py)
# 2. DB 정리 (db_cleanup.py)
# ──────────────────────────────────────────────────────────

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# .env 로드
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Python
PYTHON="$PROJECT_DIR/.venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
  PYTHON=$(which python3)
fi

LOG_DIR="$PROJECT_DIR/logs/maintenance"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/maintenance_$(date +%Y%m%d).log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 일일 유지보수 시작" >> "$LOG_FILE"

# 1. 시그널 사후 추적
echo "[$(date '+%H:%M:%S')] 사후 추적 실행" >> "$LOG_FILE"
$PYTHON scripts/scalp_retrospective.py >> "$LOG_FILE" 2>&1 || true

# 2. DB 정리
echo "[$(date '+%H:%M:%S')] DB 정리 실행" >> "$LOG_FILE"
$PYTHON scripts/db_cleanup.py >> "$LOG_FILE" 2>&1 || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 일일 유지보수 완료" >> "$LOG_FILE"

# 오래된 로그 정리 (30일+)
find "$LOG_DIR" -name "maintenance_*.log" -mtime +30 -delete 2>/dev/null || true
