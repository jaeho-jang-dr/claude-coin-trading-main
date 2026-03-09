#!/bin/bash
# =============================================================================
# 부팅 시 자동 실행 스크립트
# tmux → Claude Code (remote-control) → Dashboard → QR 페이지
# =============================================================================

set -e

PROJECT_DIR="/Users/drj00/workspace/blockchain"
REMOTE_URL_FILE="$PROJECT_DIR/data/remote_url.txt"
LOG_DIR="$PROJECT_DIR/logs"
TMUX_SESSION="blockchain"
DASHBOARD_PORT=5555

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/startup.log"
}

# --- 0. 준비 ---
mkdir -p "$LOG_DIR" "$PROJECT_DIR/data/charts" "$PROJECT_DIR/data/snapshots"
log "=== Startup begin ==="

# --- 0.1 오래된 데이터 자동 정리 ---
find "$PROJECT_DIR/data/snapshots" -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null
find "$PROJECT_DIR/data/charts" -name "*.png" -mtime +7 -delete 2>/dev/null
find "$LOG_DIR" -name "*.log" -size +10M -exec truncate -s 1M {} \; 2>/dev/null
log "Cleanup: old snapshots/charts/logs trimmed"

# --- 1. 기존 세션 정리 ---
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
# 기존 대시보드 프로세스 정리
lsof -i :$DASHBOARD_PORT -t 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# --- 2. tmux 세션 생성 ---
tmux new-session -d -s "$TMUX_SESSION" -n dashboard -c "$PROJECT_DIR"
log "tmux session created: $TMUX_SESSION"

# --- 3. 대시보드 실행 (윈도우 0: dashboard) ---
tmux send-keys -t "$TMUX_SESSION:dashboard" "source .venv/bin/activate && python scripts/dashboard.py $DASHBOARD_PORT" Enter
log "Dashboard starting on port $DASHBOARD_PORT"

# --- 4. Claude Code 원격 세션 (윈도우 1: claude) ---
tmux new-window -t "$TMUX_SESSION" -n claude -c "$PROJECT_DIR"
tmux send-keys -t "$TMUX_SESSION:claude" "unset CLAUDECODE && claude --dangerously-skip-permissions" Enter
log "Claude Code starting..."

# Claude 초기화 대기
sleep 15

# /rc (remote-control) 실행
tmux send-keys -t "$TMUX_SESSION:claude" "/rc" Enter
log "Remote Control connecting..."

# remote-control 연결 대기 및 URL 추출
MAX_WAIT=60
WAITED=0
REMOTE_URL=""

while [ $WAITED -lt $MAX_WAIT ]; do
    sleep 5
    WAITED=$((WAITED + 5))

    # tmux pane에서 URL 추출
    PANE_OUTPUT=$(tmux capture-pane -t "$TMUX_SESSION:claude" -p -S -30 2>/dev/null)
    REMOTE_URL=$(echo "$PANE_OUTPUT" | grep -o 'https://claude.ai/code/session_[A-Za-z0-9]*' | tail -1)

    if [ -n "$REMOTE_URL" ]; then
        log "Remote Control URL: $REMOTE_URL"
        echo "$REMOTE_URL" > "$REMOTE_URL_FILE"
        break
    fi

    log "Waiting for remote URL... (${WAITED}s)"
done

if [ -z "$REMOTE_URL" ]; then
    log "WARNING: Could not capture remote URL after ${MAX_WAIT}s"
    echo "https://claude.ai/code/pending" > "$REMOTE_URL_FILE"
fi

# /rc 메뉴에서 Continue 선택
sleep 2
tmux send-keys -t "$TMUX_SESSION:claude" Enter
log "Remote Control menu dismissed"

# --- 5. 텔레그램으로 새 URL 전송 ---
source "$PROJECT_DIR/.env"
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || echo '127.0.0.1')
DASHBOARD_URL="http://${LOCAL_IP}:${DASHBOARD_PORT}"
QR_URL="${DASHBOARD_URL}/qr"
REMOTE_URL_FINAL=$(cat "$REMOTE_URL_FILE")

MSG="Crypto Bot Started

Claude Code Remote:
${REMOTE_URL_FINAL}

Dashboard:
${DASHBOARD_URL}

QR Page:
${QR_URL}

$(date '+%Y-%m-%d %H:%M:%S')"

curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_USER_ID}" \
    --data-urlencode "text=${MSG}" \
    -d "disable_web_page_preview=true" > /dev/null 2>&1

log "Telegram notification sent"

# --- 6. 완료 ---
log "=== Startup complete ==="
log "  Dashboard: ${DASHBOARD_URL}"
log "  QR page:   ${QR_URL}"
log "  Remote:    ${REMOTE_URL_FINAL}"
log "  tmux:      tmux attach -t $TMUX_SESSION"
