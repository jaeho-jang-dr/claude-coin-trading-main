#!/bin/bash
# =============================================================================
# Remote Control Watchdog
# Claude Code의 remote-control 상태를 감시하고 자동 재연결
# bridge-pointer.json 존재 여부로 리모트 상태를 정확히 판단
# LaunchAgent로 3분마다 실행
# =============================================================================

PROJECT_DIR="/Users/drj00/workspace/blockchain"
REMOTE_URL_FILE="$PROJECT_DIR/data/remote_url.txt"
LOG_FILE="$PROJECT_DIR/logs/watchdog.log"
BRIDGE_FILE="$HOME/.claude/projects/-Users-drj00-workspace-blockchain/bridge-pointer.json"
TMUX_SESSION="blockchain"
TMUX_WINDOW="claude"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

send_telegram() {
    source "$PROJECT_DIR/.env" 2>/dev/null
    [ -z "$TELEGRAM_BOT_TOKEN" ] && return
    curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_USER_ID}" \
        --data-urlencode "text=$1" \
        -d "disable_web_page_preview=true" > /dev/null 2>&1
}

extract_url() {
    local pane_output
    pane_output=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p -S -30 2>/dev/null)
    local url
    # bridge=env_ 형식 (최신)
    url=$(echo "$pane_output" | grep -o 'https://claude\.ai/code?bridge=env_[A-Za-z0-9_]*' | tail -1)
    if [ -z "$url" ]; then
        # session_ 형식 (구형)
        url=$(echo "$pane_output" | grep -o 'https://claude\.ai/code/session_[A-Za-z0-9]*' | tail -1)
    fi
    echo "$url"
}

start_claude_and_rc() {
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "unset CLAUDECODE && claude --dangerously-skip-permissions" Enter
    sleep 20
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "/rc" Enter
    sleep 10
    local new_url
    new_url=$(extract_url)
    if [ -n "$new_url" ]; then
        echo "$new_url" > "$REMOTE_URL_FILE"
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Enter
        send_telegram "Watchdog: $1
$new_url"
        log "OK: $1. URL: $new_url"
    else
        log "ERROR: Could not get remote URL after $1"
    fi
}

# --- 1. tmux 세션 존재 확인 ---
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "WARN: tmux session not found. Running startup.sh..."
    bash "$PROJECT_DIR/scripts/startup.sh"
    exit 0
fi

# --- 2. claude 윈도우 존재 확인 ---
if ! tmux list-windows -t "$TMUX_SESSION" -F '#{window_name}' 2>/dev/null | grep -q "^${TMUX_WINDOW}"; then
    log "WARN: tmux window '$TMUX_WINDOW' not found. Recreating..."
    tmux new-window -t "$TMUX_SESSION" -n "$TMUX_WINDOW" -c "$PROJECT_DIR"
    sleep 1
    start_claude_and_rc "Claude window recreated"
    exit 0
fi

# --- 3. Claude 프로세스 생존 확인 ---
PANE_PID=$(tmux list-panes -t "$TMUX_SESSION:$TMUX_WINDOW" -F '#{pane_pid}' 2>/dev/null)
if [ -n "$PANE_PID" ]; then
    CLAUDE_RUNNING=$(pgrep -P "$PANE_PID" -f "claude" 2>/dev/null)
    if [ -z "$CLAUDE_RUNNING" ]; then
        log "WARN: Claude not running in pane. Restarting..."
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" C-c
        sleep 2
        start_claude_and_rc "Claude restarted"
        exit 0
    fi
fi

# --- 4. Remote Control 활성 상태 확인 (bridge-pointer.json) ---
if [ -f "$BRIDGE_FILE" ]; then
    log "OK: Remote active"
    exit 0
fi

# bridge-pointer.json 없음 = 리모트 꺼짐
log "WARN: Remote inactive (no bridge-pointer.json). Reconnecting..."

# /rc 메뉴가 이미 떠있으면 Escape로 빠져나옴
PANE_OUTPUT=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p -S -10 2>/dev/null)
if echo "$PANE_OUTPUT" | grep -q "Remote Control\|Disconnect\|Show QR"; then
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Escape
    sleep 2
fi

# /rc 실행
tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "/rc" Enter
sleep 10

# URL 추출
NEW_URL=$(extract_url)
if [ -n "$NEW_URL" ]; then
    OLD_URL=$(cat "$REMOTE_URL_FILE" 2>/dev/null)
    echo "$NEW_URL" > "$REMOTE_URL_FILE"
    # Continue 선택
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Enter
    sleep 2
    if [ -f "$BRIDGE_FILE" ]; then
        log "OK: Remote reconnected. URL: $NEW_URL"
        if [ "$NEW_URL" != "$OLD_URL" ]; then
            send_telegram "Watchdog: Remote reconnected
$NEW_URL"
        fi
    else
        log "WARN: /rc done but bridge-pointer.json still missing"
    fi
else
    PANE_NOW=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p -S -5 2>/dev/null)
    log "ERROR: Could not extract URL. Screen: $(echo "$PANE_NOW" | tr '\n' '|')"
    if echo "$PANE_NOW" | grep -q "Continue\|Remote Control"; then
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Enter
    fi
fi
