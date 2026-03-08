#!/bin/bash
# =============================================================================
# Remote Control Watchdog v3
#
# 3분마다 실행 (LaunchAgent: com.claude.watchdog-remote)
#
# 설계 철학: "치료보다 예방"
#   - 5분마다 경량 킵얼라이브 → 10분 타임아웃에 절대 걸리지 않게 함
#   - reconnecting 발생 시에도 우아한 복구 우선 (재시작은 최후 수단)
#
# 복구 우선순위:
#   1) active → 5분마다 킵얼라이브 (Escape로 경량 ping)
#   2) reconnecting → disconnect → /rc 재연결 (Claude 유지)
#   3) reconnecting 재연결 실패 → Claude 완전 재시작 (최후 수단)
#   4) Claude 없음 → 새로 시작
#   5) tmux 없음 → startup.sh
# =============================================================================

PROJECT_DIR="/Users/drj00/workspace/blockchain"
REMOTE_URL_FILE="$PROJECT_DIR/data/remote_url.txt"
LOG_FILE="$PROJECT_DIR/logs/watchdog.log"
KEEPALIVE_FILE="$PROJECT_DIR/data/.rc_keepalive_ts"
BRIDGE_FILE="$HOME/.claude/projects/-Users-drj00-workspace-blockchain/bridge-pointer.json"
TMUX_SESSION="blockchain"
TMUX_WINDOW="claude"
KEEPALIVE_INTERVAL=300   # 5분 (10분 타임아웃의 절반 → 안전 마진 충분)

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$(dirname "$LOG_FILE")" "$PROJECT_DIR/data"

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
    url=$(echo "$pane_output" | grep -o 'https://claude\.ai/code/session_[A-Za-z0-9]*' | tail -1)
    if [ -z "$url" ]; then
        url=$(echo "$pane_output" | grep -o 'https://claude\.ai/code?bridge=env_[A-Za-z0-9_]*' | tail -1)
    fi
    echo "$url"
}

get_rc_status() {
    # tmux 화면 전체에서 Remote Control 상태를 읽는다
    local screen
    screen=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p 2>/dev/null)

    if echo "$screen" | grep -q "Remote Control active"; then
        echo "active"
    elif echo "$screen" | grep -q "reconnecting"; then
        echo "reconnecting"
    else
        echo "none"
    fi
}

# ─── rating 프롬프트 자동 dismiss ───
# 세션 종료 후 "1: Bad  2: Fine  3: Good  0: Dismiss" 프롬프트가 뜨면
# 입력 대기 상태로 RC가 타임아웃된다. 0(Dismiss)을 자동 전송하여 해소.
dismiss_rating_prompt() {
    local screen
    screen=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p 2>/dev/null)
    if echo "$screen" | grep -q "1: Bad.*2: Fine.*3: Good.*0: Dismiss"; then
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" '0' Enter
        sleep 2
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" C-u
        log "OK: Rating prompt auto-dismissed"
        return 0
    fi
    return 1
}

# ─── 경량 킵얼라이브 ───
# /rc → 자동완성/메뉴 뜨면 Escape → 세션에 영향 없이 bridge traffic만 발생
send_keepalive() {
    # rating 프롬프트 먼저 해소
    dismiss_rating_prompt

    # 메뉴가 떠있으면 먼저 닫기
    local pane_check
    pane_check=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p -S -5 2>/dev/null)
    if echo "$pane_check" | grep -q "Disconnect\|Show QR\|Enter to select"; then
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Escape
        sleep 1
    fi

    # Escape 키만 전송 → bridge 통신 발생, 세션 오염 없음
    # 아무 메뉴도 없으면 Escape는 무시됨 (no-op)
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Escape
    sleep 1

    echo "$(date +%s)" > "$KEEPALIVE_FILE"
}

# ─── 우아한 복구: reconnecting → disconnect → 재연결 (Claude 유지) ───
graceful_reconnect() {
    log "ACTION: Graceful reconnect (disconnect → /rc)"

    # 0) rating 프롬프트 먼저 해소
    dismiss_rating_prompt

    # 1) stale bridge 파일 삭제
    rm -f "$BRIDGE_FILE"

    # 2) 무의미한 메뉴 조작 대신 안전한 탈출 시도
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" C-c
    sleep 2
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" C-d
    sleep 2
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "/exit" Enter
    sleep 5

    # 3) 쉘 복귀 성공 여부 판단
    local post_exit
    post_exit=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p -S -3 2>/dev/null)
    if ! echo "$post_exit" | grep -qE '(\$|%)\s*$|drj00@'; then
        log "WARN: Graceful reconnect failed to exit to shell"
        return 1
    fi

    # 4) 터미널 청소 후 재실행
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" C-u C-l
    sleep 1
    
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "unset CLAUDECODE && claude --dangerously-skip-permissions" Enter
    sleep 20

    # 5) 다시 /rc로 새 연결
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "/rc" Enter
    sleep 12

    # 자동완성 메뉴 대응
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Enter
    sleep 8

    # 6) 결과 확인
    local new_status
    new_status=$(get_rc_status)
    local new_url
    new_url=$(extract_url)

    if [ "$new_status" = "active" ] && [ -n "$new_url" ]; then
        echo "$new_url" > "$REMOTE_URL_FILE"
        echo "$(date +%s)" > "$KEEPALIVE_FILE"
        log "OK: Graceful reconnect succeeded. URL: $new_url"
        local old_url
        old_url=$(cat "$REMOTE_URL_FILE" 2>/dev/null)
        if [ "$new_url" != "$old_url" ]; then
            send_telegram "🔄 Watchdog: 우아한 재연결 성공
$new_url"
        fi
        return 0
    fi

    log "WARN: Graceful reconnect failed (status: $new_status)"
    return 1
}

# ─── 최후 수단: Claude 완전 재시작 ───
restart_claude_with_rc() {
    local reason="$1"
    log "ACTION: Full restart - last resort ($reason)"

    # 0) rating 프롬프트 먼저 해소
    dismiss_rating_prompt

    # 1) 확실한 프로세스 강제 종료 (kill -9)
    local PANE_PID
    PANE_PID=$(tmux list-panes -t "$TMUX_SESSION:$TMUX_WINDOW" -F '#{pane_pid}' 2>/dev/null)
    if [ -n "$PANE_PID" ]; then
        local CLAUDE_RUNNING
        CLAUDE_RUNNING=$(pgrep -P "$PANE_PID" -f "claude" 2>/dev/null)
        if [ -n "$CLAUDE_RUNNING" ]; then
            kill -9 $CLAUDE_RUNNING 2>/dev/null
            log "OK: Killed stuck Claude process ($CLAUDE_RUNNING)"
            sleep 3
        fi
    fi

    # 2) stale bridge 파일 삭제
    rm -f "$BRIDGE_FILE"

    # 3) 터미널 프롬프트 완벽 초기화 (C-c: 인터럽트, C-u: 줄지우기, C-l: 화면지우기)
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" C-c C-u C-l
    sleep 2

    # 4) 새 Claude 시작
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "unset CLAUDECODE && claude --dangerously-skip-permissions" Enter
    sleep 20

    # 5) /rc 활성화
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "/rc" Enter
    sleep 12

    # 자동완성 대응
    tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Enter
    sleep 8

    # 6) 결과 확인
    local new_status
    new_status=$(get_rc_status)
    local new_url
    new_url=$(extract_url)

    if [ "$new_status" = "active" ] && [ -n "$new_url" ]; then
        echo "$new_url" > "$REMOTE_URL_FILE"
        echo "$(date +%s)" > "$KEEPALIVE_FILE"
        log "OK: Full restart succeeded. URL: $new_url"
        send_telegram "🔄 Watchdog: 재시작 완료 ($reason)
Remote Control active
$new_url"
    else
        log "ERROR: Full restart failed. Status: $new_status"
        send_telegram "⚠️ Watchdog: 재시작 실패 ($reason)
Status: $new_status
수동 확인 필요"
    fi
}

# =============================================================================
# 메인 로직
# =============================================================================

# --- 0. rating 프롬프트 체크 (RC 블록 방지) ---
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    dismiss_rating_prompt
fi

# --- 1. tmux 세션 확인 ---
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "WARN: tmux session not found. Running startup.sh..."
    bash "$PROJECT_DIR/scripts/startup.sh"
    exit 0
fi

# --- 2. claude 윈도우 확인 ---
if ! tmux list-windows -t "$TMUX_SESSION" -F '#{window_name}' 2>/dev/null | grep -q "^${TMUX_WINDOW}"; then
    log "WARN: tmux window '$TMUX_WINDOW' not found. Recreating..."
    tmux new-window -t "$TMUX_SESSION" -n "$TMUX_WINDOW" -c "$PROJECT_DIR"
    sleep 1
    restart_claude_with_rc "Claude window recreated"
    exit 0
fi

# --- 3. Claude 프로세스 확인 ---
PANE_PID=$(tmux list-panes -t "$TMUX_SESSION:$TMUX_WINDOW" -F '#{pane_pid}' 2>/dev/null)
if [ -n "$PANE_PID" ]; then
    CLAUDE_RUNNING=$(pgrep -P "$PANE_PID" -f "claude" 2>/dev/null)
    if [ -z "$CLAUDE_RUNNING" ]; then
        log "WARN: Claude not running. Starting..."
        restart_claude_with_rc "Claude process not found"
        exit 0
    fi
fi

# --- 4. Remote Control 상태 판단 ---
RC_STATUS=$(get_rc_status)

case "$RC_STATUS" in
    active)
        # ✅ 정상 — 킵얼라이브만 (5분 간격)
        LAST_PING=0
        [ -f "$KEEPALIVE_FILE" ] && LAST_PING=$(cat "$KEEPALIVE_FILE" 2>/dev/null)
        NOW=$(date +%s)
        ELAPSED=$(( NOW - LAST_PING ))

        if [ "$ELAPSED" -ge "$KEEPALIVE_INTERVAL" ]; then
            send_keepalive
            log "OK: Remote active (keepalive sent)"
        else
            log "OK: Remote active (next keepalive in $(( KEEPALIVE_INTERVAL - ELAPSED ))s)"
        fi
        ;;

    reconnecting)
        # ⚠️ 장애 — 우아한 복구 시도 → 실패 시에만 재시작
        log "WARN: Remote Control reconnecting. Trying graceful reconnect..."

        if graceful_reconnect; then
            log "OK: Recovered from reconnecting without restart"
        else
            log "WARN: Graceful reconnect failed. Full restart as last resort."
            restart_claude_with_rc "reconnecting → graceful failed"
        fi
        ;;

    none)
        # Remote Control 없음 → /rc 시도
        log "WARN: No Remote Control. Trying /rc..."

        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" "/rc" Enter
        sleep 12
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" Enter
        sleep 5

        NEW_STATUS=$(get_rc_status)
        if [ "$NEW_STATUS" = "active" ]; then
            NEW_URL=$(extract_url)
            [ -n "$NEW_URL" ] && echo "$NEW_URL" > "$REMOTE_URL_FILE"
            echo "$(date +%s)" > "$KEEPALIVE_FILE"
            log "OK: Remote Control activated. URL: $NEW_URL"
            send_telegram "✅ Watchdog: Remote Control 활성화
$NEW_URL"
        elif [ "$NEW_STATUS" = "reconnecting" ]; then
            log "WARN: /rc → reconnecting. Trying graceful reconnect..."
            if graceful_reconnect; then
                log "OK: Recovered via graceful reconnect"
            else
                restart_claude_with_rc "/rc → reconnecting → graceful failed"
            fi
        else
            restart_claude_with_rc "Remote Control failed to activate"
        fi
        ;;
esac
