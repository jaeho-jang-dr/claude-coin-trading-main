#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# cron 실행 래퍼
#
# [2026-03-09 변경] Python 에이전트 파이프라인(run_agents.py)을
# PRIMARY 실행 경로로 사용한다. run_agents.py가 실패할 경우에만
# 기존 bash 파이프라인(run_analysis.sh | claude -p)으로 FALLBACK한다.
#
# Python 파이프라인: run_agents.py
#   - 데이터 수집, 전략 전환, 매매 판단, Supabase 기록, 텔레그램 알림 모두 포함
#   - claude -p 호출 불필요 (Orchestrator가 직접 판단)
#
# Bash 파이프라인 (fallback): run_analysis.sh → claude -p
#   - Python 파이프라인 실패 시에만 사용
#
# crontab/LaunchAgent 등록:
#   0 0,4,8,12,16,20 * * * /path/to/blockchain/scripts/cron_run.sh
# ──────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# .env 로드
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Python 가상환경 활성화
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# 긴급 정지 확인
if [ "${EMERGENCY_STOP:-false}" = "true" ]; then
  echo "[$(date)] EMERGENCY_STOP 활성화됨. 실행 중단." >&2
  exit 0
fi

# 로그 디렉토리 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/executions"
RESPONSE_DIR="logs/claude_responses"
mkdir -p "$LOG_DIR" "$RESPONSE_DIR"

LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
RESPONSE_FILE="${RESPONSE_DIR}/${TIMESTAMP}.txt"

echo "[$(date)] === cron 실행 시작 ===" > "$LOG_FILE"

# 에러 발생 시 텔레그램 알림
notify_error() {
  local msg="$1"
  echo "[$(date)] ERROR: ${msg}" >> "$LOG_FILE"
  python3 scripts/notify_telegram.py error "cron 실행 오류" "$msg" 2>/dev/null || true
}

# ══════════════════════════════════════════════════════════
# PRIMARY: Python 에이전트 파이프라인 (run_agents.py)
# ══════════════════════════════════════════════════════════
echo "[$(date)] Python 에이전트 파이프라인 시작..." >> "$LOG_FILE"

AGENT_OUTPUT=""
AGENT_SUCCESS=false

if AGENT_OUTPUT=$(python3 scripts/run_agents.py 2>>"$LOG_FILE"); then
  AGENT_SUCCESS=true
  echo "$AGENT_OUTPUT" > "$RESPONSE_FILE"
  echo "[$(date)] Python 에이전트 파이프라인 성공" >> "$LOG_FILE"
  echo "[$(date)] 응답 저장: ${RESPONSE_FILE}" >> "$LOG_FILE"
else
  echo "[$(date)] WARNING: Python 에이전트 파이프라인 실패 (exit $?). Bash fallback으로 전환..." >> "$LOG_FILE"
fi

# ══════════════════════════════════════════════════════════
# FALLBACK: 기존 Bash 파이프라인 (run_analysis.sh | claude -p)
# Python 파이프라인 실패 시에만 실행
# ══════════════════════════════════════════════════════════
if [ "$AGENT_SUCCESS" = "false" ]; then
  echo "[$(date)] Bash fallback 파이프라인 시작..." >> "$LOG_FILE"
  notify_error "Python 파이프라인 실패 → Bash fallback 실행 중"

  # 1. 데이터 수집 + 프롬프트 생성
  echo "[$(date)] 데이터 수집 중..." >> "$LOG_FILE"
  PROMPT=$(bash scripts/run_analysis.sh 2>>"$LOG_FILE") || true

  if [ -z "$PROMPT" ]; then
    notify_error "Fallback도 실패 — 프롬프트 생성 실패"
    exit 1
  fi

  echo "[$(date)] 프롬프트 생성 완료 ($(echo "$PROMPT" | wc -c) bytes)" >> "$LOG_FILE"

  # 2. claude -p 실행
  echo "[$(date)] claude -p 분석 시작..." >> "$LOG_FILE"
  RESPONSE=$(echo "$PROMPT" | claude -p --dangerously-skip-permissions --allowedTools "Bash(python3:*)" 2>>"$LOG_FILE") || {
    notify_error "Fallback claude -p 실행 실패"
    exit 1
  }

  # 3. 응답 저장
  echo "$RESPONSE" > "$RESPONSE_FILE"
  echo "[$(date)] claude 응답 저장: ${RESPONSE_FILE}" >> "$LOG_FILE"

  # 4. Supabase에 결정 기록 + 이전 결정 성과 업데이트
  echo "[$(date)] Supabase 저장 중..." >> "$LOG_FILE"
  echo "$RESPONSE" | python3 scripts/save_decision.py 2>>"$LOG_FILE" || {
    echo "[$(date)] WARNING: Supabase 저장 실패 (치명적 아님)" >> "$LOG_FILE"
  }
fi

# 완료
echo "[$(date)] === cron 실행 완료 ===" >> "$LOG_FILE"
