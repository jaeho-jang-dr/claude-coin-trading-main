# Remote Control 장애 분석 및 수정 이력

> Claude Code Remote Control이 반복적으로 끊기고, watchdog 자동 복구가 **단 한 번도 성공하지 못하는** 문제.
> 이 문서를 Gemini 등 다른 AI에게 전달하여 근본 원인 분석을 요청한다.

---

## 시스템 구성

```
┌─────────────────────────────────────────────────────┐
│  macOS Mac mini (M2, Darwin 24.6.0)                 │
│                                                     │
│  tmux 세션: "blockchain"                             │
│    ├── 윈도우 "dashboard" → Flask 대시보드            │
│    └── 윈도우 "claude"    → claude CLI + /rc (RC)    │
│                                                     │
│  LaunchAgent (3분 간격):                              │
│    com.claude.watchdog-remote                        │
│    → watchdog_remote.sh (RC 감시 + 킵얼라이브)       │
└─────────────────────────────────────────────────────┘
```

### RC 작동 원리

1. `claude --dangerously-skip-permissions` 실행 (tmux 윈도우)
2. `/rc` 또는 `/remote-control` 입력 → RC 활성화
3. `https://claude.ai/code/session_XXXXX` URL 발급
4. 아이폰/PC 브라우저에서 해당 URL로 원격 제어
5. **10분간 bridge 통신 없으면 자동 타임아웃** → reconnecting 상태

### watchdog_remote.sh 역할

- 3분마다 LaunchAgent가 실행
- RC 상태를 tmux 화면 텍스트에서 파싱 (`Remote Control active` / `reconnecting`)
- active → 5분마다 킵얼라이브 (`/rc` Enter → Escape)
- reconnecting → graceful reconnect (disconnect → /rc) → 실패 시 full restart

---

## 핵심 문제: 자동 복구 성공률 0%

### 통계 (2026-03-07 ~ 2026-03-09, 3일간)

```
총 watchdog 실행: ~1,580회
OK (정상):       617회 (39%)
WARN (경고):     567회 (36%)
ERROR (에러):    377회 (24%)
ACTION (복구시도): 21회

자동 복구 성공:     0회 ← 심각
Graceful 실패:    20회
Full restart 실패: 10회
```

### 날짜별 에러

```
2026-03-07: 109건 ERROR
2026-03-08: 258건 ERROR
2026-03-09:  10건 ERROR (수동 개입으로 해소)
```

### 장애 시간대 패턴

```
00시: 57건 ← 최다 (cron 단타 실행 후?)
01시: 34건
23시: 31건
04시: 31건
03시: 31건
```

밤~새벽에 집중. 사용자 부재 시간에 RC가 끊기면 복구 불가.

---

## 장애 패턴 분류

### 패턴 1: Rating 프롬프트 블로킹 (2026-03-09 확인)

**증상:** Claude 세션 종료/재시작 후 rating 프롬프트 출력

```
● How is Claude doing this session? (optional)
  1: Bad    2: Fine   3: Good   0: Dismiss
```

**문제:** 이 프롬프트가 뜨면 사용자 입력을 기다리며 세션이 블록됨.
watchdog의 `/rc` 키 입력이 rating에 먹혀서 엉뚱한 동작 발생.

**수정 (2026-03-09):** `dismiss_rating_prompt()` 함수 추가 — `0` Enter 자동 전송

```bash
dismiss_rating_prompt() {
    local screen
    screen=$(tmux capture-pane -t "$TMUX_SESSION:$TMUX_WINDOW" -p 2>/dev/null)
    if echo "$screen" | grep -q "1: Bad.*2: Fine.*3: Good.*0: Dismiss"; then
        tmux send-keys -t "$TMUX_SESSION:$TMUX_WINDOW" '0' Enter
        sleep 2
        log "OK: Rating prompt auto-dismissed"
        return 0
    fi
    return 1
}
```

4곳에서 호출: 메인 진입, send_keepalive, graceful_reconnect, restart_claude_with_rc

---

### 패턴 2: Graceful Reconnect 실패 (성공 0건)

**의도한 동작:**
```
reconnecting 감지
→ bridge 파일 삭제
→ /remote-control Enter → Disconnect 선택 (Up Up Enter)
→ /rc Enter → 새 연결
→ active + 새 URL
```

**실패 원인 추정:**
1. `/remote-control` 입력 시 자동완성 메뉴가 뜨는데, 메뉴 항목 순서가 가정과 다를 수 있음
2. `Up Up Enter`로 Disconnect를 선택하는데, 메뉴 구조가 다를 수 있음
3. reconnecting 상태에서 `/remote-control`이 정상 작동하지 않을 수 있음
4. sleep 타이밍이 맞지 않아 키 입력이 누락될 수 있음

**실제 로그:**
```
[00:44:08] ACTION: Graceful reconnect (disconnect → /rc)
[00:44:14] WARN: Graceful reconnect failed (status: none)  ← 6초만에 실패
```

6초면 sleep 합계(5+3+1+12+8=29초)에 한참 못 미침. **중간에 뭔가 짧게 끝남.**

---

### 패턴 3: Full Restart 실패 (성공 0건)

**의도한 동작:**
```
Escape → /exit Enter → 셸로 복귀 대기(8초)
→ bridge 파일 삭제
→ unset CLAUDECODE && claude --dangerously-skip-permissions Enter → 대기(20초)
→ /rc Enter → 대기(12초) → Enter → 대기(8초)
→ active + 새 URL
```

**실패 원인 추정:**
1. `/exit` 후 rating 프롬프트가 떠서 셸 복귀가 안 됨 (← 이번에 수정함)
2. `/exit`의 자동완성이 뜰 수 있음 (/exit → 선택지 목록)
3. 새 claude 시작 후 20초 대기가 부족할 수 있음 (초기 로딩)
4. `/rc` 입력 시 자동완성 `/remote-control` 목록이 뜨는데 Enter 타이밍 문제

**실제 로그:**
```
[00:44:14] ACTION: Full restart - last resort
[00:45:07] ERROR: Full restart failed. Status: reconnecting  ← 53초 소요, 그래도 실패
```

---

### 패턴 4: 킵얼라이브가 RC를 방해 (의심)

**킵얼라이브 동작:**
```bash
tmux send-keys "/rc" Enter    # /rc 입력
sleep 3
tmux send-keys Escape          # 메뉴 닫기
```

**문제:**
- `/rc` 입력 시 이미 RC active 상태면 Disconnect/Show QR 메뉴가 뜸
- 이때 Escape가 제대로 안 먹히면 메뉴가 남아있을 수 있음
- 다음 킵얼라이브 때 메뉴 위에 `/rc`가 입력되면 예측 불가 상태

---

### 패턴 5: 10분 타임아웃 (근본 원인)

RC는 **bridge 통신이 10분간 없으면 자동 연결 해제**. 킵얼라이브 5분 간격이면 이론적으로 안전하지만:

1. 킵얼라이브 자체가 실패하면 (위 패턴 4) 타임아웃 발생
2. watchdog가 3분 간격이라 킵얼라이브는 5분 이상 후에만 발송 → 실제로는 최대 8분 공백
3. 킵얼라이브 `/rc` Enter → Escape가 bridge 통신으로 인정되는지 불확실

---

## 현재 watchdog_remote.sh 전체 코드

```bash
#!/bin/bash
# Remote Control Watchdog v3 + rating dismiss patch
# 3분마다 실행 (LaunchAgent)

PROJECT_DIR="/Users/drj00/workspace/blockchain"
REMOTE_URL_FILE="$PROJECT_DIR/data/remote_url.txt"
LOG_FILE="$PROJECT_DIR/logs/watchdog.log"
KEEPALIVE_FILE="$PROJECT_DIR/data/.rc_keepalive_ts"
BRIDGE_FILE="$HOME/.claude/projects/-Users-drj00-workspace-blockchain/bridge-pointer.json"
TMUX_SESSION="blockchain"
TMUX_WINDOW="claude"
KEEPALIVE_INTERVAL=300  # 5분

get_rc_status() {
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

send_keepalive() {
    dismiss_rating_prompt
    # 메뉴 떠있으면 닫기
    # /rc Enter → bridge 통신 → Escape로 취소
    tmux send-keys "/rc" Enter
    sleep 3
    tmux send-keys Escape
}

graceful_reconnect() {
    dismiss_rating_prompt
    rm -f "$BRIDGE_FILE"
    tmux send-keys Escape; sleep 1
    tmux send-keys "/remote-control" Enter; sleep 5
    # Disconnect 선택 (Up Up Enter)
    tmux send-keys Up Up Enter; sleep 3
    # /rc 재연결
    tmux send-keys "/rc" Enter; sleep 12
    tmux send-keys Enter; sleep 8
    # 결과 확인
}

restart_claude_with_rc() {
    dismiss_rating_prompt
    tmux send-keys Escape; sleep 1
    tmux send-keys "/exit" Enter; sleep 8
    rm -f "$BRIDGE_FILE"
    tmux send-keys "unset CLAUDECODE && claude --dangerously-skip-permissions" Enter; sleep 20
    tmux send-keys "/rc" Enter; sleep 12
    tmux send-keys Enter; sleep 8
    # 결과 확인
}

# 메인:
# 0) rating dismiss
# 1) tmux 세션 확인
# 2) claude 윈도우 확인
# 3) Claude 프로세스 확인
# 4) RC 상태 → active(킵얼라이브) / reconnecting(복구) / none(/rc)
```

---

## 수정 이력

| 날짜 | 버전 | 수정 내용 | 결과 |
|------|------|----------|------|
| 2026-03-07 | v1 | 최초 watchdog: 5분 간격, /rc 재시도만 | 복구 실패 반복 |
| 2026-03-07 | v2 | bridge 파일 삭제 + disconnect 추가 | 복구 실패 반복 |
| 2026-03-08 | v3 | 킵얼라이브 예방 + 우아한 복구 + full restart | 킵얼라이브 OK, 복구 여전히 0% |
| 2026-03-09 | v3.1 | rating 프롬프트 자동 dismiss 추가 | 미검증 (방금 적용) |

---

## 질문 (근본 원인 분석 요청)

1. **왜 자동 복구 성공률이 0%인가?** graceful reconnect도 full restart도 한 번도 성공 못 함. tmux send-keys 기반 자동화의 한계인가?

2. **킵얼라이브가 진짜 RC 타임아웃을 예방하는가?** `/rc` Enter → Escape가 bridge 통신으로 인정되는지 확인 방법은?

3. **근본적으로 다른 접근이 필요한가?**
   - tmux send-keys 대신 Claude Code의 API/CLI 옵션으로 RC를 관리할 수 있나?
   - `--keepalive` 같은 내장 옵션이 있나?
   - bridge-pointer.json을 직접 조작하여 reconnect할 수 있나?

4. **reconnecting 상태에서 `/rc`를 입력하면 정확히 어떤 일이 일어나는가?** 새 bridge가 생성되나, 기존 stale bridge 때문에 실패하나?

5. **rating 프롬프트 외에 다른 블로킹 UI 요소는 없는가?** MCP 인증 프롬프트, 업데이트 알림, 에러 다이얼로그 등.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/watchdog_remote.sh` | watchdog 스크립트 (이 문서의 주인공) |
| `scripts/startup.sh` | 부팅 시 tmux + claude + RC 초기화 |
| `logs/watchdog.log` | watchdog 실행 로그 (1,580줄) |
| `logs/watchdog_launchagent.log` | LaunchAgent stdout |
| `~/.claude/projects/.../bridge-pointer.json` | RC bridge 포인터 |
| `data/remote_url.txt` | 현재 RC URL 저장 |
| `data/.rc_keepalive_ts` | 마지막 킵얼라이브 타임스탬프 |

---

## 환경 정보

```
macOS: Darwin 24.6.0 (Mac mini M2)
Claude Code: v2.1.69
tmux: 3.x
Shell: zsh
LaunchAgent: StartInterval 180 (3분)
```
