# ──────────────────────────────────────────────────────────
# Windows 작업 스케줄러용 자동매매 실행 래퍼
#
# Python 에이전트 파이프라인(run_agents.py)을 실행한다.
# 데이터 수집, 전략 전환, 매매 판단, Supabase 기록, 텔레그램 알림 모두 포함.
#
# 수동 실행:
#   powershell -ExecutionPolicy Bypass -File scripts\win_cron_run.ps1
#
# 작업 스케줄러 등록:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_win_cron.ps1
# ──────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

# UTF-8 강제
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# .env 로드
if (Test-Path ".env") {
    Get-Content ".env" -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

# Python 경로
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

# 긴급 정지 확인
if ($env:EMERGENCY_STOP -eq "true") {
    Write-Host "[$(Get-Date)] EMERGENCY_STOP 활성화됨. 실행 중단."
    exit 0
}

# 로그 디렉토리 준비
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $ProjectDir "logs\executions"
$ResponseDir = Join-Path $ProjectDir "logs\claude_responses"
New-Item -ItemType Directory -Force -Path $LogDir, $ResponseDir | Out-Null

$LogFile = Join-Path $LogDir "${Timestamp}.log"
$ResponseFile = Join-Path $ResponseDir "${Timestamp}.txt"

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

function Send-ErrorNotification($msg) {
    Write-Log "ERROR: $msg"
    try {
        & $Python scripts\notify_telegram.py error "자동매매 오류" $msg 2>$null
    } catch {}
}

Write-Log "=== 자동매매 cron 실행 시작 ==="

# ── PRIMARY: Python 에이전트 파이프라인 ──
Write-Log "Python 에이전트 파이프라인 시작..."

try {
    $AgentOutput = & $Python scripts\run_agents.py 2>&1
    $AgentExit = $LASTEXITCODE

    if ($AgentExit -eq 0) {
        $AgentOutput | Out-File -FilePath $ResponseFile -Encoding UTF8
        Write-Log "Python 에이전트 파이프라인 성공"

        # 회고 분석
        Write-Log "회고 분석 시작..."
        try {
            & $Python scripts\retrospective.py 2>&1 | ForEach-Object { Write-Log $_ }
        } catch {
            Write-Log "WARNING: 회고 분석 실패 (치명적 아님): $_"
        }
    } else {
        Write-Log "Python 에이전트 파이프라인 실패 (exit $AgentExit)"
        Send-ErrorNotification "에이전트 파이프라인 실패 (exit $AgentExit)"

        # FALLBACK: bash 파이프라인
        Write-Log "Bash fallback은 Windows에서 미지원. 다음 실행을 기다립니다."
    }
} catch {
    Write-Log "에이전트 파이프라인 예외: $_"
    Send-ErrorNotification "에이전트 예외: $_"
}

Write-Log "=== 자동매매 cron 실행 완료 ==="
