# ──────────────────────────────────────────────────────────
# Windows 작업 스케줄러용 자동매매 실행 래퍼
#
# run_analysis.ps1로 데이터 수집 + 프롬프트 생성 후,
# claude -p에 전달하고, 결과를 로그에 저장한다.
#
# 수동 실행:
#   powershell -ExecutionPolicy Bypass -File scripts\win_cron_run.ps1
# ──────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

# .env 로드
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

# 긴급 정지 확인
if ($env:EMERGENCY_STOP -eq "true") {
    Write-Host "[$(Get-Date)] EMERGENCY_STOP 활성화됨. 실행 중단."
    exit 0
}

# 로그 디렉토리 준비
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\executions"
$ResponseDir = "logs\claude_responses"
New-Item -ItemType Directory -Force -Path $LogDir, $ResponseDir | Out-Null

$LogFile = Join-Path $LogDir "${Timestamp}.log"
$ResponseFile = Join-Path $ResponseDir "${Timestamp}.txt"

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Send-ErrorNotification($msg) {
    Write-Log "ERROR: $msg"
    try {
        & ".venv\Scripts\python.exe" scripts\notify_telegram.py error "자동매매 오류" $msg 2>$null
    } catch {}
}

Write-Log "=== 자동매매 실행 시작 ==="

# 1. 데이터 수집 + 프롬프트 생성
Write-Log "데이터 수집 중..."
try {
    $Prompt = & powershell -ExecutionPolicy Bypass -File scripts\run_analysis.ps1 2>>$LogFile
} catch {
    Send-ErrorNotification "프롬프트 생성 실패: $_"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    Send-ErrorNotification "프롬프트 생성 실패 - 빈 결과"
    exit 1
}

Write-Log "프롬프트 생성 완료 ($($Prompt.Length) chars)"

# 2. claude -p 실행
Write-Log "claude -p 분석 시작..."
try {
    $Response = $Prompt | claude -p --dangerously-skip-permissions --allowedTools "Bash(python3:*),Bash(python:*),Bash(.venv*)" 2>>$LogFile
} catch {
    Send-ErrorNotification "claude -p 실행 실패: $_"
    exit 1
}

# 3. 응답 저장
Set-Content -Path $ResponseFile -Value $Response -Encoding UTF8
Write-Log "claude 응답 저장: $ResponseFile"

# 4. 완료
Write-Log "=== 자동매매 실행 완료 ==="
