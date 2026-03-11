# ──────────────────────────────────────────────────────────
# 초단타 봇 래퍼 — 크래시 시 자동 재시작 (최대 5회/일)
#
# 작업 스케줄러에서 매일 00:05 실행.
# 봇이 종료되면 30초 대기 후 재시작 (일일 최대 5회).
# DRY_RUN 모드로 실행.
#
# 수동 실행:
#   powershell -ExecutionPolicy Bypass -File scripts\win_short_term_wrapper.ps1
# ──────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectDir "scripts\short_term_trader.py"
$LogDir = Join-Path $ProjectDir "logs\short_term"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$MaxRestarts = 5
$RestartCount = 0
$RestartDelay = 30  # 초

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 초단타 봇 래퍼 시작 (DRY_RUN)"

while ($RestartCount -lt $MaxRestarts) {
    $RestartCount++
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 봇 시작 ($RestartCount/$MaxRestarts)..."

    try {
        & $Python $Script --dry-run 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "wrapper_$(Get-Date -Format 'yyyyMMdd').log") -Append
        $exitCode = $LASTEXITCODE
    } catch {
        $exitCode = -1
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 예외: $_"
    }

    if ($exitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 봇 정상 종료."
        break
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 봇 비정상 종료 (exit $exitCode). ${RestartDelay}초 후 재시작..."

    # 텔레그램 알림
    try {
        & $Python scripts\notify_telegram.py error "초단타 봇 크래시" "exit=$exitCode, 재시작 $RestartCount/$MaxRestarts" 2>$null
    } catch {}

    Start-Sleep -Seconds $RestartDelay
}

if ($RestartCount -ge $MaxRestarts) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 재시작 한도 초과 ($MaxRestarts회). 내일까지 대기."
    try {
        & $Python scripts\notify_telegram.py error "초단타 봇 중단" "일일 재시작 한도 ${MaxRestarts}회 초과. 내일 00:05 자동 재시작." 2>$null
    } catch {}
}
