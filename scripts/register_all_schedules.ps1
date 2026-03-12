# RL 학습 + 모니터링 전체 스케줄 등록 (창 숨김 모드)
# 실행: powershell -ExecutionPolicy Bypass -File scripts/register_all_schedules.ps1

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$WScript = "wscript.exe"
$VBS = Join-Path $ProjectDir "scripts\run_silent.vbs"

if (-not (Test-Path $VBS)) {
    Write-Host "ERROR: run_silent.vbs not found at $VBS" -ForegroundColor Red
    exit 1
}

Write-Host "=== RL Training & Monitoring Schedule Registration ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host "Mode:    Silent (no console windows)"
Write-Host ""

# ─── Helper: schtasks.exe 사용 (관리자 불필요) ───
function Register-SilentTask {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Args = "",
        [string]$Schedule,       # hourly, daily, weekly, monthly
        [string]$Modifier = "",  # /mo N
        [string]$Days = "",      # /d MON, /d 1
        [string]$StartTime = "00:00",
        [string]$Description = ""
    )

    # VBS 래퍼 인자: "스크립트경로" "인자들"
    $scriptPath = $Script
    if (-not [System.IO.Path]::IsPathRooted($Script)) {
        $scriptPath = Join-Path $ProjectDir "scripts\$Script"
    }

    $taskArgs = """$VBS"" ""$scriptPath"""
    if ($Args -ne "") {
        $taskArgs += " ""$Args"""
    }

    $schtasksArgs = @(
        "/create",
        "/tn", $Name,
        "/tr", "$WScript $taskArgs",
        "/sc", $Schedule,
        "/st", $StartTime,
        "/f"
    )

    if ($Modifier -ne "") {
        $schtasksArgs += "/mo"
        $schtasksArgs += $Modifier
    }
    if ($Days -ne "") {
        $schtasksArgs += "/d"
        $schtasksArgs += $Days
    }

    $result = & schtasks.exe @schtasksArgs 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $Name" -ForegroundColor Green
        Write-Host "       $Description" -ForegroundColor DarkGray
    } else {
        Write-Host "  [FAIL] $Name — $result" -ForegroundColor Red
    }
}

# ═══════════════════════════════════════
# 1. RL Training Tiers
# ═══════════════════════════════════════
Write-Host "`n--- RL Training ---" -ForegroundColor Yellow

# Tier 1: 6시간마다 (50K steps, ~6분)
Register-SilentTask -Name "CoinTrading_RL_tier1" `
    -Script "training_scheduler.py" -Args "tier1" `
    -Schedule "hourly" -Modifier "6" -StartTime "00:00" `
    -Description "6시간 간격 빠른 증분 학습 (50K steps)"

# Tier 2: 매일 03:00 (150K steps, ~15분)
Register-SilentTask -Name "CoinTrading_RL_tier2" `
    -Script "training_scheduler.py" -Args "tier2" `
    -Schedule "daily" -StartTime "03:00" `
    -Description "매일 03:00 중간 강도 학습 (150K steps)"

# Tier 3: 매주 일요일 04:00 (300K steps, ~30분)
Register-SilentTask -Name "CoinTrading_RL_tier3" `
    -Script "training_scheduler.py" -Args "tier3" `
    -Schedule "weekly" -Days "SUN" -StartTime "04:00" `
    -Description "매주 일요일 04:00 전체 재학습 (300K steps)"

# Tier 4: 매월 1일 02:00 (500K steps, ~60분)
Register-SilentTask -Name "CoinTrading_RL_tier4" `
    -Script "train_1h_intensive.py" `
    -Schedule "monthly" -Days "1" -StartTime "02:00" `
    -Description "매월 1시간 집중 훈련 5-Phase (500K steps)"

# ═══════════════════════════════════════
# 2. Trading Agent (8시간마다)
# ═══════════════════════════════════════
Write-Host "`n--- Trading Agent ---" -ForegroundColor Yellow

Register-SilentTask -Name "CoinTrading_Agent" `
    -Script "run_agents.py" `
    -Schedule "hourly" -Modifier "8" -StartTime "00:00" `
    -Description "8시간 간격 에이전트 매매 실행"

# ═══════════════════════════════════════
# 3. Performance Monitoring
# ═══════════════════════════════════════
Write-Host "`n--- Monitoring ---" -ForegroundColor Yellow

# 매시간 모니터링
Register-SilentTask -Name "CoinTrading_Monitor_Hourly" `
    -Script "performance_report.py" `
    -Schedule "hourly" -Modifier "1" -StartTime "00:30" `
    -Description "매시간 성과 모니터링"

# 주간 리포트: 매주 월요일 09:00
Register-SilentTask -Name "CoinTrading_Report_Weekly" `
    -Script "performance_report.py" -Args "--weekly" `
    -Schedule "weekly" -Days "MON" -StartTime "09:00" `
    -Description "매주 월요일 09:00 주간 성과 리포트"

# 월간 백테스트: 매월 1일 10:00
Register-SilentTask -Name "CoinTrading_Backtest_Monthly" `
    -Script "performance_report.py" -Args "--backtest" `
    -Schedule "monthly" -Days "1" -StartTime "10:00" `
    -Description "매월 백테스트 리포트"

# ═══════════════════════════════════════
# Summary
# ═══════════════════════════════════════
Write-Host "`n=== Registration Complete (Silent Mode) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Schedule Overview:" -ForegroundColor White
Write-Host "  [RL] Tier1 Quick:      Every 6 hours      (50K steps, ~6min)" -ForegroundColor DarkCyan
Write-Host "  [RL] Tier2 Daily:      Daily 03:00         (150K steps, ~15min)" -ForegroundColor DarkCyan
Write-Host "  [RL] Tier3 Weekly:     Sunday 04:00         (300K steps, ~30min)" -ForegroundColor DarkCyan
Write-Host "  [RL] Tier4 Monthly:    Monthly 02:00        (500K steps, ~60min)" -ForegroundColor DarkCyan
Write-Host "  [Agent] Trading:       Every 8 hours" -ForegroundColor DarkGreen
Write-Host "  [Monitor] Hourly:      Every hour :30" -ForegroundColor DarkYellow
Write-Host "  [Report] Weekly:       Monday 09:00" -ForegroundColor DarkYellow
Write-Host "  [Report] Backtest:     Monthly 10:00" -ForegroundColor DarkYellow
Write-Host ""
Write-Host "All tasks run via wscript.exe + pythonw.exe = NO console windows" -ForegroundColor Green
Write-Host ""
Write-Host "To check status:  schtasks /query /tn CoinTrading_RL_tier1"
Write-Host "To remove all:    Get-ScheduledTask -TaskName 'CoinTrading_*' | Unregister-ScheduledTask -Confirm:`$false"
