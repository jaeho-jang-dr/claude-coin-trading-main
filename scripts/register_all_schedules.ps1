# RL 학습 + 모니터링 전체 스케줄 등록
# 관리자 권한으로 실행: powershell -ExecutionPolicy Bypass -File scripts/register_all_schedules.ps1

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python not found at $Python" -ForegroundColor Red
    exit 1
}

Write-Host "=== RL Training & Monitoring Schedule Registration ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host "Python:  $Python"
Write-Host ""

# ─── Helper ───
function Register-Task {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Args,
        $Trigger,
        [int]$TimeLimitMin = 30,
        [string]$Description = ""
    )

    $action = New-ScheduledTaskAction `
        -Execute $Python `
        -Argument "-u `"$Script`" $Args" `
        -WorkingDirectory $ProjectDir

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $TimeLimitMin) `
        -MultipleInstances IgnoreNew

    # Try highest first, fallback to limited if access denied
    try {
        Register-ScheduledTask `
            -TaskName $Name `
            -Description $Description `
            -Action $action `
            -Trigger $Trigger `
            -Settings $settings `
            -RunLevel Highest `
            -Force | Out-Null
    } catch {
        Register-ScheduledTask `
            -TaskName $Name `
            -Description $Description `
            -Action $action `
            -Trigger $Trigger `
            -Settings $settings `
            -Force | Out-Null
    }

    $info = Get-ScheduledTaskInfo -TaskName $Name -ErrorAction SilentlyContinue
    Write-Host "  [OK] $Name" -ForegroundColor Green
    Write-Host "       Next: $($info.NextRunTime)" -ForegroundColor DarkGray
}

# ═══════════════════════════════════════
# 1. RL Training Tiers
# ═══════════════════════════════════════
Write-Host "`n--- RL Training ---" -ForegroundColor Yellow

# Tier 1: 6시간마다 Quick Incremental (50K steps, ~6분)
$t1 = New-ScheduledTaskTrigger -Once -At '00:00' `
    -RepetitionInterval (New-TimeSpan -Hours 6) `
    -RepetitionDuration (New-TimeSpan -Days 365)
Register-Task -Name "CoinTrading_RL_tier1" `
    -Script (Join-Path $ProjectDir "scripts\training_scheduler.py") `
    -Args "tier1" `
    -Trigger $t1 `
    -TimeLimitMin 30 `
    -Description "6시간 간격 빠른 증분 학습 (50K steps)"

# Tier 2: 매일 03:00 Daily Training (150K steps, ~15분)
$t2 = New-ScheduledTaskTrigger -Daily -At '03:00'
Register-Task -Name "CoinTrading_RL_tier2" `
    -Script (Join-Path $ProjectDir "scripts\training_scheduler.py") `
    -Args "tier2" `
    -Trigger $t2 `
    -TimeLimitMin 60 `
    -Description "매일 03:00 중간 강도 학습 (150K steps)"

# Tier 3: 매주 일요일 04:00 Weekly Full Retrain (300K steps, ~30분)
$t3 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '04:00'
Register-Task -Name "CoinTrading_RL_tier3" `
    -Script (Join-Path $ProjectDir "scripts\training_scheduler.py") `
    -Args "tier3" `
    -Trigger $t3 `
    -TimeLimitMin 60 `
    -Description "매주 일요일 04:00 전체 재학습 (300K steps)"

# Tier 4: 매월 1일 02:00 Monthly 1H Intensive (500K steps, ~60분)
$t4 = New-ScheduledTaskTrigger -Once -At '02:00' `
    -RepetitionInterval (New-TimeSpan -Days 30) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Register-Task -Name "CoinTrading_RL_tier4" `
    -Script (Join-Path $ProjectDir "scripts\train_1h_intensive.py") `
    -Args "" `
    -Trigger $t4 `
    -TimeLimitMin 90 `
    -Description "매월 1시간 집중 훈련 5-Phase (500K steps)"

# ═══════════════════════════════════════
# 2. Trading Agent (8시간마다)
# ═══════════════════════════════════════
Write-Host "`n--- Trading Agent ---" -ForegroundColor Yellow

$ta = New-ScheduledTaskTrigger -Once -At '00:00' `
    -RepetitionInterval (New-TimeSpan -Hours 8) `
    -RepetitionDuration (New-TimeSpan -Days 365)
Register-Task -Name "CoinTrading_Agent" `
    -Script (Join-Path $ProjectDir "scripts\run_agents.py") `
    -Args "" `
    -Trigger $ta `
    -TimeLimitMin 15 `
    -Description "8시간 간격 에이전트 매매 실행"

# ═══════════════════════════════════════
# 3. Performance Monitoring
# ═══════════════════════════════════════
Write-Host "`n--- Monitoring ---" -ForegroundColor Yellow

# 매시간 모니터링
$pm = New-ScheduledTaskTrigger -Once -At '00:30' `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 365)
Register-Task -Name "CoinTrading_Monitor_Hourly" `
    -Script (Join-Path $ProjectDir "scripts\performance_report.py") `
    -Args "" `
    -Trigger $pm `
    -TimeLimitMin 10 `
    -Description "매시간 성과 모니터링 + retrospective"

# 주간 리포트: 매주 월요일 09:00
$pw = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At '09:00'
Register-Task -Name "CoinTrading_Report_Weekly" `
    -Script (Join-Path $ProjectDir "scripts\performance_report.py") `
    -Args "--weekly" `
    -Trigger $pw `
    -TimeLimitMin 10 `
    -Description "매주 월요일 09:00 주간 성과 리포트"

# 월간 백테스트: 매월 1일 10:00
$pb = New-ScheduledTaskTrigger -Once -At '10:00' `
    -RepetitionInterval (New-TimeSpan -Days 30) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Register-Task -Name "CoinTrading_Backtest_Monthly" `
    -Script (Join-Path $ProjectDir "scripts\performance_report.py") `
    -Args "--backtest" `
    -Trigger $pb `
    -TimeLimitMin 15 `
    -Description "매월 30일 백테스트 리포트"

# ═══════════════════════════════════════
# Summary
# ═══════════════════════════════════════
Write-Host "`n=== Registration Complete ===" -ForegroundColor Cyan
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
Write-Host "To check status: python scripts/training_scheduler.py status"
Write-Host "To remove all:   Get-ScheduledTask -TaskName 'CoinTrading_*' | Unregister-ScheduledTask -Confirm:`$false"
