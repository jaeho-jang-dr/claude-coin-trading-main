<#
.SYNOPSIS
    성과 모니터링 스케줄러 등록 (Windows Task Scheduler)

.DESCRIPTION
    1. 매시간: retrospective + 성능 저하 감지
    2. 매주 월요일 09:00: 주간 종합 리포트
    3. 지속학습: 6시간 간격 SB3 PPO 증분 학습

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_monitoring.ps1 install
    powershell -ExecutionPolicy Bypass -File scripts\setup_monitoring.ps1 status
    powershell -ExecutionPolicy Bypass -File scripts\setup_monitoring.ps1 remove
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("install", "status", "remove", "run")]
    [string]$Action = "status"
)

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

$Tasks = @(
    @{
        Name = "CoinTrading_HourlyMonitor"
        Description = "매시간 성과 모니터링 (retrospective + 알림)"
        Script = "scripts\performance_report.py"
        Args = ""
        Trigger = "Hourly"
        Interval = 1
    },
    @{
        Name = "CoinTrading_WeeklyReport"
        Description = "주간 종합 성과 리포트"
        Script = "scripts\performance_report.py"
        Args = "--weekly"
        Trigger = "Weekly"
        DayOfWeek = "Monday"
        Time = "09:00"
    },
    @{
        Name = "CoinTrading_ContinuousLearn"
        Description = "SB3 PPO 지속학습 (6시간 간격)"
        Script = "scripts\continuous_learn_sb3.py"
        Args = "--steps 50000 --days 30"
        Trigger = "Hourly"
        Interval = 6
    }
)

function Install-Tasks {
    Write-Host "`n=== 모니터링 태스크 등록 ===" -ForegroundColor Cyan

    foreach ($task in $Tasks) {
        $taskName = $task.Name
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

        if ($existing) {
            Write-Host "  [SKIP] $taskName (이미 등록됨)" -ForegroundColor Yellow
            continue
        }

        $scriptPath = Join-Path $ProjectDir $task.Script
        $arguments = "-u `"$scriptPath`" $($task.Args)"

        $action = New-ScheduledTaskAction `
            -Execute $Python `
            -Argument $arguments `
            -WorkingDirectory $ProjectDir

        if ($task.Trigger -eq "Hourly") {
            $trigger = New-ScheduledTaskTrigger -Once -At "00:00" `
                -RepetitionInterval (New-TimeSpan -Hours $task.Interval) `
                -RepetitionDuration (New-TimeSpan -Days 365)
        } elseif ($task.Trigger -eq "Weekly") {
            $trigger = New-ScheduledTaskTrigger -Weekly `
                -DaysOfWeek $task.DayOfWeek `
                -At $task.Time
        }

        $settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
            -MultipleInstances IgnoreNew

        Register-ScheduledTask `
            -TaskName $taskName `
            -Description $task.Description `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -RunLevel Highest `
            -Force | Out-Null

        Write-Host "  [OK] $taskName - $($task.Description)" -ForegroundColor Green
    }

    Write-Host "`n등록 완료!" -ForegroundColor Green
}

function Show-Status {
    Write-Host "`n=== 모니터링 태스크 상태 ===" -ForegroundColor Cyan

    foreach ($task in $Tasks) {
        $taskName = $task.Name
        $st = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

        if ($st) {
            $info = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
            $lastRun = if ($info.LastRunTime -and $info.LastRunTime.Year -gt 2000) {
                $info.LastRunTime.ToString("yyyy-MM-dd HH:mm")
            } else { "아직 없음" }
            $nextRun = if ($info.NextRunTime -and $info.NextRunTime.Year -gt 2000) {
                $info.NextRunTime.ToString("yyyy-MM-dd HH:mm")
            } else { "-" }
            $status = $st.State

            $color = switch ($status) {
                "Ready"   { "Green" }
                "Running" { "Yellow" }
                "Disabled" { "Red" }
                default   { "White" }
            }

            Write-Host "  [$status] " -ForegroundColor $color -NoNewline
            Write-Host "$taskName" -NoNewline
            Write-Host " | 마지막: $lastRun | 다음: $nextRun"
        } else {
            Write-Host "  [없음] $taskName" -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

function Remove-Tasks {
    Write-Host "`n=== 모니터링 태스크 제거 ===" -ForegroundColor Cyan

    foreach ($task in $Tasks) {
        $taskName = $task.Name
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

        if ($existing) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            Write-Host "  [삭제] $taskName" -ForegroundColor Red
        } else {
            Write-Host "  [없음] $taskName" -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

function Run-Now {
    Write-Host "`n=== 수동 실행 ===" -ForegroundColor Cyan
    Write-Host "  매시간 모니터링 실행 중..."
    & $Python -u (Join-Path $ProjectDir "scripts\performance_report.py")
    Write-Host "  완료!" -ForegroundColor Green
}

switch ($Action) {
    "install" { Install-Tasks; Show-Status }
    "status"  { Show-Status }
    "remove"  { Remove-Tasks }
    "run"     { Run-Now }
}
