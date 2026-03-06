# ──────────────────────────────────────────────────────────
# Windows 작업 스케줄러 등록/해제/상태 도우미
#
# 사용법:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_task_scheduler.ps1 install [간격]
#   powershell -ExecutionPolicy Bypass -File scripts\setup_task_scheduler.ps1 status
#   powershell -ExecutionPolicy Bypass -File scripts\setup_task_scheduler.ps1 remove
#   powershell -ExecutionPolicy Bypass -File scripts\setup_task_scheduler.ps1 run
#
# 실행 시각: 09:00, 17:00, 01:00 (하루 3회)
# ──────────────────────────────────────────────────────────

param(
    [Parameter(Position=0)]
    [ValidateSet("install", "status", "remove", "run")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$TaskName = "CryptoCoinTrader"
$TaskDescription = "Claude 암호화폐 자동매매 - 매일 09:00, 17:00, 01:00 실행"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$WrapperScript = Join-Path $ProjectDir "scripts\win_cron_run.ps1"

# 고정 실행 시각
$ScheduleTimes = @("09:00", "17:00", "01:00")

function Show-Status {
    Write-Host "`n=== 작업 스케줄러 상태 ===" -ForegroundColor Cyan

    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue

        Write-Host "  작업 이름:   $($task.TaskName)" -ForegroundColor Green
        Write-Host "  상태:       $($task.State)"
        Write-Host "  설명:       $($task.Description)"

        if ($task.Triggers.Count -gt 0) {
            $times = @()
            foreach ($t in $task.Triggers) {
                if ($t.StartBoundary) {
                    $dt = [datetime]$t.StartBoundary
                    $times += $dt.ToString('HH:mm')
                }
            }
            if ($times.Count -gt 0) {
                Write-Host "  실행 시각:  $($times -join ', ')"
            }
        }

        if ($info) {
            if ($info.LastRunTime -and $info.LastRunTime.Year -gt 1999) {
                Write-Host "  마지막 실행: $($info.LastRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"
            } else {
                Write-Host "  마지막 실행: (아직 실행되지 않음)"
            }
            Write-Host "  마지막 결과: $($info.LastTaskResult)"
            if ($info.NextRunTime -and $info.NextRunTime.Year -gt 1999) {
                Write-Host "  다음 실행:   $($info.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"
            }
        }
    }
    catch {
        Write-Host "  등록된 작업 없음" -ForegroundColor Yellow
    }

    # .env에서 안전장치 상태 표시
    $envFile = Join-Path $ProjectDir ".env"
    if (Test-Path $envFile) {
        Write-Host "`n=== 안전장치 상태 ===" -ForegroundColor Cyan
        $envContent = Get-Content $envFile
        foreach ($line in $envContent) {
            if ($line -match '^\s*(DRY_RUN|EMERGENCY_STOP|MAX_TRADE_AMOUNT|MAX_DAILY_TRADES)\s*=\s*(.*)$') {
                $key = $Matches[1]
                $val = $Matches[2].Trim()
                $color = "White"
                if ($key -eq "DRY_RUN" -and $val -eq "false") { $color = "Red" }
                if ($key -eq "EMERGENCY_STOP" -and $val -eq "true") { $color = "Red" }
                Write-Host "  ${key}: $val" -ForegroundColor $color
            }
        }
    }
    Write-Host ""
}

function Install-Task {
    Write-Host "`n=== 작업 스케줄러 등록 ===" -ForegroundColor Cyan
    Write-Host "  실행 시각: $($ScheduleTimes -join ', ')" -ForegroundColor Yellow

    # 기존 작업 제거
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "  기존 작업 제거됨" -ForegroundColor Yellow
    } catch {}

    # 래퍼 스크립트 존재 확인
    if (-not (Test-Path $WrapperScript)) {
        Write-Error "래퍼 스크립트를 찾을 수 없습니다: $WrapperScript"
        exit 1
    }

    # 실행 액션 구성
    $argString = '-ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $WrapperScript + '"'
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument $argString `
        -WorkingDirectory $ProjectDir

    # 트리거 구성: 매일 고정 시각 3회 (09:00, 17:00, 01:00)
    $triggers = @()
    foreach ($time in $ScheduleTimes) {
        $triggers += New-ScheduledTaskTrigger -Daily -At $time
    }

    # 설정 (놓친 실행은 무시 - StartWhenAvailable 미설정)
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RunOnlyIfNetworkAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
        -RestartCount 1 `
        -RestartInterval (New-TimeSpan -Minutes 5)

    # 등록 (현재 사용자 권한)
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $action `
        -Trigger $triggers `
        -Settings $settings `
        -RunLevel Limited

    Write-Host "`n  등록 완료!" -ForegroundColor Green
    Write-Host "  작업 이름: $TaskName"
    Write-Host "  실행 시각: $($ScheduleTimes -join ', ')"
    Write-Host "  놓친 실행: 무시 (다음 정시에 실행)"
    Write-Host "  래퍼 스크립트: $WrapperScript"
    Write-Host ""

    # 등록 후 상태 표시
    Show-Status
}

function Remove-Task {
    Write-Host "`n=== 작업 스케줄러 해제 ===" -ForegroundColor Cyan

    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "  '$TaskName' 작업이 제거되었습니다." -ForegroundColor Green
    }
    catch {
        Write-Host "  등록된 작업이 없습니다." -ForegroundColor Yellow
    }
    Write-Host ""
}

function Run-Now {
    Write-Host "`n=== 수동 실행 ===" -ForegroundColor Cyan

    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "  '$TaskName' 작업을 수동 실행했습니다." -ForegroundColor Green
        Write-Host "  결과는 logs\executions\ 에서 확인하세요." -ForegroundColor Yellow
    }
    catch {
        Write-Host "  등록된 작업이 없습니다. 먼저 install을 실행하세요." -ForegroundColor Red
    }
    Write-Host ""
}

# 메인 실행
switch ($Action) {
    "install" { Install-Task }
    "status"  { Show-Status }
    "remove"  { Remove-Task }
    "run"     { Run-Now }
}
