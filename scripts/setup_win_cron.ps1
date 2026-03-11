# ──────────────────────────────────────────────────────────
# Windows 작업 스케줄러에 자동매매 cron 등록/해제
#
# 사용법 (관리자 권한 필요):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_win_cron.ps1              # 등록 (4시간 간격)
#   powershell -ExecutionPolicy Bypass -File scripts\setup_win_cron.ps1 -Interval 8  # 8시간 간격
#   powershell -ExecutionPolicy Bypass -File scripts\setup_win_cron.ps1 -Remove      # 해제
#   powershell -ExecutionPolicy Bypass -File scripts\setup_win_cron.ps1 -Status      # 상태 확인
# ──────────────────────────────────────────────────────────

param(
    [int]$Interval = 4,         # 실행 간격 (시간): 4, 8, 12, 24
    [switch]$Remove,
    [switch]$Status
)

$TaskName = "CoinTrading_AutoTrade"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ScriptPath = Join-Path $ProjectDir "scripts\win_cron_run.ps1"

# ── 상태 확인 ──
if ($Status) {
    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Write-Host "[활성] 자동매매 작업이 등록되어 있습니다." -ForegroundColor Green
        Write-Host ""
        Write-Host "  작업 이름: $TaskName"
        Write-Host "  상태:     $($task.State)"

        $triggers = $task.Triggers
        foreach ($t in $triggers) {
            Write-Host "  트리거:   $($t.CimClass.CimClassName) — 간격: $($t.Repetition.Interval)"
        }

        # 마지막 실행 결과
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
        Write-Host "  마지막 실행: $($info.LastRunTime)"
        Write-Host "  마지막 결과: $($info.LastTaskResult)"
    } catch {
        Write-Host "[미등록] 자동매매 작업이 등록되어 있지 않습니다." -ForegroundColor Yellow
    }
    exit 0
}

# ── 해제 ──
if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "[OK] 자동매매 작업이 해제되었습니다: $TaskName" -ForegroundColor Green
    } catch {
        Write-Host "[FAIL] 해제 실패: $_" -ForegroundColor Red
        Write-Host "  관리자 권한으로 실행해주세요."
    }

    # 주간 재학습도 같이 해제할지 확인
    try {
        $weeklyTask = Get-ScheduledTask -TaskName "CoinTrading_Weekly_Retrain" -ErrorAction Stop
        Write-Host ""
        Write-Host "  주간 재학습(CoinTrading_Weekly_Retrain)도 등록되어 있습니다."
        Write-Host "  해제하려면: powershell scripts\setup_weekly_retrain.py --remove"
    } catch {}

    exit 0
}

# ── 등록 ──
Write-Host "=== 자동매매 작업 스케줄러 등록 ===" -ForegroundColor Cyan
Write-Host ""

# 간격 유효성 검사
$ValidIntervals = @(4, 8, 12, 24)
if ($Interval -notin $ValidIntervals) {
    Write-Host "[FAIL] 유효한 간격: 4, 8, 12, 24 (시간)" -ForegroundColor Red
    exit 1
}

# 기존 작업 제거
try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
} catch {}

# 실행 액션: PowerShell로 win_cron_run.ps1 실행
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$ScriptPath`"" `
    -WorkingDirectory $ProjectDir

# 트리거: 매일 반복
if ($Interval -eq 24) {
    # 24시간 = 매일 1회 (09:00)
    $Trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
} else {
    # N시간 간격 반복 — 매일 00:00 시작, N시간마다
    $Trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
    $Trigger.Repetition = (New-ScheduledTaskTrigger -Once -At "00:00" `
        -RepetitionInterval (New-TimeSpan -Hours $Interval) `
        -RepetitionDuration (New-TimeSpan -Hours 24)).Repetition
}

# 설정
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew

# 등록
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Claude Coin Trading 자동매매 ($($Interval)시간 간격)" `
        -RunLevel Highest `
        -Force | Out-Null

    Write-Host "[OK] 작업 등록 완료!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  작업 이름: $TaskName"
    Write-Host "  실행 간격: ${Interval}시간"
    if ($Interval -eq 24) {
        Write-Host "  실행 시각: 매일 09:00"
    } else {
        $Hours = @()
        for ($h = 0; $h -lt 24; $h += $Interval) { $Hours += "${h}:00" }
        Write-Host "  실행 시각: $($Hours -join ', ')"
    }
    Write-Host "  스크립트:  $ScriptPath"
    Write-Host "  프로젝트:  $ProjectDir"
    Write-Host ""
    Write-Host "확인:" -ForegroundColor Yellow
    Write-Host "  상태 확인: powershell scripts\setup_win_cron.ps1 -Status"
    Write-Host "  수동 실행: powershell -ExecutionPolicy Bypass -File scripts\win_cron_run.ps1"
    Write-Host "  해제:      powershell scripts\setup_win_cron.ps1 -Remove"
    Write-Host ""

    # DRY_RUN 상태 경고
    if (Test-Path ".env") {
        $dryRun = Get-Content ".env" | Where-Object { $_ -match "^DRY_RUN=(.*)$" } | ForEach-Object { $Matches[1] }
        if ($dryRun -eq "true") {
            Write-Host "[안전] DRY_RUN=true — 분석만 실행, 실제 매매 없음" -ForegroundColor Green
        } else {
            Write-Host "[주의] DRY_RUN=false — 실제 매매가 실행됩니다!" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "[FAIL] 등록 실패: $_" -ForegroundColor Red
    Write-Host "  관리자 권한으로 실행해주세요." -ForegroundColor Yellow
    Write-Host "  방법: 시작 메뉴 > PowerShell > 우클릭 > '관리자 권한으로 실행'"
}

# 주간 재학습 안내
Write-Host ""
Write-Host "--- 주간 재학습 (별도) ---" -ForegroundColor DarkGray
Write-Host "  등록: python scripts\setup_weekly_retrain.py"
Write-Host "  일요일 03:00 심층 재학습 (500K 스텝)"
