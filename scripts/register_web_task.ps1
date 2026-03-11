$ProjectDir = 'D:\Projects\claude-coin-trading-main'
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
$scriptPath = Join-Path $ProjectDir 'scripts\web_server.py'

$action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-u `"$scriptPath`"" `
    -WorkingDirectory $ProjectDir

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 3

$existing = Get-ScheduledTask -TaskName 'CoinTrading_WebDashboard' -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName 'CoinTrading_WebDashboard' -Confirm:$false
    Write-Host "  [UPDATE] Removed old task" -ForegroundColor Yellow
}

Register-ScheduledTask `
    -TaskName 'CoinTrading_WebDashboard' `
    -Description 'Web Dashboard (port 5555)' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Highest `
    -Force | Out-Null

Write-Host "  [OK] CoinTrading_WebDashboard registered" -ForegroundColor Green

Start-ScheduledTask -TaskName 'CoinTrading_WebDashboard'
Write-Host "  [OK] Started now" -ForegroundColor Green
