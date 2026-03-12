$ErrorActionPreference = "SilentlyContinue"
$VBS = "D:\Projects\claude-coin-trading-main\scripts\run_silent.vbs"

# 1. Delete all existing
$names = @(
    "CoinTrading_RL_tier1", "CoinTrading_RL_tier2", "CoinTrading_RL_tier3", "CoinTrading_RL_tier4",
    "CoinTrading_Agent", "CoinTrading_Monitor_Hourly", "CoinTrading_HourlyMonitor",
    "CoinTrading_Report_Weekly", "CoinTrading_Report_Weekly2", "CoinTrading_WeeklyReport",
    "CoinTrading_Backtest_Monthly", "CoinTrading_WebDashboard"
)
foreach ($n in $names) {
    schtasks.exe /delete /tn $n /f 2>$null | Out-Null
}
Write-Host "[1] Old tasks deleted"

# 2. Re-register silent
schtasks.exe /create /tn "CoinTrading_RL_tier1" /tr "wscript.exe `"$VBS`" `"training_scheduler.py`" `"tier1`"" /sc hourly /mo 6 /st 00:00 /f
schtasks.exe /create /tn "CoinTrading_RL_tier2" /tr "wscript.exe `"$VBS`" `"training_scheduler.py`" `"tier2`"" /sc daily /st 03:00 /f
schtasks.exe /create /tn "CoinTrading_RL_tier3" /tr "wscript.exe `"$VBS`" `"training_scheduler.py`" `"tier3`"" /sc weekly /d SUN /st 04:00 /f
schtasks.exe /create /tn "CoinTrading_RL_tier4" /tr "wscript.exe `"$VBS`" `"train_1h_intensive.py`"" /sc monthly /d 1 /st 02:00 /f
schtasks.exe /create /tn "CoinTrading_Agent" /tr "wscript.exe `"$VBS`" `"run_agents.py`"" /sc hourly /mo 8 /st 00:00 /f
schtasks.exe /create /tn "CoinTrading_Monitor_Hourly" /tr "wscript.exe `"$VBS`" `"performance_report.py`"" /sc hourly /mo 1 /st 00:30 /f
schtasks.exe /create /tn "CoinTrading_Report_Weekly" /tr "wscript.exe `"$VBS`" `"performance_report.py`" `"--weekly`"" /sc weekly /d MON /st 09:00 /f
schtasks.exe /create /tn "CoinTrading_Backtest_Monthly" /tr "wscript.exe `"$VBS`" `"performance_report.py`" `"--backtest`"" /sc monthly /d 1 /st 10:00 /f
schtasks.exe /create /tn "CoinTrading_WebDashboard" /tr "wscript.exe `"$VBS`" `"web_server.py`"" /sc onlogon /f

Write-Host "`n[2] New silent tasks registered"

# 3. Verify
Write-Host "`n[3] Current tasks:"
Get-ScheduledTask -TaskName "CoinTrading_*" | Select-Object TaskName, State | Format-Table -AutoSize
