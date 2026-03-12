@echo off
REM === 모든 CoinTrading 작업을 창 숨김 모드로 재등록 ===
REM === 우클릭 → "관리자 권한으로 실행" ===

echo ============================================
echo   CoinTrading 스케줄 재등록 (Silent Mode)
echo ============================================
echo.

set PROJECT=D:\Projects\claude-coin-trading-main
set VBS=%PROJECT%\scripts\run_silent.vbs

REM 기존 작업 전부 삭제
echo [1/3] 기존 작업 삭제...
for %%T in (CoinTrading_RL_tier1 CoinTrading_RL_tier2 CoinTrading_RL_tier3 CoinTrading_RL_tier4 CoinTrading_Agent CoinTrading_Monitor_Hourly CoinTrading_HourlyMonitor CoinTrading_Report_Weekly CoinTrading_Report_Weekly2 CoinTrading_WeeklyReport CoinTrading_Backtest_Monthly CoinTrading_WebDashboard) do (
    schtasks /delete /tn "%%T" /f >nul 2>&1
)
echo   완료

REM Silent 모드로 재등록
echo.
echo [2/3] Silent 모드 재등록...

REM RL Tier1: 6시간마다
schtasks /create /tn "CoinTrading_RL_tier1" /tr "wscript.exe \"%VBS%\" \"training_scheduler.py\" \"tier1\"" /sc hourly /mo 6 /st 00:00 /f
REM RL Tier2: 매일 03:00
schtasks /create /tn "CoinTrading_RL_tier2" /tr "wscript.exe \"%VBS%\" \"training_scheduler.py\" \"tier2\"" /sc daily /st 03:00 /f
REM RL Tier3: 매주 일요일 04:00
schtasks /create /tn "CoinTrading_RL_tier3" /tr "wscript.exe \"%VBS%\" \"training_scheduler.py\" \"tier3\"" /sc weekly /d SUN /st 04:00 /f
REM RL Tier4: 매월 1일 02:00
schtasks /create /tn "CoinTrading_RL_tier4" /tr "wscript.exe \"%VBS%\" \"train_1h_intensive.py\"" /sc monthly /d 1 /st 02:00 /f
REM Trading Agent: 8시간마다
schtasks /create /tn "CoinTrading_Agent" /tr "wscript.exe \"%VBS%\" \"run_agents.py\"" /sc hourly /mo 8 /st 00:00 /f
REM Monitor: 매시간
schtasks /create /tn "CoinTrading_Monitor_Hourly" /tr "wscript.exe \"%VBS%\" \"performance_report.py\"" /sc hourly /mo 1 /st 00:30 /f
REM Weekly Report: 월요일 09:00
schtasks /create /tn "CoinTrading_Report_Weekly" /tr "wscript.exe \"%VBS%\" \"performance_report.py\" \"--weekly\"" /sc weekly /d MON /st 09:00 /f
REM Monthly Backtest: 매월 1일 10:00
schtasks /create /tn "CoinTrading_Backtest_Monthly" /tr "wscript.exe \"%VBS%\" \"performance_report.py\" \"--backtest\"" /sc monthly /d 1 /st 10:00 /f
REM Web Dashboard (keep running)
schtasks /create /tn "CoinTrading_WebDashboard" /tr "wscript.exe \"%VBS%\" \"web_server.py\"" /sc onlogon /f

echo.
echo [3/3] 등록 결과 확인...
echo.
schtasks /query /fo TABLE /nh | findstr "CoinTrading"
echo.
echo ============================================
echo   완료! 모든 작업이 창 없이 실행됩니다.
echo ============================================
pause
