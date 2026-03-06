@echo off
echo [방화벽 규칙 추가] BTC Trader Dashboard - Port 5555
netsh advfirewall firewall add rule name="BTC Trader Dashboard" dir=in action=allow protocol=TCP localport=5555
if %errorlevel%==0 (
    echo.
    echo SUCCESS: 방화벽 규칙이 추가되었습니다.
    echo 이제 아이폰에서 접속할 수 있습니다.
) else (
    echo.
    echo FAILED: 관리자 권한으로 실행해주세요.
    echo 이 파일을 우클릭 - 관리자 권한으로 실행
)
echo.
pause
