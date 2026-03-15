@echo off
chcp 65001 >nul 2>&1
title 텔레그램 멀티챗
cd /d D:\Entertainments\DevEnvironment\claude-coin-trading-main
python scripts\telegram_listener.py
pause
