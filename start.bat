@echo off
chcp 65001 >nul
title Local AI Workstation - RTX 3060 12GB

echo ============================================================
echo           Запуск Local AI Workstation (RTX 3060 12GB)
echo ============================================================
echo.

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
set OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe

echo [1/3] Проверка службы Ollama...
powershell -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:11434' -UseBasicParsing -TimeoutSec 1).StatusCode } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Запуск фонового демона Ollama...
    start "" "%OLLAMA_PATH%" serve
    timeout /t 3 /nobreak >nul
) else (
    echo Служба Ollama уже активна.
)

echo [2/3] Запуск веб-сервера Local AI Workstation...
cd /d "%~dp0"
start "" powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
"%PYTHON_PATH%" app.py

pause
