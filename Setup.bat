@echo off
REM One-time setup for the Scalable Performance Dashboard on Windows.
REM Double-click this file in Explorer.

cd /d "%~dp0"
cls

echo ================================================================
echo    Scalable Performance Dashboard  -  Setup
echo ================================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'uv' is not installed.
    echo.
    echo uv is the tool that manages Python + dependencies for you.
    echo It's free and takes 30 seconds to install.
    echo.
    echo ^>^>^> Please open "How to install.html" first and follow Step 1.
    echo.
    pause
    exit /b 1
)

echo [1/3] Installing Python and Python dependencies...
echo       ^(this may take a few minutes the first time^)
echo.
uv sync
if errorlevel 1 goto :err

echo.
echo [2/3] Installing the headless browser ^(for live gettex prices^)...
echo.
uv run playwright install chromium
if errorlevel 1 goto :err

echo.
echo [3/3] Setup complete!
echo.
echo What to do next
echo ---------------
echo   1. Export your transaction history from Scalable Capital ^(CSV^).
echo   2. Drop the CSV into the folder named "transaction_data"
echo      ^(right next to this Setup file^).
echo   3. Double-click "Run Dashboard.bat" to start the dashboard.
echo.
pause
exit /b 0

:err
echo.
echo Setup failed. Please scroll up to see the error message and try again.
echo If the problem persists, ask the friend who shared this with you.
echo.
pause
exit /b 1
