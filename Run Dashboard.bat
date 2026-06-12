@echo off
REM Starts the Scalable Performance Dashboard and opens it in your default
REM browser. Double-click this file in Explorer. Close this window or press
REM Ctrl-C to stop the dashboard.

cd /d "%~dp0"
cls

echo ================================================================
echo    Scalable Performance Dashboard
echo ================================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'uv' is not installed. Please run 'Setup.bat' first.
    echo.
    pause
    exit /b 1
)

dir /b "transaction_data\*.csv" >nul 2>&1
if errorlevel 1 (
    echo ERROR: No CSV file in "transaction_data".
    echo.
    echo Please:
    echo   1. Export your transactions from Scalable Capital as CSV.
    echo   2. Drop the file into the "transaction_data" folder.
    echo   3. Double-click this launcher again.
    echo.
    pause
    exit /b 1
)

echo Starting the dashboard...
echo.
echo   * Your browser will open at http://127.0.0.1:8050 as soon as it
echo     is ready.
echo   * The very first start can take 5-7 minutes while live prices and
echo     historical data are fetched. Later starts are fast.
echo   * To stop the dashboard, close this window or press Ctrl-C.
echo.
echo ------------------------------------------------------------------

REM Launch the browser-open helper in the background.
start "" /b cmd /c uv run python -m dashboard.open_browser >nul 2>&1

REM Run the dashboard in the foreground.
uv run python run.py
