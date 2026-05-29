@echo off
cd /d "%~dp0"
python plankton_viewer.py
if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed. Please install Python and dependencies:
    echo   pip install -r requirements.txt
    echo.
    pause
)
