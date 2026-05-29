@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
python planktoscope_segment_viewer.py
if errorlevel 1 (
    echo.
    echo [ERROR] 启动失败，请确认已安装 Python 和依赖：
    echo   pip install -r requirements.txt
    echo.
    pause
)
