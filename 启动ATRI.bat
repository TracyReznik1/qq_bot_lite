@echo off
chcp 65001 >nul
cd /d "%~dp0"

call .venv\Scripts\activate.bat

python -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt -q
    echo Done. Starting ATRI...
)

python run_bot.py
pause
