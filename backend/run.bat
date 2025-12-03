@echo off
REM Quick run script - ensures we're in the right directory
REM This script can be run from anywhere

cd /d "%~dp0"

echo ========================================
echo   GoGame Backend - Starting Server
echo ========================================
echo.
echo Current directory: %CD%
echo.

REM Check if we're in backend directory
if not exist "app\main.py" (
    echo ERROR: app\main.py not found!
    echo Please run this script from the backend directory.
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup.bat first to create virtual environment.
    echo Or create it manually: python -m venv venv
    echo.
    pause
    exit /b 1
)

REM Check if uvicorn is installed
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo WARNING: uvicorn is not installed!
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo.
    if exist "env.example" (
        echo Creating .env from env.example...
        copy env.example .env >nul
        echo Please edit .env file and configure your settings.
        echo.
        pause
    ) else (
        echo Please create .env file manually.
        echo.
        pause
    )
)

echo Starting FastAPI server...
echo.
echo API will be available at: http://localhost:8000
echo API docs will be available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause

