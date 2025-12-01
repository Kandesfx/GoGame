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
    echo Warning: Virtual environment not found.
    echo You may need to create it: python -m venv venv
    echo.
)

REM Check if .env exists
if not exist ".env" (
    echo Warning: .env file not found!
    echo Please copy env.example to .env and configure it.
    echo.
    pause
)

echo Starting FastAPI server...
echo.
echo API will be available at: http://localhost:8000
echo API docs will be available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause

