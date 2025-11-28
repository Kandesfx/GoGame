@echo off
REM Quick start script for GoGame Backend
REM This script starts the FastAPI server

echo ========================================
echo   GoGame Backend - Starting Server
echo ========================================
echo.

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

