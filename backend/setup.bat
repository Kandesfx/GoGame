@echo off
REM Setup script for GoGame Backend
REM This script sets up the development environment

echo ========================================
echo   GoGame Backend - Setup Script
echo ========================================
echo.

cd /d "%~dp0"

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version
echo.

REM Create virtual environment if it doesn't exist
echo [2/4] Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [4/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo.
echo Dependencies installed successfully!
echo.

REM Check if .env exists
if not exist ".env" (
    echo [BONUS] Setting up .env file...
    if exist "env.example" (
        copy env.example .env >nul
        echo Created .env file from env.example
        echo.
        echo IMPORTANT: Please edit .env file and configure:
        echo   - POSTGRES_DSN: PostgreSQL connection string
        echo   - MONGO_DSN: MongoDB connection string (optional)
        echo   - JWT_SECRET_KEY: Random secret key (min 32 chars)
        echo.
    ) else (
        echo WARNING: env.example not found!
        echo Please create .env file manually.
        echo.
    )
) else (
    echo .env file already exists.
    echo.
)

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Edit .env file with your database settings
echo   2. Run: run.bat (or: python -m uvicorn app.main:app --reload)
echo   3. Open: http://localhost:8000/docs
echo.
pause

