@echo off
REM Script to fix virtual environment path issues
REM This script recreates the venv with correct paths

cd /d "%~dp0"

echo ========================================
echo   Fixing Virtual Environment
echo ========================================
echo.

REM Check if we're in backend directory
if not exist "app\main.py" (
    echo ERROR: app\main.py not found!
    echo Please run this script from the backend directory.
    echo.
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Backup old venv if exists
if exist "venv" (
    echo Old virtual environment found.
    echo.
    set /p REMOVE_OLD=Remove old venv and create new one? (Y/N): 
    if /i not "%REMOVE_OLD%"=="Y" (
        echo Cancelled.
        pause
        exit /b 0
    )
    
    echo Removing old virtual environment...
    rmdir /s /q venv
    if errorlevel 1 (
        echo Warning: Could not remove old venv completely.
        echo You may need to remove it manually.
        echo.
    )
)

echo Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Virtual environment fixed!
echo ========================================
echo.
echo You can now run:
echo   run.bat
echo   or
echo   uvicorn app.main:app --reload
echo.
pause

