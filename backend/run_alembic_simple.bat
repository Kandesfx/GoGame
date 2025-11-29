@echo off
REM Simple script to run alembic - uses Python from venv directly
REM This script can be run from anywhere

cd /d "%~dp0"

echo ========================================
echo   GoGame Backend - Running Alembic
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

REM Try to find and use Python from venv
set PYTHON_EXE=

REM Check venv in root first
if exist "..\venv\Scripts\python.exe" (
    set PYTHON_EXE=..\venv\Scripts\python.exe
    echo Using Python from root venv...
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
    echo Using Python from backend venv...
) else (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    echo Or use system Python (not recommended):
    set /p USE_SYSTEM=Use system Python? (Y/N): 
    if /i not "%USE_SYSTEM%"=="Y" (
        pause
        exit /b 1
    )
    set PYTHON_EXE=python
)

REM Check if alembic is installed
echo Checking if alembic is installed...
%PYTHON_EXE% -c "import alembic" 2>nul
if errorlevel 1 (
    echo Alembic is not installed!
    echo.
    echo Installing dependencies...
    %PYTHON_EXE% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
)

echo.
echo Running alembic upgrade head...
echo.

%PYTHON_EXE% -m alembic upgrade head

if errorlevel 1 (
    echo.
    echo ERROR: Alembic command failed!
    echo.
    echo Troubleshooting:
    echo 1. Make sure PostgreSQL is running
    echo 2. Check your .env file has correct database connection
    echo 3. Make sure database 'gogame' exists
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Alembic migration completed!
echo ========================================
echo.
pause

