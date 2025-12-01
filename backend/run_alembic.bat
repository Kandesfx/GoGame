@echo off
REM Script to run alembic commands with virtual environment activated
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

REM Try to activate virtual environment from root
set VENV_ACTIVATED=0

if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment from root...
    call ..\venv\Scripts\activate.bat
    if errorlevel 1 (
        echo Warning: Failed to activate venv from root, trying backend...
    ) else (
        set VENV_ACTIVATED=1
    )
)

if %VENV_ACTIVATED%==0 (
    if exist "venv\Scripts\activate.bat" (
        echo Activating virtual environment from backend...
        call venv\Scripts\activate.bat
        if not errorlevel 1 (
            set VENV_ACTIVATED=1
        )
    )
)

if %VENV_ACTIVATED%==0 (
    echo.
    echo ERROR: Virtual environment not found!
    echo.
    echo Checking for Python...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH!
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo Would you like to create a virtual environment? (Y/N)
    set /p CREATE_VENV=
    if /i "%CREATE_VENV%"=="Y" (
        echo.
        echo Creating virtual environment in backend directory...
        python -m venv venv
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment!
            pause
            exit /b 1
        )
        echo Activating new virtual environment...
        call venv\Scripts\activate.bat
        set VENV_ACTIVATED=1
    ) else (
        echo.
        echo Please create a virtual environment manually:
        echo   python -m venv venv
        echo   venv\Scripts\activate.bat
        echo   pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

REM Check if alembic is installed
python -c "import alembic" 2>nul
if errorlevel 1 (
    echo ERROR: Alembic is not installed!
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
)

echo.
echo Running alembic upgrade head...
echo.

alembic upgrade head

if errorlevel 1 (
    echo.
    echo ERROR: Alembic command failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Alembic migration completed!
echo ========================================
echo.
pause

