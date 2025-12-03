@echo off
REM Wrapper script for parse_by_year.py using Windows Python
REM This ensures we use the correct Python with all dependencies installed

set PYTHON_PATH=C:\Users\Hai\AppData\Local\Programs\Python\Python312\python.exe

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Please update PYTHON_PATH in this script to point to your Python installation
    pause
    exit /b 1
)

REM Run the script with Windows Python
"%PYTHON_PATH%" "%~dp0parse_by_year.py" %*

