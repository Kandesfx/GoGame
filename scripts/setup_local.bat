@echo off
REM Setup script for Windows
REM Cài đặt dependencies cho local processing

echo Installing dependencies for local processing...
echo.

REM Kiểm tra Python
py --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Installing packages...
py -m pip install --upgrade pip
py -m pip install sgf numpy torch tqdm

echo.
echo Verifying installation...
py -c "import sgf; import numpy; import torch; import tqdm; print('All packages installed successfully!')"

if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install
    pause
    exit /b 1
)

echo.
echo Setup complete!
pause

