@echo off
REM Script để check và fix Node.js path trong Windows

echo Checking Node.js installation...

where node >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Node.js found!
    node --version
    npm --version
    echo.
    echo You can now run: npm install
) else (
    echo ❌ Node.js not found in PATH
    echo.
    echo Please:
    echo 1. Check if Node.js is installed
    echo 2. Add Node.js to PATH environment variable
    echo 3. Restart terminal
    echo.
    echo Common Node.js location: C:\Program Files\nodejs\
    pause
)

