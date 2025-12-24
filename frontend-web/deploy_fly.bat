@echo off
REM Script để deploy frontend lên Fly.io

echo ============================================================
echo Deploy GoGame Frontend to Fly.io
echo ============================================================
echo.

REM Kiểm tra fly CLI
where fly >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Fly CLI not found!
    echo.
    echo Cài đặt Fly CLI:
    echo   PowerShell: iwr https://fly.io/install.ps1 -useb ^| iex
    echo.
    exit /b 1
)

echo [OK] Fly CLI found
fly version
echo.

REM Kiểm tra đang ở frontend-web directory
if not exist "fly.toml" (
    echo [ERROR] fly.toml not found!
    echo.
    echo Đảm bảo bạn đang ở thư mục frontend-web.
    echo.
    exit /b 1
)

echo [OK] fly.toml found
echo.

REM Kiểm tra Dockerfile
if not exist "Dockerfile" (
    echo [ERROR] Dockerfile not found!
    echo.
    exit /b 1
)

echo [OK] Dockerfile found
echo.

REM Kiểm tra VITE_API_URL trong fly.toml
findstr /C:"VITE_API_URL" fly.toml >nul
if errorlevel 1 (
    echo [WARNING] VITE_API_URL not found in fly.toml
    echo.
    echo Đảm bảo fly.toml có build_args với VITE_API_URL
    echo.
)

echo ============================================================
echo Deploying to Fly.io...
echo ============================================================
echo.

fly deploy -a gogame-frontend

if errorlevel 1 (
    echo.
    echo [ERROR] Deploy failed!
    echo.
    echo Xem logs: fly logs -a gogame-frontend
    echo.
    exit /b 1
)

echo.
echo ============================================================
echo Deploy completed successfully!
echo ============================================================
echo.
echo Kiểm tra:
echo   - Status: fly status -a gogame-frontend
echo   - Logs: fly logs -a gogame-frontend
echo   - URL: https://gogame-frontend.fly.dev
echo.
