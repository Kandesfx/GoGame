@echo off
REM Script đơn giản để build gogame_py trên Windows (CMD)

setlocal enabledelayedexpansion

cd /d "%~dp0\.."
set PROJECT_ROOT=%CD%

echo ============================================================
echo Building gogame_py Python Module
echo ============================================================
echo.

REM 1. Kiểm tra CMake
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found!
    echo    Windows (MSYS2): pacman -S mingw-w64-x86_64-cmake
    echo    Hoặc tải từ: https://cmake.org/download/
    exit /b 1
)
echo [OK] CMake found
cmake --version | findstr /C:"version"

REM 2. Kiểm tra pybind11
echo.
echo Checking pybind11...

python -c "import pybind11" >nul 2>&1
if errorlevel 1 (
    python3 -c "import pybind11" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] pybind11 not found!
        echo.
        echo Cài đặt pybind11:
        echo   Windows (MSYS2): pacman -S mingw-w64-x86_64-pybind11
        echo   Hoặc: pip install pybind11
        exit /b 1
    ) else (
        echo [OK] pybind11 found (via python3)
    )
) else (
    echo [OK] pybind11 found (via python)
)

REM 3. Build
echo.
echo ============================================================
echo Building module...
echo ============================================================

if not exist build mkdir build
cd build

REM Configure (chỉ nếu chưa có CMakeCache.txt)
if not exist CMakeCache.txt (
    echo Running cmake ..
    cmake ..
    echo.
)

REM Build target gogame_py
echo Building target: gogame_py
cmake --build . --target gogame_py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo    Hãy kiểm tra output ở trên để xem lỗi
    exit /b 1
)

echo.

REM 4. Kiểm tra file đã được tạo
echo ============================================================
echo Checking output files...
echo ============================================================

set MODULE_FILE=
for %%f in (gogame_py*.pyd) do (
    set MODULE_FILE=%%f
    goto :found
)

:found
if "!MODULE_FILE!"=="" (
    echo [ERROR] Module file not found!
    echo.
    echo Có thể:
    echo   1. pybind11 không được CMake tìm thấy
    echo   2. Build failed (xem output ở trên)
    echo   3. File ở vị trí khác
    echo.
    echo Hãy kiểm tra:
    echo   - Output của 'cmake ..' có 'pybind11 found' không?
    echo   - Có lỗi khi build không?
    exit /b 1
) else (
    echo [OK] Module created: !MODULE_FILE!
    dir !MODULE_FILE!
)

echo.
echo ============================================================
echo Build completed successfully!
echo ============================================================
echo.
echo Để sử dụng module:
echo   1. Thêm build\ vào PYTHONPATH:
echo      set PYTHONPATH=%PROJECT_ROOT%\build;%PYTHONPATH%
echo.
echo   2. Hoặc copy vào backend\:
echo      copy "!MODULE_FILE!" "%PROJECT_ROOT%\backend\gogame_py.pyd"
echo.
echo   3. Hoặc dùng script tự động:
echo      python scripts\install_gogame_py.py
echo.

endlocal

