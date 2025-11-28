@echo off
REM Script setup database cho GoGame backend (Windows)

echo ============================================================
echo 🚀 GoGame Database Setup Script (Windows)
echo ============================================================
echo.

REM Kiểm tra file .env
if not exist .env (
    echo ⚠️  Không tìm thấy file .env
    echo 💡 Tạo file .env từ env.example:
    echo    copy env.example .env
    exit /b 1
)

echo ✅ Đã tìm thấy file .env
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python không được tìm thấy
    echo 💡 Cài đặt Python từ https://www.python.org/downloads/
    exit /b 1
)

REM Kiểm tra psql (PostgreSQL client)
where psql >nul 2>&1
if errorlevel 1 (
    echo ⚠️  psql không được tìm thấy trong PATH
    echo 💡 Có thể sử dụng Python script thay thế:
    echo    python scripts\setup_database.py
    echo.
    echo Hoặc thêm PostgreSQL bin vào PATH:
    echo    C:\Program Files\PostgreSQL\14\bin
    echo.
)

REM Sử dụng Python script (cross-platform và đáng tin cậy hơn)
echo 🔄 Đang chạy Python setup script...
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python scripts\setup_database.py

if errorlevel 1 (
    echo.
    echo ❌ Có lỗi xảy ra
    exit /b 1
)

echo.
echo ============================================================
echo ✅ Database setup hoàn tất!
echo ============================================================
echo.
echo 💡 Bạn có thể chạy backend server:
echo    python -m uvicorn app.main:app --reload
pause

