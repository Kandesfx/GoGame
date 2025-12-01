@echo off
REM Script để tạo migration đầu tiên từ models

cd /d "%~dp0\.."

echo Tạo migration đầu tiên từ SQLAlchemy models...
alembic revision --autogenerate -m "Initial schema: users, matches, coins, premium, tokens"

echo.
echo ✅ Migration đã được tạo!
echo.
echo 📝 Bước tiếp theo:
echo 1. Review file migration trong migrations/versions/
echo 2. Chỉnh sửa nếu cần (ví dụ: thêm indexes, constraints)
echo 3. Chạy: alembic upgrade head

pause

