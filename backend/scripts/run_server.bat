@echo off
REM Script để chạy FastAPI server

cd /d "%~dp0\.."

echo Starting FastAPI server...
echo API docs will be available at: http://localhost:8000/docs
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause

