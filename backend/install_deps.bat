@echo off
REM Simple dependency installer for Windows

cd /d "%~dp0"

echo Installing dependencies...
echo.

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Install core dependencies
echo Installing core dependencies...
python -m pip install --upgrade pip setuptools wheel

REM Install packages
echo.
echo Installing packages...
python -m pip install fastapi==0.111.0
python -m pip install "uvicorn[standard]==0.30.1"
python -m pip install sqlalchemy==2.0.30
python -m pip install alembic==1.13.1
python -m pip install "psycopg[binary]==3.1.19"
python -m pip install "motor>=3.5.0"
python -m pip install "pymongo>=4.5.0,<5.0.0"
python -m pip install "pydantic[email]==2.7.1"
python -m pip install pydantic-settings==2.2.1
python -m pip install python-dotenv==1.0.1
python -m pip install PyJWT==2.9.0
python -m pip install argon2-cffi==23.1.0
python -m pip install "passlib[argon2]==1.7.4"
python -m pip install httpx==0.27.0
python -m pip install pytest==8.2.2

echo.
echo Checking uvicorn installation...
python -c "import uvicorn" && echo ✅ uvicorn installed successfully! || echo ❌ uvicorn not installed

echo.
echo Done! You can now run: run.bat
pause

