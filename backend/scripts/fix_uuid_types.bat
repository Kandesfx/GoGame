@echo off
REM Script to fix UUID type conversion in database
REM This script runs the SQL fix directly

cd /d "%~dp0\.."

echo ========================================
echo   Fix UUID Types in Database
echo ========================================
echo.

REM Check if we're in backend directory
if not exist "backend\app\main.py" (
    echo ERROR: backend\app\main.py not found!
    echo Please run this script from the project root or backend directory.
    echo.
    pause
    exit /b 1
)

REM Load .env file
if not exist "backend\.env" (
    echo ERROR: .env file not found in backend directory!
    echo Please create .env file from env.example
    echo.
    pause
    exit /b 1
)

echo Loading database connection from .env...
for /f "tokens=*" %%a in ('type backend\.env ^| findstr /i "POSTGRES_DSN"') do set %%a

REM Extract connection info (simple parsing)
REM Note: This is a simple parser, may need adjustment based on your .env format
echo.
echo Connecting to database...
echo.

REM Try to use psql if available
where psql >nul 2>&1
if errorlevel 1 (
    echo ERROR: psql command not found!
    echo.
    echo Please install PostgreSQL client tools or run the SQL manually:
    echo   1. Open pgAdmin or psql
    echo   2. Connect to your database
    echo   3. Run the SQL from: backend\migrations\fix_uuid_types.sql
    echo.
    pause
    exit /b 1
)

REM Try to parse DSN (simple approach)
REM For now, ask user to provide connection details
echo Please provide database connection details:
echo.
set /p DB_HOST=Host [localhost]: 
if "%DB_HOST%"=="" set DB_HOST=localhost

set /p DB_PORT=Port [5432]: 
if "%DB_PORT%"=="" set DB_PORT=5432

set /p DB_NAME=Database [gogame]: 
if "%DB_NAME%"=="" set DB_NAME=gogame

set /p DB_USER=User [postgres]: 
if "%DB_USER%"=="" set DB_USER=postgres

set /p DB_PASS=Password: 

echo.
echo Running UUID conversion script...
echo.

REM Set password for psql
set PGPASSWORD=%DB_PASS%

psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -f backend\migrations\fix_uuid_types.sql

if errorlevel 1 (
    echo.
    echo ERROR: Failed to run SQL script!
    echo.
    echo You can run it manually:
    echo   1. Connect to your database
    echo   2. Run: backend\migrations\fix_uuid_types.sql
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   UUID conversion completed!
echo ========================================
echo.
pause

