@echo off
REM Script để activate virtual environment và chạy parse script trên Windows

REM Activate venv
call venv\Scripts\activate.bat

REM Run the script with all arguments
python scripts\parse_sgf_local.py %*

