@echo off
REM Script để deploy backend lên Fly.io
REM Chạy từ root directory của project

cd /d "%~dp0\.."

echo 🚀 Deploying backend to Fly.io...
echo 📁 Build context: %CD%
echo 📄 Using: fly.toml (root)

fly deploy

