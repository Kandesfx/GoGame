@echo off
REM Script để deploy frontend lên Fly.io

echo 🚀 Deploying frontend to Fly.io...
echo 📁 Current directory: %CD%
echo 📄 Using: fly.toml

fly deploy

