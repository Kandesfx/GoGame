@echo off
REM Deploy backend to Fly.io from root directory
cd ..
fly deploy -c backend/fly.toml
