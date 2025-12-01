#!/bin/bash
# Script để chạy FastAPI server

cd "$(dirname "$0")/.." || exit 1

echo "Starting FastAPI server..."
echo "API docs will be available at: http://localhost:8000/docs"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

