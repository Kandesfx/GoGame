#!/bin/bash
# Quick start script for GoGame Backend
# This script starts the FastAPI server

echo "========================================"
echo "  GoGame Backend - Starting Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found."
    echo "You may need to create it: python -m venv venv"
    echo ""
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found!"
    echo "Please copy env.example to .env and configure it."
    echo ""
    read -p "Press Enter to continue anyway..."
fi

echo "Starting FastAPI server..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

