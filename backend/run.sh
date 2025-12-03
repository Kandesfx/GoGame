#!/bin/bash
# Quick run script - ensures we're in the right directory
# This script can be run from anywhere

cd "$(dirname "$0")" || exit 1

echo "========================================"
echo "  GoGame Backend - Starting Server"
echo "========================================"
echo ""
echo "Current directory: $(pwd)"
echo ""

# Check if we're in backend directory
if [ ! -f "app/main.py" ]; then
    echo "ERROR: app/main.py not found!"
    echo "Please run this script from the backend directory."
    echo ""
    exit 1
fi

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment (Linux/Mac style)..."
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows style)..."
    source venv/Scripts/activate
else
    echo "ERROR: Virtual environment not found!"
    echo ""
    echo "Please run ./setup.sh first to create virtual environment."
    echo "Or create it manually: python3 -m venv venv"
    echo ""
    exit 1
fi

# Check if uvicorn is installed
python -c "import uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: uvicorn is not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies!"
        exit 1
    fi
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo ""
    if [ -f "env.example" ]; then
        echo "Creating .env from env.example..."
        cp env.example .env
        echo "Please edit .env file and configure your settings."
        echo ""
        read -p "Press Enter to continue..."
    else
        echo "Please create .env file manually."
        echo ""
        read -p "Press Enter to continue anyway..."
    fi
fi

echo "Starting FastAPI server..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

