#!/bin/bash
# Setup script for GoGame Backend
# This script sets up the development environment

echo "========================================"
echo "  GoGame Backend - Setup Script"
echo "========================================"
echo ""

cd "$(dirname "$0")" || exit 1

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "[1/4] Checking Python version..."
python3 --version
echo ""

# Create virtual environment if it doesn't exist
echo "[2/4] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment!"
        exit 1
    fi
    echo "Virtual environment created successfully!"
else
    echo "Virtual environment already exists."
fi
echo ""

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi
echo "Virtual environment activated!"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install dependencies
echo "[4/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies!"
    exit 1
fi
echo ""
echo "Dependencies installed successfully!"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[BONUS] Setting up .env file..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "Created .env file from env.example"
        echo ""
        echo "IMPORTANT: Please edit .env file and configure:"
        echo "  - POSTGRES_DSN: PostgreSQL connection string"
        echo "  - MONGO_DSN: MongoDB connection string (optional)"
        echo "  - JWT_SECRET_KEY: Random secret key (min 32 chars)"
        echo ""
    else
        echo "WARNING: env.example not found!"
        echo "Please create .env file manually."
        echo ""
    fi
else
    echo ".env file already exists."
    echo ""
fi

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your database settings"
echo "  2. Run: ./run.sh (or: python -m uvicorn app.main:app --reload)"
echo "  3. Open: http://localhost:8000/docs"
echo ""

