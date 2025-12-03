#!/bin/bash
# Simple dependency installer - skips problematic packages

cd "$(dirname "$0")" || exit 1

echo "Installing dependencies (skipping optional packages that require Rust)..."
echo ""

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Install core dependencies first
echo "Installing core dependencies..."
pip install --upgrade pip setuptools wheel

# Install FastAPI without orjson dependency
echo "Installing FastAPI (without orjson)..."
pip install --no-deps fastapi==0.111.0
pip install starlette==0.37.2 typing-extensions fastapi-cli jinja2 python-multipart ujson

# Install other packages
echo "Installing other dependencies..."
pip install "uvicorn[standard]==0.30.1" || echo "Warning: uvicorn install failed"
pip install sqlalchemy==2.0.30 || echo "Warning: sqlalchemy install failed"
pip install alembic==1.13.1 || echo "Warning: alembic install failed"
pip install "psycopg[binary]==3.1.19" || echo "Warning: psycopg install failed"
pip install "motor>=3.5.0" || echo "Warning: motor install failed"
pip install "pymongo>=4.5.0,<5.0.0" || echo "Warning: pymongo install failed"
pip install "pydantic[email]==2.7.1" || echo "Warning: pydantic install failed"
pip install pydantic-settings==2.2.1 || echo "Warning: pydantic-settings install failed"
pip install python-dotenv==1.0.1 || echo "Warning: python-dotenv install failed"
pip install PyJWT==2.9.0 || echo "Warning: PyJWT install failed"
pip install argon2-cffi==23.1.0 || echo "Warning: argon2-cffi install failed"
pip install "passlib[argon2]==1.7.4" || echo "Warning: passlib install failed"
pip install httpx==0.27.0 || echo "Warning: httpx install failed"
pip install pytest==8.2.2 || echo "Warning: pytest install failed"

echo ""
echo "Checking uvicorn installation..."
python -c "import uvicorn" && echo "✅ uvicorn installed successfully!" || echo "❌ uvicorn not installed"

echo ""
echo "Done! You can now run: ./run.sh"

