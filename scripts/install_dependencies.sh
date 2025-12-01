#!/bin/bash
# Script để cài đặt dependencies
# Có thể chạy từ Git Bash hoặc MSYS2

echo "🔧 Installing dependencies for local processing..."
echo

# Detect environment
if command -v pacman &> /dev/null; then
    echo "✅ Detected MSYS2 environment"
    IS_MSYS2=true
else
    echo "⚠️  Not in MSYS2. Will try to use Python Windows or pip directly."
    IS_MSYS2=false
fi

# Function to check if package is installed
check_package() {
    python -c "import $1" 2>/dev/null
}

# Install pip if needed
if ! check_package pip; then
    echo "📦 Installing pip..."
    if [ "$IS_MSYS2" = true ]; then
        pacman -S --noconfirm mingw-w64-x86_64-python-pip
    else
        echo "⚠️  Cannot install pip automatically. Please:"
        echo "   1. Open MSYS2 MinGW64 terminal"
        echo "   2. Run: pacman -S mingw-w64-x86_64-python-pip"
        echo "   OR use Python Windows: py -m pip install sgf torch"
        exit 1
    fi
fi

# Install sgf
if ! check_package sgf; then
    echo "📦 Installing sgf..."
    python -m pip install sgf
else
    echo "✅ sgf already installed"
fi

# Install torch
if ! check_package torch; then
    echo "📦 Installing torch (CPU version)..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    echo "✅ torch already installed"
fi

# Verify all packages
echo
echo "🔍 Verifying installation..."
python -c "
try:
    import sgf
    import numpy
    import torch
    import tqdm
    print('✅ All packages installed successfully!')
    print(f'   sgf: {sgf.__version__ if hasattr(sgf, \"__version__\") else \"installed\"}')
    print(f'   numpy: {numpy.__version__}')
    print(f'   torch: {torch.__version__}')
    print(f'   tqdm: {tqdm.__version__}')
except ImportError as e:
    print(f'❌ Error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Setup complete! You can now run:"
    echo "   python scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019"
else
    echo
    echo "❌ Setup failed. Please check errors above."
    exit 1
fi

