#!/bin/bash
# Script to check for Windows Python and help set up torch installation

echo "🔍 Checking for Windows Python installations..."
echo

# Check common locations
PYTHON_PATHS=(
    "/c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe"
    "/c/Python*/python.exe"
    "/c/Program Files/Python*/python.exe"
    "/c/Program Files (x86)/Python*/python.exe"
)

FOUND=false

for pattern in "${PYTHON_PATHS[@]}"; do
    for python_path in $pattern; do
        if [ -f "$python_path" ]; then
            echo "✅ Found Windows Python:"
            echo "   Path: $python_path"
            VERSION=$("$python_path" --version 2>&1)
            echo "   Version: $VERSION"
            echo
            FOUND=true
            
            # Check if it's CPython (not MSYS2)
            if echo "$VERSION" | grep -q "Python"; then
                echo "   Type: CPython (Windows) ✅"
                echo
                echo "💡 To use this Python:"
                echo "   1. Create new venv:"
                echo "      $python_path -m venv venv_windows"
                echo
                echo "   2. Activate and install:"
                echo "      source venv_windows/Scripts/activate"
                echo "      pip install sgf numpy torch tqdm"
                echo
            fi
        fi
    done
done

if [ "$FOUND" = false ]; then
    echo "❌ No Windows Python found in common locations"
    echo
    echo "📥 Please install Windows Python:"
    echo "   1. Download from: https://www.python.org/downloads/"
    echo "   2. During installation, check 'Add Python to PATH'"
    echo "   3. Then run this script again"
    echo
fi

# Check current Python
echo "🔍 Current Python:"
echo "   Path: $(which python)"
echo "   Version: $(python --version 2>&1)"
echo "   Type: $(python -c "import sys; print('MSYS2/GCC' if 'GCC' in sys.version else 'CPython')" 2>/dev/null || echo 'Unknown')"
echo

if python -c "import sys; print('GCC' in sys.version)" 2>/dev/null | grep -q True; then
    echo "⚠️  You are using MSYS2 Python (GCC-compiled)"
    echo "   PyTorch doesn't have wheels for MSYS2 Python"
    echo "   Please use Windows Python (CPython) instead"
    echo
fi

