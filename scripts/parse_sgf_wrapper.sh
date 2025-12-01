#!/bin/bash
# Wrapper script để tự động detect và dùng Python đúng

# Tìm Python Windows
PYTHON_WIN=""
if command -v py &> /dev/null; then
    PYTHON_WIN="py"
elif [ -f "/c/Users/$USER/AppData/Local/Programs/Python/Python312/python.exe" ]; then
    PYTHON_WIN="/c/Users/$USER/AppData/Local/Programs/Python/Python312/python.exe"
elif [ -f "/c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe" ]; then
    PYTHON_WIN="/c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe"
elif [ -f "/c/Users/$USER/AppData/Local/Programs/Python/Python310/python.exe" ]; then
    PYTHON_WIN="/c/Users/$USER/AppData/Local/Programs/Python/Python310/python.exe"
fi

# Nếu tìm thấy Python Windows, dùng nó
if [ -n "$PYTHON_WIN" ]; then
    echo "Using Python Windows: $PYTHON_WIN"
    "$PYTHON_WIN" scripts/parse_sgf_local.py "$@"
else
    # Fallback: dùng Python hiện tại (có thể là MSYS2)
    echo "Warning: Using MSYS2 Python. Make sure packages are installed via pacman."
    echo "Or install Python Windows and add to PATH."
    python scripts/parse_sgf_local.py "$@"
fi

