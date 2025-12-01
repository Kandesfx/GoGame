#!/bin/bash
# Script để build và test gogame_py trong MSYS2 MinGW shell

set -e

PROJECT_ROOT="/d/Hai/study/TTNT/GoGame"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Building gogame_py module"
echo "============================================================"

# Check if in MSYS2 MinGW shell
if ! command -v g++ &> /dev/null; then
    echo "❌ g++ not found. Hãy chạy trong MSYS2 MinGW 64-bit shell"
    exit 1
fi

echo "✅ Compiler: $(g++ --version | head -1)"

# Check pybind11
if ! pacman -Q mingw-w64-x86_64-pybind11 &> /dev/null; then
    echo "⚠️  pybind11 chưa cài. Đang cài..."
    pacman -S --noconfirm mingw-w64-x86_64-pybind11
fi

# Build
echo ""
echo "Building with CMake..."
mkdir -p build
cd build

if [ ! -f "CMakeCache.txt" ]; then
    cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
fi

cmake --build . --target gogame_py

echo ""
echo "============================================================"
echo "Testing module"
echo "============================================================"

# Test với Python từ MSYS2
PYTHON_MSYS="/c/msys64/mingw64/bin/python3"

if [ -f "$PYTHON_MSYS" ]; then
    echo "Testing với MSYS2 Python..."
    "$PYTHON_MSYS" -c "
import sys
sys.path.insert(0, 'build')
try:
    import gogame_py
    print('✅ Import thành công!')
    board = gogame_py.Board(9)
    print(f'✅ Board created: size={board.size()}')
    ai = gogame_py.AIPlayer()
    print('✅ AIPlayer created')
    print('✅ Module hoạt động!')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"
else
    echo "⚠️  MSYS2 Python không tìm thấy"
    echo "   Module đã build nhưng cần test với Python từ MSYS2"
fi

echo ""
echo "============================================================"
echo "Installation instructions"
echo "============================================================"
echo ""
echo "Để sử dụng module với venv Python:"
echo "  1. Copy module: cp build/gogame_py.*.pyd venv/Lib/site-packages/gogame_py.pyd"
echo "  2. Copy DLLs: cp /c/msys64/mingw64/bin/lib*.dll venv/Lib/site-packages/"
echo "  3. Hoặc dùng Python từ MSYS2: /c/msys64/mingw64/bin/python3"
echo ""

