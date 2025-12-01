#!/bin/bash
# Script đơn giản để build gogame_py - tự động kiểm tra và build

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Building gogame_py Python Module"
echo "============================================================"
echo ""

# 1. Kiểm tra CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found!"
    echo "   Windows (MSYS2): pacman -S mingw-w64-x86_64-cmake"
    echo "   Linux: sudo apt install cmake"
    echo "   macOS: brew install cmake"
    exit 1
fi
echo "✅ CMake: $(cmake --version | head -1)"

# 2. Kiểm tra pybind11
echo ""
echo "Checking pybind11..."

# Thử import pybind11 qua Python
if python -c "import pybind11" 2>/dev/null; then
    echo "✅ pybind11 found (via Python)"
    PYBIND11_DIR=$(python -c "import pybind11; import os; print(os.path.dirname(pybind11.__file__))")
    echo "   Location: $PYBIND11_DIR"
elif python3 -c "import pybind11" 2>/dev/null; then
    echo "✅ pybind11 found (via Python3)"
    PYBIND11_DIR=$(python3 -c "import pybind11; import os; print(os.path.dirname(pybind11.__file__))")
    echo "   Location: $PYBIND11_DIR"
elif [ -f "/c/msys64/mingw64/bin/python3.exe" ] && /c/msys64/mingw64/bin/python3.exe -c "import pybind11" 2>/dev/null; then
    echo "✅ pybind11 found (via MSYS2 Python)"
elif command -v pacman &> /dev/null && pacman -Q mingw-w64-x86_64-pybind11 &> /dev/null; then
    echo "✅ pybind11 found (via MSYS2 package)"
else
    echo "❌ pybind11 not found!"
    echo ""
    echo "Cài đặt pybind11:"
    echo "  Windows (MSYS2): pacman -S mingw-w64-x86_64-pybind11"
    echo "  Linux: pip install pybind11"
    echo "  macOS: pip install pybind11"
    exit 1
fi

# 3. Build
echo ""
echo "============================================================"
echo "Building module..."
echo "============================================================"

mkdir -p build
cd build

# Configure (chỉ nếu chưa có CMakeCache.txt)
if [ ! -f "CMakeCache.txt" ]; then
    echo "Running cmake .."
    cmake ..
    echo ""
fi

# Kiểm tra xem CMake có tìm thấy pybind11 không
if grep -q "pybind11 not found" CMakeCache.txt 2>/dev/null || ! grep -q "pybind11 found" CMakeCache.txt 2>/dev/null; then
    echo "⚠️  Warning: CMake có thể không tìm thấy pybind11"
    echo "   Hãy kiểm tra output của 'cmake ..' ở trên"
    echo ""
fi

# Build target gogame_py
echo "Building target: gogame_py"
cmake --build . --target gogame_py

echo ""

# 4. Kiểm tra file đã được tạo
echo "============================================================"
echo "Checking output files..."
echo "============================================================"

MODULE_FILE=$(find . -name "gogame_py*.pyd" -o -name "gogame_py*.so" | head -1)

if [ -z "$MODULE_FILE" ]; then
    echo "❌ Module file not found!"
    echo ""
    echo "Có thể:"
    echo "  1. pybind11 không được CMake tìm thấy"
    echo "  2. Build failed (xem output ở trên)"
    echo "  3. File ở vị trí khác"
    echo ""
    echo "Hãy kiểm tra:"
    echo "  - Output của 'cmake ..' có 'pybind11 found' không?"
    echo "  - Có lỗi khi build không?"
    exit 1
else
    echo "✅ Module created: $MODULE_FILE"
    ls -lh "$MODULE_FILE"
fi

echo ""
echo "============================================================"
echo "✅ Build completed successfully!"
echo "============================================================"
echo ""
echo "Để sử dụng module:"
echo "  1. Thêm build/ vào PYTHONPATH:"
echo "     export PYTHONPATH=\"$PROJECT_ROOT/build:\$PYTHONPATH\""
echo ""
echo "  2. Hoặc copy vào backend/:"
if [[ "$MODULE_FILE" == *.pyd ]]; then
    echo "     cp \"$MODULE_FILE\" \"$PROJECT_ROOT/backend/gogame_py.pyd\""
else
    echo "     cp \"$MODULE_FILE\" \"$PROJECT_ROOT/backend/gogame_py.so\""
fi
echo ""
echo "  3. Hoặc dùng script tự động:"
echo "     python scripts/install_gogame_py.py"
echo ""

