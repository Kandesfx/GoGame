#!/bin/bash
# Setup gogame_py để backend có thể sử dụng

set -e

PROJECT_ROOT="/d/Hai/study/TTNT/GoGame"
VENV_LIB="$PROJECT_ROOT/venv/Lib/site-packages"
MINGW_BIN="/c/msys64/mingw64/bin"
BUILD_DIR="$PROJECT_ROOT/build"

echo "============================================================"
echo "Setting up gogame_py for backend"
echo "============================================================"

# 1. Copy module
echo ""
echo "Step 1: Copying module..."
MODULE_FILE=$(find "$BUILD_DIR" -name "gogame_py*.pyd" | head -1)
if [ -z "$MODULE_FILE" ]; then
    echo "❌ Module not found in build directory"
    echo "   Hãy build trước: cmake --build build"
    exit 1
fi

cp "$MODULE_FILE" "$VENV_LIB/gogame_py.pyd"
echo "✅ Copied module: $MODULE_FILE -> $VENV_LIB/gogame_py.pyd"

# 2. Copy required DLLs
echo ""
echo "Step 2: Copying DLLs..."
DLLS=(
    "libgcc_s_seh-1.dll"
    "libstdc++-6.dll"
    "libwinpthread-1.dll"
    "libpython3.12.dll"
)

for dll in "${DLLS[@]}"; do
    if [ -f "$MINGW_BIN/$dll" ]; then
        cp "$MINGW_BIN/$dll" "$VENV_LIB/"
        echo "✅ Copied: $dll"
    else
        echo "⚠️  Not found: $dll"
    fi
done

echo ""
echo "============================================================"
echo "Setup completed!"
echo "============================================================"
echo ""
echo "💡 Note: Module được build với MinGW, có thể có conflicts"
echo "   với venv Python (MSVC). Nếu gặp lỗi, hãy:"
echo "   1. Dùng Python từ MSYS2: /c/msys64/mingw64/bin/python3"
echo "   2. Hoặc rebuild module với MSVC"
echo ""

