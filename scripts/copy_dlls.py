"""Copy MinGW DLLs vào site-packages để module có thể load."""

import os
import shutil
import sys
from pathlib import Path

# Paths
VENV_LIB = Path(sys.executable).parent.parent / "Lib" / "site-packages"
MINGW_BIN = Path("C:/msys64/mingw64/bin")

# Required DLLs
REQUIRED_DLLS = [
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll",
]


def copy_dlls():
    """Copy DLLs vào site-packages."""
    if not MINGW_BIN.exists():
        print(f"❌ MinGW bin directory không tồn tại: {MINGW_BIN}")
        print(f"   Hãy kiểm tra đường dẫn MSYS2")
        return False
    
    print(f"Copying DLLs từ {MINGW_BIN} vào {VENV_LIB}...")
    
    copied = []
    for dll in REQUIRED_DLLS:
        src = MINGW_BIN / dll
        dst = VENV_LIB / dll
        
        if src.exists():
            try:
                shutil.copy2(src, dst)
                print(f"✅ Copied: {dll}")
                copied.append(dll)
            except Exception as e:
                print(f"❌ Failed to copy {dll}: {e}")
        else:
            print(f"⚠️  DLL not found: {src}")
    
    if copied:
        print(f"\n✅ Đã copy {len(copied)} DLLs")
        return True
    else:
        print(f"\n❌ Không copy được DLL nào")
        return False


def test_import():
    """Test import sau khi copy DLLs."""
    print("\n" + "=" * 60)
    print("Testing import after copying DLLs")
    print("=" * 60)
    
    try:
        import gogame_py
        print("✅ Import thành công!")
        
        # Test basic functionality
        board = gogame_py.Board(9)
        print(f"✅ Board created: size={board.size()}")
        
        # Test Color enum
        print(f"✅ Color enum: {gogame_py.Color.Black}, {gogame_py.Color.White}")
        
        # Test AIPlayer
        ai = gogame_py.AIPlayer()
        print(f"✅ AIPlayer created")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Copying MinGW DLLs")
    print("=" * 60)
    
    if copy_dlls():
        if test_import():
            print("\n" + "=" * 60)
            print("✅ Module sẵn sàng!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️  DLLs đã copy nhưng import vẫn fail")
            print("=" * 60)
            print("\nCó thể cần:")
            print("  1. Restart terminal/shell")
            print("  2. Hoặc rebuild module với MSVC thay vì MinGW")
    else:
        print("\n" + "=" * 60)
        print("❌ Failed to copy DLLs")
        print("=" * 60)

