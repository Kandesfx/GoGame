"""Fix Python DLL dependency cho gogame_py module."""

import os
import shutil
import sys
from pathlib import Path

VENV_LIB = Path(sys.executable).parent.parent / "Lib" / "site-packages"
PYTHON_DIR = Path(sys.executable).parent.parent.parent
PYTHON_DLLS = PYTHON_DIR / "DLLs"


def find_python_dll():
    """Tìm libpython DLL."""
    # Check Python installation directory
    possible_paths = [
        PYTHON_DLLS / "libpython3.12.dll",
        PYTHON_DIR / "libpython3.12.dll",
        Path("C:/msys64/mingw64/bin/libpython3.12.dll"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Try to find in PATH
    import subprocess
    try:
        result = subprocess.run(
            ["where", "libpython3.12.dll"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    return Path(line.strip())
    except:
        pass
    
    return None


def copy_python_dll():
    """Copy Python DLL vào site-packages."""
    dll_path = find_python_dll()
    
    if not dll_path:
        print("❌ Không tìm thấy libpython3.12.dll")
        print(f"   Đã tìm trong: {PYTHON_DLLS}, {PYTHON_DIR}")
        return False
    
    print(f"✅ Tìm thấy: {dll_path}")
    
    target = VENV_LIB / "libpython3.12.dll"
    try:
        shutil.copy2(dll_path, target)
        print(f"✅ Đã copy vào: {target}")
        return True
    except Exception as e:
        print(f"❌ Failed to copy: {e}")
        return False


def test_import():
    """Test import."""
    try:
        import gogame_py
        print("✅ Import thành công!")
        
        # Quick test
        board = gogame_py.Board(9)
        print(f"✅ Board test: size={board.size()}")
        
        ai = gogame_py.AIPlayer()
        print(f"✅ AIPlayer test: OK")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Fixing Python DLL dependency")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Python dir: {PYTHON_DIR}")
    print()
    
    if copy_python_dll():
        print("\n" + "=" * 60)
        print("Testing import...")
        print("=" * 60)
        if test_import():
            print("\n" + "=" * 60)
            print("✅ Module hoạt động!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️  Vẫn còn lỗi")
            print("=" * 60)
    else:
        print("\n💡 Có thể cần:")
        print("  1. Cài Python từ python.org (có libpython DLL)")
        print("  2. Hoặc dùng Python từ MSYS2")
        print("  3. Hoặc rebuild với MSVC")

