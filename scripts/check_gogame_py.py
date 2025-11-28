"""Script để check và debug gogame_py module installation."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
VENV_LIB = Path(sys.executable).parent.parent / "Lib" / "site-packages"


def check_dlls():
    """Kiểm tra MinGW DLLs trong PATH."""
    print("=" * 60)
    print("Checking MinGW DLLs in PATH")
    print("=" * 60)
    
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    mingw_paths = [p for p in path_dirs if "mingw64" in p.lower() or "msys64" in p.lower()]
    
    if mingw_paths:
        print("✅ Tìm thấy MinGW paths trong PATH:")
        for p in mingw_paths:
            print(f"   {p}")
            
            # Check DLLs
            dlls = ["libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"]
            for dll in dlls:
                dll_path = Path(p) / dll
                if dll_path.exists():
                    print(f"      ✅ {dll}")
                else:
                    print(f"      ❌ {dll} not found")
    else:
        print("❌ Không tìm thấy MinGW trong PATH")
        print("\n💡 Hãy thêm vào PATH:")
        print("   C:\\msys64\\mingw64\\bin")
        print("\n   Hoặc chạy trong MSYS2 MinGW 64-bit shell")


def check_module():
    """Kiểm tra module file."""
    print("\n" + "=" * 60)
    print("Checking module files")
    print("=" * 60)
    
    # Check build directory
    module_file = BUILD_DIR / "gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd"
    if module_file.exists():
        print(f"✅ Module trong build: {module_file}")
        print(f"   Size: {module_file.stat().st_size / 1024:.1f} KB")
    else:
        print(f"❌ Module không tồn tại trong build")
        return False
    
    # Check site-packages
    installed = VENV_LIB / "gogame_py.pyd"
    if installed.exists():
        print(f"✅ Module trong site-packages: {installed}")
    else:
        print(f"⚠️  Module chưa được install vào site-packages")
        print(f"   Chạy: python scripts/install_gogame_py.py")
    
    return True


def test_import():
    """Test import module."""
    print("\n" + "=" * 60)
    print("Testing module import")
    print("=" * 60)
    
    # Add build to path
    sys.path.insert(0, str(BUILD_DIR))
    
    try:
        import gogame_py
        print("✅ Import thành công!")
        print(f"\nAvailable classes:")
        classes = [x for x in dir(gogame_py) if not x.startswith("_")]
        for cls in classes:
            print(f"   - {cls}")
        
        # Test basic functionality
        print(f"\nTesting Board creation...")
        board = gogame_py.Board(9)
        print(f"✅ Board created: size={board.size()}")
        print(f"   Current player: {board.current_player()}")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"\n💡 Solutions:")
        print(f"   1. Đảm bảo đang chạy trong MSYS2 MinGW 64-bit shell")
        print(f"   2. Thêm C:\\msys64\\mingw64\\bin vào PATH")
        print(f"   3. Hoặc copy DLLs vào cùng thư mục với module")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("gogame_py Module Checker")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    print()
    
    check_dlls()
    if not check_module():
        return
    
    if test_import():
        print("\n" + "=" * 60)
        print("✅ All checks passed!")
        print("=" * 60)
        print("\nModule sẵn sàng để sử dụng!")
    else:
        print("\n" + "=" * 60)
        print("❌ Module chưa hoạt động")
        print("=" * 60)


if __name__ == "__main__":
    main()

