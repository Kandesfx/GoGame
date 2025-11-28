"""Script để install gogame_py module vào Python environment."""

import os
import shutil
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
VENV_LIB = Path(sys.executable).parent.parent / "Lib" / "site-packages"

# Tên file module (có thể khác nhau tùy platform)
MODULE_NAMES = [
    "gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd",
    "gogame_py.pyd",
    "gogame_py.so",
]


def find_module_file() -> Path | None:
    """Tìm file module trong build directory."""
    for name in MODULE_NAMES:
        module_path = BUILD_DIR / name
        if module_path.exists():
            return module_path
    return None


def install_module():
    """Copy module vào site-packages."""
    module_file = find_module_file()
    
    if not module_file:
        print(f"❌ Không tìm thấy gogame_py module trong {BUILD_DIR}")
        print(f"   Hãy build module trước: cmake --build build")
        return False
    
    print(f"✅ Tìm thấy module: {module_file}")
    
    # Copy vào site-packages
    target = VENV_LIB / "gogame_py.pyd"
    
    try:
        shutil.copy2(module_file, target)
        print(f"✅ Đã copy module vào: {target}")
        
        # Test import
        import gogame_py
        print(f"✅ Module import thành công!")
        print(f"   Available classes: {[x for x in dir(gogame_py) if not x.startswith('_')]}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"   Có thể thiếu MinGW DLLs. Hãy thêm C:\\msys64\\mingw64\\bin vào PATH")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Installing gogame_py module")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Build dir: {BUILD_DIR}")
    print(f"Target: {VENV_LIB}")
    print()
    
    success = install_module()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Installation completed!")
        print("=" * 60)
        print("\nBây giờ bạn có thể:")
        print("  python -c 'import gogame_py'")
        print("  python scripts/test_premium.py  # Test với AI thực tế")
    else:
        print("\n" + "=" * 60)
        print("❌ Installation failed")
        print("=" * 60)
        sys.exit(1)

