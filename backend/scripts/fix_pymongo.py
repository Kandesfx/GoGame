"""Script để fix xung đột pymongo/motor."""

import subprocess
import sys


def main():
    print("=" * 60)
    print("Fix pymongo/motor compatibility issue")
    print("=" * 60)
    print()

    # Uninstall conflicting packages
    print("1. Uninstalling pymongo and motor...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pymongo", "motor"], check=False)

    print("\n2. Installing compatible versions...")
    # Install pymongo 4.x first
    subprocess.run([sys.executable, "-m", "pip", "install", "pymongo>=4.5.0,<5.0.0"], check=True)
    # Then install motor
    subprocess.run([sys.executable, "-m", "pip", "install", "motor==3.4.0"], check=True)

    print("\n3. Verifying installation...")
    try:
        import pymongo
        import motor
        print(f"✅ pymongo version: {pymongo.__version__}")
        print(f"✅ motor version: {motor.__version__}")
        print("\n✅ Installation successful!")
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

