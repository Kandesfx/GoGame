"""Script để fix xung đột pymongo/motor - version 2."""

import subprocess
import sys


def main():
    print("=" * 60)
    print("Fix pymongo/motor compatibility issue (v2)")
    print("=" * 60)
    print()

    # Check current versions
    try:
        import pymongo
        print(f"Current pymongo version: {pymongo.__version__}")
    except ImportError:
        print("pymongo not installed")

    # Uninstall conflicting packages
    print("\n1. Uninstalling pymongo and motor...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pymongo", "motor"], check=False)

    print("\n2. Installing compatible versions...")
    # Install pymongo 4.6.x - 4.9.x (most compatible with motor 3.4.0)
    subprocess.run([sys.executable, "-m", "pip", "install", "pymongo>=4.6.0,<4.10.0"], check=True)
    # Then install motor
    subprocess.run([sys.executable, "-m", "pip", "install", "motor==3.4.0"], check=True)

    print("\n3. Verifying installation...")
    try:
        import pymongo
        import motor
        print(f"✅ pymongo version: {pymongo.__version__}")
        # Test import
        from motor.motor_asyncio import AsyncIOMotorClient
        print("✅ motor import successful!")
        print("\n✅ Installation successful!")
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Trying alternative: upgrade motor to latest...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "motor"], check=True)
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            print("✅ motor upgrade successful!")
            return 0
        except ImportError as e2:
            print(f"❌ Still failing: {e2}")
            return 1


if __name__ == "__main__":
    sys.exit(main())

