"""Script test kết nối PostgreSQL & MongoDB."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.database import engine, get_mongo_client
from sqlalchemy import text

settings = get_settings()


def test_postgres():
    """Test kết nối PostgreSQL."""
    print("🔍 Testing PostgreSQL connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ PostgreSQL connected!")
            print(f"   Version: {version[:50]}...")
            return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False


async def test_mongodb_async():
    """Test kết nối MongoDB (async)."""
    try:
        client = get_mongo_client()
        # Test connection (async)
        await client.admin.command("ping")
        server_info = await client.server_info()
        print(f"✅ MongoDB connected!")
        print(f"   Version: {server_info.get('version', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False


def test_mongodb():
    """Test kết nối MongoDB (wrapper for async)."""
    print("\n🔍 Testing MongoDB connection...")
    try:
        # Thử lấy event loop hiện tại
        try:
            loop = asyncio.get_running_loop()
            # Nếu đã có loop đang chạy, dùng run_until_complete
            return loop.run_until_complete(test_mongodb_async())
        except RuntimeError:
            # Nếu không có loop, tạo mới
            return asyncio.run(test_mongodb_async())
    except Exception:
        # Fallback: tạo event loop mới
        return asyncio.run(test_mongodb_async())


def main():
    """Chạy tất cả tests."""
    print("=" * 60)
    print("Database Connection Test")
    print("=" * 60)
    print(f"\nPostgreSQL DSN: {settings.postgres_dsn}")
    print(f"MongoDB DSN: {settings.mongo_dsn}")
    print(f"MongoDB Database: {settings.mongo_database}\n")

    pg_ok = test_postgres()
    mongo_ok = test_mongodb()

    print("\n" + "=" * 60)
    if pg_ok and mongo_ok:
        print("✅ All database connections successful!")
        return 0
    else:
        print("❌ Some database connections failed!")
        print("\n💡 Tips:")
        if not pg_ok:
            print("   - Kiểm tra PostgreSQL đang chạy")
            print("   - Kiểm tra POSTGRES_DSN trong .env")
        if not mongo_ok:
            print("   - Kiểm tra MongoDB đang chạy")
            print("   - Kiểm tra MONGO_DSN trong .env")
        return 1


if __name__ == "__main__":
    sys.exit(main())

