#!/usr/bin/env python3
"""
Script tự động setup database cho GoGame backend.

Script này sẽ:
1. Đọc cấu hình từ .env
2. Tạo database nếu chưa tồn tại
3. Chạy migrations
4. Kiểm tra kết nối

Usage:
    python scripts/setup_database.py
    hoặc
    python -m scripts.setup_database
"""

import os
import sys
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qs

# Thêm thư mục backend vào path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

try:
    from dotenv import load_dotenv
    import psycopg
    from psycopg import sql
except ImportError as e:
    print(f"❌ Thiếu dependencies: {e}")
    print("📦 Cài đặt: pip install python-dotenv psycopg[binary]")
    sys.exit(1)


def load_env_file():
    """Load file .env từ thư mục backend."""
    env_file = backend_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Đã load file .env từ {env_file}")
        return True
    else:
        print(f"⚠️  Không tìm thấy file .env tại {env_file}")
        print("💡 Tạo file .env từ env.example:")
        print(f"   cp {backend_dir / 'env.example'} {env_file}")
        return False


def parse_postgres_dsn(dsn: str) -> dict:
    """
    Parse PostgreSQL DSN string.
    
    Format: postgresql+psycopg://user:password@host:port/database
    """
    # Loại bỏ driver prefix nếu có
    dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
    dsn = dsn.replace("postgresql://", "postgresql://")
    
    parsed = urlparse(dsn)
    
    return {
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") if parsed.path else "postgres",
    }


def get_admin_connection(config: dict) -> psycopg.Connection:
    """Kết nối đến PostgreSQL với quyền admin (không chỉ định database cụ thể)."""
    admin_config = config.copy()
    admin_config["database"] = "postgres"  # Kết nối đến database mặc định
    
    try:
        conn = psycopg.connect(
            host=admin_config["host"],
            port=admin_config["port"],
            user=admin_config["user"],
            password=admin_config["password"],
            dbname=admin_config["database"]
        )
        print(f"✅ Đã kết nối đến PostgreSQL tại {admin_config['host']}:{admin_config['port']}")
        return conn
    except psycopg.OperationalError as e:
        print(f"❌ Không thể kết nối đến PostgreSQL: {e}")
        print("\n💡 Kiểm tra:")
        print("   1. PostgreSQL đang chạy")
        print("   2. Thông tin trong .env đúng")
        print("   3. User có quyền tạo database")
        sys.exit(1)


def database_exists(conn: psycopg.Connection, dbname: str) -> bool:
    """Kiểm tra xem database có tồn tại không."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (dbname,)
        )
        return cur.fetchone() is not None


def create_database(conn: psycopg.Connection, dbname: str):
    """Tạo database mới."""
    # PostgreSQL không cho phép tạo database trong transaction
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            # Kiểm tra xem database đã tồn tại chưa
            if database_exists(conn, dbname):
                print(f"ℹ️  Database '{dbname}' đã tồn tại")
                return
            
            # Tạo database
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
            print(f"✅ Đã tạo database '{dbname}'")
    except psycopg.Error as e:
        print(f"❌ Lỗi khi tạo database: {e}")
        sys.exit(1)
    finally:
        conn.autocommit = False


def create_user_if_needed(conn: psycopg.Connection, username: str, password: str):
    """Tạo user nếu chưa tồn tại."""
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            # Kiểm tra user có tồn tại không
            cur.execute(
                "SELECT 1 FROM pg_user WHERE usename = %s",
                (username,)
            )
            if cur.fetchone():
                print(f"ℹ️  User '{username}' đã tồn tại")
                return
            
            # Tạo user
            cur.execute(
                sql.SQL("CREATE USER {} WITH PASSWORD %s").format(sql.Identifier(username)),
                (password,)
            )
            print(f"✅ Đã tạo user '{username}'")
            
            # Cấp quyền
            cur.execute(
                sql.SQL("ALTER USER {} CREATEDB").format(sql.Identifier(username))
            )
            print(f"✅ Đã cấp quyền CREATEDB cho user '{username}'")
    except psycopg.Error as e:
        print(f"⚠️  Không thể tạo user (có thể đã tồn tại): {e}")
    finally:
        conn.autocommit = False


def run_migrations(backend_dir: Path):
    """Chạy Alembic migrations."""
    import subprocess
    
    print("\n🔄 Đang chạy migrations...")
    
    # Chuyển đến thư mục backend
    os.chdir(backend_dir)
    
    try:
        # Chạy alembic upgrade head
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Migrations đã chạy thành công")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy migrations:")
        print(e.stderr)
        print("\n💡 Xem backend/migrations/TROUBLESHOOTING.md để biết cách xử lý")
        sys.exit(1)


def test_connection(config: dict):
    """Kiểm tra kết nối đến database."""
    try:
        conn = psycopg.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            dbname=config["database"]
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"\n✅ Kết nối database thành công!")
            print(f"   PostgreSQL version: {version.split(',')[0]}")
            
            # Kiểm tra các bảng đã được tạo
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
            if tables:
                print(f"\n📊 Các bảng đã được tạo ({len(tables)}):")
                for table in tables:
                    print(f"   - {table}")
            else:
                print("\n⚠️  Chưa có bảng nào trong database")
        
        conn.close()
        return True
    except psycopg.Error as e:
        print(f"❌ Không thể kết nối đến database: {e}")
        return False


def main():
    """Hàm chính."""
    print("=" * 60)
    print("🚀 GoGame Database Setup Script")
    print("=" * 60)
    print()
    
    # Load .env file
    if not load_env_file():
        print("\n❌ Không thể tiếp tục mà không có file .env")
        sys.exit(1)
    
    # Lấy DSN từ environment
    postgres_dsn = os.getenv("POSTGRES_DSN")
    if not postgres_dsn:
        print("❌ POSTGRES_DSN không được tìm thấy trong .env")
        sys.exit(1)
    
    print(f"📝 PostgreSQL DSN: {postgres_dsn.replace(postgres_dsn.split('@')[0].split(':')[-1], '***')}")
    
    # Parse DSN
    config = parse_postgres_dsn(postgres_dsn)
    dbname = config["database"]
    username = config["user"]
    password = config["password"]
    
    print(f"\n📋 Thông tin database:")
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Database: {dbname}")
    print(f"   User: {username}")
    
    # Kết nối với quyền admin
    print("\n🔌 Đang kết nối đến PostgreSQL...")
    admin_conn = get_admin_connection(config)
    
    # Tạo user nếu cần (nếu user khác postgres)
    if username != "postgres" and password:
        print(f"\n👤 Đang kiểm tra/tạo user '{username}'...")
        create_user_if_needed(admin_conn, username, password)
    
    # Tạo database
    print(f"\n💾 Đang kiểm tra/tạo database '{dbname}'...")
    create_database(admin_conn, dbname)
    
    admin_conn.close()
    
    # Chạy migrations
    run_migrations(backend_dir)
    
    # Kiểm tra kết nối
    print("\n🔍 Đang kiểm tra kết nối...")
    if test_connection(config):
        print("\n" + "=" * 60)
        print("✅ Database setup hoàn tất!")
        print("=" * 60)
        print("\n💡 Bạn có thể chạy backend server:")
        print("   python -m uvicorn app.main:app --reload")
    else:
        print("\n❌ Có lỗi xảy ra. Vui lòng kiểm tra lại.")
        sys.exit(1)


if __name__ == "__main__":
    main()

