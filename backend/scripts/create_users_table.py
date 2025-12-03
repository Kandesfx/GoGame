#!/usr/bin/env python3
"""
Script đơn giản để tạo bảng users trên Fly.io
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from app.config import get_settings

def create_users_table():
    """Tạo bảng users nếu chưa tồn tại."""
    settings = get_settings()
    engine = create_engine(settings.postgres_dsn, echo=False)
    
    sql = """
    -- Tạo extension nếu chưa có
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- Tạo bảng users
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        username VARCHAR(32) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        elo_rating INTEGER DEFAULT 1500 NOT NULL,
        coins INTEGER DEFAULT 0 NOT NULL,
        display_name VARCHAR(64),
        avatar_url VARCHAR(255),
        preferences JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
        last_login TIMESTAMP WITH TIME ZONE
    );
    
    -- Tạo indexes
    CREATE INDEX IF NOT EXISTS ix_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS ix_users_elo_rating ON users(elo_rating);
    """
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # Chia thành các statements
            for statement in sql.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        conn.execute(text(statement))
                        print(f"✅ Executed: {statement[:50]}...")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'already exists' in error_msg or 'duplicate' in error_msg:
                            print(f"⚠️  Skipped (already exists): {statement[:50]}...")
                        else:
                            print(f"❌ Error: {e}")
                            raise
            
            trans.commit()
            print("✅ Users table created successfully!")
            return True
        except Exception as e:
            trans.rollback()
            print(f"❌ Error: {e}")
            return False
        finally:
            conn.close()

if __name__ == "__main__":
    success = create_users_table()
    sys.exit(0 if success else 1)

