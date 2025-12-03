#!/usr/bin/env python3
"""
Script để setup database schema trên Fly.io
Chạy SQL script trực tiếp thay vì dùng migrations
"""

import os
import sys
from pathlib import Path

# Add parent directory to path để import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from app.config import get_settings

def setup_database():
    """Setup database schema từ SQL script."""
    settings = get_settings()
    
    # Tạo engine
    engine = create_engine(settings.postgres_dsn, echo=False)
    
    # Đọc SQL script
    # Script nằm trong /app/scripts/, SQL file cũng ở đó
    sql_file = Path(__file__).parent / "database_schema.sql"
    
    # Fallback: nếu không tìm thấy, thử path khác
    if not sql_file.exists():
        sql_file = Path("/app/scripts/database_schema.sql")
    
    if not sql_file.exists():
        print(f"❌ SQL file not found: {sql_file}")
        return False
    
    print(f"📄 Reading SQL script: {sql_file}")
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Loại bỏ các dòng CREATE DATABASE (không cần trên Fly.io)
    # và các dòng comment không cần thiết
    lines = []
    skip_next = False
    for line in sql_content.split('\n'):
        # Skip CREATE DATABASE statements
        if 'CREATE DATABASE' in line.upper() or 'DROP DATABASE' in line.upper():
            continue
        # Skip connection statements
        if '\\c' in line or 'CONNECT' in line.upper():
            continue
        lines.append(line)
    
    sql_content = '\n'.join(lines)
    
    # Chia thành các statements - parse tốt hơn
    statements = []
    current_statement = []
    in_multiline = False
    
    for line in sql_content.split('\n'):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('--'):
            continue
        
        # Skip CREATE DATABASE và DROP DATABASE (không cần trên Fly.io)
        if 'CREATE DATABASE' in line.upper() or 'DROP DATABASE' in line.upper():
            continue
        
        current_statement.append(line)
        
        # Kết thúc statement khi gặp dấu chấm phẩy (không nằm trong string)
        if line.endswith(';'):
            statement = ' '.join(current_statement)
            if statement.strip() and not statement.upper().startswith('CREATE DATABASE'):
                statements.append(statement)
            current_statement = []
    
    print(f"📊 Found {len(statements)} SQL statements")
    
    # Execute từng statement
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for i, statement in enumerate(statements, 1):
                try:
                    print(f"  [{i}/{len(statements)}] Executing statement...")
                    conn.execute(text(statement))
                except Exception as e:
                    # Ignore errors nếu table/column đã tồn tại
                    error_msg = str(e).lower()
                    if 'already exists' in error_msg or 'duplicate' in error_msg:
                        print(f"  ⚠️  Skipping (already exists): {str(e)[:100]}")
                        continue
                    else:
                        print(f"  ❌ Error: {e}")
                        # Không raise, tiếp tục với statement tiếp theo
                        continue
            
            trans.commit()
            print("✅ Database schema setup completed!")
            return True
            
        except Exception as e:
            trans.rollback()
            print(f"❌ Error setting up database: {e}")
            return False
        finally:
            conn.close()

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)

