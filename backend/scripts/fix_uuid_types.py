#!/usr/bin/env python3
"""Script to fix UUID type conversion in database."""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
import psycopg

def main():
    """Main function."""
    print("=" * 60)
    print("  Fix UUID Types in Database")
    print("=" * 60)
    print()
    
    # Load .env
    env_path = backend_dir / ".env"
    if not env_path.exists():
        print("❌ ERROR: .env file not found!")
        print(f"   Expected at: {env_path}")
        print()
        print("💡 Please create .env file from env.example")
        sys.exit(1)
    
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
    
    # Get database connection
    postgres_dsn = os.getenv("POSTGRES_DSN")
    if not postgres_dsn:
        print("❌ ERROR: POSTGRES_DSN not found in .env!")
        sys.exit(1)
    
    print(f"📝 PostgreSQL DSN: {postgres_dsn.split('@')[0]}@***")
    print()
    
    # Read SQL script
    sql_path = backend_dir / "migrations" / "fix_uuid_types.sql"
    if not sql_path.exists():
        print(f"❌ ERROR: SQL script not found at {sql_path}")
        sys.exit(1)
    
    print(f"📄 Reading SQL script from {sql_path.name}...")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()
    
    print()
    print("🔌 Connecting to database...")
    try:
        conn = psycopg.connect(postgres_dsn)
        print("✅ Connected to database")
        print()
        
        print("🔄 Running UUID conversion...")
        with conn.cursor() as cur:
            cur.execute(sql_script)
            conn.commit()
        
        print()
        print("✅ UUID conversion completed successfully!")
        print()
        
        # Verify conversion
        print("🔍 Verifying conversion...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    table_name,
                    column_name,
                    data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                    AND column_name IN ('id', 'user_id', 'match_id', 'black_player_id', 'white_player_id')
                    AND table_name IN ('users', 'matches', 'refresh_tokens', 'coin_transactions', 'premium_requests')
                ORDER BY table_name, column_name;
            """)
            
            results = cur.fetchall()
            if results:
                print()
                print("📊 Current column types:")
                for table, column, dtype in results:
                    status = "✅" if dtype == "uuid" else "⚠️ "
                    print(f"   {status} {table}.{column}: {dtype}")
        
        conn.close()
        
    except psycopg.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()

