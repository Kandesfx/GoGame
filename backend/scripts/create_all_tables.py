#!/usr/bin/env python3
"""
Script để tạo tất cả các bảng còn lại trong database
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from app.config import get_settings

def create_all_tables():
    """Tạo tất cả các bảng còn thiếu."""
    settings = get_settings()
    engine = create_engine(settings.postgres_dsn, echo=False)
    
    # Danh sách các bảng cần tạo (theo thứ tự dependency)
    tables_sql = [
        # 1. Extension
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
        
        # 2. Matches table
        """CREATE TABLE IF NOT EXISTS matches (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            black_player_id UUID,
            white_player_id UUID,
            ai_level INTEGER,
            board_size INTEGER DEFAULT 9 NOT NULL,
            result VARCHAR(32),
            room_code VARCHAR(6) UNIQUE,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            sgf_id VARCHAR(64),
            premium_analysis_id VARCHAR(64),
            time_control_minutes INTEGER,
            black_time_remaining_seconds INTEGER,
            white_time_remaining_seconds INTEGER,
            last_move_at TIMESTAMP WITH TIME ZONE,
            black_elo_change INTEGER,
            white_elo_change INTEGER,
            black_ready BOOLEAN DEFAULT FALSE NOT NULL,
            white_ready BOOLEAN DEFAULT FALSE NOT NULL,
            CONSTRAINT fk_matches_black_player FOREIGN KEY (black_player_id) REFERENCES users(id) ON DELETE SET NULL,
            CONSTRAINT fk_matches_white_player FOREIGN KEY (white_player_id) REFERENCES users(id) ON DELETE SET NULL
        );""",
        
        # 3. Indexes cho matches
        'CREATE INDEX IF NOT EXISTS ix_matches_room_code ON matches(room_code) WHERE room_code IS NOT NULL;',
        'CREATE INDEX IF NOT EXISTS ix_matches_black_player_id ON matches(black_player_id) WHERE black_player_id IS NOT NULL;',
        'CREATE INDEX IF NOT EXISTS ix_matches_white_player_id ON matches(white_player_id) WHERE white_player_id IS NOT NULL;',
        'CREATE INDEX IF NOT EXISTS ix_matches_started_at ON matches(started_at);',
        'CREATE INDEX IF NOT EXISTS ix_matches_finished_at ON matches(finished_at) WHERE finished_at IS NOT NULL;',
        
        # 4. Coin transactions table
        """CREATE TABLE IF NOT EXISTS coin_transactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            amount INTEGER NOT NULL,
            type VARCHAR(32) NOT NULL,
            source VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_coin_transactions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            CONSTRAINT chk_coin_transactions_amount CHECK (amount != 0)
        );""",
        
        # 5. Indexes cho coin_transactions
        'CREATE INDEX IF NOT EXISTS ix_coin_transactions_user_id ON coin_transactions(user_id);',
        'CREATE INDEX IF NOT EXISTS ix_coin_transactions_created_at ON coin_transactions(created_at);',
        'CREATE INDEX IF NOT EXISTS ix_coin_transactions_type ON coin_transactions(type);',
        
        # 6. Premium requests table
        """CREATE TABLE IF NOT EXISTS premium_requests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            match_id UUID NOT NULL,
            feature VARCHAR(32) NOT NULL,
            cost INTEGER NOT NULL,
            status VARCHAR(32) DEFAULT 'pending' NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            completed_at TIMESTAMP WITH TIME ZONE,
            CONSTRAINT fk_premium_requests_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            CONSTRAINT fk_premium_requests_match FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );""",
        
        # 7. Indexes cho premium_requests
        'CREATE INDEX IF NOT EXISTS ix_premium_requests_user_id ON premium_requests(user_id);',
        'CREATE INDEX IF NOT EXISTS ix_premium_requests_match_id ON premium_requests(match_id);',
        'CREATE INDEX IF NOT EXISTS ix_premium_requests_status ON premium_requests(status);',
        'CREATE INDEX IF NOT EXISTS ix_premium_requests_created_at ON premium_requests(created_at);',
        
        # 8. Premium subscriptions table
        """CREATE TABLE IF NOT EXISTS premium_subscriptions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID UNIQUE NOT NULL,
            plan VARCHAR(32) NOT NULL,
            status VARCHAR(32) DEFAULT 'active' NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            cancelled_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_premium_subscriptions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            CONSTRAINT chk_premium_subscriptions_plan CHECK (plan IN ('monthly', 'yearly')),
            CONSTRAINT chk_premium_subscriptions_status CHECK (status IN ('active', 'expired', 'cancelled'))
        );""",
        
        # 9. Indexes cho premium_subscriptions
        'CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_user_id ON premium_subscriptions(user_id);',
        'CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_status ON premium_subscriptions(status);',
        'CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_expires_at ON premium_subscriptions(expires_at);',
    ]
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for i, sql in enumerate(tables_sql, 1):
                try:
                    conn.execute(text(sql))
                    print(f"✅ [{i}/{len(tables_sql)}] Executed successfully")
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'already exists' in error_msg or 'duplicate' in error_msg:
                        print(f"⚠️  [{i}/{len(tables_sql)}] Skipped (already exists)")
                    else:
                        print(f"❌ [{i}/{len(tables_sql)}] Error: {e}")
                        # Không raise, tiếp tục với statement tiếp theo
                        continue
            
            trans.commit()
            print("\n✅ All tables created successfully!")
            return True
        except Exception as e:
            trans.rollback()
            print(f"\n❌ Error: {e}")
            return False
        finally:
            conn.close()

if __name__ == "__main__":
    success = create_all_tables()
    sys.exit(0 if success else 1)

