"""add_room_code_to_matches

Revision ID: 3aad605b648c
Revises: 9675a5a7988c
Create Date: 2025-11-24 02:02:01.414255

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3aad605b648c'
down_revision: Union[str, None] = '9675a5a7988c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Thêm cột room_code vào bảng matches (chỉ nếu table tồn tại)
    op.execute("""
        DO $$ 
        BEGIN
            -- Check if table exists first
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'matches'
            ) THEN
                -- Check if column doesn't exist
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'room_code'
                ) THEN
                    ALTER TABLE matches ADD COLUMN room_code VARCHAR(6);
                END IF;
            END IF;
        END $$;
    """)
    
    # Tạo index cho room_code (chỉ cho các row có room_code)
    op.execute("""
        DO $$ 
        BEGIN
            -- Check if table exists first
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'matches'
            ) THEN
                -- Check if index doesn't exist
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE schemaname = 'public'
                    AND tablename = 'matches' AND indexname = 'idx_matches_room_code'
                ) THEN
                    CREATE INDEX idx_matches_room_code ON matches(room_code) 
                    WHERE room_code IS NOT NULL;
                END IF;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # Xóa index (chỉ nếu table tồn tại)
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'matches'
            ) THEN
                IF EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE schemaname = 'public'
                    AND tablename = 'matches' AND indexname = 'idx_matches_room_code'
                ) THEN
                    DROP INDEX idx_matches_room_code;
                END IF;
            END IF;
        END $$;
    """)
    
    # Xóa cột room_code (chỉ nếu table tồn tại)
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'matches'
            ) THEN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'room_code'
                ) THEN
                    ALTER TABLE matches DROP COLUMN room_code;
                END IF;
            END IF;
        END $$;
    """)

