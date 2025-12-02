"""add_time_control_to_matches

Revision ID: add_time_control
Revises: 3aad605b648c
Create Date: 2024-01-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_time_control'
down_revision: Union[str, None] = '3aad605b648c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add time control columns to matches table (chỉ nếu table tồn tại)
    op.execute("""
        DO $$ 
        BEGIN
            -- Check if table exists first
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'matches'
            ) THEN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'time_control_minutes'
                ) THEN
                    ALTER TABLE matches ADD COLUMN time_control_minutes INTEGER;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'black_time_remaining_seconds'
                ) THEN
                    ALTER TABLE matches ADD COLUMN black_time_remaining_seconds INTEGER;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'white_time_remaining_seconds'
                ) THEN
                    ALTER TABLE matches ADD COLUMN white_time_remaining_seconds INTEGER;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'last_move_at'
                ) THEN
                    ALTER TABLE matches ADD COLUMN last_move_at TIMESTAMP WITH TIME ZONE;
                END IF;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # Remove time control columns from matches table (chỉ nếu table tồn tại)
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
                    AND table_name = 'matches' AND column_name = 'last_move_at'
                ) THEN
                    ALTER TABLE matches DROP COLUMN last_move_at;
                END IF;
                
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'white_time_remaining_seconds'
                ) THEN
                    ALTER TABLE matches DROP COLUMN white_time_remaining_seconds;
                END IF;
                
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'black_time_remaining_seconds'
                ) THEN
                    ALTER TABLE matches DROP COLUMN black_time_remaining_seconds;
                END IF;
                
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'time_control_minutes'
                ) THEN
                    ALTER TABLE matches DROP COLUMN time_control_minutes;
                END IF;
            END IF;
        END $$;
    """)

