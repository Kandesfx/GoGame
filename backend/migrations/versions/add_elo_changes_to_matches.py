"""Add ELO changes to matches

Revision ID: add_elo_changes
Revises: add_time_control
Create Date: 2024-01-XX XX:XX:XX.XXXXXX

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_elo_changes'
down_revision: Union[str, None] = 'add_time_control'  # Update this to the latest revision
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add ELO change columns to matches table (chỉ nếu table tồn tại)
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'matches'
            ) THEN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'black_elo_change'
                ) THEN
                    ALTER TABLE matches ADD COLUMN black_elo_change INTEGER;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'white_elo_change'
                ) THEN
                    ALTER TABLE matches ADD COLUMN white_elo_change INTEGER;
                END IF;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # Remove ELO change columns from matches table (chỉ nếu table tồn tại)
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
                    AND table_name = 'matches' AND column_name = 'white_elo_change'
                ) THEN
                    ALTER TABLE matches DROP COLUMN white_elo_change;
                END IF;
                
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'black_elo_change'
                ) THEN
                    ALTER TABLE matches DROP COLUMN black_elo_change;
                END IF;
            END IF;
        END $$;
    """)

