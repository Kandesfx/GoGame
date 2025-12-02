"""Add ready status to matches

Revision ID: add_ready_status
Revises: add_elo_changes
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_ready_status'
down_revision = 'add_elo_changes'
branch_labels = None
depends_on = None


def upgrade():
    # Add black_ready and white_ready columns to matches table (chỉ nếu table tồn tại)
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
                    AND table_name = 'matches' AND column_name = 'black_ready'
                ) THEN
                    ALTER TABLE matches ADD COLUMN black_ready BOOLEAN NOT NULL DEFAULT false;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'white_ready'
                ) THEN
                    ALTER TABLE matches ADD COLUMN white_ready BOOLEAN NOT NULL DEFAULT false;
                END IF;
            END IF;
        END $$;
    """)


def downgrade():
    # Remove black_ready and white_ready columns (chỉ nếu table tồn tại)
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
                    AND table_name = 'matches' AND column_name = 'white_ready'
                ) THEN
                    ALTER TABLE matches DROP COLUMN white_ready;
                END IF;
                
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public'
                    AND table_name = 'matches' AND column_name = 'black_ready'
                ) THEN
                    ALTER TABLE matches DROP COLUMN black_ready;
                END IF;
            END IF;
        END $$;
    """)

