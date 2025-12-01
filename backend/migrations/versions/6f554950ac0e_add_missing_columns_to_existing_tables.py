"""add_missing_columns_to_existing_tables

Revision ID: 6f554950ac0e
Revises: 06aeee49f6ae
Create Date: 2025-11-19 22:08:45.871102

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f554950ac0e'
down_revision: Union[str, None] = '06aeee49f6ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Kiểm tra xem bảng users có tồn tại không
    # Nếu không tồn tại, tạo bảng trước
    # Sử dụng schema 'public' mặc định
    op.execute("""
        DO $$ 
        BEGIN
            -- Kiểm tra xem bảng users có tồn tại không (trong schema public)
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'users'
            ) THEN
                -- Tạo bảng users với các cột cơ bản
                CREATE TABLE public.users (
                    id VARCHAR(36) PRIMARY KEY,
                    username VARCHAR(32) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    elo_rating INTEGER DEFAULT 1500,
                    coins INTEGER DEFAULT 0,
                    display_name VARCHAR(64),
                    avatar_url VARCHAR(255),
                    preferences JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_login TIMESTAMP WITH TIME ZONE
                );
                
                -- Tạo indexes
                CREATE INDEX ix_users_username ON public.users(username);
                CREATE INDEX ix_users_email ON public.users(email);
            END IF;
        END $$;
    """)
    
    # Sau khi đảm bảo bảng đã tồn tại, thêm các cột còn thiếu
    op.execute("""
        DO $$ 
        BEGIN
            -- Chỉ thêm cột nếu bảng đã tồn tại
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'users'
            ) THEN
                -- Kiểm tra và thêm display_name
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users' 
                    AND column_name = 'display_name'
                ) THEN
                    ALTER TABLE public.users ADD COLUMN display_name VARCHAR(64);
                END IF;
                
                -- Kiểm tra và thêm avatar_url
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users' 
                    AND column_name = 'avatar_url'
                ) THEN
                    ALTER TABLE public.users ADD COLUMN avatar_url VARCHAR(255);
                END IF;
                
                -- Kiểm tra và thêm preferences (JSON)
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users' 
                    AND column_name = 'preferences'
                ) THEN
                    ALTER TABLE public.users ADD COLUMN preferences JSONB;
                END IF;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # Xóa các cột đã thêm (nếu cần rollback)
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'users' 
                AND column_name = 'preferences'
            ) THEN
                ALTER TABLE public.users DROP COLUMN preferences;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'users' 
                AND column_name = 'avatar_url'
            ) THEN
                ALTER TABLE public.users DROP COLUMN avatar_url;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'users' 
                AND column_name = 'display_name'
            ) THEN
                ALTER TABLE public.users DROP COLUMN display_name;
            END IF;
        END $$;
    """)

