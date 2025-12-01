"""increase_refresh_token_length

Revision ID: 9675a5a7988c
Revises: 6f554950ac0e
Create Date: 2025-11-19 22:09:55.645629

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9675a5a7988c'
down_revision: Union[str, None] = '6f554950ac0e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tăng độ dài cột token trong refresh_tokens từ VARCHAR(255) lên TEXT
    op.execute("""
        ALTER TABLE refresh_tokens 
        ALTER COLUMN token TYPE TEXT;
    """)


def downgrade() -> None:
    # Rollback về VARCHAR(255) - có thể mất dữ liệu nếu token > 255 ký tự
    op.execute("""
        ALTER TABLE refresh_tokens 
        ALTER COLUMN token TYPE VARCHAR(255);
    """)

