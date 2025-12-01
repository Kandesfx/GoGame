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
    # Add black_ready and white_ready columns to matches table
    op.add_column('matches', sa.Column('black_ready', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('matches', sa.Column('white_ready', sa.Boolean(), nullable=False, server_default='false'))


def downgrade():
    # Remove black_ready and white_ready columns
    op.drop_column('matches', 'white_ready')
    op.drop_column('matches', 'black_ready')

