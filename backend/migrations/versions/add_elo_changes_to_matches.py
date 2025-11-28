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
    op.add_column('matches', sa.Column('black_elo_change', sa.Integer(), nullable=True))
    op.add_column('matches', sa.Column('white_elo_change', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('matches', 'white_elo_change')
    op.drop_column('matches', 'black_elo_change')

