"""add experiment columns

Revision ID: 2e1fbffa24b6
Revises: 002
Create Date: 2026-08-25 12:30:56.617057

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2e1fbffa24b6'
down_revision: Union[str, Sequence[str], None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('recovery_decisions', sa.Column('experiment_name', sa.String(length=64), nullable=True))
    op.add_column('recovery_decisions', sa.Column('experiment_variant', sa.String(length=32), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('recovery_decisions', 'experiment_variant')
    op.drop_column('recovery_decisions', 'experiment_name')
