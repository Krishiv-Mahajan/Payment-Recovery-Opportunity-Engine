"""
Migration 001: Add Phase 4 columns to recovery_decisions table.

These columns were introduced in Phase 4 via Base.metadata.create_all().
For databases that pre-date Phase 4 (or were created fresh before this
migration chain existed), we add the columns individually.

SQLite does not support ADD COLUMN IF NOT EXISTS, so we check for
column existence before issuing ALTER TABLE statements.

Downgrade: no destructive ops — columns are left in place.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    """Check if a column already exists in a table (SQLite-compatible)."""
    bind = op.get_bind()
    result = bind.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    phase4_columns = [
        ("predicted_p0", sa.Float(), True),
        ("predicted_p1", sa.Float(), True),
        ("predicted_uplift", sa.Float(), True),
        ("expected_incremental_net_paise", sa.Integer(), True),
        ("execution_reference_id", sa.String(255), True),
        ("executed_at", sa.DateTime(timezone=True), True),
        ("outcome_observed_at", sa.DateTime(timezone=True), True),
    ]

    for col_name, col_type, nullable in phase4_columns:
        if not _column_exists("recovery_decisions", col_name):
            op.add_column(
                "recovery_decisions",
                sa.Column(col_name, col_type, nullable=nullable),
            )

    # Add index on execution_reference_id if it doesn't exist
    bind = op.get_bind()
    indexes = bind.execute(
        sa.text("PRAGMA index_list(recovery_decisions)")
    ).fetchall()
    index_names = [r[1] for r in indexes]
    if "ix_recovery_decisions_execution_reference_id" not in index_names:
        op.create_index(
            "ix_recovery_decisions_execution_reference_id",
            "recovery_decisions",
            ["execution_reference_id"],
        )


def downgrade() -> None:
    # Deliberately no destructive ops on downgrade.
    # Columns remain; this migration is marked as reverted in alembic_version.
    pass
