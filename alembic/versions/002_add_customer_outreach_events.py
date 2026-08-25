"""
Migration 002: Create customer_outreach_events table.

This table backs the Phase 5 cooldown guardrail.
Records each outreach attempt (payment link sent) per customer identifier.

Downgrade: DROP TABLE customer_outreach_events.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "customer_outreach_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "customer_identifier",
            sa.String(255),
            nullable=False,
            comment="Normalized email or contact. Used for cooldown lookups.",
        ),
        sa.Column(
            "payment_record_id",
            sa.Integer(),
            sa.ForeignKey("payment_records.id"),
            nullable=False,
        ),
        sa.Column(
            "recovery_decision_id",
            sa.Integer(),
            sa.ForeignKey("recovery_decisions.id"),
            nullable=False,
        ),
        sa.Column(
            "action",
            sa.String(64),
            nullable=False,
            comment="RecoveryAction enum value, e.g. SEND_PAYMENT_LINK",
        ),
        sa.Column(
            "channel",
            sa.String(64),
            nullable=False,
            comment="Outreach channel, e.g. payment_link",
        ),
        sa.Column(
            "outreach_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_customer_outreach_events_identifier_at",
        "customer_outreach_events",
        ["customer_identifier", "outreach_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_customer_outreach_events_identifier_at", table_name="customer_outreach_events")
    op.drop_table("customer_outreach_events")
