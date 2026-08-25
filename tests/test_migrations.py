"""
Tests for Alembic migrations.
"""
from __future__ import annotations

import os
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine
import pytest


def test_migrations_up_and_down(tmp_path, monkeypatch):
    """
    Test that Alembic can successfully run all migrations up to head,
    and then downgrade back to base.
    """
    db_path = tmp_path / "test_migrations.db"
    url = f"sqlite:///{db_path}"

    from app.config import get_settings, Settings
    get_settings.cache_clear()
    def mock_settings():
        return Settings(database_url=url, app_env="test")
    monkeypatch.setattr("app.config.get_settings", mock_settings)
    monkeypatch.setattr("app.database.get_settings", mock_settings)


    # Seed the Phase 4 schema (before Alembic was introduced)
    from sqlalchemy import text, create_engine
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE payment_records (
                id INTEGER NOT NULL PRIMARY KEY,
                razorpay_payment_id VARCHAR(255) NOT NULL,
                amount INTEGER NOT NULL,
                currency VARCHAR(8) NOT NULL,
                status VARCHAR(10) NOT NULL
            );
        '''))
        conn.execute(text('''
            CREATE TABLE recovery_decisions (
                id INTEGER NOT NULL PRIMARY KEY,
                payment_record_id INTEGER NOT NULL,
                decision_status VARCHAR(16) NOT NULL,
                selected_action VARCHAR(31) NOT NULL,
                model_version VARCHAR(64) NOT NULL
            );
        '''))

    # Configure Alembic to use the temp database
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", url)

    # Upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Verify tables exist (sqlite specific check)
    from sqlalchemy import text
    engine = create_engine(url)
    with engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        table_names = {t[0] for t in tables}
        assert "recovery_decisions" in table_names
        assert "customer_outreach_events" in table_names
        assert "alembic_version" in table_names

    # Downgrade to base
    command.downgrade(alembic_cfg, "base")
