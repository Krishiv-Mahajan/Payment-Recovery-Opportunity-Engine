"""
Database engine and session management.

Uses SQLAlchemy with a synchronous SQLite connection for Phase 1.
The architecture is intentionally designed to swap SQLite → PostgreSQL later
by only changing DATABASE_URL — no application-layer code changes required.

Concurrency note:
  SQLite in WAL mode handles moderate concurrent reads safely.
  For production throughput, migrate to PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


def _get_engine():
    settings = get_settings()
    url = settings.database_url

    connect_args = {}
    if url.startswith("sqlite"):
        # SQLite requires this flag for use across threads in FastAPI
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        url,
        connect_args=connect_args,
        echo=False,  # Set True to see SQL statements during development
    )

    # Enable WAL mode for SQLite — better concurrent read performance
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_wal_mode(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


engine = _get_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """
    Declarative base for all ORM models.
    All models that inherit from this class will be tracked by SQLAlchemy
    and can be created via Base.metadata.create_all(engine).
    """

    pass


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in route:
        def my_route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database.

    In test environments (app_env='test'), uses SQLAlchemy create_all() for
    in-memory SQLite databases (Alembic is not used in tests).

    In all other environments, Alembic migrations are the authoritative schema
    manager. create_all() is NOT called to avoid conflicts with Alembic-managed
    schemas.
    """
    from app.config import get_settings

    settings = get_settings()

    # Import models so they register with Base.metadata
    import app.models  # noqa: F401

    if settings.app_env == "test":
        # Tests use in-memory SQLite — Alembic not applicable
        Base.metadata.create_all(bind=engine)
    else:
        # Alembic is authoritative: run pending migrations at startup
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command

        alembic_cfg = AlembicConfig("alembic.ini")
        alembic_command.upgrade(alembic_cfg, "head")
