"""
Shared test fixtures and configuration.

These fixtures are available to all tests via pytest's conftest discovery.

Design decisions:
  - All tests use an in-memory SQLite database (not the dev roe.db file).
  - Settings are overridden to use a deterministic test webhook secret.
  - The FastAPI TestClient is synchronous — no asyncio complexity in tests.
  - No live Razorpay credentials are required for any test.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings
from app.database import Base, get_db
from app.main import create_app

# ── Test constants ─────────────────────────────────────────────────────────────

TEST_WEBHOOK_SECRET = "test_webhook_secret_do_not_use_in_production"

# A minimal valid payment.failed payload mirroring the Razorpay webhook structure.
# All IDs are clearly fictional test values.
PAYMENT_FAILED_PAYLOAD = {
    "entity": "event",
    "account_id": "acc_test000000001",
    "event": "payment.failed",
    "contains": ["payment"],
    "payload": {
        "payment": {
            "entity": {
                "id": "pay_test000000001",
                "entity": "payment",
                "amount": 50000,
                "currency": "INR",
                "status": "failed",
                "order_id": "order_test00000001",
                "method": "card",
                "captured": False,
                "email": "test.customer@example.com",
                "contact": "+919999999999",
                "error_code": "BAD_REQUEST_ERROR",
                "error_description": "Payment failed during authentication",
                "error_source": "customer",
                "error_step": "payment_authentication",
                "error_reason": "payment_failed",
                "created_at": 1700000000,
            }
        }
    },
    "created_at": 1700000000,
    "id": "evt_test000000001",
}

PAYMENT_CAPTURED_PAYLOAD = {
    "entity": "event",
    "account_id": "acc_test000000001",
    "event": "payment.captured",
    "contains": ["payment"],
    "payload": {
        "payment": {
            "entity": {
                "id": "pay_test000000002",
                "entity": "payment",
                "amount": 75000,
                "currency": "INR",
                "status": "captured",
                "order_id": "order_test00000002",
                "method": "upi",
                "captured": True,
                "email": "another.customer@example.com",
                "contact": "+918888888888",
                "created_at": 1700000100,
            }
        }
    },
    "created_at": 1700000100,
    "id": "evt_test000000002",
}

ORDER_PAID_PAYLOAD = {
    "entity": "event",
    "account_id": "acc_test000000001",
    "event": "order.paid",
    "contains": ["payment", "order"],
    "payload": {
        "payment": {
            "entity": {
                "id": "pay_test000000003",
                "entity": "payment",
                "amount": 100000,
                "currency": "INR",
                "status": "captured",
                "order_id": "order_test00000003",
                "method": "netbanking",
                "captured": True,
                "email": "third.customer@example.com",
                "contact": "+917777777777",
                "created_at": 1700000200,
            }
        },
        "order": {
            "entity": {
                "id": "order_test00000003",
                "entity": "order",
                "amount": 100000,
                "amount_paid": 100000,
                "amount_due": 0,
                "currency": "INR",
                "status": "paid",
            }
        },
    },
    "created_at": 1700000200,
    "id": "evt_test000000003",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def db_engine():
    """
    Create a fresh named shared-memory SQLite engine for each test function.

    Uses SQLite's shared-cache mode with a unique database name per test.
    This allows multiple connections (app + test assertions) to share the
    same in-memory database, unlike plain sqlite:///:memory: which gives
    each new connection its own isolated database.
    """
    import tempfile
    import os
    import uuid
    from sqlalchemy import event

    # Use a temporary file instead of in-memory shared cache to support WAL mode
    # and accurate production locking semantics during concurrency tests.
    fd, path = tempfile.mkstemp(prefix=f"testdb_{uuid.uuid4().hex}_", suffix=".db")
    os.close(fd)
    
    url = f"sqlite:///{path}"
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 15},
    )

    @event.listens_for(engine, "connect")
    def set_wal_mode(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture(scope="function")
def db_session(db_engine) -> Session:
    """
    Yield a database session bound to the in-memory engine.
    Rolls back after each test to ensure isolation.
    """
    TestingSessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(db_engine, monkeypatch) -> TestClient:
    """
    FastAPI TestClient wired to the in-memory DB and test secrets.

    Patches:
      - get_settings in config, security, and main modules → test credentials
      - app.database.engine → test in-memory engine
      - app.database.SessionLocal → session factory bound to test engine
      - get_db FastAPI dependency → test session

    This ensures init_db() (called in lifespan) and all request-scoped DB
    sessions both target the same in-memory database.
    """
    from app.config import Settings
    import app.config as config_module
    import app.security as security_module
    import app.main as main_module
    import app.database as database_module

    settings_override = Settings(
        razorpay_key_id="rzp_test_TESTONLY",
        razorpay_key_secret="test_key_secret_TESTONLY",
        razorpay_webhook_secret=TEST_WEBHOOK_SECRET,
        database_url="sqlite:///:memory:",
        app_env="test",
        executor_mode="mock",
    )

    # Clear the lru_cache and patch get_settings everywhere it is imported
    get_settings.cache_clear()
    _override = lambda: settings_override  # noqa: E731

    monkeypatch.setattr(config_module, "get_settings", _override)
    monkeypatch.setattr(security_module, "get_settings", _override)
    monkeypatch.setattr(main_module, "get_settings", _override)

    # Patch the module-level engine and SessionLocal in database.py so that
    # init_db() uses the test engine instead of creating a new file/memory DB.
    TestingSessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database_module, "engine", db_engine)
    monkeypatch.setattr(database_module, "SessionLocal", TestingSessionLocal)

    # Build the app with the patched settings
    app = create_app()

    # Override the FastAPI get_db dependency to use the test session factory
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


# ── Helper functions ───────────────────────────────────────────────────────────


def make_signature(payload: dict, secret: str = TEST_WEBHOOK_SECRET) -> str:
    """
    Compute the correct HMAC-SHA256 signature for a payload dict.
    Mirrors the logic Razorpay uses to sign webhook deliveries.
    """
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()


def make_signed_request(
    client: TestClient,
    payload: dict,
    secret: str = TEST_WEBHOOK_SECRET,
) -> tuple:
    """
    POST a webhook payload with a valid signature.
    Returns (response, body_bytes) for assertions.
    """
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    response = client.post(
        "/webhooks/razorpay",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": signature,
        },
    )
    return response, body
