"""
Application configuration.

All sensitive values are loaded from environment variables via python-dotenv.
Secrets are never logged or exposed through API responses.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required env vars (see .env.example):
      RAZORPAY_KEY_ID
      RAZORPAY_KEY_SECRET
      RAZORPAY_WEBHOOK_SECRET
      DATABASE_URL
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Razorpay (test-mode) ────────────────────────────────────────────────
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # ── Database ───────────────────────────────────────────────────────────
    # SQLite for development; swap to postgresql+asyncpg://... for production
    database_url: str = "sqlite:///./roe.db"

    # ── Application ────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Executor ───────────────────────────────────────────────────────────
    # Controls which PaymentLinkProvider implementation is used at startup.
    # 'mock' is the permanent default — safe for tests and local development.
    # 'razorpay' requires RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to be set.
    # Real Razorpay calls NEVER happen unless EXECUTOR_MODE=razorpay is explicit.
    executor_mode: str = "mock"  # Literal["mock", "razorpay"]

    execution_stale_after_seconds: int = 600

    # ── Background Worker ──────────────────────────────────────────────────
    worker_enabled: bool = False
    worker_poll_interval_seconds: int = 5
    worker_batch_size: int = 10

    # ── Guardrails ─────────────────────────────────────────────────────────
    # Cooldown window in hours. The cooldown guardrail blocks re-outreach to
    # the same customer_identifier within this window.
    cooldown_hours: int = 48

    # ── Experimentation ────────────────────────────────────────────────────
    experiment_name: str | None = None
    control_percentage: int = 50

    def __repr__(self) -> str:
        # Deliberately omit secrets from repr/log output
        return (
            f"Settings(app_env={self.app_env!r}, "
            f"database_url={self.database_url!r}, "
            f"razorpay_key_id={'[SET]' if self.razorpay_key_id else '[MISSING]'})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache means we parse env vars exactly once per process.
    Tests can override by calling get_settings.cache_clear() before patching.
    """
    return Settings()
