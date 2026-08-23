"""
Recovery Opportunity Engine — FastAPI application entry point.

Phase 1 exposed endpoints:
  POST /webhooks/razorpay  — Razorpay webhook receiver
  GET  /health             — Liveness check
  GET  /events/{event_id}  — Retrieve a webhook event by Razorpay event ID

NOT exposed (Phase 1 scope boundary):
  - General CRUD API
  - Recovery action execution
  - Policy management
  - Dashboard / metrics endpoints
  - ML model management
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.models import WebhookEvent
from app.schemas import EventDetailResponse, HealthResponse
from app.webhooks import router as webhook_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Recovery Opportunity Engine — Phase 1")
        logger.info("Environment: %s", settings.app_env)
        logger.info("Database: %s", settings.database_url)
        # Log sanitized config — NEVER log secrets
        logger.info("Config: %r", settings)
        init_db()
        logger.info("Database initialized successfully")
        yield
        logger.info("Shutting down Recovery Opportunity Engine")

    app = FastAPI(
        title="Recovery Opportunity Engine",
        description=(
            "AI-driven revenue recovery decision engine for Razorpay Buildathon Track 03. "
            "Phase 1: Event ingestion foundation."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Routes ─────────────────────────────────────────────────────────
    app.include_router(webhook_router, tags=["Webhooks"])

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness check",
        tags=["Health"],
    )
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            environment=settings.app_env,
        )

    @app.get(
        "/events/{event_id}",
        response_model=EventDetailResponse,
        summary="Retrieve a webhook event by Razorpay event ID",
        tags=["Events"],
    )
    def get_event(
        event_id: str,
        db: Session = Depends(get_db),
    ) -> EventDetailResponse:
        """
        Look up a webhook event by its Razorpay event ID.

        Returns event metadata (NOT the raw payload, which may contain
        customer contact details that should not be exposed in API responses).
        """
        event = (
            db.query(WebhookEvent)
            .filter(WebhookEvent.event_id == event_id)
            .first()
        )
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event '{event_id}' not found",
            )
        return EventDetailResponse(
            event_id=event.event_id,
            event_type=event.event_type,
            received_at=event.received_at.isoformat(),
            processing_status=event.processing_status,
            signature_verified=event.signature_verified,
        )

    return app


app = create_app()
