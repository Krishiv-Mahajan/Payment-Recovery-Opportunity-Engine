"""
Recovery Opportunity Engine — FastAPI application entry point.

Phase 5 exposed endpoints:
  POST /webhooks/razorpay  — Razorpay webhook receiver
  GET  /health             — Liveness check
  GET  /events/{event_id}  — Retrieve a webhook event by Razorpay event ID
  GET  /metrics            — Prometheus operational metrics (low-cardinality only)

Executor mode (controlled by EXECUTOR_MODE env var):
  mock     (default) — MockPaymentLinkProvider, no network calls, safe for tests
  razorpay           — RazorpayPaymentLinkProvider, requires Razorpay credentials

The model and all dependencies are loaded once during lifespan startup and
injected into app.state. Webhook handlers never load models or select providers.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.executor import MockPaymentLinkProvider
from app.guardrails import GuardrailsEngine
from app.ml.policy import EconomicPolicyPredictor
from app.ml.predictor import PlaceholderPredictor
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
        logger.info("Starting Recovery Opportunity Engine — Phase 5")
        logger.info("Environment: %s", settings.app_env)
        logger.info("Database: %s", settings.database_url)
        logger.info("Config: %r", settings)

        init_db()
        logger.info("Database initialized successfully")

        # ── Load Phase 3 model ─────────────────────────────────────────────
        try:
            s_learner = joblib.load("artifacts/model_v1.joblib")
            predictor = EconomicPolicyPredictor(s_learner)
            logger.info("Loaded EconomicPolicyPredictor with Phase 3 model.")
        except Exception as e:
            logger.error(
                "Failed to load Phase 3 model (%s). Falling back to PlaceholderPredictor.", e
            )
            predictor = PlaceholderPredictor()

        # ── Select executor by configuration ───────────────────────────────
        # Default is always mock. Real Razorpay requires EXECUTOR_MODE=razorpay.
        executor_mode = settings.executor_mode.lower()

        if executor_mode == "razorpay":
            if not settings.razorpay_key_id or not settings.razorpay_key_secret:
                raise RuntimeError(
                    "EXECUTOR_MODE=razorpay requires RAZORPAY_KEY_ID and "
                    "RAZORPAY_KEY_SECRET to be set in the environment."
                )
            from app.executor_razorpay import RazorpayPaymentLinkProvider
            executor = RazorpayPaymentLinkProvider(
                key_id=settings.razorpay_key_id,
                key_secret=settings.razorpay_key_secret,
            )
            logger.info("Executor: RazorpayPaymentLinkProvider (real API calls enabled)")
        else:
            if executor_mode != "mock":
                logger.warning(
                    "Unknown EXECUTOR_MODE=%r. Defaulting to mock.", executor_mode
                )
            executor = MockPaymentLinkProvider()
            logger.info("Executor: MockPaymentLinkProvider (no real API calls)")

        # ── Inject all dependencies into app.state ────────────────────────
        app.state.predictor = predictor
        app.state.guardrails = GuardrailsEngine(cooldown_hours=settings.cooldown_hours)
        app.state.executor = executor

        yield
        logger.info("Shutting down Recovery Opportunity Engine")

    app = FastAPI(
        title="Recovery Opportunity Engine",
        description=(
            "AI-driven revenue recovery decision engine for Razorpay Buildathon Track 03. "
            "Phase 5: Production readiness — real execution, cooldown guardrail, metrics."
        ),
        version="0.5.0",
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
            version="0.5.0",
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

    @app.get(
        "/metrics",
        summary="Prometheus operational metrics",
        tags=["Observability"],
    )
    def metrics() -> PlainTextResponse:
        """
        Expose Prometheus text-format operational metrics.

        Labels are LOW CARDINALITY only (event_type, action, status, rule).
        Payment IDs, customer identifiers, and execution references are never
        included in metric labels or values.
        """
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()
