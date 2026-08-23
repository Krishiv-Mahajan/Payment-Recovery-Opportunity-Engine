"""
Webhook endpoint: POST /webhooks/razorpay

Responsibilities:
  1. Read the raw request body WITHOUT parsing it first.
  2. Verify the Razorpay HMAC-SHA256 signature.
  3. Parse and validate the JSON payload structure.
  4. Check for duplicate event delivery.
  5. Persist the raw event and delegate to the event processor.
  6. Always return HTTP 200 to Razorpay (non-2xx causes retry storms).

SECURITY:
  - Signature verification happens before any business logic.
  - Raw body is used for verification (never re-serialized JSON).
  - Secrets are never included in responses or logs.
  - Duplicate events are safely acknowledged and not reprocessed.

IDEMPOTENCY:
  - The Razorpay event ID (payload root "id" field) is the idempotency key.
  - Duplicate events are detected via a UNIQUE constraint on WebhookEvent.event_id.
  - Processing is skipped for duplicates; we still return 200.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.crud import append_audit_log, create_webhook_event, update_webhook_status
from app.database import get_db
from app.event_processor import process_webhook_event
from app.models import WebhookProcessingStatus
from app.schemas import RazorpayWebhookPayload, WebhookAckResponse
from app.security import verify_webhook_signature

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/webhooks/razorpay",
    response_model=WebhookAckResponse,
    summary="Razorpay webhook receiver",
    description=(
        "Receives Razorpay webhook events. "
        "Verifies HMAC-SHA256 signature before processing. "
        "Returns 200 for all valid (and duplicate) events."
    ),
)
async def receive_razorpay_webhook(
    request: Request,
    db: Session = Depends(get_db),
    x_razorpay_signature: str | None = Header(
        default=None,
        alias="X-Razorpay-Signature",
        description="HMAC-SHA256 signature from Razorpay",
    ),
) -> WebhookAckResponse:
    """
    Primary webhook handler.

    Returns HTTP 200 for:
      - Successfully processed events
      - Duplicate events (already processed)

    Returns HTTP 400 for:
      - Missing or invalid signature
      - Malformed JSON body

    Returns HTTP 500 for:
      - Unexpected processing failures (allows Razorpay to retry)
    """
    # ── Step 1: Read raw body ──────────────────────────────────────────
    raw_body: bytes = await request.body()

    if not raw_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty request body",
        )

    # ── Step 2: Signature verification ────────────────────────────────
    # CRITICAL: use raw_body, NOT re-serialized JSON
    signature_valid = verify_webhook_signature(
        raw_body=raw_body,
        signature_header=x_razorpay_signature or "",
    )

    if not signature_valid:
        logger.warning(
            "Webhook signature verification failed — rejecting event. "
            "Signature header present: %s",
            bool(x_razorpay_signature),
        )
        # Persist the rejected event for auditability, then reject
        _persist_rejected_webhook(
            db=db,
            raw_body=raw_body,
            x_razorpay_signature=x_razorpay_signature,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        )

    # ── Step 3: Parse JSON payload ─────────────────────────────────────
    try:
        payload_dict = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse webhook JSON body: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    try:
        parsed_payload = RazorpayWebhookPayload.model_validate(payload_dict)
        # Validate nested entities before persisting/processing the event so
        # invalid values (including non-positive amounts) are client errors,
        # not retriable processing failures.
        parsed_payload.extract_payment_entity()
        parsed_payload.extract_order_entity()
    except Exception as exc:
        logger.error("Webhook payload validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload does not match expected Razorpay webhook schema",
        )

    event_id = parsed_payload.id
    event_type = parsed_payload.event

    logger.info("Webhook received: event_id=%s type=%s", event_id, event_type)

    # ── Step 4: Idempotency check — persist webhook event ─────────────
    webhook_event, is_duplicate = create_webhook_event(
        db,
        event_id=event_id,
        event_type=event_type,
        raw_payload=raw_body.decode("utf-8"),
        signature_verified=True,
    )

    if is_duplicate:
        # Safely acknowledge duplicate delivery without reprocessing
        logger.info(
            "Duplicate webhook received and acknowledged: event_id=%s", event_id
        )
        db.commit()
        return WebhookAckResponse(
            status="duplicate",
            event_id=event_id,
            message="Event already received and processed.",
        )

    # ── Step 5: Persist duplicate-detection record before processing ───
    # Commit the RECEIVED record first so that even if processing fails,
    # the raw event is preserved for debugging.
    append_audit_log(
        db,
        event_type="WEBHOOK_RECEIVED",
        action="SIGNATURE_VERIFIED",
        reason=f"Webhook event {event_type} received with valid signature",
        metadata={
            "event_id": event_id,
            "event_type": event_type,
        },
    )
    db.commit()

    # ── Step 6: Process the event ──────────────────────────────────────
    try:
        process_webhook_event(db, webhook_event, parsed_payload)
        db.commit()
    except Exception as exc:
        logger.exception(
            "Event processing failed for event_id=%s: %s", event_id, exc
        )
        db.rollback()
        # Return 500 so Razorpay knows to retry this event
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Event processing failed",
        )

    return WebhookAckResponse(
        status="ok",
        event_id=event_id,
        message=f"Event {event_type} processed successfully.",
    )


def _persist_rejected_webhook(
    db: Session,
    raw_body: bytes,
    x_razorpay_signature: str | None,
) -> None:
    """
    Attempt to persist a signature-rejected webhook for auditability.

    This is best-effort: if we can't parse the event_id, we generate a unique
    local identifier so separately rejected deliveries remain auditable.
    We never want to crash on a rejection — just record it.
    """
    try:
        payload_dict = json.loads(raw_body)
        event_id = payload_dict.get("id")
        event_type = payload_dict.get("event", "unknown")
    except Exception:
        event_id = None
        event_type = "unknown"

    if not isinstance(event_id, str) or not event_id:
        event_id = f"invalid_signature_{uuid4().hex}"
    if not isinstance(event_type, str) or not event_type:
        event_type = "unknown"

    try:
        create_webhook_event(
            db,
            event_id=event_id,
            event_type=event_type,
            raw_payload=raw_body.decode("utf-8", errors="replace"),
            signature_verified=False,
        )
        append_audit_log(
            db,
            event_type="WEBHOOK_REJECTED",
            action="INVALID_SIGNATURE",
            reason="Webhook rejected due to signature mismatch",
            metadata={
                "event_id": event_id,
                "signature_header_present": bool(x_razorpay_signature),
            },
        )
        db.commit()
    except Exception as persist_exc:
        logger.warning(
            "Could not persist rejected webhook record: %s", persist_exc
        )
        db.rollback()
