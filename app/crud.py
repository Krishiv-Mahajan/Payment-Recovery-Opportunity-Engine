"""
Database CRUD operations.

All database writes go through these functions.
This layer exists to:
  1. Keep the webhook handler thin and readable.
  2. Centralize idempotency logic.
  3. Make database behavior independently testable.

All functions accept an open Session and do NOT commit.
The caller is responsible for committing/rolling back transactions.
This design keeps transaction boundaries explicit and avoids
hidden nested transactions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    AuditLog,
    CustomerOutreachEvent,
    DecisionStatus,
    PaymentRecord,
    PaymentStatus,
    RecoveryAction,
    RecoveryDecision,
    WebhookEvent,
    WebhookProcessingStatus,
)
from app.ml.features import PaymentFeatures
from app.ml.predictor import PolicyPrediction
from app.schemas import RazorpayPaymentEntity


def create_webhook_event(
    db: Session,
    *,
    event_id: str,
    event_type: str,
    raw_payload: str,
    signature_verified: bool,
) -> tuple[WebhookEvent, bool]:
    """
    Persist a received webhook event.

    Idempotency: If a WebhookEvent with the same event_id already exists,
    returns the existing record and is_duplicate=True without inserting.

    Args:
        db: Active database session (caller owns transaction).
        event_id: Razorpay's event ID from the payload root.
        event_type: e.g., "payment.failed"
        raw_payload: The raw JSON string exactly as received.
        signature_verified: True if HMAC-SHA256 check passed.

    Returns:
        Tuple of (WebhookEvent, is_duplicate).
        is_duplicate=True means this event_id was already seen.
    """
    status = (
        WebhookProcessingStatus.RECEIVED
        if signature_verified
        else WebhookProcessingStatus.SIGNATURE_INVALID
    )

    event = WebhookEvent(
        event_id=event_id,
        event_type=event_type,
        raw_payload=raw_payload,
        signature_verified=signature_verified,
        processing_status=status,
    )
    try:
        # Let the database's UNIQUE constraint be the source of truth.  A
        # pre-insert SELECT has a race window when requests use separate DB
        # sessions.  The savepoint limits a duplicate-key rollback to this
        # insert, preserving any work in the caller's transaction.
        with db.begin_nested():
            db.add(event)
            db.flush()
    except IntegrityError:
        # Another delivery (or an earlier delivery) owns this event ID.
        existing = (
            db.query(WebhookEvent).filter(WebhookEvent.event_id == event_id).first()
        )
        return existing, True

    return event, False


def update_webhook_status(
    db: Session,
    webhook_event: WebhookEvent,
    status: WebhookProcessingStatus,
) -> None:
    """Update the processing status of a WebhookEvent."""
    webhook_event.processing_status = status
    db.add(webhook_event)


def upsert_payment_record_from_failed(
    db: Session,
    *,
    webhook_event_id: int,
    payment: RazorpayPaymentEntity,
) -> tuple[PaymentRecord, bool]:
    """
    Create or retrieve a PaymentRecord from a payment.failed payload.

    If a record for this razorpay_payment_id already exists:
      - If it is already 'captured' or 'authorized', do NOT downgrade to 'failed'.
        Return the existing record and created=False.
      - If it is in any other state, update it to 'failed' with the new error details.

    Args:
        db: Active session (caller owns transaction).
        webhook_event_id: The ID of the parent WebhookEvent row.
        payment: Validated RazorpayPaymentEntity from the webhook.

    Returns:
        Tuple of (PaymentRecord, created).
        created=True if a new row was inserted.
    """
    existing = (
        db.query(PaymentRecord)
        .filter(PaymentRecord.razorpay_payment_id == payment.id)
        .first()
    )

    # Define status priority order — only allow forward transitions
    # A captured payment should never regress to failed
    STATUS_PRIORITY = {
        PaymentStatus.CREATED: 0,
        PaymentStatus.FAILED: 1,
        PaymentStatus.AUTHORIZED: 2,
        PaymentStatus.CAPTURED: 3,
        PaymentStatus.REFUNDED: 4,
    }

    payment_created_at = None
    if payment.created_at:
        payment_created_at = datetime.fromtimestamp(payment.created_at, tz=timezone.utc)

    if existing is not None:
        existing_priority = STATUS_PRIORITY.get(existing.status, 0)
        failed_priority = STATUS_PRIORITY.get(PaymentStatus.FAILED, 0)

        if existing_priority > failed_priority:
            # Already in a terminal forward state — do not regress
            return existing, False

        # Update error context from the new failed event
        existing.status = PaymentStatus.FAILED
        existing.error_code = payment.error_code
        existing.error_description = payment.error_description
        existing.error_source = payment.error_source
        existing.error_step = payment.error_step
        existing.error_reason = payment.error_reason
        existing.method = payment.method or existing.method
        db.add(existing)
        return existing, False

    record = PaymentRecord(
        webhook_event_id=webhook_event_id,
        razorpay_payment_id=payment.id,
        razorpay_order_id=payment.order_id,
        amount=payment.amount,
        currency=payment.currency,
        status=PaymentStatus.FAILED,
        method=payment.method,
        error_code=payment.error_code,
        error_description=payment.error_description,
        error_source=payment.error_source,
        error_step=payment.error_step,
        error_reason=payment.error_reason,
        customer_email=payment.email,
        customer_contact=payment.contact,
        payment_created_at=payment_created_at,
    )
    db.add(record)
    db.flush()
    return record, True


def upsert_payment_record_from_captured(
    db: Session,
    *,
    webhook_event_id: int,
    payment: RazorpayPaymentEntity,
) -> tuple[PaymentRecord, bool]:
    """
    Update (or create) a PaymentRecord from a payment.captured payload.

    A captured payment takes precedence over earlier failed or authorized
    records, but must not regress a later refunded state.

    Args:
        db: Active session.
        webhook_event_id: The ID of the parent WebhookEvent row.
        payment: Validated RazorpayPaymentEntity from the webhook.

    Returns:
        Tuple of (PaymentRecord, created).
    """
    existing = (
        db.query(PaymentRecord)
        .filter(PaymentRecord.razorpay_payment_id == payment.id)
        .first()
    )

    payment_created_at = None
    if payment.created_at:
        payment_created_at = datetime.fromtimestamp(payment.created_at, tz=timezone.utc)

    if existing is not None:
        status_priority = {
            PaymentStatus.CREATED: 0,
            PaymentStatus.FAILED: 1,
            PaymentStatus.AUTHORIZED: 2,
            PaymentStatus.CAPTURED: 3,
            PaymentStatus.REFUNDED: 4,
        }
        if status_priority.get(existing.status, 0) > status_priority[PaymentStatus.CAPTURED]:
            return existing, False

        existing.status = PaymentStatus.CAPTURED
        existing.razorpay_order_id = payment.order_id or existing.razorpay_order_id
        existing.method = payment.method or existing.method
        existing.customer_email = payment.email or existing.customer_email
        existing.customer_contact = payment.contact or existing.customer_contact
        # Clear error fields — captured means it succeeded
        existing.error_code = None
        existing.error_description = None
        existing.error_source = None
        existing.error_step = None
        existing.error_reason = None
        db.add(existing)
        return existing, False

    record = PaymentRecord(
        webhook_event_id=webhook_event_id,
        razorpay_payment_id=payment.id,
        razorpay_order_id=payment.order_id,
        amount=payment.amount,
        currency=payment.currency,
        status=PaymentStatus.CAPTURED,
        method=payment.method,
        customer_email=payment.email,
        customer_contact=payment.contact,
        payment_created_at=payment_created_at,
    )
    db.add(record)
    db.flush()
    return record, True


def upsert_payment_record_from_order_paid(
    db: Session,
    *,
    webhook_event_id: int,
    payment: RazorpayPaymentEntity | None,
    razorpay_order_id: str | None,
) -> PaymentRecord | None:
    """
    Update a PaymentRecord when an order.paid event arrives.

    order.paid events contain both an order entity and (usually) a payment entity.
    We look up the payment by ID if available, otherwise by order_id.

    Returns the updated PaymentRecord, or None if we cannot identify the payment.
    """
    if payment is not None:
        record, _ = upsert_payment_record_from_captured(
            db, webhook_event_id=webhook_event_id, payment=payment
        )
        return record

    if razorpay_order_id:
        existing = (
            db.query(PaymentRecord)
            .filter(PaymentRecord.razorpay_order_id == razorpay_order_id)
            .first()
        )
        if existing:
            # An order.paid event confirms capture, but must not overwrite a
            # later terminal state such as REFUNDED.
            if existing.status != PaymentStatus.REFUNDED:
                existing.status = PaymentStatus.CAPTURED
            db.add(existing)
            return existing

    return None


def create_recovery_decision(
    db: Session,
    *,
    payment_record_id: int,
    prediction: PolicyPrediction,
    experiment_name: str | None = None,
    experiment_variant: str | None = None,
) -> RecoveryDecision:
    """
    Insert a RecoveryDecision record.

    Phase 1 always produces PENDING_POLICY / NO_ACTION.

    Args:
        db: Active session.
        payment_record_id: FK to the PaymentRecord this decision is for.
        prediction: Output from the policy predictor (or control baseline).
        experiment_name: Configured experiment name (e.g. "ml_policy_v1")
        experiment_variant: Deterministic variant ("control" or "treatment")
    """
    decision = RecoveryDecision(
        payment_record_id=payment_record_id,
        decision_status=prediction.decision_status,
        selected_action=prediction.selected_action,
        model_version=prediction.model_version,
        experiment_name=experiment_name,
        experiment_variant=experiment_variant,
        predicted_p0=prediction.predicted_p0,
        predicted_p1=prediction.predicted_p1,
        predicted_uplift=prediction.predicted_uplift,
        expected_incremental_net_paise=prediction.expected_incremental_net_paise,
    )
    db.add(decision)
    db.flush()
    return decision


def update_recovery_decision_execution(
    db: Session,
    decision: RecoveryDecision,
    execution_reference_id: str,
) -> None:
    """
    Update a decision after successful execution dispatch.
    """
    decision.decision_status = DecisionStatus.EXECUTED
    decision.execution_reference_id = execution_reference_id
    decision.executed_at = datetime.now(timezone.utc)
    db.add(decision)
    db.flush()


def update_recovery_decision_outcome(
    db: Session,
    decision: RecoveryDecision,
) -> None:
    """
    Update a decision after observing a downstream outcome (e.g., payment_link.paid).
    """
    decision.decision_status = DecisionStatus.OUTCOME_OBSERVED
    decision.outcome_observed_at = datetime.now(timezone.utc)
    db.add(decision)
    db.flush()


def append_audit_log(
    db: Session,
    *,
    event_type: str,
    action: str,
    payment_record_id: int | None = None,
    reason: str | None = None,
    metadata: dict | None = None,
) -> AuditLog:
    """
    Append an immutable audit log entry.

    Audit rows are NEVER updated or deleted.

    Args:
        db: Active session.
        event_type: Category of event being audited (e.g., "WEBHOOK_RECEIVED").
        action: Specific action taken (e.g., "CREATE_PAYMENT_RECORD").
        payment_record_id: FK to affected PaymentRecord, if applicable.
        reason: Human-readable description.
        metadata: Structured dict — serialized to JSON for storage.
    """
    log = AuditLog(
        payment_record_id=payment_record_id,
        event_type=event_type,
        action=action,
        reason=reason,
        metadata_json=json.dumps(metadata) if metadata is not None else None,
    )
    db.add(log)
    db.flush()
    return log


def create_customer_outreach_event(
    db: Session,
    *,
    customer_identifier: str,
    payment_record_id: int,
    recovery_decision_id: int,
    action: str,
    channel: str,
) -> CustomerOutreachEvent:
    """
    Record a customer outreach action for cooldown tracking.

    Args:
        db: Active database session (caller owns transaction).
        customer_identifier: Normalized email or contact (never None — callers must check).
        payment_record_id: FK to the associated PaymentRecord.
        recovery_decision_id: FK to the associated RecoveryDecision.
        action: RecoveryAction enum value string (e.g., 'SEND_PAYMENT_LINK').
        channel: Outreach channel string (e.g., 'payment_link').

    Returns:
        The newly created CustomerOutreachEvent (not yet committed).
    """
    event = CustomerOutreachEvent(
        customer_identifier=customer_identifier,
        payment_record_id=payment_record_id,
        recovery_decision_id=recovery_decision_id,
        action=action,
        channel=channel,
    )
    db.add(event)
    db.flush()
    return event


def get_recent_outreach_for_customer(
    db: Session,
    customer_identifier: str,
    since: datetime,
) -> CustomerOutreachEvent | None:
    """
    Return the most recent CustomerOutreachEvent for a customer within a time window.

    Returns None if no outreach occurred after `since`.

    Args:
        db: Active database session.
        customer_identifier: Normalized email or contact string.
        since: Cutoff datetime — outreach at or after this time is considered recent.
    """
    return (
        db.query(CustomerOutreachEvent)
        .filter(
            CustomerOutreachEvent.customer_identifier == customer_identifier,
            CustomerOutreachEvent.outreach_at >= since,
        )
        .order_by(CustomerOutreachEvent.outreach_at.desc())
        .first()
    )
