"""
Webhook event processing pipeline.

This module orchestrates the complete processing flow for each inbound
Razorpay webhook event:

  1. Parse and validate the JSON payload (after signature verification).
  2. Route to the appropriate event-type handler.
  3. Normalize the payment entity into PaymentRecord.
  4. Extract features.
  5. Run the (placeholder) policy predictor.
  6. Persist the RecoveryDecision.
  7. Append audit log entries at each step.

Each event-type handler is an isolated function, making the routing logic
simple and the individual handlers independently testable.

Supported events (Phase 1):
  - payment.failed
  - payment.captured
  - order.paid

Extensible to:
  - payment.authorized
  - payment_link.paid
  - invoice.paid
  - subscription.charged
  - subscription.halted

Unsupported events are acknowledged (HTTP 200) but not processed,
with an audit log entry recording the unhandled type.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.crud import (
    append_audit_log,
    create_recovery_decision,
    update_recovery_decision_execution,
    update_recovery_decision_outcome,
    update_webhook_status,
    upsert_payment_record_from_captured,
    upsert_payment_record_from_failed,
    upsert_payment_record_from_order_paid,
)
from app.ml.features import extract_features
from app.models import RecoveryAction, RecoveryDecision, WebhookEvent, WebhookProcessingStatus
from app.schemas import RazorpayWebhookPayload

logger = logging.getLogger(__name__)


def process_webhook_event(
    db: Session,
    webhook_event: WebhookEvent,
    parsed_payload: RazorpayWebhookPayload,
    app_state: Any,
) -> None:
    """
    Process a validated, non-duplicate webhook event end-to-end.

    This function is called ONLY after:
      - Signature has been verified.
      - The event has been confirmed non-duplicate.
      - The WebhookEvent row has been persisted with status=RECEIVED.

    Successful work remains part of the caller's transaction. On failure,
    partial processing work is rolled back and the FAILED status plus error
    audit entry are committed in a separate transaction before the exception
    is re-raised.

    Args:
        db: Active database session. Caller owns the outer transaction.
        webhook_event: The persisted WebhookEvent row.
        parsed_payload: Validated Pydantic model of the webhook body.
    """
    event_type = parsed_payload.event

    try:
        update_webhook_status(db, webhook_event, WebhookProcessingStatus.PROCESSING)
        db.flush()

        if event_type == "payment.failed":
            _handle_payment_failed(db, webhook_event, parsed_payload, app_state)
        elif event_type == "payment.captured":
            _handle_payment_captured(db, webhook_event, parsed_payload)
        elif event_type == "order.paid":
            _handle_order_paid(db, webhook_event, parsed_payload)
        elif event_type == "payment_link.paid":
            _handle_payment_link_paid(db, webhook_event, parsed_payload)
        else:
            _handle_unknown_event(db, webhook_event, event_type)

        update_webhook_status(db, webhook_event, WebhookProcessingStatus.PROCESSED)

    except Exception as exc:
        logger.exception(
            "Unhandled error processing webhook event_id=%s type=%s",
            webhook_event.event_id,
            event_type,
        )
        # Discard partial payment/decision/audit writes from this attempt.
        # The caller must receive the exception so Razorpay retries, but the
        # failure outcome itself must survive that retry-triggering rollback.
        db.rollback()
        try:
            failed_event = (
                db.query(WebhookEvent)
                .filter(WebhookEvent.event_id == webhook_event.event_id)
                .one()
            )
            update_webhook_status(db, failed_event, WebhookProcessingStatus.FAILED)
            append_audit_log(
                db,
                event_type="PROCESSING_ERROR",
                action="UNHANDLED_EXCEPTION",
                reason=f"Unexpected error: {type(exc).__name__}: {exc}",
                metadata={
                    "webhook_event_id": failed_event.id,
                    "rzp_event_type": event_type,
                },
            )
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Could not persist failure state for webhook event_id=%s",
                webhook_event.event_id,
            )
        raise


# ── Event-type handlers ────────────────────────────────────────────────────────


def _handle_payment_failed(
    db: Session,
    webhook_event: WebhookEvent,
    payload: RazorpayWebhookPayload,
    app_state: Any,
) -> None:
    """
    Handle a payment.failed event.

    Steps:
      1. Extract the payment entity from the payload.
      2. Upsert a PaymentRecord (status=failed).
      3. Extract features.
      4. Run the placeholder policy predictor.
      5. Persist RecoveryDecision (PENDING_POLICY / NO_ACTION).
      6. Append audit logs.
    """
    payment_entity = payload.extract_payment_entity()
    if payment_entity is None:
        append_audit_log(
            db,
            event_type="PAYMENT_FAILED",
            action="SKIP_MISSING_PAYMENT_ENTITY",
            reason="payment.failed webhook did not contain a payment entity in payload",
            metadata={"webhook_event_id": webhook_event.id},
        )
        logger.warning(
            "payment.failed event %s has no payment entity", webhook_event.event_id
        )
        return

    payment_record, created = upsert_payment_record_from_failed(
        db,
        webhook_event_id=webhook_event.id,
        payment=payment_entity,
    )

    action_taken = "CREATE_PAYMENT_RECORD" if created else "UPDATE_PAYMENT_RECORD"

    append_audit_log(
        db,
        event_type="PAYMENT_FAILED",
        action=action_taken,
        payment_record_id=payment_record.id,
        reason=f"payment.failed event received; record {'created' if created else 'updated'}",
        metadata={
            "razorpay_payment_id": payment_entity.id,
            "error_code": payment_entity.error_code,
            "error_reason": payment_entity.error_reason,
            "error_source": payment_entity.error_source,
            "amount": payment_entity.amount,
            "method": payment_entity.method,
        },
    )

    # Feature extraction
    features = extract_features(payment_record, db)

    append_audit_log(
        db,
        event_type="FEATURE_EXTRACTION",
        action="EXTRACT_FEATURES",
        payment_record_id=payment_record.id,
        reason="Feature vector extracted from payment record and history",
        metadata=features.to_dict(),
    )

    # 1. Economic Prediction
    prediction = app_state.predictor.predict(features)
    
    # 2. Guardrails (Operational Safety)
    safe_prediction = app_state.guardrails.evaluate(db, payment_record, prediction)

    recovery_decision = create_recovery_decision(
        db,
        payment_record_id=payment_record.id,
        prediction=safe_prediction,
    )

    append_audit_log(
        db,
        event_type="POLICY_DECISION",
        action="PLACEHOLDER_DECISION",
        payment_record_id=payment_record.id,
        reason=prediction.reasoning,
        metadata={
            "decision_status": safe_prediction.decision_status,
            "selected_action": safe_prediction.selected_action,
            "model_version": safe_prediction.model_version,
            "recovery_decision_id": recovery_decision.id,
        },
    )

    # 3. Action Execution (Synchronous Outbox Simulation)
    if safe_prediction.selected_action != RecoveryAction.NO_ACTION:
        # Step A: Persist execution intent first to avoid dual-write vulnerability.
        # This commits the WebhookEvent, PaymentRecord, and RecoveryDecision(DECIDED).
        db.commit()

        try:
            # Step B: Execute Mock Provider
            idempotency_key = f"exec_rd_{recovery_decision.id}"
            execution_reference_id = app_state.executor.create_payment_link(
                recovery_decision, idempotency_key
            )
            
            # Step C: Update Execution Result
            update_recovery_decision_execution(db, recovery_decision, execution_reference_id)
            
            append_audit_log(
                db,
                event_type="ACTION_EXECUTED",
                action="CREATE_PAYMENT_LINK",
                payment_record_id=payment_record.id,
                reason=f"Successfully created payment link: {execution_reference_id}",
                metadata={"execution_reference_id": execution_reference_id},
            )
            # The outer webhooks.py loop will commit this final state update.
        except Exception as exc:
            logger.error("Action execution failed: %s", exc)
            append_audit_log(
                db,
                event_type="ACTION_EXECUTION_FAILED",
                action="CREATE_PAYMENT_LINK_FAILED",
                payment_record_id=payment_record.id,
                reason=f"Failed to create payment link: {exc}",
                metadata={"error": str(exc)},
            )
            db.commit() # Ensure failure log is persisted
            # In a full outbox pattern, an async worker would retry DECIDED records.
            # We swallow the error here so the webhook succeeds and Razorpay doesn't retry,
            # as our guardrails would incorrectly block a direct webhook retry.

    logger.info(
        "payment.failed processed: payment_id=%s decision=%s action=%s",
        payment_entity.id,
        safe_prediction.decision_status,
        safe_prediction.selected_action,
    )


def _handle_payment_captured(
    db: Session,
    webhook_event: WebhookEvent,
    payload: RazorpayWebhookPayload,
) -> None:
    """
    Handle a payment.captured event.

    Updates or creates a PaymentRecord with status=captured.
    Clears error fields on the record — captured means success.
    """
    payment_entity = payload.extract_payment_entity()
    if payment_entity is None:
        append_audit_log(
            db,
            event_type="PAYMENT_CAPTURED",
            action="SKIP_MISSING_PAYMENT_ENTITY",
            reason="payment.captured webhook did not contain a payment entity in payload",
            metadata={"webhook_event_id": webhook_event.id},
        )
        logger.warning(
            "payment.captured event %s has no payment entity", webhook_event.event_id
        )
        return

    payment_record, created = upsert_payment_record_from_captured(
        db,
        webhook_event_id=webhook_event.id,
        payment=payment_entity,
    )

    action_taken = "CREATE_PAYMENT_RECORD" if created else "UPDATE_PAYMENT_RECORD"

    append_audit_log(
        db,
        event_type="PAYMENT_CAPTURED",
        action=action_taken,
        payment_record_id=payment_record.id,
        reason=f"payment.captured event received; record {'created' if created else 'updated to captured'}",
        metadata={
            "razorpay_payment_id": payment_entity.id,
            "amount": payment_entity.amount,
            "method": payment_entity.method,
        },
    )

    logger.info("payment.captured processed: payment_id=%s", payment_entity.id)


def _handle_order_paid(
    db: Session,
    webhook_event: WebhookEvent,
    payload: RazorpayWebhookPayload,
) -> None:
    """
    Handle an order.paid event.

    order.paid contains an order entity and optionally a payment entity.
    We use the payment entity if present (more specific); otherwise
    fall back to the order ID to locate the existing PaymentRecord.
    """
    payment_entity = payload.extract_payment_entity()
    order_entity = payload.extract_order_entity()
    order_id = order_entity.id if order_entity else None

    payment_record = upsert_payment_record_from_order_paid(
        db,
        webhook_event_id=webhook_event.id,
        payment=payment_entity,
        razorpay_order_id=order_id,
    )

    if payment_record is None:
        append_audit_log(
            db,
            event_type="ORDER_PAID",
            action="SKIP_UNMATCHED",
            reason=(
                "order.paid event received but no matching PaymentRecord found. "
                "The order may have been created outside this system's scope."
            ),
            metadata={
                "webhook_event_id": webhook_event.id,
                "order_id": order_id,
            },
        )
        logger.warning(
            "order.paid event %s: no matching payment record for order_id=%s",
            webhook_event.event_id,
            order_id,
        )
        return

    append_audit_log(
        db,
        event_type="ORDER_PAID",
        action="UPDATE_PAYMENT_RECORD",
        payment_record_id=payment_record.id,
        reason="order.paid event updated payment record to captured",
        metadata={
            "order_id": order_id,
            "payment_id": payment_entity.id if payment_entity else None,
        },
    )

    logger.info("order.paid processed: order_id=%s", order_id)


def _handle_unknown_event(
    db: Session,
    webhook_event: WebhookEvent,
    event_type: str,
) -> None:
    """
    Handle an event type that is not currently supported.

    We acknowledge receipt (HTTP 200) but do no processing.
    The audit log records that it was received and skipped.
    """
    append_audit_log(
        db,
        event_type="UNSUPPORTED_EVENT",
        action="SKIP_UNSUPPORTED",
        reason=f"Event type '{event_type}' is not handled in Phase 4",
        metadata={"webhook_event_id": webhook_event.id, "rzp_event_type": event_type},
    )
    logger.info("Unsupported event type received and acknowledged: %s", event_type)


def _handle_payment_link_paid(
    db: Session,
    webhook_event: WebhookEvent,
    payload: RazorpayWebhookPayload,
) -> None:
    """
    Handle a payment_link.paid event.
    
    Extracts the execution_reference_id and closes the loop by setting OUTCOME_OBSERVED.
    """
    plink_entity = payload.extract_payment_link_entity()
    if not plink_entity:
        append_audit_log(
            db,
            event_type="PAYMENT_LINK_PAID",
            action="SKIP_MISSING_ENTITY",
            reason="payment_link.paid webhook did not contain a payment_link entity",
            metadata={"webhook_event_id": webhook_event.id},
        )
        return

    # Require explicit correlation reference. Never guess correlation.
    ref_id = plink_entity.notes.get("execution_reference_id")
    
    if not ref_id:
        append_audit_log(
            db,
            event_type="PAYMENT_LINK_PAID",
            action="SKIP_MISSING_REFERENCE",
            reason="payment_link.paid webhook did not contain an execution_reference_id in notes",
            metadata={"webhook_event_id": webhook_event.id},
        )
        return
    
    decision = (
        db.query(RecoveryDecision)
        .filter(RecoveryDecision.execution_reference_id == ref_id)
        .first()
    )
    
    if not decision:
        append_audit_log(
            db,
            event_type="PAYMENT_LINK_PAID",
            action="SKIP_UNMATCHED_REFERENCE",
            reason=f"No RecoveryDecision found for reference {ref_id}",
            metadata={"webhook_event_id": webhook_event.id, "execution_reference_id": ref_id},
        )
        return
        
    update_recovery_decision_outcome(db, decision)
    
    append_audit_log(
        db,
        event_type="PAYMENT_LINK_PAID",
        action="OUTCOME_OBSERVED",
        payment_record_id=decision.payment_record_id,
        reason="payment_link.paid successfully mapped to recovery decision.",
        metadata={"recovery_decision_id": decision.id, "execution_reference_id": ref_id},
    )
    logger.info("payment_link.paid processed. loop closed for decision %s", decision.id)

