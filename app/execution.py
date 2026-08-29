import logging
from sqlalchemy.orm import Session

from app.models import RecoveryDecision, PaymentRecord
from app.crud import (
    mark_execution_successful,
    mark_execution_failed_retryable,
    mark_execution_failed_permanent,
    append_audit_log,
    create_customer_outreach_event,
)
from app.executor import (
    PaymentLinkProvider,
    DuplicateReferenceExecutionError,
    ReconciliationError,
    TransientExecutionError,
    PermanentExecutionError,
)
from app.retry_policy import calculate_retry
from app.metrics import executions_total

logger = logging.getLogger(__name__)


def execute_recovery_decision(
    db: Session,
    recovery_decision: RecoveryDecision,
    executor: PaymentLinkProvider,
    payment_record: PaymentRecord,
) -> None:
    """
    Execute a recovery decision. Handles link creation, duplicate reconciliation,
    retry policy calculation, and state transitions (EXECUTED, DECIDED+retry, FAILED, EXECUTING).
    """
    execution_reference_id = f"exec_rd_{recovery_decision.id}"
    customer_identifier = payment_record.customer_email or payment_record.customer_contact

    try:
        plink_id = executor.create_payment_link(
            recovery_decision, execution_reference_id
        )
        mark_execution_successful(db, recovery_decision, execution_reference_id)
        
        if customer_identifier is not None:
            normalized = (
                customer_identifier.lower()
                if payment_record.customer_email
                else customer_identifier
            )
            create_customer_outreach_event(
                db,
                customer_identifier=normalized,
                payment_record_id=payment_record.id,
                recovery_decision_id=recovery_decision.id,
                action=recovery_decision.selected_action.value,
                channel="payment_link",
            )
            
        executions_total.labels(status="success").inc()
        append_audit_log(
            db,
            event_type="ACTION_EXECUTED",
            action="CREATE_PAYMENT_LINK",
            payment_record_id=payment_record.id,
            reason=f"Payment link created: {plink_id} (execution_reference_id={execution_reference_id})",
            metadata={"execution_reference_id": execution_reference_id, "plink_id": plink_id},
        )
        db.commit()

    except DuplicateReferenceExecutionError as exc:
        logger.info("Duplicate reference detected for decision %s, attempting reconciliation...", recovery_decision.id)
        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_RECONCILING",
            action="RECONCILE_DUPLICATE_REFERENCE",
            payment_record_id=payment_record.id,
            reason="Duplicate reference ID error received; attempting to fetch existing payment link.",
            metadata={"error": str(exc), "recovery_decision_id": recovery_decision.id},
        )
        db.commit()
        try:
            plink_id = executor.reconcile_duplicate_reference(
                recovery_decision, execution_reference_id
            )
            mark_execution_successful(db, recovery_decision, execution_reference_id)
            
            if customer_identifier is not None:
                normalized = (
                    customer_identifier.lower()
                    if payment_record.customer_email
                    else customer_identifier
                )
                create_customer_outreach_event(
                    db,
                    customer_identifier=normalized,
                    payment_record_id=payment_record.id,
                    recovery_decision_id=recovery_decision.id,
                    action=recovery_decision.selected_action.value,
                    channel="payment_link",
                )
            
            executions_total.labels(status="success").inc()
            append_audit_log(
                db,
                event_type="ACTION_EXECUTED",
                action="RECONCILED_PAYMENT_LINK",
                payment_record_id=payment_record.id,
                reason=f"Payment link reconciled successfully: {plink_id} (execution_reference_id={execution_reference_id})",
                metadata={"execution_reference_id": execution_reference_id, "plink_id": plink_id},
            )
            db.commit()
        except TransientExecutionError as rec_exc:
            executions_total.labels(status="failure").inc()
            logger.warning("Transient reconciliation failure for decision %s: %s", recovery_decision.id, rec_exc)
            
            retry_result = calculate_retry(recovery_decision.execution_attempts, rec_exc)
            if retry_result.retryable:
                mark_execution_failed_retryable(
                    db, recovery_decision, str(rec_exc), error_type=rec_exc.error_type, next_retry_at=retry_result.next_retry_at
                )
                action_taken = "RECONCILE_TRANSIENT_FAILURE"
            else:
                mark_execution_failed_permanent(
                    db, recovery_decision, str(rec_exc), error_type=rec_exc.error_type
                )
                action_taken = "RECONCILE_PERMANENT_FAILURE_LIMIT_REACHED"

            append_audit_log(
                db,
                event_type="ACTION_EXECUTION_FAILED",
                action=action_taken,
                payment_record_id=payment_record.id,
                reason=f"Failed to reconcile payment link (transient): {rec_exc}. {retry_result.reason}",
                metadata={"error": str(rec_exc), "error_type": rec_exc.error_type, "recovery_decision_id": recovery_decision.id, "next_retry_at": retry_result.next_retry_at.isoformat() if retry_result.next_retry_at else None},
            )
            db.commit()
        except (PermanentExecutionError, ReconciliationError) as rec_exc:
            executions_total.labels(status="failure").inc()
            logger.error("Permanent reconciliation failure for decision %s: %s", recovery_decision.id, rec_exc)
            mark_execution_failed_permanent(
                db, recovery_decision, str(rec_exc), error_type=rec_exc.error_type
            )
            append_audit_log(
                db,
                event_type="ACTION_EXECUTION_FAILED",
                action="RECONCILE_PERMANENT_FAILURE",
                payment_record_id=payment_record.id,
                reason=f"Failed to reconcile payment link (permanent): {rec_exc}",
                metadata={"error": str(rec_exc), "error_type": rec_exc.error_type, "recovery_decision_id": recovery_decision.id},
            )
            db.commit()

    except TransientExecutionError as exc:
        executions_total.labels(status="failure").inc()
        logger.warning("Transient execution failure for decision %s: %s", recovery_decision.id, exc)
        
        retry_result = calculate_retry(recovery_decision.execution_attempts, exc)
        if retry_result.retryable:
            mark_execution_failed_retryable(
                db, recovery_decision, str(exc), error_type=exc.error_type, next_retry_at=retry_result.next_retry_at
            )
            action_taken = "CREATE_PAYMENT_LINK_TRANSIENT_FAILURE"
        else:
            mark_execution_failed_permanent(
                db, recovery_decision, str(exc), error_type=exc.error_type
            )
            action_taken = "CREATE_PAYMENT_LINK_PERMANENT_FAILURE_LIMIT_REACHED"

        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_FAILED",
            action=action_taken,
            payment_record_id=payment_record.id,
            reason=f"Failed to create payment link (transient): {exc}. {retry_result.reason}",
            metadata={"error": str(exc), "error_type": exc.error_type, "recovery_decision_id": recovery_decision.id, "next_retry_at": retry_result.next_retry_at.isoformat() if retry_result.next_retry_at else None},
        )
        db.commit()

    except PermanentExecutionError as exc:
        executions_total.labels(status="failure").inc()
        logger.error("Permanent execution failure for decision %s: %s", recovery_decision.id, exc)
        mark_execution_failed_permanent(
            db, recovery_decision, str(exc), error_type=exc.error_type
        )
        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_FAILED",
            action="CREATE_PAYMENT_LINK_PERMANENT_FAILURE",
            payment_record_id=payment_record.id,
            reason=f"Failed to create payment link (permanent): {exc}",
            metadata={"error": str(exc), "error_type": exc.error_type, "recovery_decision_id": recovery_decision.id},
        )
        db.commit()

    except Exception as exc:
        executions_total.labels(status="failure").inc()
        logger.error("Unexpected execution failure for decision %s: %s", recovery_decision.id, exc)
        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_FAILED",
            action="CREATE_PAYMENT_LINK_UNEXPECTED_FAILURE",
            payment_record_id=payment_record.id,
            reason=f"Unexpected failure during execution: {type(exc).__name__}: {exc}",
            metadata={"error": str(exc), "recovery_decision_id": recovery_decision.id},
        )
        db.commit()  # Leaves decision in EXECUTING state as a stale/crashed record
