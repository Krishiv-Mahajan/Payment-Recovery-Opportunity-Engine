import logging
from sqlalchemy.orm import Session

from app.models import RecoveryDecision
from app.executor import PaymentLinkProvider, TransientExecutionError, PermanentExecutionError, ReconciliationError, AmbiguousStateError
from app.crud import (
    mark_execution_successful,
    mark_execution_failed_retryable,
    mark_execution_failed_permanent,
    update_execution_error_only,
    append_audit_log,
)
from app.retry_policy import calculate_retry

logger = logging.getLogger(__name__)


def reconcile_stale_decision(db: Session, decision: RecoveryDecision, executor: PaymentLinkProvider) -> None:
    """
    Safely recover a stale EXECUTING decision by reconciling with the external provider.
    """
    # The reference_id used to create the payment link
    execution_reference_id = f"exec_rd_{decision.id}"
    
    logger.info("Attempting stale recovery for decision %s", decision.id)

    try:
        # Case A / Case B / Case C (Lookup mismatch/multiple)
        plink_id = executor.reconcile_duplicate_reference(decision, execution_reference_id)
        
        # If we reach here, we found a single matching link (Case A)
        mark_execution_successful(db, decision, execution_reference_id)
        
        append_audit_log(
            db,
            event_type="ACTION_EXECUTED",
            action="STALE_RECOVERY_SUCCESS",
            payment_record_id=decision.payment_record_id,
            reason=f"Stale recovery found matching link: {plink_id}",
            metadata={"execution_reference_id": execution_reference_id, "plink_id": plink_id, "recovery_decision_id": decision.id},
        )
        db.commit()

    except AmbiguousStateError as exc:
        # Case B: Lookup succeeded but no link exists after bounded retries.
        # Since this is stale recovery (e.g. 15+ mins later), the indexing delay is over.
        # This definitively means the POST failed or was lost before reaching the provider.
        # We can safely schedule a retry if attempts < MAX.
        retry_result = calculate_retry(
            decision.execution_attempts, 
            TransientExecutionError("Stale execution timeout (no link found)", error_type="STALE_TIMEOUT")
        )
        
        if retry_result.retryable:
            mark_execution_failed_retryable(
                db, decision, "Stale execution timeout (no link found)", error_type="STALE_TIMEOUT", next_retry_at=retry_result.next_retry_at
            )
            action_taken = "STALE_RECOVERY_RETRYABLE"
        else:
            mark_execution_failed_permanent(
                db, decision, "Stale execution timeout (no link found)", error_type="STALE_TIMEOUT"
            )
            action_taken = "STALE_RECOVERY_PERMANENT_LIMIT_REACHED"

        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_FAILED",
            action=action_taken,
            payment_record_id=decision.payment_record_id,
            reason=f"Stale recovery confirmed no link exists. {retry_result.reason}",
            metadata={"error": str(exc), "recovery_decision_id": decision.id},
        )
        db.commit()

    except ReconciliationError as exc:
        # Case C: Lookup succeeded but found a mismatch or multiple links.
        # This is a permanent integrity failure.
        mark_execution_failed_permanent(db, decision, str(exc), error_type=exc.error_type)
        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_FAILED",
            action="STALE_RECOVERY_MISMATCH",
            payment_record_id=decision.payment_record_id,
            reason=f"Stale recovery found mismatched links: {exc}",
            metadata={"error": str(exc), "recovery_decision_id": decision.id},
        )
        db.commit()

    except (TransientExecutionError, PermanentExecutionError) as exc:
        # Case D: Lookup failed entirely (timeout, 5xx, or 401/403).
        # We DO NOT transition out of EXECUTING because the external state is ambiguous.
        # We just update the error. It will be picked up again in 10 minutes.
        update_execution_error_only(db, decision, str(exc), error_type=exc.error_type)
        
        append_audit_log(
            db,
            event_type="ACTION_EXECUTION_FAILED",
            action="STALE_RECOVERY_LOOKUP_FAILED",
            payment_record_id=decision.payment_record_id,
            reason=f"Stale recovery lookup failed, leaving in EXECUTING: {exc}",
            metadata={"error": str(exc), "error_type": exc.error_type, "recovery_decision_id": decision.id},
        )
        db.commit()
