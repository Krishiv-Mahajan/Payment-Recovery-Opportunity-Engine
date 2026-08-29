import logging
import threading
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.executor import PaymentLinkProvider
from app.crud import (
    claim_stale_execution,
    get_eligible_recovery_decisions,
    claim_decision_for_execution,
)
from app.models import RecoveryDecision
from app.stale_recovery import reconcile_stale_decision
from app.execution import execute_recovery_decision

logger = logging.getLogger(__name__)


def run_worker_cycle(db_engine, settings: Settings, executor: PaymentLinkProvider) -> None:
    """
    Run exactly one deterministic cycle of the background retry worker.
    1. Reconcile stale EXECUTING decisions.
    2. Process up to `batch_size` eligible DECIDED decisions.
    """
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    
    # 1. Recover stale EXECUTING decisions
    while True:
        with SessionLocal() as session:
            try:
                stale_decision = claim_stale_execution(session, settings.execution_stale_after_seconds)
                if not stale_decision:
                    break
                
                # Commit the claim so the lock is released during reconciliation
                session.commit()
                # Reload the decision
                stale_decision = session.get(RecoveryDecision, stale_decision.id)
                
                logger.info("Worker claimed stale decision %s for recovery", stale_decision.id)
                reconcile_stale_decision(session, stale_decision, executor)
            except Exception as e:
                logger.exception("Unexpected error during stale recovery: %s", e)
                # Break to avoid infinite loop on a bad record
                break

    # 2. Find eligible DECIDED candidates
    candidate_ids = []
    with SessionLocal() as session:
        try:
            candidates = get_eligible_recovery_decisions(session, limit=settings.worker_batch_size)
            candidate_ids = [c.id for c in candidates]
        except Exception as e:
            logger.exception("Unexpected error querying eligible decisions: %s", e)
            return

    # 3. Process eligible decisions individually
    for decision_id in candidate_ids:
        with SessionLocal() as session:
            try:
                # Atomically claim to prevent races with webhooks or other workers
                if not claim_decision_for_execution(session, decision_id):
                    logger.debug("Decision %s was claimed by another process, skipping", decision_id)
                    continue

                # Commit the claim immediately so the write lock is released
                session.commit()

                # Load decision and payment_record for execution
                decision = session.get(RecoveryDecision, decision_id)
                if not decision:
                    logger.warning("Decision %s not found after claim", decision_id)
                    continue
                    
                payment_record = decision.payment_record

                logger.info("Worker executing decision %s", decision.id)
                execute_recovery_decision(session, decision, executor, payment_record)
            except Exception as e:
                logger.exception("Unexpected error executing decision %s in worker: %s", decision_id, e)
                # Ensure session rollback in case of dirty state
                session.rollback()


def start_worker(db_engine, settings: Settings, executor: PaymentLinkProvider, stop_event: threading.Event) -> None:
    """
    Worker loop to run in a background thread.
    """
    logger.info("Background retry worker starting")
    try:
        while not stop_event.is_set():
            try:
                run_worker_cycle(db_engine, settings, executor)
            except Exception as e:
                logger.exception("Worker cycle encountered an unexpected error: %s", e)
            
            # Wait for poll interval, interruptible by shutdown
            stop_event.wait(timeout=settings.worker_poll_interval_seconds)
    finally:
        logger.info("Background retry worker stopped")
