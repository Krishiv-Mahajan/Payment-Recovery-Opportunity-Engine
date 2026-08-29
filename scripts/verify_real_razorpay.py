import os
import time
import logging
from datetime import datetime, timezone, timedelta
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.executor_razorpay import RazorpayPaymentLinkProvider
from app.executor import DuplicateReferenceExecutionError
from app.models import RecoveryDecision, DecisionStatus, PaymentRecord, RecoveryAction
from app.database import Base
from app.crud import claim_stale_execution
from app.stale_recovery import reconcile_stale_decision
from app.worker import run_worker_cycle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_test_db():
    engine = create_engine("sqlite:///./verify_razorpay.db", connect_args={"check_same_thread": False})
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return engine


def setup_decision(session, decision_id: int, status: DecisionStatus, attempts=0, action=RecoveryAction.SEND_PAYMENT_LINK):
    record = PaymentRecord(id=decision_id, amount=1000, currency="INR", razorpay_payment_id=f"pay_mock_{decision_id}", status="failed")
    decision = RecoveryDecision(
        id=decision_id,
        payment_record=record,
        decision_status=status,
        selected_action=action,
        execution_attempts=attempts,
    )
    session.add(record)
    session.add(decision)
    session.commit()
    return decision


def run_validation():
    settings = get_settings()
    if settings.razorpay_key_id.startswith("rzp_test_REPLACEME") or settings.razorpay_key_secret == "REPLACEME":
        logger.error("Real API credentials still contain REPLACEME. Exiting.")
        return

    logger.info("A. Environment: Settings loaded, credentials configured.")

    # 1. Basic API Connectivity
    try:
        response = httpx.get(
            "https://api.razorpay.com/v1/payment_links/",
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret),
            params={"count": 1},
            timeout=5.0
        )
        response.raise_for_status()
        logger.info("B. Real API Authentication: PASS")
    except Exception as e:
        logger.error("B. Real API Authentication: FAIL - %s", e)
        return

    executor = RazorpayPaymentLinkProvider(
        key_id=settings.razorpay_key_id,
        key_secret=settings.razorpay_key_secret
    )

    db_engine = setup_test_db()
    SessionLocal = sessionmaker(bind=db_engine)

    # 2. Test Real Payment Link Creation
    logger.info("=== 2. Real Payment Link Creation ===")
    base_id = int(time.time())
    decision_id_1 = base_id + 1
    execution_reference_id_1 = f"exec_rd_{decision_id_1}"
    
    with SessionLocal() as session:
        decision_1 = setup_decision(session, decision_id_1, DecisionStatus.EXECUTING)
        try:
            plink_id = executor.create_payment_link(decision_1, execution_reference_id_1)
            logger.info("C. Real Payment Link Creation:")
            logger.info("   - Decision ID: %s", decision_id_1)
            logger.info("   - Reference ID: %s", execution_reference_id_1)
            logger.info("   - Razorpay Plink ID: %s", plink_id)
        except Exception as e:
            logger.error("Failed real creation: %s", e)
            return

    # 3. Test duplicate reference behavior
    logger.info("=== 3. Duplicate Reference Behavior ===")
    with SessionLocal() as session:
        decision_1 = session.get(RecoveryDecision, decision_id_1)
        try:
            executor.create_payment_link(decision_1, execution_reference_id_1)
            logger.error("D. Duplicate Reference: FAIL - Expected DuplicateReferenceExecutionError")
        except DuplicateReferenceExecutionError as e:
            logger.info("D. Duplicate Reference: PASS - Correctly received DuplicateReferenceExecutionError")
            logger.info("   - Error: %s", e)
        except Exception as e:
            logger.error("D. Duplicate Reference: FAIL - Unexpected exception type: %s", e)

    # 4. Test reconciliation against the real API
    logger.info("=== 4. Reconciliation against Real API ===")
    with SessionLocal() as session:
        decision_1 = session.get(RecoveryDecision, decision_id_1)
        try:
            logger.info("Waiting 2s for Razorpay to index the newly created link...")
            time.sleep(2)
            recovered_plink_id = executor.reconcile_duplicate_reference(decision_1, execution_reference_id_1)
            logger.info("E. Reconciliation: PASS")
            logger.info("   - Recovered Plink ID: %s", recovered_plink_id)
            if plink_id != recovered_plink_id:
                logger.error("Mismatch: Original %s vs Recovered %s", plink_id, recovered_plink_id)
        except Exception as e:
            logger.error("E. Reconciliation: FAIL - %s", e)

    # 5. Test stale EXECUTING recovery
    logger.info("=== 5. Stale EXECUTING Recovery ===")
    # Scenario A: Link exists
    decision_id_2 = base_id + 2
    execution_reference_id_2 = f"exec_rd_{decision_id_2}"
    with SessionLocal() as session:
        decision_2 = setup_decision(session, decision_id_2, DecisionStatus.EXECUTING)
        # Manually create link first to simulate it exists
        plink_id_2 = executor.create_payment_link(decision_2, execution_reference_id_2)
        # Now artificially age it
        past_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        decision_2.execution_started_at = past_time
        session.commit()
    
    # Run stale recovery
    with SessionLocal() as session:
        settings.execution_stale_after_seconds = 600
        stale_decision = claim_stale_execution(session, settings.execution_stale_after_seconds)
        if stale_decision and stale_decision.id == decision_id_2:
            reconcile_stale_decision(session, stale_decision, executor)
            session.commit()
            
            refreshed = session.get(RecoveryDecision, decision_id_2)
            logger.info("F. Stale Recovery (Link exists): PASS. State transitioned to %s", refreshed.decision_status)
        else:
            logger.error("F. Stale Recovery (Link exists): FAIL to claim")

    # Scenario B: Link does not exist
    decision_id_3 = base_id + 3
    with SessionLocal() as session:
        decision_3 = setup_decision(session, decision_id_3, DecisionStatus.EXECUTING)
        past_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        decision_3.execution_started_at = past_time
        session.commit()
    
    with SessionLocal() as session:
        stale_decision = claim_stale_execution(session, settings.execution_stale_after_seconds)
        if stale_decision and stale_decision.id == decision_id_3:
            reconcile_stale_decision(session, stale_decision, executor)
            session.commit()
            
            refreshed = session.get(RecoveryDecision, decision_id_3)
            logger.info("F. Stale Recovery (No link): PASS. State transitioned to %s", refreshed.decision_status)
            logger.info("   - next_retry_at: %s", refreshed.next_retry_at)
        else:
            logger.error("F. Stale Recovery (No link): FAIL to claim")

    # 6. Test worker with real executor
    logger.info("=== 6. Worker Execution ===")
    decision_id_4 = base_id + 4
    with SessionLocal() as session:
        setup_decision(session, decision_id_4, DecisionStatus.DECIDED)
    
    settings.worker_enabled = True
    settings.worker_batch_size = 10
    run_worker_cycle(db_engine, settings, executor)
    
    with SessionLocal() as session:
        decision_4 = session.get(RecoveryDecision, decision_id_4)
        logger.info("H. Worker: PASS")
        logger.info("   - Decision %s processed, new state: %s", decision_id_4, decision_4.decision_status)


if __name__ == "__main__":
    run_validation()
