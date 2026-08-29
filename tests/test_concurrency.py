"""
Concurrency tests for atomic claiming.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from app.models import RecoveryDecision, DecisionStatus, RecoveryAction, PaymentRecord, WebhookEvent, WebhookProcessingStatus, PaymentStatus
from app.crud import claim_decision_for_execution


def test_atomic_claim_concurrency(db_engine):
    """
    Test that two concurrent claims for the same decision result in exactly
    one success and one failure.
    """
    # 1. Setup a DECIDED decision
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    
    with Session() as session:
        webhook = WebhookEvent(
            event_id="evt_concurrency_test",
            event_type="payment.failed",
            raw_payload="{}",
            signature_verified=True,
            processing_status=WebhookProcessingStatus.PROCESSED
        )
        session.add(webhook)
        session.flush()
        
        payment = PaymentRecord(
            webhook_event_id=webhook.id,
            razorpay_payment_id="pay_concurrency_test",
            amount=50000,
            currency="INR",
            status=PaymentStatus.FAILED
        )
        session.add(payment)
        session.flush()
        
        decision = RecoveryDecision(
            payment_record_id=payment.id,
            decision_status=DecisionStatus.DECIDED,
            selected_action=RecoveryAction.SEND_PAYMENT_LINK,
            model_version="test"
        )
        session.add(decision)
        session.commit()
        
        decision_id = decision.id
        
    # 2. Attempt two concurrent claims using two different sessions
    claim_1_success = None
    claim_2_success = None
    
    def worker_1():
        nonlocal claim_1_success
        with Session() as s1:
            claim_1_success = claim_decision_for_execution(s1, decision_id)
            s1.commit()
            
    def worker_2():
        nonlocal claim_2_success
        with Session() as s2:
            claim_2_success = claim_decision_for_execution(s2, decision_id)
            s2.commit()
            
    t1 = threading.Thread(target=worker_1)
    t2 = threading.Thread(target=worker_2)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    # 3. Verify exactly one claim succeeds
    assert (claim_1_success and not claim_2_success) or (not claim_1_success and claim_2_success)
    
    # 4. Verify final state and attempts
    with Session() as session:
        updated_decision = session.query(RecoveryDecision).get(decision_id)
        assert updated_decision.decision_status == DecisionStatus.EXECUTING
        assert updated_decision.execution_attempts == 1
        assert updated_decision.execution_started_at is not None
