import pytest
import copy
from datetime import datetime, timezone, timedelta

from app.models import RecoveryDecision, DecisionStatus, PaymentRecord, RecoveryAction
from app.crud import claim_stale_execution, claim_decision_for_execution
from app.stale_recovery import reconcile_stale_decision
from app.executor import PaymentLinkProvider, TransientExecutionError, PermanentExecutionError, ReconciliationError, AmbiguousStateError


def test_atomic_stale_claim(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        # Create a stale decision
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=101, amount=1000, currency="INR", razorpay_payment_id="pay_101", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=201, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=1
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        # Claim A succeeds
        claimed_decision = claim_stale_execution(session, stale_threshold_seconds=600)
        assert claimed_decision is not None
        assert claimed_decision.id == 201
        
        # execution_started_at is updated
        assert claimed_decision.execution_started_at.replace(tzinfo=timezone.utc) > now - timedelta(seconds=10)
        session.commit()
        
        # Claim B fails
        second_claim = claim_stale_execution(session, stale_threshold_seconds=600)
        assert second_claim is None


def test_fresh_executing_not_claimed(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=102, amount=1000, currency="INR", razorpay_payment_id="pay_102", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=202, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=100), # Not stale yet
            execution_attempts=1
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        claimed = claim_stale_execution(session, stale_threshold_seconds=600)
        assert claimed is None


class MockFoundLinkExecutor:
    def reconcile_duplicate_reference(self, decision, ref_id):
        return "plink_found_123"

def test_stale_recovery_matching_link(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=103, amount=1000, currency="INR", razorpay_payment_id="pay_103", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=203, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=1
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        decision = claim_stale_execution(session, 600)
        reconcile_stale_decision(session, decision, MockFoundLinkExecutor())
        
        db_decision = session.query(RecoveryDecision).get(203)
        assert db_decision.decision_status == DecisionStatus.EXECUTED
        assert db_decision.execution_reference_id == "exec_rd_203"


class MockNoLinkExecutor:
    def reconcile_duplicate_reference(self, decision, ref_id):
        raise AmbiguousStateError("No payment link found", error_type="MISSING_LINK")

def test_stale_recovery_no_link(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=104, amount=1000, currency="INR", razorpay_payment_id="pay_104", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=204, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=2 # Attempt count 2
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        decision = claim_stale_execution(session, 600)
        reconcile_stale_decision(session, decision, MockNoLinkExecutor())
        
        db_decision = session.query(RecoveryDecision).get(204)
        assert db_decision.decision_status == DecisionStatus.DECIDED
        # Attempt count should remain 2, not incremented or reset
        assert db_decision.execution_attempts == 2
        assert db_decision.next_retry_at is not None


def test_stale_recovery_no_link_at_max_attempt(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=105, amount=1000, currency="INR", razorpay_payment_id="pay_105", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=205, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=5 # MAX ATTEMPTS
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        decision = claim_stale_execution(session, 600)
        reconcile_stale_decision(session, decision, MockNoLinkExecutor())
        
        db_decision = session.query(RecoveryDecision).get(205)
        # Cannot be retried since attempts=5
        assert db_decision.decision_status == DecisionStatus.FAILED
        assert db_decision.execution_attempts == 5
        assert db_decision.next_retry_at is None


class MockTransientLookupExecutor:
    def reconcile_duplicate_reference(self, decision, ref_id):
        raise TransientExecutionError("HTTP 502", error_type="NETWORK")

def test_stale_recovery_lookup_transient_failure(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=106, amount=1000, currency="INR", razorpay_payment_id="pay_106", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=206, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=3
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        decision = claim_stale_execution(session, 600)
        reconcile_stale_decision(session, decision, MockTransientLookupExecutor())
        
        db_decision = session.query(RecoveryDecision).get(206)
        # Should remain in EXECUTING with updated error
        assert db_decision.decision_status == DecisionStatus.EXECUTING
        assert db_decision.last_error == "HTTP 502"
        assert db_decision.last_error_type == "NETWORK"
        assert db_decision.next_retry_at is None


class MockPermanentLookupExecutor:
    def reconcile_duplicate_reference(self, decision, ref_id):
        raise PermanentExecutionError("HTTP 401 Unauthorized", error_type="AUTH")

def test_stale_recovery_lookup_permanent_failure(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=107, amount=1000, currency="INR", razorpay_payment_id="pay_107", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=207, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=1
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        decision = claim_stale_execution(session, 600)
        reconcile_stale_decision(session, decision, MockPermanentLookupExecutor())
        
        db_decision = session.query(RecoveryDecision).get(207)
        # Should REMAIN in EXECUTING, NOT FAILED.
        assert db_decision.decision_status == DecisionStatus.EXECUTING
        assert db_decision.last_error == "HTTP 401 Unauthorized"
        assert db_decision.last_error_type == "AUTH"


class MockMismatchLookupExecutor:
    def reconcile_duplicate_reference(self, decision, ref_id):
        raise ReconciliationError("Mismatched amount", error_type="MISMATCH")

def test_stale_recovery_reconciliation_mismatch(db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        now = datetime.now(timezone.utc)
        record = PaymentRecord(id=108, amount=1000, currency="INR", razorpay_payment_id="pay_108", status=DecisionStatus.FAILED.value)
        decision = RecoveryDecision(
            id=208, 
            payment_record=record,
            decision_status=DecisionStatus.EXECUTING,
            execution_started_at=now - timedelta(seconds=700),
            execution_attempts=1
        )
        session.add(record)
        session.add(decision)
        session.commit()
        
    with Session() as session:
        decision = claim_stale_execution(session, 600)
        reconcile_stale_decision(session, decision, MockMismatchLookupExecutor())
        
        db_decision = session.query(RecoveryDecision).get(208)
        # A confirmed mismatch is a terminal state failure
        assert db_decision.decision_status == DecisionStatus.FAILED
        assert db_decision.last_error == "Mismatched amount"
        assert db_decision.last_error_type == "MISMATCH"
