import threading
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import sessionmaker

from app.models import RecoveryDecision, DecisionStatus, PaymentRecord, RecoveryAction
from app.worker import run_worker_cycle, start_worker
from app.config import Settings
from app.executor import PaymentLinkProvider, TransientExecutionError, PermanentExecutionError
from app.execution import execute_recovery_decision
from app.crud import get_eligible_recovery_decisions


class MockWorkerExecutor(PaymentLinkProvider):
    def __init__(self, should_fail=None):
        self.calls = []
        self.should_fail = should_fail

    def create_payment_link(self, decision: RecoveryDecision, execution_reference_id: str) -> str:
        self.calls.append(decision.id)
        if self.should_fail == "transient":
            raise TransientExecutionError("Transient mock error")
        elif self.should_fail == "permanent":
            raise PermanentExecutionError("Permanent mock error")
        elif self.should_fail == "unknown":
            raise ValueError("Unknown mock error")
        return f"plink_mock_{decision.id}"

    def reconcile_duplicate_reference(self, decision: RecoveryDecision, execution_reference_id: str) -> str:
        return f"plink_reconciled_{decision.id}"


def setup_decision(session, decision_id: int, status: DecisionStatus, next_retry_at=None, action=RecoveryAction.SEND_PAYMENT_LINK, attempts=0):
    record = PaymentRecord(id=decision_id, amount=1000, currency="INR", razorpay_payment_id=f"pay_{decision_id}", status="failed")
    decision = RecoveryDecision(
        id=decision_id,
        payment_record=record,
        decision_status=status,
        selected_action=action,
        next_retry_at=next_retry_at,
        execution_attempts=attempts,
    )
    session.add(record)
    session.add(decision)
    session.commit()
    return decision


def test_run_worker_cycle_eligible(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0)
    executor = MockWorkerExecutor()
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        setup_decision(session, 101, DecisionStatus.DECIDED)
        
    run_worker_cycle(db_engine, settings, executor)
    
    with Session() as session:
        decision = session.get(RecoveryDecision, 101)
        assert decision.decision_status == DecisionStatus.EXECUTED
        assert 101 in executor.calls


def test_run_worker_cycle_future_retry_ignored(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0)
    executor = MockWorkerExecutor()
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        future_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        setup_decision(session, 102, DecisionStatus.DECIDED, next_retry_at=future_time)
        
    run_worker_cycle(db_engine, settings, executor)
    
    with Session() as session:
        decision = session.get(RecoveryDecision, 102)
        assert decision.decision_status == DecisionStatus.DECIDED # Unchanged
        assert 102 not in executor.calls


def test_run_worker_cycle_eligible_retry(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0)
    executor = MockWorkerExecutor()
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        past_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        setup_decision(session, 103, DecisionStatus.DECIDED, next_retry_at=past_time)
        
    run_worker_cycle(db_engine, settings, executor)
    
    with Session() as session:
        decision = session.get(RecoveryDecision, 103)
        assert decision.decision_status == DecisionStatus.EXECUTED
        assert 103 in executor.calls


def test_run_worker_cycle_ignores_no_action_and_failed(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0)
    executor = MockWorkerExecutor()
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        setup_decision(session, 104, DecisionStatus.DECIDED, action=RecoveryAction.NO_ACTION)
        setup_decision(session, 105, DecisionStatus.FAILED)
        
    run_worker_cycle(db_engine, settings, executor)
    
    with Session() as session:
        assert 104 not in executor.calls
        assert 105 not in executor.calls


def test_run_worker_cycle_failure_isolation(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0)
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        setup_decision(session, 110, DecisionStatus.DECIDED)
        setup_decision(session, 111, DecisionStatus.DECIDED)
        setup_decision(session, 112, DecisionStatus.DECIDED)
    
    # Custom executor that fails completely on 110
    class IsolatingExecutor(PaymentLinkProvider):
        def __init__(self):
            self.calls = []
        def create_payment_link(self, decision, execution_reference_id):
            self.calls.append(decision.id)
            if decision.id == 110:
                raise ValueError("Unexpected error on 110")
            if decision.id == 111:
                raise PermanentExecutionError("Permanent error on 111", error_type="MOCK")
            return f"plink_{decision.id}"
            
        def reconcile_duplicate_reference(self, decision, ref_id):
            return "x"
            
    executor = IsolatingExecutor()
    run_worker_cycle(db_engine, settings, executor)
    
    with Session() as session:
        d110 = session.get(RecoveryDecision, 110)
        assert d110.decision_status == DecisionStatus.EXECUTING # Leaves it in EXECUTING state
        
        d111 = session.get(RecoveryDecision, 111)
        assert d111.decision_status == DecisionStatus.FAILED
        
        d112 = session.get(RecoveryDecision, 112)
        assert d112.decision_status == DecisionStatus.EXECUTED
        
        assert len(executor.calls) == 3


def test_worker_startup_shutdown(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=1)
    executor = MockWorkerExecutor()
    stop_event = threading.Event()
    
    thread = threading.Thread(target=start_worker, args=(db_engine, settings, executor, stop_event))
    thread.start()
    
    assert thread.is_alive()
    stop_event.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_stale_recovery_first(db_engine):
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0, execution_stale_after_seconds=600)
    executor = MockWorkerExecutor()
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        # Create a stale executing decision
        past_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        decision = setup_decision(session, 120, DecisionStatus.EXECUTING)
        decision.execution_started_at = past_time
        session.commit()
        
    run_worker_cycle(db_engine, settings, executor)
    
    with Session() as session:
        decision = session.get(RecoveryDecision, 120)
        # Assuming MockWorkerExecutor returns reconciled link successfully
        assert decision.decision_status == DecisionStatus.EXECUTED


def test_webhook_vs_worker_race(db_engine):
    from app.crud import claim_decision_for_execution
    
    settings = Settings(worker_enabled=True, worker_batch_size=10, worker_poll_interval_seconds=0)
    executor = MockWorkerExecutor()
    Session = sessionmaker(bind=db_engine)
    
    with Session() as session:
        setup_decision(session, 130, DecisionStatus.DECIDED)
        
    def worker_action():
        run_worker_cycle(db_engine, settings, executor)
        
    def webhook_action():
        with Session() as session:
            if claim_decision_for_execution(session, 130):
                session.commit()
                decision = session.get(RecoveryDecision, 130)
                execute_recovery_decision(session, decision, executor, decision.payment_record)

    t1 = threading.Thread(target=worker_action)
    t2 = threading.Thread(target=webhook_action)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    with Session() as session:
        decision = session.get(RecoveryDecision, 130)
        assert decision.decision_status == DecisionStatus.EXECUTED
        assert decision.execution_attempts == 1
        assert len(executor.calls) == 1
