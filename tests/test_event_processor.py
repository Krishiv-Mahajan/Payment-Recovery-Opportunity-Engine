"""
Tests for event processor lifecycle and execution flow.
"""
from __future__ import annotations

import pytest
from app.models import RecoveryDecision, DecisionStatus, PaymentRecord
from tests.conftest import make_signed_request, PAYMENT_FAILED_PAYLOAD

def test_lifecycle_and_payment_link_paid(client, db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    # Force the predictor to return an action so we trigger execution
    from app.ml.predictor import PolicyPrediction
    from app.models import RecoveryAction
    class ForceActionPredictor:
        def predict(self, features):
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                model_version="test-mock",
                reasoning="mock"
            )
    client.app.state.predictor = ForceActionPredictor()
    
    # 1. Fire payment.failed
    r = make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
    assert r[0].status_code == 200
    
    with Session() as session:
        decision = session.query(RecoveryDecision).first()
        assert decision.decision_status == DecisionStatus.EXECUTED
        ref_id = decision.execution_reference_id
        assert ref_id is not None
        
    # 2. Fire payment_link.paid with correct reference
    plink_payload = {
        "entity": "event",
        "event": "payment_link.paid",
        "id": "evt_plink_paid_1",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": "plink_123",
                    "status": "paid",
                    "notes": {
                        "execution_reference_id": ref_id
                    }
                }
            }
        }
    }
    
    r2, _ = make_signed_request(client, plink_payload)
    assert r2.status_code == 200
    
    with Session() as session:
        decision = session.query(RecoveryDecision).first()
        assert decision.decision_status == DecisionStatus.OUTCOME_OBSERVED
        
def test_payment_link_paid_missing_reference(client, db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    plink_payload = {
        "entity": "event",
        "event": "payment_link.paid",
        "id": "evt_plink_paid_2",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": "plink_124",
                    "status": "paid",
                    "notes": {} # Missing execution_reference_id
                }
            }
        }
    }
    
    r, _ = make_signed_request(client, plink_payload)
    assert r.status_code == 200
    
    with Session() as session:
        from app.models import AuditLog
        log = session.query(AuditLog).filter(AuditLog.action == "SKIP_MISSING_REFERENCE").first()
        assert log is not None

def test_executor_transient_failure(client, db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    from app.ml.predictor import PolicyPrediction
    from app.models import RecoveryAction
    class ForceActionPredictor:
        def predict(self, features):
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                model_version="test-mock",
                reasoning="mock"
            )
    client.app.state.predictor = ForceActionPredictor()
    
    from app.executor import TransientExecutionError
    class FailingExecutor:
        def create_payment_link(self, decision, ref_id):
            raise TransientExecutionError("Transient mock error", error_type="TIMEOUT")
            
    client.app.state.executor = FailingExecutor()
    
    r = make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
    assert r[0].status_code == 200
    
    with Session() as session:
        decision = session.query(RecoveryDecision).first()
        # Should have transitioned from DECIDED -> EXECUTING -> DECIDED
        assert decision.decision_status == DecisionStatus.DECIDED
        assert decision.execution_attempts == 1
        assert decision.last_error == "Transient mock error"
        assert decision.last_error_type == "TIMEOUT"

def test_executor_permanent_failure(client, db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    
    from app.ml.predictor import PolicyPrediction
    from app.models import RecoveryAction
    class ForceActionPredictor:
        def predict(self, features):
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                model_version="test-mock",
                reasoning="mock"
            )
    client.app.state.predictor = ForceActionPredictor()
    
    from app.executor import PermanentExecutionError
    class FailingExecutor:
        def create_payment_link(self, decision, ref_id):
            raise PermanentExecutionError("Permanent mock error", error_type="HTTP_400")
            
    client.app.state.executor = FailingExecutor()
    
    import copy
    PAYMENT_FAILED_PAYLOAD_2 = copy.deepcopy(PAYMENT_FAILED_PAYLOAD)
    PAYMENT_FAILED_PAYLOAD_2["payload"]["payment"]["entity"]["id"] = "pay_perm_fail"
    
    r = make_signed_request(client, PAYMENT_FAILED_PAYLOAD_2)
    assert r[0].status_code == 200
    
    with Session() as session:
        # Assuming our DB gets cleared or we just find by payment_id
        decision = session.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_perm_fail").first()
        # Should have transitioned from DECIDED -> EXECUTING -> FAILED
        assert decision.decision_status == DecisionStatus.FAILED
        assert decision.execution_attempts == 1
        assert decision.last_error == "Permanent mock error"
        assert decision.last_error_type == "HTTP_400"

def test_executor_duplicate_reference_success(client, db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)

    from app.ml.predictor import PolicyPrediction
    from app.models import RecoveryAction
    class ForceActionPredictor:
        def predict(self, features):
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                model_version="test-mock",
                reasoning="mock"
            )
    client.app.state.predictor = ForceActionPredictor()

    class ReconcilingExecutor:
        def create_payment_link(self, decision, ref_id):
            from app.executor import DuplicateReferenceExecutionError
            raise DuplicateReferenceExecutionError("Duplicate mock error", error_type="DUPLICATE_REFERENCE_ID")
        
        def reconcile_duplicate_reference(self, decision, ref_id):
            return "plink_reconciled"
            
    client.app.state.executor = ReconcilingExecutor()
    
    import copy
    PAYMENT_FAILED_PAYLOAD_DUP = copy.deepcopy(PAYMENT_FAILED_PAYLOAD)
    PAYMENT_FAILED_PAYLOAD_DUP["payload"]["payment"]["entity"]["id"] = "pay_dup_success"
    
    r = make_signed_request(client, PAYMENT_FAILED_PAYLOAD_DUP)
    assert r[0].status_code == 200
    
    with Session() as session:
        decision = session.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_dup_success").first()
        assert decision.decision_status == DecisionStatus.EXECUTED
        assert decision.execution_attempts == 1

def test_executor_duplicate_reference_reconciliation_failure(client, db_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)

    from app.ml.predictor import PolicyPrediction
    from app.models import RecoveryAction
    class ForceActionPredictor:
        def predict(self, features):
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                model_version="test-mock",
                reasoning="mock"
            )
    client.app.state.predictor = ForceActionPredictor()

    class FailedReconcilingExecutor:
        def create_payment_link(self, decision, ref_id):
            from app.executor import DuplicateReferenceExecutionError
            raise DuplicateReferenceExecutionError("Duplicate mock error", error_type="DUPLICATE_REFERENCE_ID")
        
        def reconcile_duplicate_reference(self, decision, ref_id):
            from app.executor import ReconciliationError
            raise ReconciliationError("Mock missing link", error_type="MISSING_LINK")
            
    client.app.state.executor = FailedReconcilingExecutor()
    
    import copy
    PAYMENT_FAILED_PAYLOAD_DUP2 = copy.deepcopy(PAYMENT_FAILED_PAYLOAD)
    PAYMENT_FAILED_PAYLOAD_DUP2["payload"]["payment"]["entity"]["id"] = "pay_dup_fail"
    
    r = make_signed_request(client, PAYMENT_FAILED_PAYLOAD_DUP2)
    assert r[0].status_code == 200
    
    with Session() as session:
        decision = session.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_dup_fail").first()
        assert decision.decision_status == DecisionStatus.FAILED
        assert decision.last_error == "Mock missing link"
        assert decision.last_error_type == "MISSING_LINK"
