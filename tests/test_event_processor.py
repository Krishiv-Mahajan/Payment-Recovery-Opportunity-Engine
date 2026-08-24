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
