import pytest
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.models import (
    PaymentRecord, RecoveryDecision, PaymentStatus, DecisionStatus, 
    RecoveryAction, AuditLog
)
from app.dashboard_metrics import (
    get_live_summary, get_experiment_metrics, 
    get_economic_impact, get_guardrail_metrics
)

# Helper to create a dummy payment record
def create_payment(db: Session, status: PaymentStatus, amount: int = 1000):
    p = PaymentRecord(
        razorpay_payment_id=f"pay_{datetime.now().timestamp()}",
        amount=amount,
        currency="INR",
        status=status,
    )
    db.add(p)
    db.flush()
    return p

# Helper to create a dummy recovery decision
def create_decision(db: Session, payment_record_id: int, variant: str, status: DecisionStatus, action: RecoveryAction, expected_net: int = 0):
    d = RecoveryDecision(
        payment_record_id=payment_record_id,
        decision_status=status,
        selected_action=action,
        experiment_variant=variant,
        expected_incremental_net_paise=expected_net
    )
    db.add(d)
    db.flush()
    return d

def test_zero_decisions(db_session: Session):
    """Test dashboard metrics handle zero-data gracefully."""
    summary = get_live_summary(db_session)
    assert summary["total_failed"] == 0
    assert summary["total_opportunities"] == 0
    assert summary["links_executed"] == 0
    assert summary["links_paid"] == 0
    assert summary["recovered_revenue_inr"] == 0.0

    exp = get_experiment_metrics(db_session)
    assert exp["control_n"] == 0
    assert exp["control_rate"] == 0
    assert exp["treatment_n"] == 0
    assert exp["treatment_rate"] == 0
    assert exp["observed_uplift"] is None

    econ = get_economic_impact(db_session)
    assert econ["expected_incremental_net_inr"] == 0.0

def test_treatment_only(db_session: Session):
    """Test when only treatment data exists (no div/0 errors)."""
    p1 = create_payment(db_session, PaymentStatus.FAILED, 50000)
    create_decision(db_session, p1.id, "treatment", DecisionStatus.EXECUTED, RecoveryAction.SEND_PAYMENT_LINK, 1000)

    exp = get_experiment_metrics(db_session)
    assert exp["treatment_n"] == 1
    assert exp["control_n"] == 0
    assert exp["treatment_rate"] == 0.0
    assert exp["control_rate"] == 0.0
    assert exp["observed_uplift"] is None

def test_control_and_treatment(db_session: Session):
    """Test treatment and control aggregation and natural recovery."""
    # Control naturally recovered
    p_ctrl = create_payment(db_session, PaymentStatus.CAPTURED, 20000)
    create_decision(db_session, p_ctrl.id, "control", DecisionStatus.DECIDED, RecoveryAction.NO_ACTION, 0)
    
    # Treatment not recovered
    p_trt = create_payment(db_session, PaymentStatus.FAILED, 30000)
    create_decision(db_session, p_trt.id, "treatment", DecisionStatus.EXECUTED, RecoveryAction.SEND_PAYMENT_LINK, 500)

    exp = get_experiment_metrics(db_session)
    assert exp["control_n"] == 1
    assert exp["control_recovered"] == 1
    assert exp["control_rate"] == 1.0
    
    assert exp["treatment_n"] == 1
    assert exp["treatment_recovered"] == 0
    assert exp["treatment_rate"] == 0.0
    
    assert exp["observed_uplift"] == -1.0 # 0.0 - 1.0

def test_executed_but_unpaid_link(db_session: Session):
    """Test link sent but not paid."""
    p = create_payment(db_session, PaymentStatus.FAILED, 50000)
    create_decision(db_session, p.id, "treatment", DecisionStatus.EXECUTED, RecoveryAction.SEND_PAYMENT_LINK, 1500)

    summary = get_live_summary(db_session)
    assert summary["links_executed"] == 1
    assert summary["links_paid"] == 0
    assert summary["recovered_revenue_inr"] == 0.0

    econ = get_economic_impact(db_session)
    assert econ["expected_incremental_net_inr"] == 15.0

def test_executed_and_paid_link(db_session: Session):
    """Test link sent and paid."""
    p = create_payment(db_session, PaymentStatus.FAILED, 50000)
    create_decision(db_session, p.id, "treatment", DecisionStatus.OUTCOME_OBSERVED, RecoveryAction.SEND_PAYMENT_LINK, 1500)

    summary = get_live_summary(db_session)
    assert summary["links_executed"] == 1
    assert summary["links_paid"] == 1
    assert summary["recovered_revenue_inr"] == 500.0

    econ = get_economic_impact(db_session)
    assert econ["expected_incremental_net_inr"] == 15.0

    exp = get_experiment_metrics(db_session)
    assert exp["treatment_recovered"] == 1
    assert exp["treatment_rate"] == 1.0

def test_guardrail_blocked_decision(db_session: Session):
    """Test guardrail metrics."""
    log = AuditLog(
        event_type="GUARDRAIL_COOLDOWN",
        action="SKIP_OUTREACH",
        reason="cooldown"
    )
    db_session.add(log)
    
    log2 = AuditLog(
        event_type="POLICY_DECISION",
        action="PLACEHOLDER",
        reason="Safety override: duplicate"
    )
    db_session.add(log2)
    db_session.flush()

    guardrails = get_guardrail_metrics(db_session)
    assert guardrails["cooldown_blocks"] == 1
    assert guardrails["duplicate_blocks"] == 1
    assert guardrails["fallback_blocks"] == 0

def test_expected_incremental_net_aggregation(db_session: Session):
    """Test expected incremental net value sums only executed/observed decisions."""
    p1 = create_payment(db_session, PaymentStatus.FAILED, 1000)
    create_decision(db_session, p1.id, "treatment", DecisionStatus.EXECUTED, RecoveryAction.SEND_PAYMENT_LINK, 1000)
    
    p2 = create_payment(db_session, PaymentStatus.FAILED, 1000)
    create_decision(db_session, p2.id, "treatment", DecisionStatus.OUTCOME_OBSERVED, RecoveryAction.SEND_PAYMENT_LINK, 2000)
    
    # Not executed, should not be summed
    p3 = create_payment(db_session, PaymentStatus.FAILED, 1000)
    create_decision(db_session, p3.id, "treatment", DecisionStatus.DECIDED, RecoveryAction.SEND_PAYMENT_LINK, 5000)
    
    econ = get_economic_impact(db_session)
    assert econ["expected_incremental_net_inr"] == 30.0 # 3000 paise
