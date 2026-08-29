import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, case
from app.models import RecoveryDecision, PaymentRecord, AuditLog, DecisionStatus, PaymentStatus, RecoveryAction

logger = logging.getLogger(__name__)

def is_recovered_condition():
    """Returns the SQLAlchemy condition for determining if a payment was recovered."""
    return or_(
        RecoveryDecision.decision_status == DecisionStatus.OUTCOME_OBSERVED,
        PaymentRecord.status == PaymentStatus.CAPTURED
    )

def get_live_summary(db: Session) -> dict:
    """Calculates overall metrics from the live database."""
    total_failed = db.query(PaymentRecord).filter(PaymentRecord.status == PaymentStatus.FAILED).count()
    
    # Decisions executed
    links_executed = db.query(RecoveryDecision).filter(
        RecoveryDecision.decision_status.in_([DecisionStatus.EXECUTED, DecisionStatus.OUTCOME_OBSERVED])
    ).count()
    
    # Links paid (specifically via the link, which means OUTCOME_OBSERVED)
    links_paid = db.query(RecoveryDecision).filter(
        RecoveryDecision.decision_status == DecisionStatus.OUTCOME_OBSERVED
    ).count()
    
    # Recovered gross amount (from ANY recovery: OUTCOME_OBSERVED or CAPTURED)
    recovered_amount_paise = db.query(func.sum(PaymentRecord.amount)).join(
        RecoveryDecision, RecoveryDecision.payment_record_id == PaymentRecord.id
    ).filter(
        is_recovered_condition()
    ).scalar() or 0
    
    # Total opportunities (decisions made)
    total_opportunities = db.query(RecoveryDecision).count()
    
    return {
        "total_failed": total_failed,
        "total_opportunities": total_opportunities,
        "links_executed": links_executed,
        "links_paid": links_paid,
        "recovered_revenue_inr": recovered_amount_paise / 100.0
    }

def get_experiment_metrics(db: Session) -> dict:
    """Calculates experiment metrics (Treatment vs Control) from the live database."""
    
    # Calculate for control
    control_stats = db.query(
        func.count().label("n"),
        func.sum(case((is_recovered_condition(), 1), else_=0)).label("recovered")
    ).join(
        PaymentRecord, RecoveryDecision.payment_record_id == PaymentRecord.id
    ).filter(
        RecoveryDecision.experiment_variant == "control"
    ).first()
    
    # Calculate for treatment
    treatment_stats = db.query(
        func.count().label("n"),
        func.sum(case((is_recovered_condition(), 1), else_=0)).label("recovered")
    ).join(
        PaymentRecord, RecoveryDecision.payment_record_id == PaymentRecord.id
    ).filter(
        RecoveryDecision.experiment_variant == "treatment"
    ).first()
    
    control_n = control_stats.n or 0
    control_recovered = control_stats.recovered or 0
    control_rate = (control_recovered / control_n) if control_n > 0 else 0
    
    treatment_n = treatment_stats.n or 0
    treatment_recovered = treatment_stats.recovered or 0
    treatment_rate = (treatment_recovered / treatment_n) if treatment_n > 0 else 0
    
    # For intervention rate, we need to know how many times we actually intervened in treatment
    interventions = db.query(RecoveryDecision).filter(
        RecoveryDecision.experiment_variant == "treatment",
        RecoveryDecision.decision_status.in_([DecisionStatus.EXECUTED, DecisionStatus.OUTCOME_OBSERVED])
    ).count()
    
    intervention_rate = (interventions / treatment_n) if treatment_n > 0 else 0
    
    return {
        "control_n": control_n,
        "control_recovered": control_recovered,
        "control_rate": control_rate,
        "treatment_n": treatment_n,
        "treatment_recovered": treatment_recovered,
        "treatment_rate": treatment_rate,
        "intervention_rate": intervention_rate,
        "observed_uplift": treatment_rate - control_rate if (control_n > 0 and treatment_n > 0) else None
    }

def get_economic_impact(db: Session) -> dict:
    """
    Calculates expected incremental net value.
    Treats the stored `expected_incremental_net_paise` as authoritative.
    Only sums this value for decisions where an action was actually taken.
    """
    # Sum the expected incremental net paise for decisions that were executed or paid
    expected_incremental_paise = db.query(func.sum(RecoveryDecision.expected_incremental_net_paise)).filter(
        RecoveryDecision.decision_status.in_([DecisionStatus.EXECUTED, DecisionStatus.OUTCOME_OBSERVED]),
        RecoveryDecision.selected_action != RecoveryAction.NO_ACTION
    ).scalar() or 0
    
    return {
        "expected_incremental_net_inr": expected_incremental_paise / 100.0
    }

def get_guardrail_metrics(db: Session) -> dict:
    """Calculates guardrail block metrics from audit logs."""
    duplicate_blocks = db.query(AuditLog).filter(
        AuditLog.action.in_(["SKIP_DUPLICATE", "IDEMPOTENT_ACK"])
    ).count()
    
    cooldown_blocks = db.query(AuditLog).filter(
        AuditLog.event_type == "GUARDRAIL_COOLDOWN"
    ).count()
    
    fallback_blocks = db.query(AuditLog).filter(
        AuditLog.event_type == "GUARDRAIL_MODEL_FALLBACK"
    ).count()
    
    # Alternative way for blocks based on placeholder decision logs:
    override_cooldown_blocks = db.query(AuditLog).filter(
        AuditLog.event_type == "POLICY_DECISION",
        AuditLog.reason.ilike("%cooldown%")
    ).count()
    
    override_duplicate_blocks = db.query(AuditLog).filter(
        AuditLog.event_type == "POLICY_DECISION",
        AuditLog.reason.ilike("%duplicate%")
    ).count()
    
    override_fallback_blocks = db.query(AuditLog).filter(
        AuditLog.event_type == "POLICY_DECISION",
        AuditLog.reason.ilike("%fallback%")
    ).count()
    
    return {
        "duplicate_blocks": duplicate_blocks + override_duplicate_blocks,
        "cooldown_blocks": cooldown_blocks + override_cooldown_blocks,
        "fallback_blocks": fallback_blocks + override_fallback_blocks,
    }
