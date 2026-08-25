"""
Tests for GuardrailsEngine.
"""
from __future__ import annotations

import pytest
from app.guardrails import GuardrailsEngine
from app.models import PaymentRecord, RecoveryDecision, DecisionStatus, RecoveryAction, CustomerOutreachEvent
from app.ml.predictor import PolicyPrediction
import datetime

def test_guardrails_duplicate_execution(db_session):
    guardrails = GuardrailsEngine()
    record = PaymentRecord(
        id=1,
        razorpay_payment_id="pay_guard_1",
        amount=1000,
        currency="INR",
        status="failed",
        customer_email="a@example.com",
        customer_contact="9999999999"
    )
    db_session.add(record)
    db_session.commit()

    decision = RecoveryDecision(
        payment_record_id=record.id,
        decision_status=DecisionStatus.EXECUTED,
        selected_action=RecoveryAction.SEND_PAYMENT_LINK,
        model_version="test-v1",
    )
    db_session.add(decision)
    db_session.commit()

    new_pred = PolicyPrediction(
        decision_status=DecisionStatus.DECIDED,
        selected_action=RecoveryAction.SEND_PAYMENT_LINK,
        model_version="test-v2",
        reasoning="mock",
    )

    safe_pred = guardrails.evaluate(db_session, record, new_pred)
    assert safe_pred.selected_action == RecoveryAction.NO_ACTION
    assert "Duplicate execution prevented" in safe_pred.reasoning

def test_guardrails_model_fallback(db_session):
    guardrails = GuardrailsEngine()
    record = PaymentRecord(
        id=2,
        razorpay_payment_id="pay_guard_2",
        amount=1000,
        currency="INR",
        status="failed",
        customer_email="a@example.com",
        customer_contact="9999999999"
    )
    db_session.add(record)
    db_session.commit()

    safe_pred = guardrails.evaluate(db_session, record, None)
    assert safe_pred.selected_action == RecoveryAction.NO_ACTION
    assert safe_pred.model_version == "fallback-v0"

def test_guardrails_cooldown_allow_first_time(db_session):
    guardrails = GuardrailsEngine(cooldown_hours=48)
    record = PaymentRecord(
        id=3,
        razorpay_payment_id="pay_guard_3",
        amount=1000,
        currency="INR",
        status="failed",
        customer_email="a@example.com",
    )
    db_session.add(record)
    db_session.commit()

    new_pred = PolicyPrediction(
        decision_status=DecisionStatus.DECIDED,
        selected_action=RecoveryAction.SEND_PAYMENT_LINK,
        model_version="test-v3",
        reasoning="mock",
    )

    safe_pred = guardrails.evaluate(db_session, record, new_pred)
    assert safe_pred.selected_action == RecoveryAction.SEND_PAYMENT_LINK


def test_guardrails_cooldown_block_recent(db_session):
    guardrails = GuardrailsEngine(cooldown_hours=48)
    record = PaymentRecord(
        id=4,
        razorpay_payment_id="pay_guard_4",
        amount=1000,
        currency="INR",
        status="failed",
        customer_email="a@example.com",
    )
    db_session.add(record)

    # Add recent outreach
    outreach = CustomerOutreachEvent(
        customer_identifier="a@example.com",
        payment_record_id=record.id,
        recovery_decision_id=1,
        action="SEND_PAYMENT_LINK",
        channel="payment_link",
        outreach_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    )
    db_session.add(outreach)
    db_session.commit()

    new_pred = PolicyPrediction(
        decision_status=DecisionStatus.DECIDED,
        selected_action=RecoveryAction.SEND_PAYMENT_LINK,
        model_version="test-v3",
        reasoning="mock",
    )

    safe_pred = guardrails.evaluate(db_session, record, new_pred)
    assert safe_pred.selected_action == RecoveryAction.NO_ACTION
    assert "Cooldown active" in safe_pred.reasoning


def test_guardrails_cooldown_allow_expired(db_session):
    guardrails = GuardrailsEngine(cooldown_hours=48)
    record = PaymentRecord(
        id=5,
        razorpay_payment_id="pay_guard_5",
        amount=1000,
        currency="INR",
        status="failed",
        customer_email="a@example.com",
    )
    db_session.add(record)

    # Add expired outreach
    outreach = CustomerOutreachEvent(
        customer_identifier="a@example.com",
        payment_record_id=record.id,
        recovery_decision_id=1,
        action="SEND_PAYMENT_LINK",
        channel="payment_link",
        outreach_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=72)
    )
    db_session.add(outreach)
    db_session.commit()

    new_pred = PolicyPrediction(
        decision_status=DecisionStatus.DECIDED,
        selected_action=RecoveryAction.SEND_PAYMENT_LINK,
        model_version="test-v3",
        reasoning="mock",
    )

    safe_pred = guardrails.evaluate(db_session, record, new_pred)
    assert safe_pred.selected_action == RecoveryAction.SEND_PAYMENT_LINK


def test_guardrails_cooldown_fail_open_no_identifier(db_session):
    guardrails = GuardrailsEngine(cooldown_hours=48)
    record = PaymentRecord(
        id=6,
        razorpay_payment_id="pay_guard_6",
        amount=1000,
        currency="INR",
        status="failed",
        customer_email=None,
        customer_contact=None,
    )
    db_session.add(record)
    db_session.commit()

    new_pred = PolicyPrediction(
        decision_status=DecisionStatus.DECIDED,
        selected_action=RecoveryAction.SEND_PAYMENT_LINK,
        model_version="test-v3",
        reasoning="mock",
    )

    safe_pred = guardrails.evaluate(db_session, record, new_pred)
    assert safe_pred.selected_action == RecoveryAction.SEND_PAYMENT_LINK
