import os
import sys
import json
import hashlib
import hmac
import time

# Set up environment for the test
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./roe_validation.db"
os.environ["RAZORPAY_WEBHOOK_SECRET"] = "test_secret"
os.environ["EXPERIMENT_NAME"] = "buildathon_validation"
os.environ["EXECUTOR_MODE"] = "mock"

# Ensure we have a fresh database
if os.path.exists("roe_validation.db"):
    os.remove("roe_validation.db")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from app.main import app
from app.database import Base
from app.models import RecoveryDecision, AuditLog, WebhookEvent, PaymentRecord
from sqlalchemy.orm import sessionmaker

# Create tables
engine = create_engine("sqlite:///./roe_validation.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

client = TestClient(app)

def make_signed_request(payload: dict):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(b"test_secret", body, hashlib.sha256).hexdigest()
    with TestClient(app) as client:
        return client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": sig})

def get_failed_payload(evt_id, pay_id, email, amount=5000000): # 50000 INR
    return {
        "entity": "event",
        "event": "payment.failed",
        "id": evt_id,
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": pay_id,
                    "entity": "payment",
                    "amount": amount,
                    "currency": "INR",
                    "status": "failed",
                    "method": "card",
                    "email": email,
                    "contact": "+919999999999",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Payment cancelled",
                    "error_source": "customer",
                    "error_step": "payment_authentication",
                    "error_reason": "customer_cancelled",
                    "created_at": 1700000000
                }
            }
        },
        "created_at": 1700000000
    }

def get_plink_paid_payload(evt_id, plink_id, ref_id):
    return {
        "entity": "event",
        "event": "payment_link.paid",
        "id": evt_id,
        "contains": ["payment_link"],
        "payload": {
            "payment_link": {
                "entity": {
                    "id": plink_id,
                    "entity": "payment_link",
                    "status": "paid",
                    "notes": {
                        "execution_reference_id": ref_id
                    }
                }
            }
        },
        "created_at": 1700001000
    }

def get_captured_payload(evt_id, pay_id, email, amount=5000000):
    return {
        "entity": "event",
        "event": "payment.captured",
        "id": evt_id,
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": pay_id,
                    "entity": "payment",
                    "amount": amount,
                    "currency": "INR",
                    "status": "captured",
                    "method": "card",
                    "email": email,
                    "contact": "+919999999999",
                    "created_at": 1600000000
                }
            }
        },
        "created_at": 1600000000
    }

print("=== PART 1: CONTROL VALIDATION ===")
os.environ["CONTROL_PERCENTAGE"] = "100"
from app.config import get_settings
get_settings.cache_clear() # Clear cache to pick up env var

r1 = make_signed_request(get_failed_payload("evt_control_1", "pay_control_1", "control@example.com"))
print("Webhook Response:", r1.status_code, r1.json())

with SessionLocal() as db:
    decision = db.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_control_1").first()
    print("Variant:", decision.experiment_variant)
    print("Action:", decision.selected_action.value)
    print("Model Version:", decision.model_version)
    assert decision.experiment_variant == "control"
    assert decision.selected_action.value == "NO_ACTION"
    assert decision.model_version == "control_baseline"
    
    logs = db.query(AuditLog).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_control_1").all()
    print("Audit Logs:", [l.event_type for l in logs])
print("PART 1 SUCCESS\n")

print("=== PART 2: TREATMENT VALIDATION ===")
os.environ["CONTROL_PERCENTAGE"] = "0"
get_settings.cache_clear()

# Setup prior success to trigger Persuadable rules in the DGP/Model
r_setup = make_signed_request(get_captured_payload("evt_treat_setup", "pay_treat_setup", "treat@example.com"))

r2 = make_signed_request(get_failed_payload("evt_treat_1", "pay_treat_1", "treat@example.com", 50000))
print("Webhook Response:", r2.status_code, r2.json())

with SessionLocal() as db:
    decision = db.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_treat_1").first()
    print("Variant:", decision.experiment_variant)
    print("Action:", decision.selected_action.value)
    print("Predicted P0:", decision.predicted_p0)
    print("Predicted P1:", decision.predicted_p1)
    print("Predicted Uplift:", decision.predicted_uplift)
    print("Expected Inc Net Paise:", decision.expected_incremental_net_paise)
    print("Decision Status:", decision.decision_status.value)
    print("Execution Ref ID:", decision.execution_reference_id)
    
    assert decision.experiment_variant == "treatment"
    assert decision.predicted_p0 is not None
    assert decision.decision_status.value == "EXECUTED"
    assert decision.execution_reference_id is not None
    
    ref_id = decision.execution_reference_id

print("Simulating payment_link.paid...")
r3 = make_signed_request(get_plink_paid_payload("evt_plink_1", "plink_1", ref_id))
print("Webhook Response:", r3.status_code, r3.json())

with SessionLocal() as db:
    decision = db.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_treat_1").first()
    print("Decision Status After Paid:", decision.decision_status.value)
    print("Outcome Observed At:", decision.outcome_observed_at)
    assert decision.decision_status.value == "OUTCOME_OBSERVED"
    assert decision.outcome_observed_at is not None
print("PART 2 SUCCESS\n")

print("=== PART 3: COOLDOWN VALIDATION ===")
# Same email as Part 2 ("treat@example.com")
r4 = make_signed_request(get_failed_payload("evt_treat_2", "pay_treat_2", "treat@example.com", 5000000))
print("Webhook Response:", r4.status_code, r4.json())

with SessionLocal() as db:
    decision = db.query(RecoveryDecision).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_treat_2").first()
    print("Variant:", decision.experiment_variant)
    print("Action (should be downgraded):", decision.selected_action.value)
    print("Decision Status:", decision.decision_status.value)
    
    assert decision.selected_action.value == "NO_ACTION"
    assert decision.decision_status.value == "DECIDED"
    
    logs = db.query(AuditLog).join(PaymentRecord).filter(PaymentRecord.razorpay_payment_id == "pay_treat_2").all()
    print("Audit Logs:", [(l.event_type, l.action) for l in logs])
print("PART 3 SUCCESS\n")

# Cleanup
os.remove("roe_validation.db")
