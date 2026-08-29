import json
import hashlib
import hmac
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(".env")

SECRET = os.environ["RAZORPAY_WEBHOOK_SECRET"]

def send(payload):
    body = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(
        SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    response = requests.post(
        "http://127.0.0.1:8000/webhooks/razorpay",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": signature,
        },
        timeout=30,
    )

    print("HTTP:", response.status_code)
    print("Response:", response.text)
    return response

def payment_captured(event_id, payment_id):
    return {
        "entity": "event",
        "event": "payment.captured",
        "id": event_id,
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "entity": "payment",
                    "amount": 50000,
                    "currency": "INR",
                    "status": "captured",
                    "method": "card",
                    "email": "live-test@example.com",
                    "contact": "+919876543210",
                    "created_at": int(time.time()),
                }
            }
        },
        "created_at": int(time.time()),
    }

def payment_failed(event_id, payment_id):
    return {
        "entity": "event",
        "event": "payment.failed",
        "id": event_id,
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "entity": "payment",
                    "amount": 50000,
                    "currency": "INR",
                    "status": "failed",
                    "method": "card",
                    "email": "live-test@example.com",
                    "contact": "+919876543210",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Payment cancelled",
                    "error_source": "customer",
                    "error_step": "payment_authentication",
                    "error_reason": "customer_cancelled",
                    "created_at": int(time.time()),
                }
            }
        },
        "created_at": int(time.time()),
    }

print("=== LIVE WEBHOOK TEST ===")

print("\n1. Creating prior successful payment...")
send(payment_captured(
    "evt_live_setup_003",
    "pay_live_setup_003",
))

print("\n2. Sending payment.failed...")
send(payment_failed(
    "evt_live_failed_003",
    "pay_live_failed_003",
))

print("\nDone.")
