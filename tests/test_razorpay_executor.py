"""
Tests for RazorpayPaymentLinkProvider.
"""
from __future__ import annotations

import httpx
import pytest

from app.executor_razorpay import RazorpayPaymentLinkProvider
from app.executor import PaymentLinkCreationError, TransientExecutionError, PermanentExecutionError
from app.models import PaymentRecord, RecoveryDecision


class MockTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)


def test_razorpay_provider_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.razorpay.com/v1/payment_links/"
        assert request.method == "POST"
        assert "Basic " in request.headers["authorization"]

        body = request.read().decode()
        import json
        data = json.loads(body)

        assert data["amount"] == 1000
        assert data["currency"] == "INR"
        assert data["notes"]["execution_reference_id"] == "exec_rd_42"
        assert data["customer"]["email"] == "a@example.com"

        return httpx.Response(200, json={"id": "plink_success123"})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR", customer_email="a@example.com")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    plink_id = provider.create_payment_link(decision, "exec_rd_42")
    assert plink_id == "plink_success123"


def test_razorpay_provider_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "Bad request"})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(PermanentExecutionError, match="HTTP 400"):
        provider.create_payment_link(decision, "exec_rd_42")

@pytest.mark.parametrize("status,expected_error", [
    (400, PermanentExecutionError),
    (401, PermanentExecutionError),
    (403, PermanentExecutionError),
    (422, PermanentExecutionError),
    (429, TransientExecutionError),
    (500, TransientExecutionError),
    (502, TransientExecutionError),
    (503, TransientExecutionError),
    (504, TransientExecutionError),
])
def test_razorpay_provider_http_classification(status, expected_error):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "test"})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(expected_error) as excinfo:
        provider.create_payment_link(decision, "exec_rd_42")
        
    assert f"HTTP_{status}" in excinfo.value.error_type


def test_razorpay_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(TransientExecutionError, match="timed out"):
        provider.create_payment_link(decision, "exec_rd_42")

def test_razorpay_provider_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError("network failure")

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(TransientExecutionError, match="network failure"):
        provider.create_payment_link(decision, "exec_rd_42")


def test_razorpay_provider_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        # Missing 'id' field in JSON
        return httpx.Response(200, json={"status": "created"})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(PermanentExecutionError, match="missing 'id' field"):
        provider.create_payment_link(decision, "exec_rd_42")

def test_razorpay_provider_duplicate_reference_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={
            "error": {
                "code": "BAD_REQUEST_ERROR",
                "description": "reference_id already exists",
                "field": "reference_id"
            }
        })

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    from app.executor import DuplicateReferenceExecutionError
    with pytest.raises(DuplicateReferenceExecutionError) as excinfo:
        provider.create_payment_link(decision, "exec_rd_42")
    assert excinfo.value.error_type == "DUPLICATE_REFERENCE_ID"

def test_reconcile_duplicate_reference_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert "reference_id=roe_rd_42" in str(request.url)
        return httpx.Response(200, json={
            "items": [
                {
                    "id": "plink_matched_123",
                    "reference_id": "roe_rd_42",
                    "amount": 1000,
                    "currency": "INR"
                }
            ]
        })

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    plink_id = provider.reconcile_duplicate_reference(decision, "exec_rd_42")
    assert plink_id == "plink_matched_123"

def test_reconcile_duplicate_reference_no_match():
    from app.executor import AmbiguousStateError
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(AmbiguousStateError, match="No payment link found"):
        provider.reconcile_duplicate_reference(decision, "exec_rd_42")

def test_reconcile_duplicate_reference_mismatch():
    from app.executor import ReconciliationError
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "items": [
                {
                    "id": "plink_matched_123",
                    "reference_id": "roe_rd_42",
                    "amount": 500, # mismatch!
                    "currency": "INR"
                }
            ]
        })

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(ReconciliationError, match="Mismatched amount"):
        provider.reconcile_duplicate_reference(decision, "exec_rd_42")

def test_reconcile_duplicate_reference_multiple():
    from app.executor import ReconciliationError
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "items": [
                {"id": "plink_1", "reference_id": "roe_rd_42", "amount": 1000, "currency": "INR"},
                {"id": "plink_2", "reference_id": "roe_rd_42", "amount": 1000, "currency": "INR"},
            ]
        })

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(ReconciliationError, match="Multiple payment links found"):
        provider.reconcile_duplicate_reference(decision, "exec_rd_42")

def test_reconcile_duplicate_reference_transient_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": "gateway"})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(TransientExecutionError, match="HTTP 502"):
        provider.reconcile_duplicate_reference(decision, "exec_rd_42")

def test_reconcile_duplicate_reference_permanent_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(PermanentExecutionError, match="HTTP 401"):
        provider.reconcile_duplicate_reference(decision, "exec_rd_42")
