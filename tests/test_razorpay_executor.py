"""
Tests for RazorpayPaymentLinkProvider.
"""
from __future__ import annotations

import httpx
import pytest

from app.executor_razorpay import RazorpayPaymentLinkProvider, PaymentLinkCreationError
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

    with pytest.raises(PaymentLinkCreationError, match="HTTP 400"):
        provider.create_payment_link(decision, "exec_rd_42")


def test_razorpay_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    client = httpx.Client(transport=MockTransport(handler))
    provider = RazorpayPaymentLinkProvider("key", "secret", http_client=client)

    record = PaymentRecord(id=1, amount=1000, currency="INR")
    decision = RecoveryDecision(id=42)
    decision.payment_record = record

    with pytest.raises(PaymentLinkCreationError, match="timed out"):
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

    with pytest.raises(PaymentLinkCreationError, match="missing 'id' field"):
        provider.create_payment_link(decision, "exec_rd_42")
