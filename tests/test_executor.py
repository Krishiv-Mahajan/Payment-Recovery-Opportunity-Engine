"""
Tests for Action Executor.
"""
from __future__ import annotations

import pytest
from app.executor import MockPaymentLinkProvider
from app.models import RecoveryDecision

def test_executor_idempotency():
    provider = MockPaymentLinkProvider()
    decision = RecoveryDecision(id=100)
    
    key = "exec_rd_123"
    ref1 = provider.create_payment_link(decision, key)
    ref2 = provider.create_payment_link(decision, key)
    
    assert ref1 == ref2
    assert "plink_mock_" in ref1

def test_executor_abstraction():
    # Verify it can be replaced
    from app.executor import PaymentLinkProvider
    
    class DummyProvider:
        def create_payment_link(self, decision: RecoveryDecision, execution_reference_id: str) -> str:
            return "dummy_id"
            
    # DummyProvider must be compatible with PaymentLinkProvider protocol
    provider: PaymentLinkProvider = DummyProvider()
    assert provider.create_payment_link(RecoveryDecision(id=1), "k") == "dummy_id"
