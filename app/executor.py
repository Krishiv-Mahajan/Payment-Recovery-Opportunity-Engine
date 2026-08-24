"""
Action Execution Layer.

This module is responsible for dispatching approved recovery actions to external services
(e.g., Razorpay APIs for Payment Links).

Phase 4 introduces the abstractions but uses MockPaymentLinkProvider to avoid external calls.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.models import RecoveryDecision

logger = logging.getLogger(__name__)


class PaymentLinkProvider(Protocol):
    """Protocol for creating payment links."""

    def create_payment_link(self, decision: RecoveryDecision, idempotency_key: str) -> str:
        """
        Create a payment link for the given recovery decision.

        Args:
            decision: The approved RecoveryDecision to execute.
            idempotency_key: Stable idempotency key for external API.

        Returns:
            The external reference ID (e.g., payment link ID).
            
        Raises:
            Exception if execution fails.
        """
        ...


class MockPaymentLinkProvider:
    """
    Mock implementation of PaymentLinkProvider for Phase 4.
    Simulates API execution without making real HTTP calls.
    """

    def create_payment_link(self, decision: RecoveryDecision, idempotency_key: str) -> str:
        # Use idempotency key to generate a stable mock ID
        # If the same idempotency key is passed, we return the same mock ID.
        import hashlib
        key_hash = hashlib.sha1(idempotency_key.encode()).hexdigest()[:8]
        mock_plink_id = f"plink_mock_{key_hash}"
        
        # In a real provider, we would pass `decision.id` or `payment_record.id`
        # in the 'notes' field of the Razorpay request so it comes back to us in the webhook.
        
        logger.info(
            "Mock execution: created payment link %s for decision %s using key %s",
            mock_plink_id,
            decision.id,
            idempotency_key,
        )
        return mock_plink_id
