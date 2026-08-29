"""
Action Execution Layer — PaymentLinkProvider abstraction.

This module defines the PaymentLinkProvider Protocol and the
MockPaymentLinkProvider used by default (EXECUTOR_MODE=mock).

The real Razorpay implementation lives in app/executor_razorpay.py.
The event_processor depends only on the PaymentLinkProvider Protocol —
it never imports RazorpayPaymentLinkProvider directly.

Executor mode is controlled by EXECUTOR_MODE environment variable:
  mock    (default) — no network calls, safe for tests and local dev
  razorpay          — real Razorpay API (requires credentials)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from app.models import RecoveryDecision

logger = logging.getLogger(__name__)

class PaymentLinkCreationError(Exception):
    """Base exception for all executor failures."""

class TransientExecutionError(PaymentLinkCreationError):
    """Raised for retryable failures (e.g. 5xx, timeout, 429, network)."""
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type

class PermanentExecutionError(PaymentLinkCreationError):
    """Raised for non-retryable failures (e.g. 400, 401, 403, 422, malformed response)."""
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type

class DuplicateReferenceExecutionError(PaymentLinkCreationError):
    """Raised when the executor indicates the reference ID has already been attempted."""
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type

class AmbiguousStateError(Exception):
    """Raised when an external action's side-effect cannot be definitively proven."""
    def __init__(self, message: str, error_type: str = "AMBIGUOUS_STATE"):
        super().__init__(message)
        self.error_type = error_type

class ReconciliationError(PaymentLinkCreationError):
    """Raised when duplicate reference reconciliation fails (e.g., missing, mismatched, or multiple links)."""
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type

class PaymentLinkProvider(Protocol):
    """
    Protocol for creating payment links.

    Implementations must be safe to swap at dependency injection time.
    The event_processor depends on this protocol, not on any concrete class.
    """

    def create_payment_link(
        self,
        decision: RecoveryDecision,
        execution_reference_id: str,
    ) -> str:
        """
        Create a payment link for the given recovery decision.

        Args:
            decision: The approved RecoveryDecision to execute.
            execution_reference_id: Stable internal token used to:
                1. Embed in payment link notes for payment_link.paid correlation.
                2. Generate deterministic mock link IDs for safe retries.
                This is NOT a Razorpay idempotency header (unsupported by this endpoint).

        Returns:
            The external reference ID (e.g., Razorpay plink_... ID or mock ID).

        Raises:
            Exception if execution fails.
        """
        ...

    def reconcile_duplicate_reference(
        self,
        decision: RecoveryDecision,
        execution_reference_id: str,
    ) -> str:
        """
        Reconcile a duplicate reference error by finding the existing link.
        
        Args:
            decision: The decision that encountered the duplicate error.
            execution_reference_id: The execution reference ID.
            
        Returns:
            The external reference ID (e.g., Razorpay plink_...).
            
        Raises:
            TransientExecutionError on retryable lookup failures.
            PermanentExecutionError on non-retryable lookup failures.
            ReconciliationError if the link cannot be safely reconciled.
        """
        ...


class MockPaymentLinkProvider:
    """
    Mock implementation of PaymentLinkProvider.

    Default for EXECUTOR_MODE=mock (the permanent default).
    Simulates payment link creation without making any network calls.

    The returned mock ID is deterministic: same execution_reference_id
    always produces the same mock plink ID, making retries safe without
    Razorpay's API involvement.
    """

    def create_payment_link(
        self,
        decision: RecoveryDecision,
        execution_reference_id: str,
    ) -> str:
        # Deterministic mock ID derived from execution_reference_id
        key_hash = hashlib.sha1(execution_reference_id.encode()).hexdigest()[:8]
        mock_plink_id = f"plink_mock_{key_hash}"

        # In a real provider, execution_reference_id would be passed in
        # the 'notes' field of the Razorpay API request so it comes back
        # unchanged in the payment_link.paid webhook.

        logger.info(
            "Mock execution: created payment link %s for decision %s "
            "(execution_reference_id=%s)",
            mock_plink_id,
            decision.id,
            execution_reference_id,
        )
        return mock_plink_id

    def reconcile_duplicate_reference(
        self,
        decision: RecoveryDecision,
        execution_reference_id: str,
    ) -> str:
        # For mock, we just generate the same deterministic ID.
        key_hash = hashlib.sha1(execution_reference_id.encode()).hexdigest()[:8]
        mock_plink_id = f"plink_mock_{key_hash}"
        return mock_plink_id
