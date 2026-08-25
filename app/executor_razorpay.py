"""
RazorpayPaymentLinkProvider — real Razorpay API execution layer.

This module implements the PaymentLinkProvider protocol using the
Razorpay Payment Links REST API.

Key design decisions:
  - HTTP client is injectable for testability (no real network in tests).
  - Uses HTTP Basic Auth (key_id:key_secret) — no separate idempotency header.
  - execution_reference_id is embedded in the 'notes' field of the API request
    so it comes back unchanged in the payment_link.paid webhook for correlation.
  - Raises PaymentLinkCreationError on any API failure (4xx, 5xx, timeout,
    malformed response). Callers must NOT retry inside the webhook path.
  - The Razorpay Payment Links API does NOT support an Idempotency-Key header.
    Duplicate execution prevention is handled upstream by GuardrailsEngine Rule 2.

API Reference (verified 2026-08-25):
  POST https://api.razorpay.com/v1/payment_links/
  Auth: Basic base64(key_id:key_secret)
  Response: { "id": "plink_...", "short_url": "...", "status": "created", ... }
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.models import RecoveryDecision

logger = logging.getLogger(__name__)

RAZORPAY_API_BASE = "https://api.razorpay.com"
PAYMENT_LINKS_ENDPOINT = "/v1/payment_links/"
REQUEST_TIMEOUT_SECONDS = 10.0


class PaymentLinkCreationError(Exception):
    """Raised when the Razorpay API call fails for any reason."""


class RazorpayPaymentLinkProvider:
    """
    Production implementation of PaymentLinkProvider for Phase 5+.

    Creates Razorpay Payment Links via the REST API. Requires
    EXECUTOR_MODE=razorpay to be active (enforced by lifespan).

    Args:
        key_id: Razorpay API key ID (rzp_test_... or rzp_live_...).
        key_secret: Razorpay API key secret.
        http_client: Injectable httpx.Client for testing. Defaults to a
                     real HTTP client when None. Tests MUST inject a mock.
    """

    def __init__(
        self,
        key_id: str,
        key_secret: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._key_id = key_id
        self._key_secret = key_secret
        self._http_client = http_client

    def _get_auth_header(self) -> str:
        """Build HTTP Basic Auth header value."""
        credentials = f"{self._key_id}:{self._key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def _build_request_body(
        self,
        decision: RecoveryDecision,
        execution_reference_id: str,
    ) -> dict[str, Any]:
        """
        Build the Razorpay API request body.

        The notes field carries execution_reference_id so the payment_link.paid
        webhook can correlate back to the RecoveryDecision. This is our internal
        correlation token — not a Razorpay idempotency mechanism.
        """
        payment_record = decision.payment_record
        body: dict[str, Any] = {
            "amount": payment_record.amount,   # paise
            "currency": payment_record.currency,
            "description": "Payment recovery link",
            "reference_id": f"roe_rd_{decision.id}",
            "notes": {
                "execution_reference_id": execution_reference_id,
                "recovery_decision_id": str(decision.id),
            },
        }

        # Include customer details if available for pre-filling the checkout form.
        # Razorpay may or may not auto-fill these for security reasons.
        customer: dict[str, str] = {}
        if payment_record.customer_email:
            customer["email"] = payment_record.customer_email
        if payment_record.customer_contact:
            customer["contact"] = payment_record.customer_contact
        if customer:
            body["customer"] = customer

        return body

    def create_payment_link(
        self,
        decision: RecoveryDecision,
        execution_reference_id: str,
    ) -> str:
        """
        Create a Razorpay Payment Link and return its external ID (plink_...).

        The execution_reference_id is embedded in the notes of the created link.
        This allows payment_link.paid webhooks to be correlated back to the
        originating RecoveryDecision via notes["execution_reference_id"].

        Args:
            decision: The approved RecoveryDecision to execute.
                      Must have a loaded payment_record relationship.
            execution_reference_id: Stable internal token derived from
                      decision.id (e.g., "exec_rd_42"). Embedded in notes —
                      NOT used as a Razorpay idempotency header (unsupported).

        Returns:
            The Razorpay Payment Link ID (e.g., "plink_ABC123XYZ").

        Raises:
            PaymentLinkCreationError: On any API error, timeout, or malformed response.
        """
        url = RAZORPAY_API_BASE + PAYMENT_LINKS_ENDPOINT
        headers = {
            "Authorization": self._get_auth_header(),
            "Content-Type": "application/json",
        }
        body = self._build_request_body(decision, execution_reference_id)

        logger.info(
            "Creating Razorpay payment link for decision_id=%s reference=%s",
            decision.id,
            execution_reference_id,
        )

        try:
            if self._http_client is not None:
                response = self._http_client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            else:
                with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                    response = client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise PaymentLinkCreationError(
                f"Razorpay API request timed out after {REQUEST_TIMEOUT_SECONDS}s"
            ) from exc
        except httpx.RequestError as exc:
            raise PaymentLinkCreationError(
                f"Razorpay API request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise PaymentLinkCreationError(
                f"Razorpay API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise PaymentLinkCreationError(
                f"Razorpay API returned non-JSON response: {response.text[:200]}"
            ) from exc

        plink_id = data.get("id")
        if not plink_id:
            raise PaymentLinkCreationError(
                f"Razorpay API response missing 'id' field: {data}"
            )

        logger.info(
            "Razorpay payment link created: %s for decision_id=%s",
            plink_id,
            decision.id,
        )
        return plink_id
