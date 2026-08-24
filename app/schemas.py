"""
Pydantic schemas for webhook payload validation and internal data structures.

These schemas validate the structure of inbound Razorpay webhook payloads.

IMPORTANT:
  Razorpay delivers webhook events as JSON. The signature must be verified
  against the RAW request body bytes BEFORE we parse the JSON.
  These schemas are used AFTER signature verification to normalize the data.

  We do not trust that the Razorpay payload structure is exactly what we expect.
  All fields use Optional where Razorpay documentation indicates possible absence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Razorpay Payment Entity ────────────────────────────────────────────────────


class RazorpayPaymentEntity(BaseModel):
    """
    Normalized representation of a Razorpay payment entity.

    Maps to the payload.payment.entity structure in payment.failed,
    payment.captured, and payment.authorized webhook events.

    Razorpay docs reference:
      https://razorpay.com/docs/api/payments/

    Fields not listed here are silently ignored (model_config extra="ignore").
    """

    model_config = {"extra": "ignore"}

    id: str
    entity: str = "payment"
    amount: int = Field(gt=0, description="Amount in the smallest currency unit")
    currency: str
    status: str
    order_id: str | None = None
    invoice_id: str | None = None
    method: str | None = None
    captured: bool = False
    description: str | None = None

    # Card details — present when method == "card"
    card_id: str | None = None

    # UPI — present when method == "upi"
    vpa: str | None = None

    # Netbanking/wallet — present when method == "netbanking" / "wallet"
    bank: str | None = None
    wallet: str | None = None

    # Customer contact — Razorpay attaches these to the payment object
    email: str | None = None
    contact: str | None = None

    # Error fields — present only when status == "failed"
    error_code: str | None = None
    error_description: str | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None

    # Unix epoch timestamp
    created_at: int | None = None


# ── Razorpay Order Entity ─────────────────────────────────────────────────────


class RazorpayOrderEntity(BaseModel):
    """
    Normalized representation of a Razorpay order entity.

    Maps to the payload.order.entity structure in order.paid webhook events.
    """

    model_config = {"extra": "ignore"}

    id: str
    entity: str = "order"
    amount: int = Field(gt=0, description="Amount in the smallest currency unit")
    amount_paid: int | None = None
    amount_due: int | None = None
    currency: str
    receipt: str | None = None
    status: str


# ── Razorpay Payment Link Entity ──────────────────────────────────────────────


class RazorpayPaymentLinkEntity(BaseModel):
    """
    Normalized representation of a Razorpay payment link entity.

    Maps to the payload.payment_link.entity structure in payment_link.paid webhook events.
    """

    model_config = {"extra": "ignore"}

    id: str
    entity: str = "payment_link"
    amount: int | None = None
    currency: str | None = None
    status: str
    notes: dict[str, str] = Field(default_factory=dict)


# ── Webhook Payload Container ─────────────────────────────────────────────────


class RazorpayEntityPayload(BaseModel):
    """
    Container for a single entity inside a webhook payload.
    Razorpay wraps each entity in {"entity": {...}} under payload.
    """

    model_config = {"extra": "ignore"}

    entity: dict[str, Any]


class RazorpayWebhookPayload(BaseModel):
    """
    Top-level Razorpay webhook payload.

    Structure:
    {
        "entity": "event",
        "account_id": "acc_...",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {"entity": {...}},
            "order":   {"entity": {...}}   # only in order.paid
        },
        "created_at": 1691735748,
        "id": "evt_..."
    }

    The "id" field at the root is Razorpay's own event identifier.
    This is our primary idempotency key.
    """

    model_config = {"extra": "ignore"}

    # Razorpay's unique event ID — our idempotency key
    id: str = Field(..., description="Razorpay event ID — the idempotency key")
    entity: str = "event"
    account_id: str | None = None
    event: str  # e.g., "payment.failed", "payment.captured"
    contains: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: int | None = None  # Unix epoch

    def extract_payment_entity(self) -> RazorpayPaymentEntity | None:
        """
        Extract and validate the payment entity from the nested payload.
        Returns None if the payload does not contain a payment entity.
        """
        payment_data = self.payload.get("payment", {})
        entity_data = payment_data.get("entity")
        if not entity_data:
            return None
        return RazorpayPaymentEntity.model_validate(entity_data)

    def extract_order_entity(self) -> RazorpayOrderEntity | None:
        """
        Extract and validate the order entity from the nested payload.
        Returns None if the payload does not contain an order entity.
        """
        order_data = self.payload.get("order", {})
        entity_data = order_data.get("entity")
        if not entity_data:
            return None
        return RazorpayOrderEntity.model_validate(entity_data)

    def extract_payment_link_entity(self) -> RazorpayPaymentLinkEntity | None:
        """
        Extract and validate the payment link entity from the nested payload.
        Returns None if the payload does not contain a payment link entity.
        """
        plink_data = self.payload.get("payment_link", {})
        entity_data = plink_data.get("entity")
        if not entity_data:
            return None
        return RazorpayPaymentLinkEntity.model_validate(entity_data)


# ── Internal API Response Schemas ─────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    version: str
    environment: str


class WebhookAckResponse(BaseModel):
    """
    Response returned from POST /webhooks/razorpay.

    Always returns HTTP 200 with a status field.
    Razorpay retries on non-2xx responses, so we acknowledge even if
    we detected a duplicate — we just don't reprocess.
    """

    status: str
    event_id: str | None = None
    message: str | None = None


class EventDetailResponse(BaseModel):
    """Response schema for GET /events/{event_id}."""

    event_id: str
    event_type: str
    received_at: str
    processing_status: str
    signature_verified: bool
