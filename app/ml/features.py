"""
Feature extraction for the Recovery Opportunity Engine.

This module constructs a deterministic feature vector from:
  1. The normalized payment record (from the webhook payload).
  2. Historical context from the database (prior attempts for the same customer).

Design principles:
  - Features are ONLY derived from data that is actually available.
  - No synthetic features. No invented behavioral signals.
  - Every feature has a documented source.
  - The feature vector is a plain Python dict — no framework dependencies.
  - The function is pure given its inputs — deterministic and testable.

Phase 1 features:
  - Payment dimensions: amount, currency, method
  - Error context: error_code, error_reason, error_source, error_step
  - Historical context: prior failure count, prior success count
    (only if customer_email or customer_contact is available)

NOT included (requires data not yet available):
  - Time-of-day behavioral patterns (needs more data)
  - Card network features (requires enrichment not in Phase 1)
  - Subscription recurrence signals (requires subscription event integration)
  - Customer lifetime value (requires order history API calls)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import PaymentRecord, PaymentStatus


@dataclass
class PaymentFeatures:
    """
    A structured, typed feature vector for one payment event.

    All fields are present (no Optional). Missing data is represented
    as None within the field value, making the schema predictable.

    This dataclass is the contract between feature extraction and
    the policy engine. Changing it should be treated as a schema migration.
    """

    # ── Payment dimensions ─────────────────────────────────────────────
    payment_id: str
    amount_paise: int  # Raw amount in paise (smallest INR unit)
    amount_inr: float  # Converted to INR for human readability
    currency: str
    method: str | None  # "card", "upi", "netbanking", "wallet", None

    # ── Error context ──────────────────────────────────────────────────
    # These are present only for payment.failed events.
    # For captured/authorized, all error fields will be None.
    error_code: str | None
    error_reason: str | None
    error_source: str | None
    error_step: str | None

    # ── Historical context ─────────────────────────────────────────────
    # Counts of prior payment attempts for the same customer identifier.
    # "customer identifier" = email if available, else contact, else None.
    # If no identifier is available, both counts are None (not 0) to
    # distinguish "unknown history" from "no history".
    prior_failure_count: int | None
    prior_success_count: int | None

    # The identifier used to look up history (for auditability)
    customer_identifier: str | None

    # ── Metadata ────────────────────────────────────────────────────────
    # Version string for this feature schema. Increment when fields change.
    feature_schema_version: str = field(default="v1")

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dict representation.
        Used for storage in AuditLog.metadata_json.
        """
        return {
            "payment_id": self.payment_id,
            "amount_paise": self.amount_paise,
            "amount_inr": self.amount_inr,
            "currency": self.currency,
            "method": self.method,
            "error_code": self.error_code,
            "error_reason": self.error_reason,
            "error_source": self.error_source,
            "error_step": self.error_step,
            "prior_failure_count": self.prior_failure_count,
            "prior_success_count": self.prior_success_count,
            "customer_identifier": self.customer_identifier,
            "feature_schema_version": self.feature_schema_version,
        }


def extract_features(payment: PaymentRecord, db: Session) -> PaymentFeatures:
    """
    Build a PaymentFeatures vector from a PaymentRecord and database history.

    Args:
        payment: The PaymentRecord to extract features for. It must have been
                 flushed so it has a database ID; it need not be committed.
        db: An active SQLAlchemy session for historical lookups.

    Returns:
        A PaymentFeatures dataclass with all available features populated.

    Side effects: None — this function is read-only with respect to the database.
    """
    # ── Customer identifier resolution ─────────────────────────────────
    # Use email as the primary identifier; fall back to contact number.
    # Razorpay attaches both to the payment entity when available.
    customer_identifier: str | None = (
        payment.customer_email or payment.customer_contact
    )

    # ── Historical context lookup ──────────────────────────────────────
    prior_failure_count: int | None = None
    prior_success_count: int | None = None

    if customer_identifier:
        identifier_column = (
            PaymentRecord.customer_email
            if payment.customer_email
            else PaymentRecord.customer_contact
        )
        # Count prior failures for this customer identifier
        # Exclude the current payment (by ID) to avoid counting itself
        failure_query = select(func.count(PaymentRecord.id)).where(
            identifier_column == customer_identifier,
            PaymentRecord.status == PaymentStatus.FAILED,
            PaymentRecord.id != payment.id,
        )
        prior_failure_count = db.execute(failure_query).scalar() or 0

        # Count prior successes (captured payments)
        success_query = select(func.count(PaymentRecord.id)).where(
            identifier_column == customer_identifier,
            PaymentRecord.status == PaymentStatus.CAPTURED,
            PaymentRecord.id != payment.id,
        )
        prior_success_count = db.execute(success_query).scalar() or 0

    # ── Construct and return feature vector ────────────────────────────
    return PaymentFeatures(
        payment_id=payment.razorpay_payment_id,
        amount_paise=payment.amount,
        amount_inr=round(payment.amount / 100, 2),
        currency=payment.currency,
        method=payment.method,
        error_code=payment.error_code,
        error_reason=payment.error_reason,
        error_source=payment.error_source,
        error_step=payment.error_step,
        prior_failure_count=prior_failure_count,
        prior_success_count=prior_success_count,
        customer_identifier=customer_identifier,
    )
