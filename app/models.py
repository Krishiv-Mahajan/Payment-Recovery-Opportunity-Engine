"""
SQLAlchemy ORM models — the database schema for Phase 1.

Design decisions:
  - WebhookEvent stores every inbound webhook with its raw payload for auditability.
  - PaymentRecord normalizes the payment entity from the webhook payload.
  - RecoveryDecision holds a placeholder decision record (PENDING_POLICY) in Phase 1.
  - AuditLog records every processing step for observability.

Idempotency strategy:
  - WebhookEvent.event_id has a UNIQUE constraint.
    Attempting to insert a duplicate event_id raises IntegrityError → we treat
    this as a duplicate delivery and skip reprocessing.
  - PaymentRecord.razorpay_payment_id has a UNIQUE constraint.
    Updates to captured/authorized state use UPSERT semantics rather than
    blind inserts to avoid duplicating records on out-of-order delivery.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ── Enumerations ──────────────────────────────────────────────────────────────


class WebhookProcessingStatus(str, enum.Enum):
    """Processing lifecycle for a received webhook event."""

    RECEIVED = "RECEIVED"  # Stored, not yet processed
    PROCESSING = "PROCESSING"  # Processing in flight
    PROCESSED = "PROCESSED"  # Fully handled
    DUPLICATE = "DUPLICATE"  # Received more than once; skipped
    FAILED = "FAILED"  # Unrecoverable processing error
    SIGNATURE_INVALID = "SIGNATURE_INVALID"  # Rejected before any processing


class PaymentStatus(str, enum.Enum):
    """Payment status as reported by Razorpay."""

    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"
    FAILED = "failed"


class DecisionStatus(str, enum.Enum):
    """
    Decision lifecycle for a RecoveryDecision record.

    Phase 1 always produces PENDING_POLICY.
    Future phases will transition to DECIDED → EXECUTED → OUTCOME_OBSERVED.
    """

    PENDING_POLICY = "PENDING_POLICY"  # Awaiting ML model / policy engine
    DECIDED = "DECIDED"  # Policy has selected an action
    EXECUTED = "EXECUTED"  # Action has been dispatched to Razorpay
    OUTCOME_OBSERVED = "OUTCOME_OBSERVED"  # Post-action outcome received


class RecoveryAction(str, enum.Enum):
    """
    Recovery actions the policy engine may eventually select.

    Phase 1 only ever uses NO_ACTION (the placeholder).
    Additional actions will be added in later phases after the
    policy engine and action executor are built.
    """

    NO_ACTION = "NO_ACTION"
    SEND_PAYMENT_LINK = "SEND_PAYMENT_LINK"  # Future phase
    SEND_PAYMENT_LINK_WITH_DISCOUNT = "SEND_PAYMENT_LINK_WITH_DISCOUNT"  # Future phase


# ── ORM Models ────────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WebhookEvent(Base):
    """
    Stores every inbound Razorpay webhook delivery.

    The raw_payload column retains the exact bytes received so that
    we can re-inspect, replay, or re-process events without data loss.
    Signature verification result is stored per delivery attempt.
    """

    __tablename__ = "webhook_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_webhook_events_event_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Razorpay's own event identifier from the payload ("id" field in root object)
    # This is the idempotency key — duplicate deliveries share the same event_id.
    event_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    event_type: Mapped[str] = mapped_column(String(128), nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )

    signature_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # The raw JSON string from Razorpay — stored for auditability and replay
    raw_payload: Mapped[str] = mapped_column(Text, nullable=False)

    processing_status: Mapped[str] = mapped_column(
        Enum(WebhookProcessingStatus, name="webhook_processing_status"),
        nullable=False,
        default=WebhookProcessingStatus.RECEIVED,
    )

    # Relationships
    payment_records: Mapped[list["PaymentRecord"]] = relationship(
        "PaymentRecord", back_populates="webhook_event"
    )

    def __repr__(self) -> str:
        return (
            f"<WebhookEvent id={self.id} event_id={self.event_id!r} "
            f"type={self.event_type!r} status={self.processing_status!r}>"
        )


class PaymentRecord(Base):
    """
    Normalized payment entity extracted from Razorpay webhook payloads.

    One PaymentRecord per razorpay_payment_id. May be updated by subsequent
    events (e.g., a payment.failed record updated when payment.captured arrives
    for the same payment ID — unusual but possible in Razorpay's flow).

    Out-of-order safety:
      Updates only move status forward in the lifecycle
      (failed → captured is allowed; captured → failed is ignored).
    """

    __tablename__ = "payment_records"
    __table_args__ = (
        UniqueConstraint(
            "razorpay_payment_id", name="uq_payment_records_razorpay_payment_id"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to the webhook event that first created this record
    webhook_event_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("webhook_events.id"), nullable=True
    )

    # Razorpay identifiers
    razorpay_payment_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    razorpay_order_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )

    # Payment dimensions
    amount: Mapped[int] = mapped_column(Integer, nullable=False)  # paise
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(PaymentStatus, name="payment_status"), nullable=False
    )
    method: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Error context — present only when status = failed
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Customer identifiers — used for velocity checks in later phases
    # These are contact details Razorpay attaches to the payment entity
    customer_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    customer_contact: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Timestamps from Razorpay (Unix epoch in payload) stored as UTC datetimes
    payment_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Row management timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )

    # Relationships
    webhook_event: Mapped["WebhookEvent | None"] = relationship(
        "WebhookEvent", back_populates="payment_records"
    )
    recovery_decisions: Mapped[list["RecoveryDecision"]] = relationship(
        "RecoveryDecision", back_populates="payment_record"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="payment_record"
    )

    def __repr__(self) -> str:
        return (
            f"<PaymentRecord id={self.id} "
            f"razorpay_payment_id={self.razorpay_payment_id!r} "
            f"status={self.status!r} amount={self.amount}>"
        )


class RecoveryDecision(Base):
    """
    Records the recovery decision made for a failed payment.

    Phase 1: Always decision_status=PENDING_POLICY, selected_action=NO_ACTION.
    This table is designed to receive real policy decisions in later phases.

    One decision record is created per payment.failed event.
    Decision history is preserved — new decisions are added, old ones are not
    overwritten. The most recent decision by created_at is the active one.
    """

    __tablename__ = "recovery_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    payment_record_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("payment_records.id"), nullable=False
    )

    decision_status: Mapped[str] = mapped_column(
        Enum(DecisionStatus, name="decision_status"),
        nullable=False,
        default=DecisionStatus.PENDING_POLICY,
    )

    selected_action: Mapped[str] = mapped_column(
        Enum(RecoveryAction, name="recovery_action"),
        nullable=False,
        default=RecoveryAction.NO_ACTION,
    )

    # Identifies which model/policy version produced this decision.
    # "placeholder-v0" for Phase 1.
    model_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="placeholder-v0"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )

    # Relationship
    payment_record: Mapped["PaymentRecord"] = relationship(
        "PaymentRecord", back_populates="recovery_decisions"
    )

    def __repr__(self) -> str:
        return (
            f"<RecoveryDecision id={self.id} "
            f"payment_record_id={self.payment_record_id} "
            f"status={self.decision_status!r} action={self.selected_action!r}>"
        )


class AuditLog(Base):
    """
    Immutable audit trail.

    Every significant processing step appends a row here.
    Rows are never updated or deleted — only appended.

    This allows reconstruction of exactly what happened to any event,
    including failed attempts, retries, and decision rationale.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    payment_record_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("payment_records.id"), nullable=True
    )

    # The type of event being audited (e.g., "WEBHOOK_RECEIVED", "SIGNATURE_VERIFIED")
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)

    # The action taken or attempted (e.g., "CREATE_PAYMENT_RECORD", "SKIP_DUPLICATE")
    action: Mapped[str] = mapped_column(String(128), nullable=False)

    # Human-readable reason or context — not a secret
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # JSON-serialized metadata for structured context (feature vector, policy inputs, etc.)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )

    # Relationship
    payment_record: Mapped["PaymentRecord | None"] = relationship(
        "PaymentRecord", back_populates="audit_logs"
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} "
            f"event_type={self.event_type!r} "
            f"action={self.action!r}>"
        )
