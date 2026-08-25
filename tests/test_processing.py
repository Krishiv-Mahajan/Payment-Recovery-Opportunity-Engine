"""
Tests for webhook event processing.

Covers:
  C. Duplicate webhook handling
  D. payment.failed normalization
  E. payment.captured processing
  F. order.paid processing
  G. Database persistence
  J. Invalid/malformed payload
  K. Out-of-order event handling
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy.orm import Session

from app.models import (
    PaymentRecord,
    PaymentStatus,
    RecoveryDecision,
    WebhookEvent,
    WebhookProcessingStatus,
    AuditLog,
    DecisionStatus,
    RecoveryAction,
)
from app.crud import append_audit_log, create_webhook_event, upsert_payment_record_from_captured
from app.schemas import RazorpayPaymentEntity
from tests.conftest import (
    PAYMENT_FAILED_PAYLOAD,
    PAYMENT_CAPTURED_PAYLOAD,
    ORDER_PAID_PAYLOAD,
    make_signed_request,
)


class TestDuplicateWebhookHandling:
    """Test C: Duplicate webhook deliveries are handled safely."""

    def test_duplicate_event_returns_200(self, client):
        """Razorpay expects 200 even for duplicate events."""
        # First delivery
        r1, _ = make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["status"] == "ok"

        # Second delivery (same event_id)
        r2, _ = make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["status"] == "duplicate"

    def test_duplicate_does_not_create_duplicate_records(self, client, db_engine):
        """Duplicate webhook must not produce duplicate PaymentRecord or RecoveryDecision."""
        from sqlalchemy.orm import sessionmaker

        # First delivery
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
        # Second delivery
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            # Exactly one WebhookEvent row
            webhook_count = (
                session.query(WebhookEvent)
                .filter(WebhookEvent.event_id == PAYMENT_FAILED_PAYLOAD["id"])
                .count()
            )
            assert webhook_count == 1, "Duplicate event should produce exactly one WebhookEvent row"

            # Exactly one PaymentRecord
            payment_count = (
                session.query(PaymentRecord)
                .filter(
                    PaymentRecord.razorpay_payment_id
                    == PAYMENT_FAILED_PAYLOAD["payload"]["payment"]["entity"]["id"]
                )
                .count()
            )
            assert payment_count == 1, "Duplicate event should produce exactly one PaymentRecord"

            # Exactly one RecoveryDecision
            record = (
                session.query(PaymentRecord)
                .filter(
                    PaymentRecord.razorpay_payment_id
                    == PAYMENT_FAILED_PAYLOAD["payload"]["payment"]["entity"]["id"]
                )
                .first()
            )
            decision_count = (
                session.query(RecoveryDecision)
                .filter(RecoveryDecision.payment_record_id == record.id)
                .count()
            )
            assert decision_count == 1, "Duplicate event should produce exactly one RecoveryDecision"


class TestPaymentFailedProcessing:
    """Test D: payment.failed event normalization and persistence."""

    def test_payment_failed_creates_payment_record(self, client, db_engine):
        """payment.failed must create a PaymentRecord with status=failed."""
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000001")
                .first()
            )
            assert record is not None, "PaymentRecord must be created for payment.failed"
            assert record.status == PaymentStatus.FAILED
            assert record.amount == 50000
            assert record.currency == "INR"
            assert record.method == "card"
            assert record.customer_email == "test.customer@example.com"
            assert record.customer_contact == "+919999999999"
            assert record.razorpay_order_id == "order_test00000001"

    def test_payment_failed_captures_error_fields(self, client, db_engine):
        """payment.failed must preserve all error context fields."""
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000001")
                .first()
            )
            assert record.error_code == "BAD_REQUEST_ERROR"
            assert record.error_source == "customer"
            assert record.error_step == "payment_authentication"
            assert record.error_reason == "payment_failed"
            assert record.error_description == "Payment failed during authentication"

    def test_payment_failed_creates_recovery_decision(self, client, db_engine):
        """payment.failed must create a PENDING_POLICY RecoveryDecision."""
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000001")
                .first()
            )
            decision = (
                session.query(RecoveryDecision)
                .filter(RecoveryDecision.payment_record_id == record.id)
                .first()
            )
            assert decision is not None, "RecoveryDecision must be created for payment.failed"
            assert decision.decision_status in [DecisionStatus.DECIDED, DecisionStatus.EXECUTED]
            assert decision.selected_action in [RecoveryAction.NO_ACTION, RecoveryAction.SEND_PAYMENT_LINK]
            assert decision.model_version != "placeholder-v0"

    def test_payment_failed_creates_audit_logs(self, client, db_engine):
        """payment.failed processing must produce audit log entries."""
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000001")
                .first()
            )
            logs = (
                session.query(AuditLog)
                .filter(AuditLog.payment_record_id == record.id)
                .all()
            )
            assert len(logs) > 0, "Audit logs must be created"
            event_types = {log.event_type for log in logs}
            assert "PAYMENT_FAILED" in event_types
            assert "FEATURE_EXTRACTION" in event_types
            assert "POLICY_DECISION" in event_types

    def test_payment_failed_stores_raw_payload(self, client, db_engine):
        """Raw payload must be stored in WebhookEvent.raw_payload."""
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            event = (
                session.query(WebhookEvent)
                .filter(WebhookEvent.event_id == "evt_test000000001")
                .first()
            )
            assert event is not None
            assert event.raw_payload is not None
            # Verify raw payload is valid JSON and contains the payment ID
            parsed = json.loads(event.raw_payload)
            assert parsed["payload"]["payment"]["entity"]["id"] == "pay_test000000001"

    def test_payment_failed_webhook_status_is_processed(self, client, db_engine):
        """WebhookEvent.processing_status must be PROCESSED after successful handling."""
        make_signed_request(client, PAYMENT_FAILED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            event = (
                session.query(WebhookEvent)
                .filter(WebhookEvent.event_id == "evt_test000000001")
                .first()
            )
            assert event.processing_status == WebhookProcessingStatus.PROCESSED


class TestPaymentCapturedProcessing:
    """Test E: payment.captured event processing."""

    def test_payment_captured_creates_record_with_captured_status(self, client, db_engine):
        """payment.captured for a new payment must create a PaymentRecord with status=captured."""
        make_signed_request(client, PAYMENT_CAPTURED_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000002")
                .first()
            )
            assert record is not None
            assert record.status == PaymentStatus.CAPTURED
            assert record.amount == 75000
            assert record.method == "upi"

    def test_payment_captured_clears_error_fields(self, client, db_engine):
        """
        Out-of-order test: payment.failed arrives first, then payment.captured.
        Captured must update status and clear error fields.
        """
        # Build a failed payload for the same payment ID that will be captured
        failed_payload = {
            **PAYMENT_FAILED_PAYLOAD,
            "id": "evt_ooo_failed",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        **PAYMENT_FAILED_PAYLOAD["payload"]["payment"]["entity"],
                        "id": "pay_test_ooo",  # Same payment ID
                        "status": "failed",
                    }
                }
            },
        }
        captured_payload = {
            **PAYMENT_CAPTURED_PAYLOAD,
            "id": "evt_ooo_captured",
            "event": "payment.captured",
            "payload": {
                "payment": {
                    "entity": {
                        **PAYMENT_CAPTURED_PAYLOAD["payload"]["payment"]["entity"],
                        "id": "pay_test_ooo",  # Same payment ID
                        "status": "captured",
                        "captured": True,
                    }
                }
            },
        }

        # Failed arrives first
        r1, _ = make_signed_request(client, failed_payload)
        assert r1.status_code == 200

        # Captured arrives second (could be out of order in production)
        r2, _ = make_signed_request(client, captured_payload)
        assert r2.status_code == 200

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test_ooo")
                .first()
            )
            # Status must have transitioned forward
            assert record.status == PaymentStatus.CAPTURED
            # Error fields must be cleared
            assert record.error_code is None
            assert record.error_reason is None


class TestOrderPaidProcessing:
    """Test F: order.paid event processing."""

    def test_order_paid_creates_captured_record(self, client, db_engine):
        """order.paid must create/update a PaymentRecord with status=captured."""
        make_signed_request(client, ORDER_PAID_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000003")
                .first()
            )
            assert record is not None
            assert record.status == PaymentStatus.CAPTURED
            assert record.razorpay_order_id == "order_test00000003"

    def test_order_paid_updates_existing_failed_record(self, client, db_engine):
        """
        If a payment.failed record exists for the same payment,
        order.paid must update it to captured.
        """
        # First, create a failed record for the same payment ID
        failed_payload = {
            **PAYMENT_FAILED_PAYLOAD,
            "id": "evt_orderpaid_failed",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        **PAYMENT_FAILED_PAYLOAD["payload"]["payment"]["entity"],
                        "id": "pay_test000000003",  # Same ID as ORDER_PAID_PAYLOAD
                        "status": "failed",
                    }
                }
            },
        }
        make_signed_request(client, failed_payload)
        # Then order.paid arrives
        make_signed_request(client, ORDER_PAID_PAYLOAD)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            records = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_test000000003")
                .all()
            )
            # Must be exactly one record, not two
            assert len(records) == 1, "Must not create duplicate PaymentRecord"
            assert records[0].status == PaymentStatus.CAPTURED


class TestOutOfOrderEventHandling:
    """Test K: Out-of-order event handling."""

    def test_captured_before_failed_does_not_regress(self, client, db_engine):
        """
        If payment.captured arrives BEFORE payment.failed (network reorder),
        the subsequent payment.failed must NOT downgrade the status to 'failed'.
        """
        captured_first_payload = {
            **PAYMENT_CAPTURED_PAYLOAD,
            "id": "evt_ooo_cap_first",
            "payload": {
                "payment": {
                    "entity": {
                        **PAYMENT_CAPTURED_PAYLOAD["payload"]["payment"]["entity"],
                        "id": "pay_ooo_cap_first",
                        "status": "captured",
                    }
                }
            },
        }
        failed_late_payload = {
            **PAYMENT_FAILED_PAYLOAD,
            "id": "evt_ooo_fail_late",
            "payload": {
                "payment": {
                    "entity": {
                        **PAYMENT_FAILED_PAYLOAD["payload"]["payment"]["entity"],
                        "id": "pay_ooo_cap_first",
                        "status": "failed",
                    }
                }
            },
        }

        # Captured arrives first
        make_signed_request(client, captured_first_payload)
        # Failed arrives late (stale event, out of order)
        make_signed_request(client, failed_late_payload)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            record = (
                session.query(PaymentRecord)
                .filter(PaymentRecord.razorpay_payment_id == "pay_ooo_cap_first")
                .first()
            )
            # Must remain captured — do not regress
            assert record.status == PaymentStatus.CAPTURED, (
                "A captured payment must never regress to failed due to late-arriving events"
            )

    def test_captured_event_does_not_regress_refunded_record(self, db_engine):
        """A late captured event must not overwrite a later REFUNDED state."""
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            webhook_event, _ = create_webhook_event(
                session,
                event_id="evt_refunded_guard",
                event_type="payment.captured",
                raw_payload="{}",
                signature_verified=True,
            )
            session.commit()

            record = PaymentRecord(
                webhook_event_id=webhook_event.id,
                razorpay_payment_id="pay_refunded_guard",
                amount=50000,
                currency="INR",
                status=PaymentStatus.REFUNDED,
                customer_email="refund@example.com",
            )
            session.add(record)
            session.commit()

            payment = RazorpayPaymentEntity.model_validate(
                {
                    "id": "pay_refunded_guard",
                    "amount": 50000,
                    "currency": "INR",
                    "status": "captured",
                }
            )
            updated, created = upsert_payment_record_from_captured(
                session,
                webhook_event_id=webhook_event.id,
                payment=payment,
            )
            session.commit()

            assert created is False
            assert updated.status == PaymentStatus.REFUNDED


class TestFailureDurability:
    """Processing failures must remain auditable after their work is rolled back."""

    def test_processing_failure_persists_failed_status_and_audit(
        self, client, db_engine, monkeypatch
    ):
        import app.event_processor as event_processor
        from sqlalchemy.orm import sessionmaker

        def raise_processing_error(*args, **kwargs):
            raise RuntimeError("forced processing failure")

        monkeypatch.setattr(event_processor, "_handle_payment_failed", raise_processing_error)
        response, _ = make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
        assert response.status_code == 500

        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            webhook_event = (
                session.query(WebhookEvent)
                .filter(WebhookEvent.event_id == PAYMENT_FAILED_PAYLOAD["id"])
                .one()
            )
            assert webhook_event.processing_status == WebhookProcessingStatus.FAILED
            error_log = (
                session.query(AuditLog)
                .filter(AuditLog.event_type == "PROCESSING_ERROR")
                .one()
            )
            assert "RuntimeError" in error_log.reason


class TestWebhookIdempotencyPersistence:
    """Database uniqueness, not a pre-insert SELECT, governs idempotency."""

    def test_duplicate_insert_keeps_the_callers_transaction_usable(self, db_engine):
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            existing, _ = create_webhook_event(
                session,
                event_id="evt_duplicate_insert",
                event_type="payment.failed",
                raw_payload="{}",
                signature_verified=True,
            )
            session.commit()

            append_audit_log(
                session,
                event_type="TEST",
                action="OUTER_TRANSACTION_SURVIVES",
            )
            duplicate, is_duplicate = create_webhook_event(
                session,
                event_id="evt_duplicate_insert",
                event_type="payment.failed",
                raw_payload="{}",
                signature_verified=True,
            )
            session.commit()

            assert is_duplicate is True
            assert duplicate.id == existing.id
            assert session.query(AuditLog).filter(AuditLog.action == "OUTER_TRANSACTION_SURVIVES").count() == 1


class TestMalformedPayload:
    """Test J: Invalid/malformed payload handling."""

    def test_payload_missing_event_id_returns_400(self, client):
        """Payload missing the root 'id' field must return 400."""
        import hashlib, hmac as hmac_mod
        from tests.conftest import TEST_WEBHOOK_SECRET

        # Build a payload without the required 'id' field
        bad_payload = {
            "entity": "event",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {},
            # 'id' is missing
        }
        body = json.dumps(bad_payload, separators=(",", ":")).encode()
        sig = hmac_mod.new(
            key=TEST_WEBHOOK_SECRET.encode(),
            msg=body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig,
            },
        )
        assert response.status_code == 400

    def test_zero_payment_amount_returns_400(self, client):
        """Payment amounts must be positive before the event is processed."""
        zero_amount_payload = {
            **PAYMENT_FAILED_PAYLOAD,
            "id": "evt_zero_amount",
            "payload": {
                "payment": {
                    "entity": {
                        **PAYMENT_FAILED_PAYLOAD["payload"]["payment"]["entity"],
                        "amount": 0,
                    }
                }
            },
        }
        response, _ = make_signed_request(client, zero_amount_payload)
        assert response.status_code == 400

    def test_rejected_webhooks_without_event_ids_get_unique_audit_rows(self, client, db_engine):
        """Distinct malformed rejections must not collide on a fallback event ID."""
        raw_body = b"not-json"
        assert client.post("/webhooks/razorpay", content=raw_body).status_code == 400
        assert client.post("/webhooks/razorpay", content=raw_body).status_code == 400

        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=db_engine)
        with Session() as session:
            rejected = (
                session.query(WebhookEvent)
                .filter(WebhookEvent.signature_verified.is_(False))
                .all()
            )
            assert len(rejected) == 2
            assert len({event.event_id for event in rejected}) == 2

    def test_unsupported_event_type_returns_200(self, client):
        """
        An unsupported but structurally valid event type must return 200.
        Razorpay should not retry events that we intentionally don't handle.
        """
        import hashlib, hmac as hmac_mod
        from tests.conftest import TEST_WEBHOOK_SECRET

        unsupported_payload = {
            "entity": "event",
            "event": "subscription.charged",  # Not handled in Phase 1
            "contains": ["subscription"],
            "payload": {},
            "id": "evt_unsupported_001",
            "created_at": 1700000999,
        }
        body = json.dumps(unsupported_payload, separators=(",", ":")).encode()
        sig = hmac_mod.new(
            key=TEST_WEBHOOK_SECRET.encode(),
            msg=body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig,
            },
        )
        assert response.status_code == 200

class TestPaymentLinkPaidProcessing:
    """Test payment_link.paid event processing (Phase 5)."""

    def test_payment_link_paid_updates_outcome(self, client, db_engine):
        """payment_link.paid maps to RecoveryDecision and sets OUTCOME_OBSERVED."""
        # 1. Manually setup a decision with execution_reference_id
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)

        with Session() as session:
            record = PaymentRecord(
                razorpay_payment_id="pay_test000000004",
                amount=1000,
                currency="INR",
                status="failed"
            )
            session.add(record)
            session.commit()

            decision = RecoveryDecision(
                payment_record_id=record.id,
                decision_status=DecisionStatus.EXECUTED,
                selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                model_version="test",
                execution_reference_id="exec_rd_999"
            )
            session.add(decision)
            session.commit()

            ref_id = "exec_rd_999"

        # 2. Build payment_link.paid payload
        plink_payload = {
            "entity": "event",
            "account_id": "acc_test000000001",
            "event": "payment_link.paid",
            "contains": ["payment_link"],
            "payload": {
                "payment_link": {
                    "entity": {
                        "id": "plink_test000000001",
                        "entity": "payment_link",
                        "status": "paid",
                        "notes": {
                            "execution_reference_id": ref_id
                        }
                    }
                }
            },
            "created_at": 1700001000,
            "id": "evt_test000000004",
        }

        # 3. Process payment_link.paid
        r2, _ = make_signed_request(client, plink_payload)
        assert r2.status_code == 200

        # 4. Verify outcome observed
        with Session() as session:
            decision = session.query(RecoveryDecision).first()
            assert decision.decision_status == DecisionStatus.OUTCOME_OBSERVED
            assert decision.outcome_observed_at is not None
