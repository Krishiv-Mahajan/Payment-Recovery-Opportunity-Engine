"""
Tests for feature extraction.

Covers:
  H. Feature extraction from payment context and history
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.database import Base
from app.models import PaymentRecord, PaymentStatus
from app.ml.features import extract_features, PaymentFeatures


@pytest.fixture
def feature_db():
    """
    In-memory database for feature extraction tests.
    Provides a session pre-populated with test payment records.
    """
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    LocalSession = sessionmaker(bind=engine)
    session = LocalSession()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def _make_payment(
    razorpay_id: str,
    status: PaymentStatus,
    email: str = "test@example.com",
    contact: str = "+919999999999",
    amount: int = 50000,
    currency: str = "INR",
    method: str = "card",
    error_code: str | None = None,
    error_reason: str | None = None,
    error_source: str | None = None,
    error_step: str | None = None,
) -> PaymentRecord:
    """Helper to build a PaymentRecord without hitting the DB."""
    return PaymentRecord(
        razorpay_payment_id=razorpay_id,
        amount=amount,
        currency=currency,
        status=status,
        method=method,
        customer_email=email,
        customer_contact=contact,
        error_code=error_code,
        error_reason=error_reason,
        error_source=error_source,
        error_step=error_step,
    )


class TestFeatureExtraction:
    """Test H: Feature extraction."""

    def test_basic_fields_extracted(self, feature_db):
        """All payment dimensions must be extracted correctly."""
        record = _make_payment(
            razorpay_id="pay_feat_001",
            status=PaymentStatus.FAILED,
            amount=120000,
            currency="INR",
            method="upi",
        )
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)

        assert features.payment_id == "pay_feat_001"
        assert features.amount_paise == 120000
        assert features.amount_inr == 1200.0
        assert features.currency == "INR"
        assert features.method == "upi"

    def test_error_fields_extracted(self, feature_db):
        """Error context fields must be extracted from a failed payment."""
        record = _make_payment(
            razorpay_id="pay_feat_002",
            status=PaymentStatus.FAILED,
            error_code="BAD_REQUEST_ERROR",
            error_reason="payment_failed",
            error_source="customer",
            error_step="payment_authentication",
        )
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)

        assert features.error_code == "BAD_REQUEST_ERROR"
        assert features.error_reason == "payment_failed"
        assert features.error_source == "customer"
        assert features.error_step == "payment_authentication"

    def test_no_prior_history_returns_zero_counts(self, feature_db):
        """A customer with no prior records must have prior counts of 0."""
        record = _make_payment(
            razorpay_id="pay_feat_003",
            status=PaymentStatus.FAILED,
            email="newcustomer@example.com",
        )
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)

        assert features.prior_failure_count == 0
        assert features.prior_success_count == 0
        assert features.customer_identifier == "newcustomer@example.com"

    def test_prior_failure_count_is_accurate(self, feature_db):
        """Prior failure count must count historical failures for same customer."""
        # Two prior failures for the same email
        prior1 = _make_payment("pay_prior_001", PaymentStatus.FAILED, email="repeat@example.com")
        prior2 = _make_payment("pay_prior_002", PaymentStatus.FAILED, email="repeat@example.com")
        feature_db.add_all([prior1, prior2])
        feature_db.commit()

        # The payment we're extracting features FOR
        current = _make_payment("pay_current_001", PaymentStatus.FAILED, email="repeat@example.com")
        feature_db.add(current)
        feature_db.commit()
        feature_db.refresh(current)

        features = extract_features(current, feature_db)

        assert features.prior_failure_count == 2
        assert features.prior_success_count == 0

    def test_prior_success_count_is_accurate(self, feature_db):
        """Prior success count must count captured payments for the same customer."""
        prior_success = _make_payment("pay_succ_001", PaymentStatus.CAPTURED, email="loyal@example.com")
        feature_db.add(prior_success)
        feature_db.commit()

        current = _make_payment("pay_current_002", PaymentStatus.FAILED, email="loyal@example.com")
        feature_db.add(current)
        feature_db.commit()
        feature_db.refresh(current)

        features = extract_features(current, feature_db)

        assert features.prior_success_count == 1
        assert features.prior_failure_count == 0

    def test_no_customer_identifier_returns_none_counts(self, feature_db):
        """A payment with no email or contact must have None prior counts."""
        record = PaymentRecord(
            razorpay_payment_id="pay_anon_001",
            amount=10000,
            currency="INR",
            status=PaymentStatus.FAILED,
            method="card",
            customer_email=None,
            customer_contact=None,
        )
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)

        assert features.customer_identifier is None
        assert features.prior_failure_count is None
        assert features.prior_success_count is None

    def test_does_not_count_current_payment_in_history(self, feature_db):
        """The current payment must not count itself in the prior failure count."""
        record = _make_payment("pay_self_001", PaymentStatus.FAILED, email="solo@example.com")
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)

        # Only one record exists — it's the current one — so prior_failure_count must be 0
        assert features.prior_failure_count == 0

    def test_history_uses_email_without_matching_an_unrelated_contact(self, feature_db):
        """Email history must not also query the contact column for that email value."""
        matching_email = _make_payment(
            "pay_email_match", PaymentStatus.FAILED, email="alice@example.com"
        )
        unrelated_contact = _make_payment(
            "pay_contact_collision",
            PaymentStatus.FAILED,
            email="other@example.com",
            contact="alice@example.com",
        )
        current = _make_payment(
            "pay_email_current", PaymentStatus.FAILED, email="alice@example.com"
        )
        feature_db.add_all([matching_email, unrelated_contact, current])
        feature_db.commit()
        feature_db.refresh(current)

        features = extract_features(current, feature_db)

        assert features.customer_identifier == "alice@example.com"
        assert features.prior_failure_count == 1

    def test_to_dict_is_json_serializable(self, feature_db):
        """PaymentFeatures.to_dict() must return a JSON-serializable dict."""
        import json

        record = _make_payment("pay_json_001", PaymentStatus.FAILED)
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)
        d = features.to_dict()

        # Must not raise
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["payment_id"] == "pay_json_001"
        assert parsed["feature_schema_version"] == "v1"

    def test_feature_schema_version_is_set(self, feature_db):
        """Feature schema version must be set to a non-empty string."""
        record = _make_payment("pay_ver_001", PaymentStatus.FAILED)
        feature_db.add(record)
        feature_db.commit()
        feature_db.refresh(record)

        features = extract_features(record, feature_db)
        assert features.feature_schema_version == "v1"
