"""
Tests for Phase 3: Training dataset construction.

Verifies:
  - Feature extraction produces correct columns.
  - None vs 0 semantics are preserved (sentinel + indicator).
  - PaymentFeatures is never mutated.
  - GroundTruth fields are never present in the training DataFrame.
  - Action encoding is correct.
  - DataFrame is correctly structured for S-Learner input.
"""
from __future__ import annotations

import pytest

from app.ml.features import PaymentFeatures
from app.ml.training import (
    FEATURE_COLS,
    MISSING_SENTINEL,
    ACTION_ENCODING,
    build_training_dataframe,
    features_to_row,
)
from app.models import RecoveryAction
from app.simulator.runner import generate_logging_dataset


def _make_features(
    prior_failure_count=None,
    prior_success_count=None,
    amount_paise=50000,
) -> PaymentFeatures:
    return PaymentFeatures(
        payment_id="test_pay_001",
        amount_paise=amount_paise,
        amount_inr=amount_paise / 100,
        currency="INR",
        method="card",
        error_code="BAD_REQUEST_ERROR",
        error_reason="insufficient_funds",
        error_source="customer",
        error_step="payment_authorization",
        prior_failure_count=prior_failure_count,
        prior_success_count=prior_success_count,
        customer_identifier="test@example.com",
    )


class TestFeaturesRow:
    def test_none_maps_to_sentinel_and_indicator(self):
        features = _make_features(prior_failure_count=None, prior_success_count=None)
        row = features_to_row(features, RecoveryAction.NO_ACTION)

        assert row["prior_failure_count"] == MISSING_SENTINEL
        assert row["prior_success_count"] == MISSING_SENTINEL
        assert row["prior_failure_count_missing"] == 1
        assert row["prior_success_count_missing"] == 1

    def test_zero_maps_to_zero_not_sentinel(self):
        """0 (known zero) must NOT be confused with None (unknown)."""
        features = _make_features(prior_failure_count=0, prior_success_count=0)
        row = features_to_row(features, RecoveryAction.NO_ACTION)

        assert row["prior_failure_count"] == 0
        assert row["prior_success_count"] == 0
        assert row["prior_failure_count_missing"] == 0
        assert row["prior_success_count_missing"] == 0

    def test_nonzero_count_preserved(self):
        features = _make_features(prior_failure_count=3, prior_success_count=7)
        row = features_to_row(features, RecoveryAction.NO_ACTION)

        assert row["prior_failure_count"] == 3
        assert row["prior_success_count"] == 7
        assert row["prior_failure_count_missing"] == 0
        assert row["prior_success_count_missing"] == 0

    def test_action_encoding_no_action(self):
        features = _make_features()
        row = features_to_row(features, RecoveryAction.NO_ACTION)
        assert row["action"] == ACTION_ENCODING[RecoveryAction.NO_ACTION]

    def test_action_encoding_send_link(self):
        features = _make_features()
        row = features_to_row(features, RecoveryAction.SEND_PAYMENT_LINK)
        assert row["action"] == ACTION_ENCODING[RecoveryAction.SEND_PAYMENT_LINK]

    def test_amount_paise_is_canonical(self):
        """amount_paise (integer) is the canonical representation, not amount_inr."""
        features = _make_features(amount_paise=150000)
        row = features_to_row(features, RecoveryAction.NO_ACTION)
        assert row["amount_paise"] == 150000
        assert "amount_inr" not in row

    def test_payment_features_not_mutated(self):
        """features_to_row must not modify the input PaymentFeatures."""
        features = _make_features(prior_failure_count=None)
        original_failure_count = features.prior_failure_count
        _ = features_to_row(features, RecoveryAction.NO_ACTION)
        assert features.prior_failure_count == original_failure_count  # still None


class TestBuildTrainingDataframe:
    def test_has_all_feature_cols_and_outcome(self):
        records = generate_logging_dataset(
            50,
            seed=1000,
            policy_weights={
                RecoveryAction.NO_ACTION: 0.5,
                RecoveryAction.SEND_PAYMENT_LINK: 0.5,
            },
        )
        df = build_training_dataframe(records)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing column: {col}"
        assert "outcome" in df.columns

    def test_no_ground_truth_leakage(self):
        """Training DataFrame must contain NO ground truth fields."""
        records = generate_logging_dataset(
            20,
            seed=1000,
            policy_weights={
                RecoveryAction.NO_ACTION: 0.5,
                RecoveryAction.SEND_PAYMENT_LINK: 0.5,
            },
        )
        df = build_training_dataframe(records)

        # These field names must NEVER appear in the training data
        forbidden = [
            "p_recovery_do_nothing",
            "p_recovery_link",
            "p_recovery_discount",
            "y_do_nothing",
            "y_link",
            "y_discount",
            "derived_quadrant",
            "latent_responsiveness",
            "latent_annoyance",
            "ground_truth",
        ]
        for field in forbidden:
            assert field not in df.columns, f"Ground truth leakage: {field} in training df"

    def test_outcome_is_binary(self):
        records = generate_logging_dataset(
            50,
            seed=1000,
            policy_weights={
                RecoveryAction.NO_ACTION: 0.5,
                RecoveryAction.SEND_PAYMENT_LINK: 0.5,
            },
        )
        df = build_training_dataframe(records)
        assert df["outcome"].isin([0, 1]).all()

    def test_row_count_matches_records(self):
        records = generate_logging_dataset(
            100,
            seed=1000,
            policy_weights={
                RecoveryAction.NO_ACTION: 0.5,
                RecoveryAction.SEND_PAYMENT_LINK: 0.5,
            },
        )
        df = build_training_dataframe(records)
        assert len(df) == 100
