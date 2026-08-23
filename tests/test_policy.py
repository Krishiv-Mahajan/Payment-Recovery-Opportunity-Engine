"""
Tests for the placeholder policy predictor.

Covers:
  I. Placeholder policy returns expected values
"""

from __future__ import annotations

from app.ml.features import PaymentFeatures
from app.ml.predictor import PlaceholderPredictor, PolicyPrediction
from app.models import DecisionStatus, RecoveryAction


class TestPlaceholderPredictor:
    """Test I: Placeholder policy interface."""

    def _make_features(self, payment_id: str = "pay_test") -> PaymentFeatures:
        return PaymentFeatures(
            payment_id=payment_id,
            amount_paise=50000,
            amount_inr=500.0,
            currency="INR",
            method="card",
            error_code="BAD_REQUEST_ERROR",
            error_reason="payment_failed",
            error_source="customer",
            error_step="payment_authentication",
            prior_failure_count=0,
            prior_success_count=0,
            customer_identifier="test@example.com",
        )

    def test_returns_policy_prediction(self):
        """predict() must return a PolicyPrediction instance."""
        predictor = PlaceholderPredictor()
        features = self._make_features()
        result = predictor.predict(features)
        assert isinstance(result, PolicyPrediction)

    def test_decision_status_is_pending_policy(self):
        """Phase 1 placeholder must always return PENDING_POLICY."""
        predictor = PlaceholderPredictor()
        features = self._make_features()
        result = predictor.predict(features)
        assert result.decision_status == DecisionStatus.PENDING_POLICY

    def test_selected_action_is_no_action(self):
        """Phase 1 placeholder must always select NO_ACTION."""
        predictor = PlaceholderPredictor()
        features = self._make_features()
        result = predictor.predict(features)
        assert result.selected_action == RecoveryAction.NO_ACTION

    def test_model_version_is_placeholder(self):
        """Model version must identify this as the placeholder."""
        predictor = PlaceholderPredictor()
        features = self._make_features()
        result = predictor.predict(features)
        assert result.model_version == "placeholder-v0"

    def test_reasoning_is_non_empty(self):
        """Reasoning must be present and non-empty for audit trail."""
        predictor = PlaceholderPredictor()
        features = self._make_features()
        result = predictor.predict(features)
        assert result.reasoning
        assert len(result.reasoning) > 10

    def test_prediction_is_deterministic(self):
        """Identical inputs must produce identical outputs."""
        predictor = PlaceholderPredictor()
        f1 = self._make_features("pay_001")
        f2 = self._make_features("pay_002")
        r1 = predictor.predict(f1)
        r2 = predictor.predict(f2)
        assert r1.decision_status == r2.decision_status
        assert r1.selected_action == r2.selected_action
        assert r1.model_version == r2.model_version

    def test_works_with_none_feature_fields(self):
        """Predictor must handle features with None optional fields without error."""
        predictor = PlaceholderPredictor()
        features = PaymentFeatures(
            payment_id="pay_sparse",
            amount_paise=10000,
            amount_inr=100.0,
            currency="INR",
            method=None,
            error_code=None,
            error_reason=None,
            error_source=None,
            error_step=None,
            prior_failure_count=None,
            prior_success_count=None,
            customer_identifier=None,
        )
        # Must not raise
        result = predictor.predict(features)
        assert result.decision_status == DecisionStatus.PENDING_POLICY


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_ok(self, client):
        """GET /health must return status=ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data

    def test_health_environment_is_test(self, client):
        """In test mode, environment must be 'test'."""
        response = client.get("/health")
        data = response.json()
        assert data["environment"] == "test"
