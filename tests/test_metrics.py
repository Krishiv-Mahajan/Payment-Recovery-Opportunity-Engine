"""
Tests for operational metrics.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_endpoint_exposed(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200

    # Check that basic metrics exist in the output
    content = response.text
    assert "roe_webhook_received_total" in content
    assert "roe_decision_total" in content
    assert "roe_execution_total" in content
    assert "roe_outcome_observed_total" in content
    assert "roe_guardrail_override_total" in content
    assert "roe_model_prediction_duration_seconds" in content

    # Ensure high-cardinality data is NEVER in the metrics
    # (By asserting the labels shown in the HELP/TYPE text)
    assert "payment_id" not in content
    assert "customer_identifier" not in content
    assert "execution_reference_id" not in content
