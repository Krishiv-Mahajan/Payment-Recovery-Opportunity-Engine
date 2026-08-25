"""
Prometheus operational metrics for the Recovery Opportunity Engine.

All metric labels are LOW CARDINALITY. High-cardinality identifiers
(payment_id, customer_identifier, execution_reference_id) are NEVER used
as label values.

This module must NOT expose any sensitive payment or customer data.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ── Webhook ingestion ──────────────────────────────────────────────────────────
webhooks_received_total = Counter(
    "roe_webhook_received_total",
    "Total Razorpay webhooks received, by event type",
    labelnames=["event_type"],
)

# ── Policy decisions ───────────────────────────────────────────────────────────
decisions_total = Counter(
    "roe_decision_total",
    "Recovery decisions made, labelled by selected action and variant",
    labelnames=["action", "variant"],
)

# ── Executions ─────────────────────────────────────────────────────────────────
executions_total = Counter(
    "roe_execution_total",
    "Payment link execution attempts, labelled by outcome status",
    labelnames=["status"],  # "success" | "failure"
)

# ── Closed-loop outcomes ───────────────────────────────────────────────────────
outcomes_observed_total = Counter(
    "roe_outcome_observed_total",
    "Closed-loop payment outcomes observed, labelled by variant",
    labelnames=["variant"],
)

# ── Guardrail overrides ────────────────────────────────────────────────────────
guardrail_overrides_total = Counter(
    "roe_guardrail_override_total",
    "Guardrail rule overrides applied, labelled by rule name",
    labelnames=["rule"],  # "duplicate" | "cooldown" | "model_fallback"
)

# ── Model prediction latency ───────────────────────────────────────────────────
model_prediction_duration_seconds = Histogram(
    "roe_model_prediction_duration_seconds",
    "Latency of the EconomicPolicyPredictor.predict() call",
)
