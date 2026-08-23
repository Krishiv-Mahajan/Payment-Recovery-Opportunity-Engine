"""
Data structures for the Controlled Recovery Simulator.

These structures are completely isolated from Phase 1 database models to ensure
the simulator does not pollute production pathways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from app.models import RecoveryAction
from app.ml.features import PaymentFeatures


INTERVENTION_COSTS_PAISE: Dict[RecoveryAction, int] = {
    RecoveryAction.NO_ACTION: 0,
    RecoveryAction.SEND_PAYMENT_LINK: 300,  # 3.00 INR = 300 paise
    RecoveryAction.SEND_PAYMENT_LINK_WITH_DISCOUNT: 300,
}


@dataclass
class SimulatedContext:
    """
    The model-visible features. 
    This is equivalent to the data available before feature extraction.
    We provide a helper to convert this to the canonical PaymentFeatures.
    """
    transaction_id: str
    amount_paise: int
    currency: str
    method: str | None
    error_code: str | None
    error_reason: str | None
    error_source: str | None
    error_step: str | None
    prior_failure_count: int | None
    prior_success_count: int | None
    customer_identifier: str | None

    def to_payment_features(self) -> PaymentFeatures:
        """Adapter to the canonical Phase 1 feature interface."""
        return PaymentFeatures(
            payment_id=self.transaction_id,
            amount_paise=self.amount_paise,
            amount_inr=round(self.amount_paise / 100, 2),
            currency=self.currency,
            method=self.method,
            error_code=self.error_code,
            error_reason=self.error_reason,
            error_source=self.error_source,
            error_step=self.error_step,
            prior_failure_count=self.prior_failure_count,
            prior_success_count=self.prior_success_count,
            customer_identifier=self.customer_identifier,
        )


@dataclass
class GroundTruth:
    """
    The simulator-visible hidden truth.
    Contains the true potential outcomes and the dynamically derived quadrant.
    NEVER exposed to the ML context or policy model.
    """
    p_recovery_do_nothing: float
    p_recovery_link: float
    p_recovery_discount: float
    derived_quadrant: str
    
    # Hidden empirical potential outcomes (sampled deterministically per transaction)
    y_do_nothing: bool
    y_link: bool
    y_discount: bool
    
    # Hidden latent traits (for reproducibility and testing)
    latent_responsiveness: float
    latent_annoyance: float


@dataclass
class SimulatedRecord:
    """
    A full logged event from the simulator.
    Contains the context, the hidden truth, the actual treatment applied,
    the intervention cost, and the explicitly observed binary outcome.
    """
    context: SimulatedContext
    ground_truth: GroundTruth
    assigned_action: RecoveryAction
    intervention_cost_paise: int
    observed_outcome: bool

