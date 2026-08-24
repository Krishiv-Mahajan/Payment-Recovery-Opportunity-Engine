"""
Economic Policy Predictor — Phase 3 implementation of the PolicyPredictor interface.

This class:
  1. Receives a PaymentFeatures vector.
  2. Calls SLearner.predict_probabilities() → (p0, p1).
  3. Computes expected incremental net recovery in integer paise.
  4. Selects the action that maximizes expected incremental net recovery.
  5. Returns a PolicyPrediction using the canonical Phase 1 interface.

The SLearner knows nothing about RecoveryAction or PolicyPrediction.
All economic logic lives here, not in the statistical model.

Guardrails (e.g., quiet hours, recency limits) will be added in future phases
as a separate layer. They are NOT mixed into this class.
"""
from __future__ import annotations

from app.ml.features import PaymentFeatures
from app.ml.models.s_learner import SLearner, MODEL_VERSION as SLEARNER_MODEL_VERSION
from app.ml.predictor import PolicyPrediction
from app.models import DecisionStatus, RecoveryAction
from app.simulator.schemas import INTERVENTION_COSTS_PAISE


class EconomicPolicyPredictor:
    """
    Production-ready policy predictor for Phase 3.

    Implements the same .predict(PaymentFeatures) -> PolicyPrediction interface
    as PlaceholderPredictor, making it a drop-in replacement.

    Economic decision rule:
        expected_incremental_gross_paise = amount_paise × (P1 - P0)
        expected_incremental_net_paise   = gross - intervention_cost

        if expected_incremental_net_paise > 0:
            select SEND_PAYMENT_LINK
        else:
            select NO_ACTION
    """

    def __init__(self, s_learner: SLearner) -> None:
        if not s_learner.is_fitted:
            raise ValueError("SLearner must be fitted before EconomicPolicyPredictor.")
        self._model = s_learner

    def predict(self, features: PaymentFeatures) -> PolicyPrediction:
        """
        Predict the optimal recovery action for a payment context.

        Args:
            features: The canonical PaymentFeatures vector.
                      Must NOT contain GroundTruth or simulator internals.

        Returns:
            PolicyPrediction with the economically optimal action.
        """
        p0, p1 = self._model.predict_probabilities(features)
        predicted_uplift = p1 - p0

        link_cost_paise = INTERVENTION_COSTS_PAISE[RecoveryAction.SEND_PAYMENT_LINK]

        expected_incremental_gross_paise = features.amount_paise * predicted_uplift
        expected_incremental_net_paise = (
            expected_incremental_gross_paise - link_cost_paise
        )

        if expected_incremental_net_paise > 0:
            selected_action = RecoveryAction.SEND_PAYMENT_LINK
        else:
            selected_action = RecoveryAction.NO_ACTION

        reasoning = (
            f"P0={p0:.4f}, P1={p1:.4f}, uplift={predicted_uplift:.4f}, "
            f"amount={features.amount_paise} paise, "
            f"E[inc_gross]={expected_incremental_gross_paise:.1f} paise, "
            f"E[inc_net]={expected_incremental_net_paise:.1f} paise, "
            f"intervention_cost={link_cost_paise} paise"
        )

        return PolicyPrediction(
            decision_status=DecisionStatus.DECIDED,
            selected_action=selected_action,
            model_version=SLEARNER_MODEL_VERSION,
            reasoning=reasoning,
            predicted_p0=p0,
            predicted_p1=p1,
            predicted_uplift=predicted_uplift,
            expected_incremental_net_paise=int(expected_incremental_net_paise),
        )

    @property
    def model_version(self) -> str:
        return SLEARNER_MODEL_VERSION
