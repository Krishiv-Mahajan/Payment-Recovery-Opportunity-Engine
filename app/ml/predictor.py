"""
Placeholder policy interface.

This module defines the boundary between the event ingestion pipeline
and the (not-yet-built) ML policy engine.

Phase 1 behavior:
  PolicyPredictor.predict() ALWAYS returns:
    - decision_status = "PENDING_POLICY"
    - selected_action = "NO_ACTION"
    - model_version = "placeholder-v0"

This is NOT an ML prediction. It is a deliberate placeholder that:
  1. Establishes the clean interface contract for future phases.
  2. Ensures the data pipeline (ingestion → features → decision record) works
     end-to-end before any model exists.
  3. Makes the audit trail honest: every failed payment records that
     no intelligent decision was made yet.

Future phases will replace PlaceholderPredictor with a trained S-learner
or uplift model, but the PolicyPrediction return type will remain the same.

IMPORTANT:
  The policy engine must NEVER authorize financial actions directly.
  All money-moving actions go through deterministic guardrails first.
  The LLM (if used for communication) must NOT be given authority to
  bypass guardrails or approve recovery actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.features import PaymentFeatures
from app.models import DecisionStatus, RecoveryAction


@dataclass
class PolicyPrediction:
    """
    The output of the policy engine for one payment event.

    In Phase 1, this is always a placeholder.
    In future phases, this will contain:
      - P(recovery | context, action) estimates per candidate action
      - Expected incremental net recovery values
      - The selected action with highest expected value
      - Confidence intervals and model metadata
    """

    decision_status: DecisionStatus
    selected_action: RecoveryAction
    model_version: str
    reasoning: str  # Human-readable explanation — for audit trail


class PlaceholderPredictor:
    """
    Phase 1 placeholder policy predictor.

    Always returns PENDING_POLICY / NO_ACTION.
    Records the fact that no real policy decision was made.

    This class satisfies the PolicyPredictor interface contract.
    When the real model is built, it will implement the same .predict() method.
    """

    MODEL_VERSION = "placeholder-v0"

    def predict(self, features: PaymentFeatures) -> PolicyPrediction:
        """
        Return a placeholder policy decision.

        Args:
            features: The extracted payment feature vector.
                      Accepted to establish the interface; not used in Phase 1.

        Returns:
            A PolicyPrediction with status=PENDING_POLICY and action=NO_ACTION.
        """
        return PolicyPrediction(
            decision_status=DecisionStatus.PENDING_POLICY,
            selected_action=RecoveryAction.NO_ACTION,
            model_version=self.MODEL_VERSION,
            reasoning=(
                "Phase 1 placeholder: no ML model or policy engine has been trained. "
                "This event has been ingested and features extracted. "
                "A real decision will be produced when the policy engine is implemented."
            ),
        )
