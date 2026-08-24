"""
Guardrails Layer for operational safety.

This module enforces safety constraints before an action is dispatched to the executor.
It does NOT contain economic policy logic (which lives in EconomicPolicyPredictor).
It enforces:
  1. Duplicate Execution Prevention
  2. Model Unavailable Fallback
  3. Execution Idempotency
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import DecisionStatus, PaymentRecord, RecoveryAction, RecoveryDecision
from app.ml.predictor import PolicyPrediction


class GuardrailsEngine:
    """
    Enforces operational safety constraints before allowing an action to be executed.
    """

    def evaluate(
        self,
        db: Session,
        payment_record: PaymentRecord,
        prediction: PolicyPrediction | None,
    ) -> PolicyPrediction:
        """
        Evaluate operational safety constraints against a proposed action.

        Returns either the original prediction (if safe) or a modified
        prediction overriding the action to NO_ACTION with a safety reasoning.
        """
        # Rule 1: Model Unavailable Fallback
        if prediction is None:
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.NO_ACTION,
                model_version="fallback-v0",
                reasoning="Safety override: Policy predictor was unavailable or returned None.",
            )

        # Doing nothing is always safe
        if prediction.selected_action == RecoveryAction.NO_ACTION:
            return prediction

        # Rule 2: Duplicate Execution Prevention
        # Check if we already decided or executed an action for THIS payment.
        existing_action = (
            db.query(RecoveryDecision)
            .filter(
                RecoveryDecision.payment_record_id == payment_record.id,
                RecoveryDecision.decision_status.in_(
                    [DecisionStatus.DECIDED, DecisionStatus.EXECUTED, DecisionStatus.OUTCOME_OBSERVED]
                ),
                RecoveryDecision.selected_action != RecoveryAction.NO_ACTION,
            )
            .first()
        )
        
        if existing_action:
            prediction.selected_action = RecoveryAction.NO_ACTION
            prediction.reasoning = (
                f"Safety override: Duplicate execution prevented. "
                f"Action {existing_action.selected_action.value} already taken on this payment."
            )
            return prediction

        # Rule 3: Cooldown/Frequency Limits
        # Cooldown enforcement requires customer communication history table 
        # and is deferred to future phase.

        return prediction
