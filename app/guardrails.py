"""
Guardrails Layer for operational safety.

This module enforces safety constraints before an action is dispatched to the executor.
It does NOT contain economic policy logic (which lives in EconomicPolicyPredictor).

Enforced rules:
  Rule 1: Model Unavailable Fallback
  Rule 2: Duplicate Execution Prevention
  Rule 3: Customer Outreach Cooldown (operational frequency limit)

None of these rules inspect predicted_p0, predicted_p1, predicted_uplift,
or expected_incremental_net_paise. Economic policy belongs exclusively to
EconomicPolicyPredictor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.crud import get_recent_outreach_for_customer
from app.models import DecisionStatus, PaymentRecord, RecoveryAction, RecoveryDecision
from app.ml.predictor import PolicyPrediction

logger = logging.getLogger(__name__)


class GuardrailsEngine:
    """
    Enforces operational safety constraints before allowing an action to be executed.

    Args:
        cooldown_hours: Hours to block re-outreach to the same customer identifier.
                        Default 48. Configurable via Settings.
    """

    def __init__(self, cooldown_hours: int = 48) -> None:
        self._cooldown_hours = cooldown_hours

    def evaluate(
        self,
        db: Session,
        payment_record: PaymentRecord,
        prediction: PolicyPrediction | None,
    ) -> PolicyPrediction:
        """
        Evaluate operational safety constraints against a proposed action.

        Returns either the original prediction (if all rules pass) or a modified
        prediction overriding the action to NO_ACTION with a safety reasoning.

        Does NOT inspect any economic/probability fields on prediction.
        """
        # Rule 1: Model Unavailable Fallback
        if prediction is None:
            return PolicyPrediction(
                decision_status=DecisionStatus.DECIDED,
                selected_action=RecoveryAction.NO_ACTION,
                model_version="fallback-v0",
                reasoning="Safety override: Policy predictor was unavailable or returned None.",
            )

        # Doing nothing is always safe — skip remaining rules
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

        # Rule 3: Customer Outreach Cooldown
        # Purely operational: "Is this customer eligible to receive another outreach?"
        # Does not inspect any economic/probability values.
        customer_identifier = (
            payment_record.customer_email or payment_record.customer_contact
        )

        if customer_identifier is None:
            # Fail open: cannot determine identity → allow outreach, log warning
            logger.warning(
                "Cooldown check skipped: no customer_identifier available for "
                "payment_record_id=%s. Outreach allowed.",
                payment_record.id,
            )
            # Caller (event_processor) will audit this case
        else:
            # Normalize email to lowercase for consistent lookups
            normalized = (
                customer_identifier.lower()
                if payment_record.customer_email
                else customer_identifier
            )
            cooldown_since = datetime.now(timezone.utc) - timedelta(hours=self._cooldown_hours)
            recent = get_recent_outreach_for_customer(db, normalized, since=cooldown_since)

            if recent:
                prediction.selected_action = RecoveryAction.NO_ACTION
                prediction.reasoning = (
                    f"Safety override: Cooldown active. Customer was last contacted at "
                    f"{recent.outreach_at.isoformat()} "
                    f"(cooldown window: {self._cooldown_hours}h)."
                )
                logger.info(
                    "Guardrail COOLDOWN: blocked outreach for customer (last contacted %s)",
                    recent.outreach_at.isoformat(),
                )
                return prediction

        return prediction
