"""
Training dataset builder for Phase 3.

Converts SimulatedRecord lists into a pandas DataFrame suitable for training
the S-Learner. Handles:
  - Feature extraction from SimulatedContext (via PaymentFeatures)
  - Missing-value semantics: None (unknown) vs 0 (known zero)
  - Sentinel imputation (-1) + binary missing indicators
  - Action column appended to X for the S-Learner
  - Outcome column as label Y

Design principles:
  - PaymentFeatures is NEVER mutated.
  - GroundTruth is NEVER accessed here.
  - This module is an implementation detail — pandas does not leak
    into domain interfaces (PaymentFeatures, PolicyPredictor, etc.).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from app.ml.features import PaymentFeatures
from app.models import RecoveryAction
from app.simulator.schemas import SimulatedRecord

# Sentinel value for unknown (None) integer history counts.
# Must be distinguishable from 0 (known zero) and any valid count.
MISSING_SENTINEL: int = -1

# Action encoding: the S-learner treats action as a numeric feature column.
ACTION_ENCODING = {
    RecoveryAction.NO_ACTION: 0,
    RecoveryAction.SEND_PAYMENT_LINK: 1,
    RecoveryAction.SEND_PAYMENT_LINK_WITH_DISCOUNT: 2,
}

# Categorical feature columns (will be one-hot encoded in the pipeline)
CATEGORICAL_COLS = [
    "method",
    "error_code",
    "error_reason",
    "error_source",
    "error_step",
]

# Numeric feature columns passed through as-is
NUMERIC_COLS = [
    "amount_paise",
    "prior_failure_count",
    "prior_success_count",
    "prior_failure_count_missing",
    "prior_success_count_missing",
    "action",
]

# All feature columns (X). Outcome and identifiers are NOT in this list.
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def features_to_row(features: PaymentFeatures, action: RecoveryAction) -> dict:
    """
    Convert a PaymentFeatures + action into a flat dict row for the S-Learner.

    PaymentFeatures is NOT mutated. None is converted to sentinel + indicator
    within this function only.

    Args:
        features: The canonical payment feature vector.
        action: The treatment action (encoded as integer for the model).

    Returns:
        A dict representing one S-Learner input row.
    """
    failure_missing = int(features.prior_failure_count is None)
    success_missing = int(features.prior_success_count is None)

    return {
        "amount_paise": features.amount_paise,
        "method": features.method,
        "error_code": features.error_code,
        "error_reason": features.error_reason,
        "error_source": features.error_source,
        "error_step": features.error_step,
        "prior_failure_count": (
            MISSING_SENTINEL
            if features.prior_failure_count is None
            else features.prior_failure_count
        ),
        "prior_success_count": (
            MISSING_SENTINEL
            if features.prior_success_count is None
            else features.prior_success_count
        ),
        "prior_failure_count_missing": failure_missing,
        "prior_success_count_missing": success_missing,
        "action": ACTION_ENCODING[action],
    }


def build_training_dataframe(records: List[SimulatedRecord]) -> pd.DataFrame:
    """
    Build the S-Learner training DataFrame from a list of SimulatedRecords.

    Each row contains:
      - Feature columns (X + action)
      - Label column: 'outcome' (0 or 1)

    GroundTruth is NEVER accessed here. The only label is observed_outcome.

    Args:
        records: List of SimulatedRecord from generate_logging_dataset().

    Returns:
        pandas DataFrame with FEATURE_COLS + ['outcome'] columns.
    """
    rows = []
    for record in records:
        # Access only model-visible context via the canonical adapter
        features = record.context.to_payment_features()
        row = features_to_row(features, record.assigned_action)
        row["outcome"] = int(record.observed_outcome)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
