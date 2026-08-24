"""
S-Learner: Single-model uplift estimator.

Trains one sklearn Pipeline on (X, action) → Y.

At inference time, for a given context X, predicts:
    P0 = P(recovery | X, action=NO_ACTION)
    P1 = P(recovery | X, action=SEND_PAYMENT_LINK)

The uplift is then computed by the calling layer:
    predicted_uplift = P1 - P0

The S-Learner knows nothing about RecoveryAction semantics,
PolicyPredictor, monetary values, or business guardrails.
Those concerns belong to app/ml/policy.py.

Estimator choice: GradientBoostingClassifier
    - Learns nonlinear X × action interactions via sequential residual fitting.
    - No external dependencies beyond scikit-learn.
    - Effective at N=10k–50k without deep hyperparameter tuning.
    - max_depth=4 provides sufficient interaction depth without overfitting.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.ml.training import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MISSING_SENTINEL,
    NUMERIC_COLS,
    ACTION_ENCODING,
    features_to_row,
)
from app.ml.features import PaymentFeatures
from app.models import RecoveryAction

# Model version identifier — increment when architecture changes
MODEL_VERSION = "slearner-gbm-v1"


def _build_pipeline(n_estimators: int = 200) -> Pipeline:
    """
    Constructs the sklearn Pipeline:
      ColumnTransformer → GradientBoostingClassifier

    OHE with handle_unknown='ignore' so unseen categories at inference
    time produce an all-zero row rather than raising an error.

    Args:
        n_estimators: Number of boosting stages. Use a small value (e.g. 20)
                      for fast unit tests; use 200 for production quality.
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", ohe, CATEGORICAL_COLS),
            ("num", "passthrough", [c for c in NUMERIC_COLS if c != "action"]),
            ("action", "passthrough", ["action"]),
        ],
        remainder="drop",
    )

    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


class SLearner:
    """
    S-Learner wrapper around a sklearn Pipeline.

    Training:
        fit(df) — df must contain FEATURE_COLS + ['outcome']

    Inference:
        predict_probabilities(features, action_encoding) → (p0, p1)
        where p0 = P(recovery | X, NO_ACTION)
              p1 = P(recovery | X, SEND_PAYMENT_LINK)
    """

    def __init__(self, n_estimators: int = 200) -> None:
        self._pipeline: Pipeline | None = None
        self.is_fitted: bool = False
        self._n_estimators = n_estimators

    def fit(self, df: pd.DataFrame) -> "SLearner":
        """
        Fit the S-Learner on the training DataFrame.

        Args:
            df: DataFrame with columns = FEATURE_COLS + ['outcome'].
                'outcome' is the binary label (0 or 1).

        Returns:
            self (for chaining)
        """
        X = df[FEATURE_COLS]
        y = df["outcome"].astype(int)

        self._pipeline = _build_pipeline(n_estimators=self._n_estimators)
        self._pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_probabilities(
        self, features: PaymentFeatures
    ) -> Tuple[float, float]:
        """
        Predict P(recovery | X, A=0) and P(recovery | X, A=1) for one context.

        Args:
            features: The canonical PaymentFeatures for the payment.

        Returns:
            (p0, p1) — floats in (0, 1).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted or self._pipeline is None:
            raise RuntimeError("SLearner must be fitted before predict_probabilities.")

        row_0 = features_to_row(features, RecoveryAction.NO_ACTION)
        row_1 = features_to_row(features, RecoveryAction.SEND_PAYMENT_LINK)

        df_both = pd.DataFrame([row_0, row_1])[FEATURE_COLS]
        probs = self._pipeline.predict_proba(df_both)[:, 1]

        p0 = float(probs[0])
        p1 = float(probs[1])
        return p0, p1

    def predict_batch_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict P(recovery = 1) for a batch of feature rows represented in a DataFrame.

        Args:
            df: DataFrame containing preprocessed FEATURE_COLS.

        Returns:
            1D numpy array of probabilities in [0, 1].

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted or self._pipeline is None:
            raise RuntimeError("SLearner must be fitted before predict_batch_probabilities.")

        return self._pipeline.predict_proba(df)[:, 1]

    def predict_uplift_variance_diagnostic(self, df: pd.DataFrame) -> dict:
        """
        Compute uplift statistics over a feature DataFrame (without outcome label).
        Used to detect S-Learner collapse (uplift collapses to near-zero everywhere).

        Args:
            df: DataFrame of FEATURE_COLS rows (action column will be overwritten).

        Returns:
            Dict with uplift variance diagnostics.
        """
        if not self.is_fitted or self._pipeline is None:
            raise RuntimeError("SLearner must be fitted before diagnostics.")

        # Build two copies with action=0 and action=1
        df_0 = df[FEATURE_COLS].copy()
        df_0["action"] = ACTION_ENCODING[RecoveryAction.NO_ACTION]

        df_1 = df[FEATURE_COLS].copy()
        df_1["action"] = ACTION_ENCODING[RecoveryAction.SEND_PAYMENT_LINK]

        p0_arr = self.predict_batch_probabilities(df_0)
        p1_arr = self.predict_batch_probabilities(df_1)
        uplift_arr = p1_arr - p0_arr

        near_zero_threshold = 0.02
        n = len(uplift_arr)

        return {
            "n": n,
            "mean_uplift": float(np.mean(uplift_arr)),
            "median_uplift": float(np.median(uplift_arr)),
            "std_uplift": float(np.std(uplift_arr)),
            "min_uplift": float(np.min(uplift_arr)),
            "max_uplift": float(np.max(uplift_arr)),
            "fraction_positive": float(np.sum(uplift_arr > 0) / n),
            "fraction_negative": float(np.sum(uplift_arr < 0) / n),
            "fraction_near_zero": float(
                np.sum(np.abs(uplift_arr) < near_zero_threshold) / n
            ),
            "collapsed": bool(np.std(uplift_arr) < near_zero_threshold),
        }
