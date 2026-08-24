#!/usr/bin/env python
"""
Phase 3 Training Script.

Generates a randomized training dataset from the simulator (seed=1000),
trains the S-Learner, saves the model artifact locally, and reports
training diagnostics.

Usage:
    python scripts/train_phase3.py

Output:
    artifacts/model_v1.joblib  (git-ignored, reproducible from seed=1000)
"""
from __future__ import annotations

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib

from app.ml.models.s_learner import SLearner
from app.ml.training import build_training_dataframe, FEATURE_COLS
from app.models import RecoveryAction
from app.simulator.runner import generate_logging_dataset

TRAIN_N = 10_000
TRAIN_SEED = 1000
ARTIFACT_PATH = "artifacts/model_v1.joblib"

POLICY_WEIGHTS = {
    RecoveryAction.NO_ACTION: 0.5,
    RecoveryAction.SEND_PAYMENT_LINK: 0.5,
}


def main() -> None:
    print("=" * 60)
    print("Phase 3 — S-Learner Training")
    print("=" * 60)
    print(f"N={TRAIN_N:,}  seed={TRAIN_SEED}  weights={{'NO_ACTION':0.5, 'LINK':0.5}}")
    print()

    # ── 1. Generate RCT data ────────────────────────────────────────────
    print("Generating training data...")
    records = generate_logging_dataset(TRAIN_N, TRAIN_SEED, POLICY_WEIGHTS)

    df = build_training_dataframe(records)
    print(f"  Training rows:   {len(df):,}")
    print(f"  Feature columns: {len(FEATURE_COLS)}")
    print(f"  Class balance:   outcome=1 {df['outcome'].mean():.3f}, "
          f"outcome=0 {1 - df['outcome'].mean():.3f}")
    print(f"  Action balance:  action=0  {(df['action'] == 0).mean():.3f}, "
          f"action=1  {(df['action'] == 1).mean():.3f}")
    print()

    # ── 2. Train ────────────────────────────────────────────────────────
    print("Training S-Learner (GradientBoostingClassifier)...")
    model = SLearner()
    model.fit(df)
    print("  Training complete.")
    print()

    # ── 3. Uplift variance diagnostic on training set ──────────────────
    print("Computing uplift variance diagnostic on training set...")
    diag = model.predict_uplift_variance_diagnostic(df)

    print(f"  Mean uplift:       {diag['mean_uplift']:+.4f}")
    print(f"  Median uplift:     {diag['median_uplift']:+.4f}")
    print(f"  Std uplift:        {diag['std_uplift']:.4f}")
    print(f"  Min/Max uplift:    [{diag['min_uplift']:+.4f}, {diag['max_uplift']:+.4f}]")
    print(f"  Fraction positive: {diag['fraction_positive']:.3f}")
    print(f"  Fraction negative: {diag['fraction_negative']:.3f}")
    print(f"  Fraction near-zero (|u|<0.02): {diag['fraction_near_zero']:.3f}")
    print()

    if diag["collapsed"]:
        print("  *** CRITICAL WARNING: Uplift variance collapsed (std < 0.02). ***")
        print("  *** The S-Learner is not learning treatment heterogeneity.    ***")
        print()
    else:
        print("  Uplift variance is meaningful — model is not collapsed.")
        print()

    # ── 4. Save artifact ────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"Model saved to: {ARTIFACT_PATH}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
