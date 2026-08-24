#!/usr/bin/env python
"""
Phase 3 Evaluation Script.

Evaluates three policies on held-out data (seed=2000):
  1. AlwaysDoNothing baseline
  2. AlwaysSendLink baseline
  3. EconomicPolicyPredictor (trained S-Learner)

Reports:
  - Recovery rate, gross recovery, intervention cost
  - Empirical and analytical incremental net recovery
  - Uplift diagnostics (mean/median/std/variance)
  - Predicted vs true uplift: MAE, rank correlation, sign agreement
  - P0/P1 probability quality: MAE vs ground truth, log-loss, Brier score
  - Action rates by quadrant
  - Baseline comparison

Usage:
    python scripts/evaluate_phase3.py
    python scripts/evaluate_phase3.py --model artifacts/model_v1.joblib
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import joblib
import numpy as np
from scipy.stats import spearmanr

from app.ml.models.s_learner import SLearner
from app.ml.policy import EconomicPolicyPredictor
from app.ml.predictor import PolicyPrediction, PlaceholderPredictor
from app.ml.features import PaymentFeatures
from app.models import RecoveryAction, DecisionStatus
from app.simulator.evaluator import evaluate_metrics
from app.simulator.runner import evaluate_policy, generate_contexts
from app.simulator.engine import generate_ground_truth
from app.simulator.schemas import SimulatedRecord

EVAL_N = 10_000
EVAL_SEED = 2000
DEFAULT_MODEL_PATH = "artifacts/model_v1.joblib"


# ── Baseline predictors ─────────────────────────────────────────────────────

class AlwaysDoNothingPredictor:
    def predict(self, features: PaymentFeatures) -> PolicyPrediction:
        return PolicyPrediction(
            decision_status=DecisionStatus.DECIDED,
            selected_action=RecoveryAction.NO_ACTION,
            model_version="baseline-always-nothing",
            reasoning="Baseline: always do nothing.",
        )


class AlwaysSendLinkPredictor:
    def predict(self, features: PaymentFeatures) -> PolicyPrediction:
        return PolicyPrediction(
            decision_status=DecisionStatus.DECIDED,
            selected_action=RecoveryAction.SEND_PAYMENT_LINK,
            model_version="baseline-always-link",
            reasoning="Baseline: always send payment link.",
        )


# ── Probability diagnostics (evaluation-only, uses GroundTruth) ────────────

def compute_probability_diagnostics(
    records: list[SimulatedRecord],
    s_learner: SLearner,
) -> dict:
    """
    Computes P0/P1 diagnostics against simulator ground truth.
    These are EVALUATION-ONLY. GroundTruth never enters training.
    """
    p0_hats, p1_hats = [], []
    p0_trues, p1_trues = [], []
    predicted_uplifts, true_uplifts = [], []

    # For log-loss / Brier: split by assigned action
    # (use only rows where we observed the arm, to avoid counterfactual labels)
    y_do_nothing_obs, p0_hat_do_nothing = [], []
    y_link_obs, p1_hat_link = [], []

    for record in records:
        features = record.context.to_payment_features()
        p0, p1 = s_learner.predict_probabilities(features)

        gt = record.ground_truth
        p0_hats.append(p0)
        p1_hats.append(p1)
        p0_trues.append(gt.p_recovery_do_nothing)
        p1_trues.append(gt.p_recovery_link)
        predicted_uplifts.append(p1 - p0)
        true_uplifts.append(gt.p_recovery_link - gt.p_recovery_do_nothing)

        # RCT-style arm-specific calibration
        if record.assigned_action == RecoveryAction.NO_ACTION:
            y_do_nothing_obs.append(int(record.observed_outcome))
            p0_hat_do_nothing.append(p0)
        elif record.assigned_action == RecoveryAction.SEND_PAYMENT_LINK:
            y_link_obs.append(int(record.observed_outcome))
            p1_hat_link.append(p1)

    p0_arr = np.array(p0_hats)
    p1_arr = np.array(p1_hats)
    pred_u = np.array(predicted_uplifts)
    true_u = np.array(true_uplifts)

    # Uplift rank correlation
    spear_corr, spear_pval = spearmanr(pred_u, true_u)

    # Sign agreement
    sign_agree = np.mean(np.sign(pred_u) == np.sign(true_u))

    # Probability MAE vs ground truth
    p0_mae = float(np.mean(np.abs(p0_arr - np.array(p0_trues))))
    p1_mae = float(np.mean(np.abs(p1_arr - np.array(p1_trues))))

    # Uplift MAE vs true uplift
    uplift_mae = float(np.mean(np.abs(pred_u - true_u)))

    # RCT arm log-loss and Brier
    def _log_loss(y, p):
        y, p = np.array(y), np.clip(np.array(p), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _brier(y, p):
        y, p = np.array(y), np.array(p)
        return float(np.mean((p - y) ** 2))

    return {
        "p0_mae_vs_true": p0_mae,
        "p1_mae_vs_true": p1_mae,
        "uplift_mae_vs_true": uplift_mae,
        "uplift_spearman_corr": float(spear_corr),
        "uplift_spearman_pval": float(spear_pval),
        "uplift_sign_agreement": float(sign_agree),
        "predicted_uplift_mean": float(np.mean(pred_u)),
        "predicted_uplift_median": float(np.median(pred_u)),
        "predicted_uplift_std": float(np.std(pred_u)),
        "predicted_uplift_min": float(np.min(pred_u)),
        "predicted_uplift_max": float(np.max(pred_u)),
        "fraction_positive_uplift": float(np.mean(pred_u > 0)),
        "fraction_negative_uplift": float(np.mean(pred_u < 0)),
        "fraction_near_zero_uplift": float(np.mean(np.abs(pred_u) < 0.02)),
        # RCT arm calibration
        "n_do_nothing_arm": len(y_do_nothing_obs),
        "log_loss_do_nothing_arm": _log_loss(y_do_nothing_obs, p0_hat_do_nothing) if y_do_nothing_obs else None,
        "brier_do_nothing_arm": _brier(y_do_nothing_obs, p0_hat_do_nothing) if y_do_nothing_obs else None,
        "n_link_arm": len(y_link_obs),
        "log_loss_link_arm": _log_loss(y_link_obs, p1_hat_link) if y_link_obs else None,
        "brier_link_arm": _brier(y_link_obs, p1_hat_link) if y_link_obs else None,
    }


def compute_quadrant_action_rates(records: list[SimulatedRecord]) -> dict:
    """Compute action rates per hidden quadrant (evaluation-only)."""
    quadrant_stats: dict[str, dict] = {}
    for record in records:
        q = record.ground_truth.derived_quadrant
        if q not in quadrant_stats:
            quadrant_stats[q] = {"total": 0, "send_link": 0}
        quadrant_stats[q]["total"] += 1
        if record.assigned_action == RecoveryAction.SEND_PAYMENT_LINK:
            quadrant_stats[q]["send_link"] += 1

    return {
        q: {
            "count": stats["total"],
            "action_rate": stats["send_link"] / stats["total"],
            "send_link_count": stats["send_link"],
        }
        for q, stats in quadrant_stats.items()
    }


def print_section(title: str) -> None:
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def print_policy_report(label: str, records: list[SimulatedRecord]) -> None:
    metrics = evaluate_metrics(records)
    q_rates = compute_quadrant_action_rates(records)
    n = metrics["n_transactions"]
    total_link = sum(
        1 for r in records if r.assigned_action == RecoveryAction.SEND_PAYMENT_LINK
    )

    print(f"  Policy:                   {label}")
    print(f"  N transactions:           {n:,}")
    print(f"  Action rate (LINK):       {total_link/n:.3f} ({total_link:,}/{n:,})")
    print(f"  Recovery rate:            {metrics['recovery_rate']:.4f}")
    print(f"  Empirical gross (INR):    {metrics['empirical_gross_recovery_inr']:,.2f}")
    print(f"  Intervention cost (INR):  {metrics['intervention_cost_inr']:,.2f}")
    print(f"  Emp inc gross (INR):      {metrics['empirical_incremental_gross_inr']:,.2f}")
    print(f"  Emp inc net (INR):        {metrics['empirical_incremental_net_inr']:,.2f}")
    print(f"  Ana inc net (INR):        {metrics['analytical_incremental_net_inr']:,.2f}")
    print(f"  Emp inc net / 1000 (INR): {metrics['empirical_incremental_net_inr_per_1000']:,.2f}")
    print()
    print("  Quadrant action rates:")
    for q in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
        if q in q_rates:
            s = q_rates[q]
            print(f"    {q:<16}: {s['action_rate']:.3f}  (count={s['count']:,})")
        else:
            print(f"    {q:<16}: (not observed)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 — Full Evaluation Report")
    print("=" * 60)
    print(f"Eval N={EVAL_N:,}  seed={EVAL_SEED}")
    print(f"Model path: {args.model}")

    # Load model
    if not os.path.exists(args.model):
        print(f"\nERROR: Model artifact not found at {args.model}")
        print("Run: python scripts/train_phase3.py first.")
        sys.exit(1)

    model: SLearner = joblib.load(args.model)
    economic_predictor = EconomicPolicyPredictor(model)

    # Evaluate all three policies
    print_section("Policy 1: Always DO_NOTHING (Baseline)")
    records_nothing = evaluate_policy(AlwaysDoNothingPredictor(), EVAL_N, EVAL_SEED)
    print_policy_report("Always DO_NOTHING", records_nothing)

    print_section("Policy 2: Always SEND_PAYMENT_LINK (Baseline)")
    records_link = evaluate_policy(AlwaysSendLinkPredictor(), EVAL_N, EVAL_SEED)
    print_policy_report("Always SEND_PAYMENT_LINK", records_link)

    print_section("Policy 3: EconomicPolicyPredictor (S-Learner)")
    records_model = evaluate_policy(economic_predictor, EVAL_N, EVAL_SEED)
    print_policy_report("EconomicPolicyPredictor (S-Learner)", records_model)

    # Probability & uplift diagnostics (evaluation-only — uses GroundTruth)
    print_section("Probability & Uplift Diagnostics (S-Learner, eval set)")
    diag = compute_probability_diagnostics(records_model, model)

    print("  Ground-truth probability calibration:")
    print(f"    MAE(P0_hat, P0_true):   {diag['p0_mae_vs_true']:.4f}")
    print(f"    MAE(P1_hat, P1_true):   {diag['p1_mae_vs_true']:.4f}")
    print(f"    MAE(uplift):            {diag['uplift_mae_vs_true']:.4f}")
    print()
    print("  Uplift rank diagnostics:")
    print(f"    Spearman corr:          {diag['uplift_spearman_corr']:+.4f}  (p={diag['uplift_spearman_pval']:.4f})")
    print(f"    Sign agreement:         {diag['uplift_sign_agreement']:.4f}")
    print()
    print("  Predicted uplift distribution:")
    print(f"    Mean:                   {diag['predicted_uplift_mean']:+.4f}")
    print(f"    Median:                 {diag['predicted_uplift_median']:+.4f}")
    print(f"    Std:                    {diag['predicted_uplift_std']:.4f}")
    print(f"    Min / Max:              [{diag['predicted_uplift_min']:+.4f}, {diag['predicted_uplift_max']:+.4f}]")
    print(f"    Fraction positive:      {diag['fraction_positive_uplift']:.3f}")
    print(f"    Fraction negative:      {diag['fraction_negative_uplift']:.3f}")
    print(f"    Fraction near-zero:     {diag['fraction_near_zero_uplift']:.3f}")
    print()
    print("  RCT arm calibration (observed outcomes only):")
    print(f"    DO_NOTHING arm (n={diag['n_do_nothing_arm']:,}):")
    print(f"      Log-loss: {diag['log_loss_do_nothing_arm']:.4f}")
    print(f"      Brier:    {diag['brier_do_nothing_arm']:.4f}")
    print(f"    LINK arm (n={diag['n_link_arm']:,}):")
    print(f"      Log-loss: {diag['log_loss_link_arm']:.4f}")
    print(f"      Brier:    {diag['brier_link_arm']:.4f}")

    # Acceptance criteria summary
    print_section("Acceptance Criteria Summary")
    metrics_model = evaluate_metrics(records_model)
    metrics_nothing = evaluate_metrics(records_nothing)
    metrics_link = evaluate_metrics(records_link)

    model_inc_net = metrics_model["analytical_incremental_net_inr"]
    nothing_inc_net = metrics_nothing["analytical_incremental_net_inr"]
    link_inc_net = metrics_link["analytical_incremental_net_inr"]

    q_rates_model = compute_quadrant_action_rates(records_model)

    criteria = [
        (
            "Uplift variance meaningful (std >= 0.02)",
            diag["predicted_uplift_std"] >= 0.02,
            f"std={diag['predicted_uplift_std']:.4f}",
        ),
        (
            "Spearman corr > 0.20",
            diag["uplift_spearman_corr"] > 0.20,
            f"corr={diag['uplift_spearman_corr']:+.4f}",
        ),
        (
            "Sign agreement > 55%",
            diag["uplift_sign_agreement"] > 0.55,
            f"agree={diag['uplift_sign_agreement']:.4f}",
        ),
        (
            "Model beats Always-DO-NOTHING (ana inc net)",
            model_inc_net > nothing_inc_net,
            f"model={model_inc_net:.2f} > nothing={nothing_inc_net:.2f}",
        ),
        (
            "Model beats Always-SEND-LINK (ana inc net)",
            model_inc_net > link_inc_net,
            f"model={model_inc_net:.2f} > link={link_inc_net:.2f}",
        ),
        (
            "Persuadable action rate > 50%",
            q_rates_model.get("Persuadable", {}).get("action_rate", 0) > 0.50,
            f"rate={q_rates_model.get('Persuadable', {}).get('action_rate', 0):.3f}",
        ),
        (
            "Sleeping Dog action rate < 25%",
            q_rates_model.get("Sleeping Dog", {}).get("action_rate", 1) < 0.25,
            f"rate={q_rates_model.get('Sleeping Dog', {}).get('action_rate', 1):.3f}",
        ),
    ]

    all_pass = True
    for name, passed, detail in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
        print(f"         {detail}")

    print()
    if all_pass:
        print("  All primary acceptance criteria MET.")
    else:
        print("  One or more acceptance criteria NOT MET — see details above.")

    print()
    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
