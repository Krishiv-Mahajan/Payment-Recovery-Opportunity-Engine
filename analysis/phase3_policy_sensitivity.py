#!/usr/bin/env python
"""
Policy sensitivity analysis: minimum P1 threshold sweep.
Vectorised: pre-computes all P0/P1 in one batch, then sweeps thresholds in-memory.

No model retraining. No GroundTruth in policy decisions.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from app.ml.models.s_learner import SLearner
from app.ml.training import features_to_row, FEATURE_COLS, ACTION_ENCODING
from app.models import RecoveryAction
from app.simulator.runner import generate_contexts
from app.simulator.engine import generate_ground_truth
from app.simulator.schemas import INTERVENTION_COSTS_PAISE

EVAL_N    = 10_000
EVAL_SEED = 2000
MODEL_PATH = "artifacts/model_v1.joblib"

THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LINK_COST_PAISE = INTERVENTION_COSTS_PAISE[RecoveryAction.SEND_PAYMENT_LINK]


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run train_phase3.py first.")
        sys.exit(1)

    model: SLearner = joblib.load(MODEL_PATH)
    print(f"Model loaded. N={EVAL_N:,}, seed={EVAL_SEED}\n")

    # ── 1. Generate all contexts and ground truths ────────────────────────
    print("Generating evaluation contexts and ground truths...")
    contexts = generate_contexts(EVAL_N, EVAL_SEED)
    ground_truths = [generate_ground_truth(c) for c in contexts]
    print(f"  {len(contexts):,} contexts ready.\n")

    # ── 2. Vectorised prediction: compute P0, P1 for all contexts ─────────
    print("Computing P0, P1 for all contexts (vectorised)...")
    features_list = [c.to_payment_features() for c in contexts]

    rows_0 = [features_to_row(f, RecoveryAction.NO_ACTION)        for f in features_list]
    rows_1 = [features_to_row(f, RecoveryAction.SEND_PAYMENT_LINK) for f in features_list]

    df_0 = pd.DataFrame(rows_0)[FEATURE_COLS]
    df_1 = pd.DataFrame(rows_1)[FEATURE_COLS]

    p0_arr = model.predict_batch_probabilities(df_0)
    p1_arr = model.predict_batch_probabilities(df_1)
    print(f"  Done. P0 mean={p0_arr.mean():.3f}, P1 mean={p1_arr.mean():.3f}\n")

    # ── 3. Build per-transaction arrays ───────────────────────────────────
    amounts       = np.array([c.amount_paise             for c in contexts])
    quadrants     = np.array([g.derived_quadrant          for g in ground_truths])
    p0_true       = np.array([g.p_recovery_do_nothing     for g in ground_truths])
    p1_true       = np.array([g.p_recovery_link           for g in ground_truths])
    y_do_nothing  = np.array([int(g.y_do_nothing)         for g in ground_truths])
    y_link        = np.array([int(g.y_link)               for g in ground_truths])

    uplift_hat   = p1_arr - p0_arr
    inc_gross_if_link = amounts * uplift_hat          # expected incremental gross
    inc_net_if_link   = inc_gross_if_link - LINK_COST_PAISE

    QUAD_NAMES = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]

    # ── 4. Sweep thresholds ───────────────────────────────────────────────
    def evaluate_threshold(min_p1: float) -> dict:
        # Policy decision: SEND_LINK iff P1>=threshold AND inc_net>0
        send_link = (p1_arr >= min_p1) & (inc_net_if_link > 0)

        # Observed outcomes under this policy
        observed = np.where(send_link, y_link, y_do_nothing)
        obs_baseline = y_do_nothing

        # Costs
        cost_paise = np.where(send_link, LINK_COST_PAISE, 0)

        # Empirical incremental
        emp_inc_gross = (observed - obs_baseline) * amounts
        emp_inc_net   = emp_inc_gross - cost_paise

        # Analytical incremental (using true probs from ground truth)
        p_chosen_true = np.where(send_link, p1_true, p0_true)
        ana_inc_gross = amounts * (p_chosen_true - p0_true)
        ana_inc_net   = ana_inc_gross - cost_paise

        # Global
        result = {
            "action_rate":       float(send_link.mean()),
            "link_count":        int(send_link.sum()),
            "cost_inr":          float(cost_paise.sum() / 100),
            "emp_inc_gross_inr": float(emp_inc_gross.sum() / 100),
            "emp_inc_net_inr":   float(emp_inc_net.sum() / 100),
            "ana_inc_gross_inr": float(ana_inc_gross.sum() / 100),
            "ana_inc_net_inr":   float(ana_inc_net.sum() / 100),
            "quadrants":         {},
        }

        # Per-quadrant
        for q in QUAD_NAMES:
            mask = quadrants == q
            if mask.sum() == 0:
                continue
            q_link  = send_link[mask]
            q_ana   = ana_inc_net[mask]
            q_emp   = emp_inc_net[mask]
            result["quadrants"][q] = {
                "count":          int(mask.sum()),
                "action_rate":    float(q_link.mean()),
                "ana_inc_net_inr": float(q_ana.sum() / 100),
                "emp_inc_net_inr": float(q_emp.sum() / 100),
            }

        return result

    # Baselines
    res_nothing = evaluate_threshold(-1.0)    # never sends (inc_net never > 0 with P1<0)
    # Force DO_NOTHING properly
    def evaluate_always_nothing() -> dict:
        send_link = np.zeros(EVAL_N, dtype=bool)
        observed  = y_do_nothing
        obs_baseline = y_do_nothing
        cost_paise   = np.zeros(EVAL_N, dtype=int)
        emp_inc_gross = (observed - obs_baseline) * amounts
        emp_inc_net   = emp_inc_gross - cost_paise
        ana_inc_gross = amounts * (p0_true - p0_true)
        ana_inc_net   = ana_inc_gross - cost_paise
        result = {
            "action_rate": 0.0, "link_count": 0,
            "cost_inr": 0.0,
            "emp_inc_gross_inr": 0.0, "emp_inc_net_inr": 0.0,
            "ana_inc_gross_inr": 0.0, "ana_inc_net_inr": 0.0,
            "quadrants": {},
        }
        for q in QUAD_NAMES:
            mask = quadrants == q
            if mask.sum() == 0:
                continue
            result["quadrants"][q] = {
                "count": int(mask.sum()), "action_rate": 0.0,
                "ana_inc_net_inr": 0.0, "emp_inc_net_inr": 0.0,
            }
        return result

    def evaluate_always_link() -> dict:
        send_link = np.ones(EVAL_N, dtype=bool)
        observed  = y_link
        cost_paise = np.full(EVAL_N, LINK_COST_PAISE)
        emp_inc_gross = (observed - y_do_nothing) * amounts
        emp_inc_net   = emp_inc_gross - cost_paise
        ana_inc_gross = amounts * (p1_true - p0_true)
        ana_inc_net   = ana_inc_gross - cost_paise
        result = {
            "action_rate": 1.0, "link_count": EVAL_N,
            "cost_inr": float(cost_paise.sum() / 100),
            "emp_inc_gross_inr": float(emp_inc_gross.sum() / 100),
            "emp_inc_net_inr":   float(emp_inc_net.sum() / 100),
            "ana_inc_gross_inr": float(ana_inc_gross.sum() / 100),
            "ana_inc_net_inr":   float(ana_inc_net.sum() / 100),
            "quadrants": {},
        }
        for q in QUAD_NAMES:
            mask = quadrants == q
            if mask.sum() == 0:
                continue
            q_ana = ana_inc_net[mask]
            q_emp = emp_inc_net[mask]
            result["quadrants"][q] = {
                "count": int(mask.sum()), "action_rate": 1.0,
                "ana_inc_net_inr": float(q_ana.sum() / 100),
                "emp_inc_net_inr": float(q_emp.sum() / 100),
            }
        return result

    s_nothing = evaluate_always_nothing()
    s_link    = evaluate_always_link()
    results   = {}
    for thr in THRESHOLDS:
        results[thr] = evaluate_threshold(thr)

    nothing_ana = s_nothing["ana_inc_net_inr"]
    link_ana    = s_link["ana_inc_net_inr"]

    def q_rate(s, q):
        return s["quadrants"].get(q, {}).get("action_rate", float("nan"))

    def q_ana(s, q):
        return s["quadrants"].get(q, {}).get("ana_inc_net_inr", 0.0)

    def q_emp(s, q):
        return s["quadrants"].get(q, {}).get("emp_inc_net_inr", 0.0)

    # ── 5. Print tables ───────────────────────────────────────────────────
    print("=" * 130)
    print("POLICY SENSITIVITY ANALYSIS — MINIMUM P1 THRESHOLD SWEEP")
    print("=" * 130)
    print()

    COL = 130
    hdr = (f"{'Label':>12} | {'Act%':>5} | {'Cost INR':>10} | "
           f"{'AnaIncNet':>12} | {'EmpIncNet':>12} | "
           f"{'Persuadabl':>10} | {'SureThing':>9} | "
           f"{'LostCause':>9} | {'SleepDog':>8}")
    print(hdr)
    print("-" * COL)

    def row(label, s):
        print(
            f"{label:>12} | "
            f"{s['action_rate']*100:>4.1f}% | "
            f"{s['cost_inr']:>10,.0f} | "
            f"{s['ana_inc_net_inr']:>12,.0f} | "
            f"{s['emp_inc_net_inr']:>12,.0f} | "
            f"{q_rate(s,'Persuadable')*100:>9.1f}% | "
            f"{q_rate(s,'Sure Thing')*100:>8.1f}% | "
            f"{q_rate(s,'Lost Cause')*100:>8.1f}% | "
            f"{q_rate(s,'Sleeping Dog')*100:>7.1f}%"
        )

    row("DO_NOTHING", s_nothing)
    row("ALW_LINK",   s_link)
    print("-" * COL)
    for thr in THRESHOLDS:
        row(f"p1>={thr:.2f}", results[thr])

    # ── 6. Quadrant net recovery table ───────────────────────────────────
    print()
    print("-" * 100)
    print("ANALYTICAL INCREMENTAL NET RECOVERY BY QUADRANT (INR)")
    print("-" * 100)
    print(f"{'Label':>12} | {'Persuadable':>12} | {'Sure Thing':>12} | "
          f"{'Lost Cause':>12} | {'Sleeping Dog':>12}")
    print("-" * 70)

    def qrow(label, s):
        print(
            f"{label:>12} | "
            f"{q_ana(s,'Persuadable'):>12,.0f} | "
            f"{q_ana(s,'Sure Thing'):>12,.0f} | "
            f"{q_ana(s,'Lost Cause'):>12,.0f} | "
            f"{q_ana(s,'Sleeping Dog'):>12,.0f}"
        )

    qrow("DO_NOTHING", s_nothing)
    qrow("ALW_LINK",   s_link)
    print("-" * 70)
    for thr in THRESHOLDS:
        qrow(f"p1>={thr:.2f}", results[thr])

    # ── 7. vs-baselines comparison ───────────────────────────────────────
    print()
    print("-" * 100)
    print("COMPARISON vs BASELINES (Analytical Incremental Net)")
    print("-" * 100)
    print(f"  {'Label':>12} | {'Ana Inc Net':>12} | {'vs DO_NOTH':>12} | {'vs ALW_LINK':>12}")
    print("  " + "-" * 55)
    for thr, s in results.items():
        v = s["ana_inc_net_inr"]
        print(f"  p1>={thr:.2f}    | {v:>12,.0f} | {v - nothing_ana:>+12,.0f} | {v - link_ana:>+12,.0f}")

    # ── 8. Monotonicity check ────────────────────────────────────────────
    print()
    print("-" * 100)
    print("MONOTONICITY CHECK (should ana_inc_net decrease as threshold rises?)")
    print("-" * 100)
    prev_net = None
    for thr, s in results.items():
        net = s["ana_inc_net_inr"]
        delta = "" if prev_net is None else f"  ({net - prev_net:+,.0f} vs prev)"
        print(f"  p1>={thr:.2f}: {net:>12,.0f}{delta}")
        prev_net = net

    # ── 9. Recommendation ────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    unconstrained = results[0.00]["ana_inc_net_inr"]

    best_thr, best_net = None, -float("inf")
    for thr, s in results.items():
        lc  = q_rate(s, "Lost Cause")
        prs = q_rate(s, "Persuadable")
        net = s["ana_inc_net_inr"]
        if prs > 0.80 and lc < 0.50 and net > unconstrained and net > best_net:
            best_net = net
            best_thr = thr

    if best_thr is not None:
        s = results[best_thr]
        print(f"\n  RECOMMENDED threshold: P1 >= {best_thr:.2f}")
        print(f"  Ana Inc Net:  {s['ana_inc_net_inr']:,.0f} INR  "
              f"({s['ana_inc_net_inr'] - unconstrained:+,.0f} vs unconstrained)")
        print(f"  Action rate:  {s['action_rate']*100:.1f}%")
        print(f"  Persuadable:  {q_rate(s,'Persuadable')*100:.1f}%  |  "
              f"Lost Cause:  {q_rate(s,'Lost Cause')*100:.1f}%  |  "
              f"Sleeping Dog:  {q_rate(s,'Sleeping Dog')*100:.1f}%")
    else:
        # Find where LC starts to drop without too much Persuadable loss
        print()
        print("  No threshold simultaneously beats unconstrained AND reduces Lost Cause <50%")
        print("  while keeping Persuadable >80%.\n")
        print("  Detailed trade-off at each threshold:")
        for thr, s in results.items():
            print(f"    p1>={thr:.2f}: ana_net={s['ana_inc_net_inr']:>12,.0f}  "
                  f"LC={q_rate(s,'Lost Cause')*100:>5.1f}%  "
                  f"Pers={q_rate(s,'Persuadable')*100:>5.1f}%  "
                  f"net_delta={s['ana_inc_net_inr']-unconstrained:>+10,.0f}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
