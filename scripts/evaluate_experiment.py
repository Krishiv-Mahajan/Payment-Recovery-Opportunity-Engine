#!/usr/bin/env python3
"""
Lightweight causal evaluation of the A/B Experimentation Framework.

Calculates and reports:
- Sample count per variant
- Recovery rate per variant (OUTCOME_OBSERVED / total assigned)
- Absolute and relative recovery-rate differences
- Total recovered INR per variant
- Intervention cost (proxy based on links sent)
- Incremental net INR
- Treatment vs Control incremental value
"""

import sys
import os

# Ensure the app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import func
from app.database import engine, SessionLocal
from app.models import RecoveryDecision, DecisionStatus, RecoveryAction, PaymentRecord

def evaluate_experiment(experiment_name: str | None = None):
    with SessionLocal() as db:
        query = db.query(RecoveryDecision)

        if experiment_name:
            query = query.filter(RecoveryDecision.experiment_name == experiment_name)

        decisions = query.all()

        if not decisions:
            print(f"No decisions found for experiment '{experiment_name or 'ALL'}'.")
            return

        # Group by variant
        stats = {
            "control": {"assigned": 0, "recovered": 0, "links_sent": 0, "recovered_inr": 0.0},
            "treatment": {"assigned": 0, "recovered": 0, "links_sent": 0, "recovered_inr": 0.0},
            "legacy": {"assigned": 0, "recovered": 0, "links_sent": 0, "recovered_inr": 0.0},
        }

        for d in decisions:
            v = d.experiment_variant or "legacy"
            if v not in stats:
                stats[v] = {"assigned": 0, "recovered": 0, "links_sent": 0, "recovered_inr": 0.0}

            stats[v]["assigned"] += 1

            if d.selected_action == RecoveryAction.SEND_PAYMENT_LINK:
                stats[v]["links_sent"] += 1

            if d.decision_status == DecisionStatus.OUTCOME_OBSERVED:
                stats[v]["recovered"] += 1
                # Fetch payment amount for recovered INR
                pr = db.query(PaymentRecord).filter(PaymentRecord.id == d.payment_record_id).first()
                if pr and pr.amount:
                    stats[v]["recovered_inr"] += pr.amount / 100.0 # Assuming amount is in paise

        print(f"=== Experiment Evaluation: {experiment_name or 'ALL'} ===")
        print("-" * 60)

        for v, s in stats.items():
            if s["assigned"] == 0:
                continue

            rate = (s["recovered"] / s["assigned"]) * 100
            print(f"Variant: {v.upper()}")
            print(f"  Assigned:      {s['assigned']}")
            print(f"  Recovered:     {s['recovered']} ({rate:.2f}%)")
            print(f"  Links Sent:    {s['links_sent']}")
            print(f"  Recovered INR: {s['recovered_inr']:.2f}")
            print("-" * 60)

        ctrl = stats.get("control", {})
        trt = stats.get("treatment", {})

        if ctrl.get("assigned", 0) > 0 and trt.get("assigned", 0) > 0:
            ctrl_rate = ctrl["recovered"] / ctrl["assigned"]
            trt_rate = trt["recovered"] / trt["assigned"]

            abs_diff = trt_rate - ctrl_rate
            rel_diff = (abs_diff / ctrl_rate) if ctrl_rate > 0 else float('inf')

            # Simple intervention cost proxy: 0.50 INR per link
            ctrl_cost = ctrl["links_sent"] * 0.50
            trt_cost = trt["links_sent"] * 0.50

            ctrl_net = ctrl["recovered_inr"] - ctrl_cost
            trt_net = trt["recovered_inr"] - trt_cost

            # Per-user net
            ctrl_net_per_user = ctrl_net / ctrl["assigned"]
            trt_net_per_user = trt_net / trt["assigned"]

            incremental_value_per_user = trt_net_per_user - ctrl_net_per_user
            total_incremental_value = incremental_value_per_user * trt["assigned"]

            print("=== Causal Uplift (Treatment vs Control) ===")
            print(f"Absolute Rate Difference: {abs_diff * 100:+.2f}%")
            if ctrl_rate > 0:
                print(f"Relative Rate Difference: {rel_diff * 100:+.2f}%")
            print(f"Net INR (Control):        {ctrl_net:.2f}")
            print(f"Net INR (Treatment):      {trt_net:.2f}")
            print(f"Incremental Net INR:      {total_incremental_value:+.2f}")
            print("============================================")
        else:
            print("Insufficient data for Treatment vs Control comparison.")

if __name__ == "__main__":
    import sys
    exp_name = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_experiment(exp_name)
