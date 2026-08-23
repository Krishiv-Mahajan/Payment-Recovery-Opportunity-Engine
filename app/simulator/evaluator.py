"""
Simulator Evaluator.
Computes metrics (gross, net, segmented) from simulated records.
"""
from __future__ import annotations

from typing import List, Dict, Any

from app.models import RecoveryAction
from app.simulator.schemas import SimulatedRecord


def evaluate_metrics(records: List[SimulatedRecord]) -> Dict[str, Any]:
    """
    Computes global and segmented evaluation metrics for a batch of simulated records.
    Explicitly tracks both empirical and analytical metrics, ensuring the DO_NOTHING
    baseline is evaluated on identical counterfactual contexts.
    All internal monetary math is done in integer paise, then converted to INR for reporting.
    """
    n = len(records)
    if n == 0:
        return {}

    total_gross_recovery_paise = 0
    total_intervention_cost_paise = 0
    recovered_count = 0
    
    # Empirical tracking (using sampled structural outcomes y_do_nothing, y_link, etc)
    empirical_incremental_gross_paise = 0
    empirical_incremental_net_paise = 0
    
    # Analytical tracking (using true expected probabilities)
    analytical_incremental_gross_paise = 0.0
    analytical_incremental_net_paise = 0.0
    
    # Segmented by quadrant
    quadrant_metrics = {}

    for record in records:
        ctx = record.context
        gt = record.ground_truth
        
        # Quadrant tracking
        quadrant = gt.derived_quadrant
        if quadrant not in quadrant_metrics:
            quadrant_metrics[quadrant] = {
                "count": 0,
                "assigned_links": 0,
                "empirical_gross_recovery_inr": 0.0,
                "intervention_cost_inr": 0.0,
                "empirical_incremental_net_inr": 0.0,
                "analytical_incremental_net_inr": 0.0
            }
        
        q_metrics = quadrant_metrics[quadrant]
        q_metrics["count"] += 1
        
        # 1. Empirical observed (assigned action)
        if record.observed_outcome:
            recovered_count += 1
            total_gross_recovery_paise += ctx.amount_paise
            q_metrics["empirical_gross_recovery_inr"] += ctx.amount_paise / 100.0
            
        total_intervention_cost_paise += record.intervention_cost_paise
        q_metrics["intervention_cost_inr"] += record.intervention_cost_paise / 100.0
        
        if record.assigned_action == RecoveryAction.SEND_PAYMENT_LINK:
            q_metrics["assigned_links"] += 1
            
        # 2. Empirical Counterfactual (DO_NOTHING baseline)
        y_assigned = record.observed_outcome
        y_baseline = gt.y_do_nothing
        
        emp_inc_gross = (int(y_assigned) - int(y_baseline)) * ctx.amount_paise
        emp_inc_net = emp_inc_gross - record.intervention_cost_paise
        
        empirical_incremental_gross_paise += emp_inc_gross
        empirical_incremental_net_paise += emp_inc_net
        q_metrics["empirical_incremental_net_inr"] += emp_inc_net / 100.0
            
        # 3. Analytical expected values
        if record.assigned_action == RecoveryAction.NO_ACTION:
            p_chosen = gt.p_recovery_do_nothing
        elif record.assigned_action == RecoveryAction.SEND_PAYMENT_LINK:
            p_chosen = gt.p_recovery_link
        elif record.assigned_action == RecoveryAction.SEND_PAYMENT_LINK_WITH_DISCOUNT:
            p_chosen = gt.p_recovery_discount
        else:
            p_chosen = 0.0
            
        expected_gross = ctx.amount_paise * p_chosen
        expected_baseline_gross = ctx.amount_paise * gt.p_recovery_do_nothing
        
        ana_inc_gross = expected_gross - expected_baseline_gross
        ana_inc_net = ana_inc_gross - record.intervention_cost_paise
        
        analytical_incremental_gross_paise += ana_inc_gross
        analytical_incremental_net_paise += ana_inc_net
        q_metrics["analytical_incremental_net_inr"] += ana_inc_net / 100.0
        
    recovery_rate = recovered_count / n
    emp_inc_net_per_1000 = (empirical_incremental_net_paise / 100.0 / n) * 1000

    return {
        "n_transactions": n,
        "recovery_rate": recovery_rate,
        "empirical_gross_recovery_inr": total_gross_recovery_paise / 100.0,
        "intervention_cost_inr": total_intervention_cost_paise / 100.0,
        "empirical_incremental_gross_inr": empirical_incremental_gross_paise / 100.0,
        "empirical_incremental_net_inr": empirical_incremental_net_paise / 100.0,
        "analytical_incremental_gross_inr": analytical_incremental_gross_paise / 100.0,
        "analytical_incremental_net_inr": analytical_incremental_net_paise / 100.0,
        "empirical_incremental_net_inr_per_1000": emp_inc_net_per_1000,
        "quadrants": quadrant_metrics
    }

