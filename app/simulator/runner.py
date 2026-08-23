"""
Simulator Runner.
Orchestrates data generation (RCT logging) and policy evaluation.
"""
from __future__ import annotations

import random
from typing import List, Dict, Any

from app.models import RecoveryAction
from app.ml.features import PaymentFeatures
from app.simulator.schemas import SimulatedContext, SimulatedRecord, INTERVENTION_COSTS_PAISE
from app.simulator.engine import generate_contexts, generate_ground_truth


def generate_logging_dataset(
    n: int, 
    seed: int, 
    policy_weights: Dict[RecoveryAction, float]
) -> List[SimulatedRecord]:
    """
    Generates a randomized control trial (RCT) logging dataset.
    Assigns actions explicitly based on configured policy_weights.
    """
    rng = random.Random(seed)
    contexts = generate_contexts(n, seed)
    records = []
    
    actions = list(policy_weights.keys())
    weights = list(policy_weights.values())
    
    for ctx in contexts:
        ground_truth = generate_ground_truth(ctx)
        
        # Randomly assign action
        assigned_action = rng.choices(actions, weights=weights)[0]
        
        # Get true structural empirical outcome for the assigned action
        if assigned_action == RecoveryAction.NO_ACTION:
            observed_outcome = ground_truth.y_do_nothing
        elif assigned_action == RecoveryAction.SEND_PAYMENT_LINK:
            observed_outcome = ground_truth.y_link
        elif assigned_action == RecoveryAction.SEND_PAYMENT_LINK_WITH_DISCOUNT:
            observed_outcome = ground_truth.y_discount
        else:
            raise ValueError(f"Unknown action {assigned_action}")
            
        records.append(SimulatedRecord(
            context=ctx,
            ground_truth=ground_truth,
            assigned_action=assigned_action,
            intervention_cost_paise=INTERVENTION_COSTS_PAISE[assigned_action],
            observed_outcome=observed_outcome
        ))
        
    return records


def evaluate_policy(
    predictor: Any, 
    n: int, 
    seed: int
) -> List[SimulatedRecord]:
    """
    Evaluates a candidate policy predictor.
    The predictor MUST expose a .predict(PaymentFeatures) method returning a PolicyPrediction.
    It never receives the ground truth or the SimulatedContext directly.
    """
    contexts = generate_contexts(n, seed)
    records = []
    
    for ctx in contexts:
        ground_truth = generate_ground_truth(ctx)
        
        # Convert simulator context to the canonical Phase 1 feature vector
        features = ctx.to_payment_features()
        
        # Evaluate policy
        prediction = predictor.predict(features)
        assigned_action = prediction.selected_action
        
        # Simulator samples outcome using hidden structural outcomes
        if assigned_action == RecoveryAction.NO_ACTION:
            observed_outcome = ground_truth.y_do_nothing
        elif assigned_action == RecoveryAction.SEND_PAYMENT_LINK:
            observed_outcome = ground_truth.y_link
        elif assigned_action == RecoveryAction.SEND_PAYMENT_LINK_WITH_DISCOUNT:
            observed_outcome = ground_truth.y_discount
        else:
            raise ValueError(f"Unknown action {assigned_action}")
            
        records.append(SimulatedRecord(
            context=ctx,
            ground_truth=ground_truth,
            assigned_action=assigned_action,
            intervention_cost_paise=INTERVENTION_COSTS_PAISE[assigned_action],
            observed_outcome=observed_outcome
        ))
        
    return records

