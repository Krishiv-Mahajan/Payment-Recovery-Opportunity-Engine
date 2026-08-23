"""
Simulator Engine.
Generates synthetic contexts and calculates hidden potential outcomes.
"""
from __future__ import annotations

import math
import random
from typing import List

from app.simulator.schemas import SimulatedContext, GroundTruth


def generate_contexts(n: int, seed: int) -> List[SimulatedContext]:
    """Generates synthetic transactions for the simulator."""
    rng = random.Random(seed)
    contexts = []
    
    for i in range(n):
        tx_id = f"sim_tx_{seed}_{i}"
        customer_id = f"cust_{rng.randint(1000, 9999)}@example.com"
        
        # Distributions for synthetic features
        amount_paise = rng.choice([10000, 25000, 50000, 150000, 500000])
        currency = "INR"
        method = rng.choices(["card", "upi", "netbanking", "wallet"], weights=[0.4, 0.4, 0.1, 0.1])[0]
        
        # Group errors into transient vs hard vs persuadable categories for realism
        error_category = rng.choices(["hard", "transient", "persuadable"], weights=[0.2, 0.4, 0.4])[0]
        
        if error_category == "hard":
            error_reason = rng.choice(["card_expired", "invalid_otp", "fraud_blocked"])
            error_code = "BAD_REQUEST_ERROR"
            error_source = "customer"
            error_step = "payment_authentication"
        elif error_category == "transient":
            error_reason = rng.choice(["gateway_timed_out", "processing_failed"])
            error_code = "GATEWAY_ERROR"
            error_source = "gateway"
            error_step = "payment_processing"
        else:
            error_reason = "insufficient_funds"
            error_code = "BAD_REQUEST_ERROR"
            error_source = "customer"
            error_step = "payment_authorization"
            
        prior_failure_count = rng.randint(0, 10)
        prior_success_count = rng.randint(0, 30)
        
        ctx = SimulatedContext(
            transaction_id=tx_id,
            amount_paise=amount_paise,
            currency=currency,
            method=method,
            error_code=error_code,
            error_reason=error_reason,
            error_source=error_source,
            error_step=error_step,
            prior_failure_count=prior_failure_count,
            prior_success_count=prior_success_count,
            customer_identifier=customer_id
        )
        contexts.append(ctx)
        
    return contexts


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_ground_truth(context: SimulatedContext) -> GroundTruth:
    """
    Computes hidden ground truth probabilities using simulation-specific rules.
    Introduces hidden latent traits (unobserved heterogeneity) to ensure the 
    data-generating process is not trivially learnable by ML models.
    Derives the quadrant from the calculated potential outcomes.
    This ground truth is never passed to the ML policy.
    """
    # Use the transaction_id to seed the hidden generation, ensuring
    # determinism without leaking state into the observable context.
    # We use a cryptographic hash to ensure cross-process reproducibility.
    import hashlib
    seed_material = hashlib.sha256(context.transaction_id.encode("utf-8")).digest()
    hidden_seed = int.from_bytes(seed_material[:8], byteorder="big", signed=False)
    hidden_rng = random.Random(hidden_seed)
    
    # 1. Generate unobserved latent traits (structural heterogeneity)
    # responsiveness: High means they are generally likely to pay if nudged.
    # annoyance: High means nudging them makes them angry and LESS likely to pay.
    latent_responsiveness = hidden_rng.gauss(0, 1.0)
    latent_annoyance = hidden_rng.gauss(0, 1.0)
    
    z0 = 0.0 # Baseline logit
    z1 = 0.0 # Treatment effect logit (SEND_PAYMENT_LINK)
    
    # 2. Base structural logic based on observables
    if context.error_reason in ("gateway_timed_out", "processing_failed") and context.prior_success_count > 3:
        # Sure Thing scenario
        z0 = 2.0
        z1 = 0.1
    elif context.error_reason in ("card_expired", "invalid_otp", "fraud_blocked") or (context.prior_failure_count > 5 and context.prior_success_count == 0):
        # Lost Cause scenario
        z0 = -3.5
        z1 = 0.2
    elif context.error_reason == "insufficient_funds" and context.prior_success_count > 0 and context.amount_paise >= 50000:
        # Persuadable scenario
        z0 = -1.5
        z1 = 2.5
    elif context.amount_paise < 20000 and context.method in ("upi", "wallet"):
        # Sleeping Dog scenario
        z0 = 1.5
        z1 = -2.5
    else:
        # Average generic case
        z0 = -1.0
        z1 = 1.0

    # 3. Apply latent unobserved heterogeneity
    # This ensures ML models have irreducible error (cannot achieve 100% accuracy)
    z0 += latent_responsiveness * 0.5
    z1 += latent_responsiveness * 1.0
    z1 -= latent_annoyance * 1.5
        
    p_recovery_do_nothing = _sigmoid(z0)
    p_recovery_link = _sigmoid(z0 + z1)
    
    # Discount is inactive for Phase 2. Keep structurally compliant but zeroed.
    p_recovery_discount = 0.0
    
    uplift = p_recovery_link - p_recovery_do_nothing
    
    # 4. Derive quadrants from potential outcomes (Simulation-specific definitions)
    if uplift < -0.05:
        derived_quadrant = "Sleeping Dog"
    elif uplift > 0.10:
        derived_quadrant = "Persuadable"
    elif p_recovery_do_nothing > 0.50:
        derived_quadrant = "Sure Thing"
    else:
        derived_quadrant = "Lost Cause"
        
    # 5. Sample structural empirical potential outcomes
    # By using a single uniform draw `u`, we preserve rank correlation between Y0 and Y1
    u = hidden_rng.random()
    y_do_nothing = u < p_recovery_do_nothing
    y_link = u < p_recovery_link
    y_discount = False
        
    return GroundTruth(
        p_recovery_do_nothing=p_recovery_do_nothing,
        p_recovery_link=p_recovery_link,
        p_recovery_discount=p_recovery_discount,
        derived_quadrant=derived_quadrant,
        y_do_nothing=y_do_nothing,
        y_link=y_link,
        y_discount=y_discount,
        latent_responsiveness=latent_responsiveness,
        latent_annoyance=latent_annoyance
    )
