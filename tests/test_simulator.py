"""
Tests for Phase 2: Controlled Recovery Simulator.
"""
from __future__ import annotations

import pytest

from app.models import RecoveryAction
from app.ml.predictor import PlaceholderPredictor
from app.ml.features import PaymentFeatures
from app.simulator.schemas import SimulatedContext
from app.simulator.engine import generate_contexts, generate_ground_truth
from app.simulator.runner import generate_logging_dataset, evaluate_policy
from app.simulator.evaluator import evaluate_metrics


class TestSimulatorEngine:
    def test_reproducibility(self):
        """Context generation with the same seed must produce identical contexts."""
        contexts_1 = generate_contexts(10, seed=42)
        contexts_2 = generate_contexts(10, seed=42)
        
        for c1, c2 in zip(contexts_1, contexts_2):
            assert c1 == c2

    def test_different_seeds_produce_disjoint_contexts(self):
        """Train and evaluation sets must have disjoint IDs based on seed."""
        c1 = generate_contexts(10, seed=1000)[0]
        c2 = generate_contexts(10, seed=2000)[0]
        assert c1.transaction_id != c2.transaction_id

    def test_potential_outcome_bounds(self):
        """Ground truth probabilities must be strictly between 0 and 1."""
        contexts = generate_contexts(100, seed=123)
        for ctx in contexts:
            gt = generate_ground_truth(ctx)
            assert 0.0 < gt.p_recovery_do_nothing < 1.0
            assert 0.0 < gt.p_recovery_link < 1.0
            assert 0.0 <= gt.p_recovery_discount < 1.0

    def test_all_quadrants_generated(self):
        """Over a large enough sample, all 4 derived quadrants should emerge."""
        contexts = generate_contexts(2000, seed=999)
        quadrants_seen = set()
        for ctx in contexts:
            gt = generate_ground_truth(ctx)
            quadrants_seen.add(gt.derived_quadrant)
        
        assert "Sleeping Dog" in quadrants_seen
        assert "Persuadable" in quadrants_seen
        assert "Sure Thing" in quadrants_seen
        assert "Lost Cause" in quadrants_seen
        
    def test_latent_variables_hidden_but_active(self):
        """
        Latent variables must not be in context, but they should alter 
        the potential outcomes for identical observable contexts if the 
        transaction IDs differ (since tx_id seeds the hidden rng).
        """
        ctx1 = generate_contexts(1, seed=42)[0]
        
        # Clone ctx1 but give it a new transaction_id
        import copy
        ctx2 = copy.deepcopy(ctx1)
        ctx2.transaction_id = "completely_different_id"
        
        gt1 = generate_ground_truth(ctx1)
        gt2 = generate_ground_truth(ctx2)
        
        # Observable contexts are identical
        assert ctx1.amount_paise == ctx2.amount_paise
        assert ctx1.method == ctx2.method
        
        # Latents are not in context
        assert not hasattr(ctx1, "latent_responsiveness")
        
        # But probabilities differ because of unobserved heterogeneity
        assert gt1.p_recovery_do_nothing != gt2.p_recovery_do_nothing
        
    def test_latent_variables_deterministic_by_txid(self):
        """
        Prove that ground truth generation is deterministic for a given transaction_id.
        """
        ctx1 = generate_contexts(1, seed=42)[0]
        import copy
        ctx2 = copy.deepcopy(ctx1)
        
        gt1 = generate_ground_truth(ctx1)
        gt2 = generate_ground_truth(ctx2)
        
        assert gt1.latent_responsiveness == gt2.latent_responsiveness
        assert gt1.latent_annoyance == gt2.latent_annoyance
        assert gt1.p_recovery_do_nothing == gt2.p_recovery_do_nothing


class TestSimulatorRunner:
    def test_randomized_assignment_proportions(self):
        """RCT logging should respect the configured policy weights."""
        n = 5000
        weights = {
            RecoveryAction.NO_ACTION: 0.5,
            RecoveryAction.SEND_PAYMENT_LINK: 0.5
        }
        records = generate_logging_dataset(n, seed=42, policy_weights=weights)
        
        assert len(records) == n
        
        do_nothing_count = sum(1 for r in records if r.assigned_action == RecoveryAction.NO_ACTION)
        link_count = sum(1 for r in records if r.assigned_action == RecoveryAction.SEND_PAYMENT_LINK)
        
        # Should be roughly 50%
        assert 0.47 * n < do_nothing_count < 0.53 * n
        assert 0.47 * n < link_count < 0.53 * n

    def test_policy_isolation_and_compatibility(self):
        """
        PolicyPredictor must be able to be evaluated directly.
        It should only receive PaymentFeatures, never GroundTruth or Latents.
        """
        class SpyPredictor:
            def predict(self, features: PaymentFeatures):
                # Assert we only see PaymentFeatures
                assert not hasattr(features, "ground_truth")
                assert not hasattr(features, "latent_responsiveness")
                assert hasattr(features, "amount_paise")
                
                # Mock a PolicyPrediction response
                from app.ml.predictor import PolicyPrediction
                from app.models import DecisionStatus
                return PolicyPrediction(
                    decision_status=DecisionStatus.PENDING_POLICY,
                    selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                    model_version="test",
                    reasoning="test"
                )

        records = evaluate_policy(SpyPredictor(), n=10, seed=42)
        assert len(records) == 10
        assert all(r.assigned_action == RecoveryAction.SEND_PAYMENT_LINK for r in records)

    def test_placeholder_predictor_evaluation(self):
        """Verify the Phase 1 PlaceholderPredictor works in the simulator out-of-the-box."""
        predictor = PlaceholderPredictor()
        records = evaluate_policy(predictor, n=5, seed=42)
        assert len(records) == 5
        assert all(r.assigned_action == RecoveryAction.NO_ACTION for r in records)
        
    def test_observed_matches_structural_empirical(self):
        """Observed outcome must exactly match the pre-sampled structural outcome for the assigned action."""
        records = generate_logging_dataset(10, seed=42, policy_weights={RecoveryAction.SEND_PAYMENT_LINK: 1.0})
        for r in records:
            assert r.observed_outcome == r.ground_truth.y_link


class TestSimulatorEvaluator:
    def test_metrics_calculation_logic(self):
        """Evaluator should correctly compute incremental metrics and costs."""
        # Simple predictor that always does SEND_PAYMENT_LINK
        class AlwaysLinkPredictor:
            def predict(self, f):
                from app.ml.predictor import PolicyPrediction
                from app.models import DecisionStatus
                return PolicyPrediction(
                    decision_status=DecisionStatus.PENDING_POLICY,
                    selected_action=RecoveryAction.SEND_PAYMENT_LINK,
                    model_version="test",
                    reasoning="test"
                )
            
        records = evaluate_policy(AlwaysLinkPredictor(), n=1000, seed=42)
        metrics = evaluate_metrics(records)
        
        assert metrics["n_transactions"] == 1000
        assert "intervention_cost_inr" in metrics
        
        # Since we ALWAYS sent a link, intervention cost should be exactly 1000 * 3.0 INR
        assert metrics["intervention_cost_inr"] == 3000.0
        
        assert "recovery_rate" in metrics
        assert "analytical_incremental_net_inr" in metrics
        assert "empirical_incremental_net_inr" in metrics
        assert "empirical_incremental_net_inr_per_1000" in metrics
        
        # Check that quadrants are populated in metrics
        assert "quadrants" in metrics
        assert len(metrics["quadrants"]) > 0
