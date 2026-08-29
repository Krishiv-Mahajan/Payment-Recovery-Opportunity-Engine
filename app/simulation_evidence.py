"""Documented Phase 3 simulation evidence used by the demo dashboard.

This is an analytical, synthetic-RCT evaluation snapshot, not observed database
data or live revenue. It was produced by ``scripts/evaluate_phase3.py`` with
``EVAL_N=10_000``, ``EVAL_SEED=2000``, and ``artifacts/model_v1.joblib``.

Keep this module dependency-free so the dashboard remains usable even when the
evaluation model artifact is not present. Update the values only after rerunning
the Phase 3 evaluation with an intentionally approved evaluation configuration.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase3SimulationEvidence:
    """Analytical INR results from the documented Phase 3 evaluation snapshot."""

    transaction_count: int = 10_000
    evaluation_seed: int = 2_000
    ai_policy_incremental_net_inr: float = 2_830_137.84
    always_link_incremental_net_inr: float = 2_335_001.95
    always_link_sleeping_dog_loss_inr: float = -720_146.00
    ai_policy_sleeping_dog_loss_inr: float = -53_346.00

    @property
    def incremental_value_vs_always_link_inr(self) -> float:
        return self.ai_policy_incremental_net_inr - self.always_link_incremental_net_inr

    @property
    def sleeping_dog_loss_avoided_inr(self) -> float:
        return abs(self.always_link_sleeping_dog_loss_inr) - abs(
            self.ai_policy_sleeping_dog_loss_inr
        )


PHASE3_SIMULATION_EVIDENCE = Phase3SimulationEvidence()
