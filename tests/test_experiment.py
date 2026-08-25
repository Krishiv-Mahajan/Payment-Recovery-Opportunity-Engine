"""
Tests for A/B Experimentation Framework.
"""

from __future__ import annotations

from app.experiment import ExperimentEngine


def test_deterministic_assignment():
    engine = ExperimentEngine(experiment_name="test_v1", control_percentage=50)

    # Same identifier should always yield the same variant
    v1 = engine.assign_variant("user_123")
    v2 = engine.assign_variant("user_123")
    assert v1 == v2

    # Different identifiers should hash differently
    # Let's find one that goes to control and one that goes to treatment
    counts = {"control": 0, "treatment": 0}
    for i in range(100):
        v = engine.assign_variant(f"user_{i}")
        counts[v] += 1

    assert counts["control"] > 0
    assert counts["treatment"] > 0


def test_distribution_fairness():
    engine = ExperimentEngine(experiment_name="test_v2", control_percentage=50)

    control_count = 0
    total = 10000
    for i in range(total):
        if engine.assign_variant(f"user_fair_{i}") == "control":
            control_count += 1

    # Should be close to 5000 (within 2% = 4800 to 5200)
    assert 4800 <= control_count <= 5200


def test_no_experiment_name_defaults_to_treatment():
    engine = ExperimentEngine(experiment_name=None, control_percentage=50)
    assert engine.assign_variant("user_123") == "treatment"


def test_100_percent_control():
    engine = ExperimentEngine(experiment_name="test_v3", control_percentage=100)
    assert engine.assign_variant("user_123") == "control"
    assert engine.assign_variant("user_456") == "control"


def test_0_percent_control():
    engine = ExperimentEngine(experiment_name="test_v4", control_percentage=0)
    assert engine.assign_variant("user_123") == "treatment"
    assert engine.assign_variant("user_456") == "treatment"
