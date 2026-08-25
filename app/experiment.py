"""
A/B Experimentation Framework.

Deterministically assigns traffic to treatment or control based on a stable
identifier (customer_identifier or payment_id) using SHA-256.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

logger = logging.getLogger(__name__)

VariantType = Literal["control", "treatment"]


class ExperimentEngine:
    """
    Deterministically assigns a variant based on a stable identifier.

    If experiment_name is None, it defaults to 'treatment' to allow normal
    ML policy execution for un-experimented traffic.
    """
    def __init__(self, experiment_name: str | None, control_percentage: int = 50):
        self.experiment_name = experiment_name
        self.control_percentage = max(0, min(100, control_percentage))

    def assign_variant(self, identifier: str) -> VariantType:
        """
        Assign a variant deterministically based on the identifier.
        """
        if not self.experiment_name or self.control_percentage == 0:
            return "treatment"

        if self.control_percentage == 100:
            return "control"

        # Hash the string: experiment_name + identifier
        hash_input = f"{self.experiment_name}:{identifier}".encode("utf-8")

        # Use sha256 to get a stable hex digest
        hex_digest = hashlib.sha256(hash_input).hexdigest()

        # Convert first 8 characters to integer
        hash_int = int(hex_digest[:8], 16)

        # Modulo 100 gives 0-99
        bucket = hash_int % 100

        if bucket < self.control_percentage:
            return "control"

        return "treatment"
