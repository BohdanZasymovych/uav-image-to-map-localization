# =============================================================================
# evaluation/base.py
# -----------------------------------------------------------------------------
# OWNER: Person C.
#
# PURPOSE:
#   Abstract base class for dataset generators and the SyntheticFrame
#   dataclass that carries a single test sample.
#   The Evaluator depends on DatasetGenerator — not on any concrete generator.
#   This allows a RealFlightDatasetLoader to be added later without any
#   changes to evaluation code.
#
# CONTAINS:
#   - SyntheticFrame   : dataclass with uav_img, ground_truth_px, transform_matrix.
#   - DatasetGenerator : ABC with a single abstract method generate(n).
#
# CONSUMED BY:
#   evaluation/dataset.py  — SyntheticDatasetGenerator implements DatasetGenerator.
#   evaluation/metrics.py  — Evaluator depends on DatasetGenerator interface.
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticFrame:
    """A single generated test sample."""

    uav_img: NDArray
    """Warped crop simulating a UAV camera frame. Shape (H, W, 3), BGR."""

    ground_truth_px: NDArray
    """True UAV center position in map pixel coordinates. Shape (2,)."""

    transform_matrix: NDArray
    """The 3x3 matrix applied to produce this frame. Stored for debugging."""


class DatasetGenerator(ABC):
    """
    Abstracts synthetic test dataset generation.

    The Evaluator depends on this interface, not on any concrete generator,
    so real-flight data loaders can be added without touching evaluation code.
    """

    @abstractmethod
    def generate(self, n: int) -> list[SyntheticFrame]:
        """
        Generate n synthetic UAV frames from a reference satellite map.

        Parameters
        ----------
        n : int

        Returns
        -------
        list[SyntheticFrame]
        """
        ...
