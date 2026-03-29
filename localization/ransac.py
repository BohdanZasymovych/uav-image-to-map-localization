# =============================================================================
# localization/ransac.py
# -----------------------------------------------------------------------------
# OWNER: Person B.
#
# PURPOSE:
#   Model-agnostic RANSAC implementation.
#   Depends only on TransformationModel — has zero knowledge of affine,
#   projective, or any other specific model.
#
# IMPLEMENT:
#   class RANSAC
#
#   __init__(self, model: TransformationModel, epsilon: float,
#            confidence: float, max_iterations: int):
#     epsilon        — inlier threshold in pixels (reprojection error < epsilon)
#     confidence     — desired probability of finding a correct model (e.g. 0.99)
#     max_iterations — hard upper bound on iteration count
#
#   run(src_pts, dst_pts) -> tuple[NDArray, NDArray, int]:
#     Returns: (best_M, inlier_mask, iterations_done)
#       best_M       — 3x3 matrix re-estimated on the full final inlier set
#       inlier_mask  — boolean array of shape (N,)
#       iterations_done — actual number of iterations executed
#
#     Algorithm:
#       1. Initialise best_mask = empty boolean array, n_iters = max_iterations.
#       2. Loop up to n_iters times:
#            a. Sample model.min_points random indices without replacement.
#            b. Call model.estimate() on the sample.
#               Catch np.linalg.LinAlgError (degenerate sample) and skip.
#            c. Call model.reprojection_error() on ALL N points.
#            d. Build boolean mask: error < epsilon.
#            e. If inlier count > best count: update best_mask,
#               recompute n_iters = min(_adaptive_iters(...), max_iterations).
#       3. Re-estimate best_M on full best_mask inlier set.
#       4. Return (best_M, best_mask, iterations_done).
#
#   _adaptive_iters(n_inliers, n_total) -> int:
#     w = n_inliers / n_total
#     N = ceil( log(1 - confidence) / log(1 - w ^ min_points) )
#     Guard: if w == 0 return max_iterations. Clamp result to max_iterations.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class RANSAC:
    def __init__(
        self,
        model: TransformationModel,
        epsilon: float = 3.0,
        confidence: float = 0.99,
        max_iterations: int = 2000,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.confidence = confidence
        self.max_iterations = max_iterations

    def run(
        self,
        src_pts: NDArray,
        dst_pts: NDArray,
    ) -> tuple[NDArray, NDArray, int]:
        # TODO: implement
        ...

    def _adaptive_iters(self, n_inliers: int, n_total: int) -> int:
        # TODO: implement
        ...
