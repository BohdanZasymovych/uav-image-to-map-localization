# =============================================================================
# evaluation/dataset.py
# -----------------------------------------------------------------------------
# OWNER: Person C.
#
# PURPOSE:
#   Concrete dataset generator. Produces synthetic UAV frames by applying
#   random affine transforms to a reference satellite map.
#   _random_affine() is the linear algebra contribution of Person C.
#
# IMPLEMENT:
#   class SyntheticDatasetGenerator(DatasetGenerator)
#
#   __init__(self, map_img: NDArray, seed: int = 42):
#     - Store map_img.
#     - Create self.rng = np.random.default_rng(seed) for reproducibility.
#
#   generate(self, n, scale_range=(0.4, 0.8),
#            rotation_range=(-30, 30),
#            shear_range=(-0.1, 0.1)) -> list[SyntheticFrame]:
#     For each of n frames:
#       1. Call _random_affine() -> M, ground_truth_px.
#       2. Warp map: cv2.warpAffine(map_img, M[:2], (W, H)).
#       3. Append SyntheticFrame(warped, ground_truth_px, M).
#
#   _random_affine(scale_range, rotation_range, shear_range)
#       -> tuple[NDArray, NDArray]:
#     Build 3x3 affine matrix by composing four primitive matrices:
#
#       s     = rng.uniform(*scale_range)
#       theta = radians(rng.uniform(*rotation_range))
#       shx   = rng.uniform(*shear_range)
#       tx,ty = random offsets within map bounds
#
#       S  = [[s,   0,  0],   uniform scale
#             [0,   s,  0],
#             [0,   0,  1]]
#
#       R  = [[cos, -sin, 0],  rotation
#             [sin,  cos, 0],
#             [0,    0,   1]]
#
#       Sh = [[1, shx, 0],    horizontal shear
#             [0,  1,  0],
#             [0,  0,  1]]
#
#       T  = [[1, 0, tx],     translation
#             [0, 1, ty],
#             [0, 0,  1]]
#
#       M  = T @ R @ Sh @ S   <-- ORDER MATTERS
#
#     Compute ground truth:
#       center_h       = [W/2, H/2, 1]
#       ground_truth_h = M @ center_h
#       ground_truth   = ground_truth_h[:2]
#
#     Return (M, ground_truth).
#
# VALIDATION TIP:
#   After implementing, verify that applying M to the UAV image center
#   manually gives ground_truth_px before running the full evaluation.
#   Getting the composition order wrong is the most common bug here.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from evaluation.base import DatasetGenerator, SyntheticFrame


class SyntheticDatasetGenerator(DatasetGenerator):
    def __init__(self, map_img: NDArray, seed: int = 42) -> None:
        # TODO: implement
        ...

    def generate(self, n: int, **kwargs) -> list[SyntheticFrame]:
        # TODO: implement
        ...

    def __random_affine(
        self,
        scale_range: tuple,
        rotation_range: tuple,
        shear_range: tuple,
    ) -> tuple[NDArray, NDArray]:
        # TODO: implement — this is the LA core of Person C's work
        ...
