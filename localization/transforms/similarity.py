# =============================================================================
# localization/transforms/similarity.py
# -----------------------------------------------------------------------------
# OWNER: Person A.
#
# PURPOSE:
#   Concrete 4-DOF similarity transformation model.
#   Handles translation, rotation, and uniform scaling only (no shear).
#   Simpler than affine; useful as a baseline or for near-nadir imagery.
#
# IMPLEMENT:
#   class SimilarityModel(TransformationModel)
#
#   min_points -> 2
#   dof        -> 4
#
#   estimate(src_pts, dst_pts) -> NDArray (3x3):
#     Parameterisation: [[a, -b, tx], [b, a, ty], [0, 0, 1]]
#     where a = s*cos(theta), b = s*sin(theta).
#     Build (2N x 4) system and solve via normal equations, same pattern
#     as AffineModel but with reduced parameter vector [a, b, tx, ty].
#
#   project(M, pts) -> NDArray (N, 2):
#     Identical to AffineModel — drop homogeneous coordinate, no perspective
#     divide needed because the last row is always [0, 0, 1].
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class SimilarityModel(TransformationModel):
    # TODO: implement
    ...
