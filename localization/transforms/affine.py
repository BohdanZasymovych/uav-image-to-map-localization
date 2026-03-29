# =============================================================================
# localization/transforms/affine.py
# -----------------------------------------------------------------------------
# OWNER: Person A.
#
# PURPOSE:
#   Concrete 6-DOF affine transformation model.
#   This file contains the core linear algebra of the project.
#
# IMPLEMENT:
#   class AffineModel(TransformationModel)
#
#   min_points -> 3
#   dof        -> 6
#
#   estimate(src_pts, dst_pts) -> NDArray (3x3):
#     1. Validate N >= 3, raise ValueError otherwise.
#     2. Build overdetermined system Av = b where:
#          v = [a1, a2, tx, a3, a4, ty]^T
#          For each correspondence i with src=(x,y), dst=(X,Y):
#            row 2i:   [x, y, 1, 0, 0, 0]  -> X
#            row 2i+1: [0, 0, 0, x, y, 1]  -> Y
#          A has shape (2N, 6), b has shape (2N,).
#     3. Solve normal equations: v = np.linalg.solve(A.T @ A, A.T @ b)
#     4. Assemble and return the 3x3 matrix:
#          [[a1, a2, tx],
#           [a3, a4, ty],
#           [0,  0,  1 ]]
#
#   project(M, pts) -> NDArray (N, 2):
#     1. Convert pts to homogeneous: hpts = hstack([pts, ones((N, 1))])
#     2. result = (M @ hpts.T).T
#     3. Return result[:, :2]  — drop homogeneous coordinate (always 1)
#
# MATHEMATICAL REFERENCE:
#   Report equations (1) through (7).
#   The normal equations derivation is equations (4)-(7).
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class AffineModel(TransformationModel):
    # TODO: implement
    ...
