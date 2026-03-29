# =============================================================================
# localization/transforms/projective.py
# -----------------------------------------------------------------------------
# OWNER: Person A.
#
# PURPOSE:
#   Concrete 8-DOF projective (homography) transformation model.
#   Estimated via the Direct Linear Transform (DLT) algorithm using SVD.
#
# IMPLEMENT:
#   class ProjectiveModel(TransformationModel)
#
#   min_points -> 4
#   dof        -> 8
#
#   estimate(src_pts, dst_pts) -> NDArray (3x3):
#     1. Validate N >= 4, raise ValueError otherwise.
#     2. Build constraint matrix A of shape (2N, 9).
#        For each correspondence i with src=(x,y), dst=(X,Y):
#          row 2i:   [-x, -y, -1,  0,  0,  0, X*x, X*y, X]
#          row 2i+1: [ 0,  0,  0, -x, -y, -1, Y*x, Y*y, Y]
#     3. Decompose: _, _, Vt = np.linalg.svd(A)
#     4. Solution h = Vt[-1]  (eigenvector of smallest singular value)
#     5. Return h.reshape(3, 3)
#
#   project(M, pts) -> NDArray (N, 2):
#     1. Convert pts to homogeneous.
#     2. result = (M @ hpts.T).T  — shape (N, 3)
#     3. Return result[:, :2] / result[:, 2:3]  — perspective divide by w
#        (this is what distinguishes projective from affine projection)
#
# MATHEMATICAL REFERENCE:
#   Hartley & Zisserman "Multiple View Geometry", Chapter 4 (DLT algorithm).
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class ProjectiveModel(TransformationModel):
    # TODO: implement
    ...
