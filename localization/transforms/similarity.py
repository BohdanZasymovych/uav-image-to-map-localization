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
    DOF = 4
    MIN_POINTS = 2

    @property
    def min_points(self) -> int:
        """
        Minimum correspondences required to call estimate().
        RANSAC uses this as the random sample size per iteration.
        """
        return self.MIN_POINTS

    @property
    def dof(self) -> int:
        """
        Degrees of freedom of the model. Informational only.
        """
        return self.DOF

    def estimate(
        self,
        src_pts: NDArray,
        dst_pts: NDArray,
    ) -> NDArray:
        """
        Estimate the 3x3 transformation matrix from point correspondences.

        Parameters
        ----------
        src_pts : NDArray  shape (N, 2)  — UAV image coordinates
        dst_pts : NDArray  shape (N, 2)  — map image coordinates

        Returns
        -------
        M : NDArray  shape (3, 3)

        Raises
        ------
        ValueError            if N < min_points
        np.linalg.LinAlgError if the configuration is degenerate
        """
        n = src_pts.shape[0]

        if n < self.min_points:
            raise ValueError(f"Need at least {self.min_points} points, got {n}")

        A = np.empty((2*n, 4), dtype=np.float64)
        b = np.empty(2*n, dtype=float)

        for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
            x_src, y_src  = src
            x_dst, y_dst  = dst
            
            A[2*i] = [x_src, -y_src, 1, 0]
            A[2*i + 1] = [y_src, x_src, 0, 1]

            b[2*i] = x_dst
            b[2*i + 1] = y_dst
        
        solution_vector = np.linalg.solve(a=A.T @ A, b=A.T @ b)
        k, l, tx, ty = solution_vector

        transformation_matrix = np.array([[k, -l, tx],
                                         [l, k, ty],
                                         [0, 0, 1]])
        
        return transformation_matrix