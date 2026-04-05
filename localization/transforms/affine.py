from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class AffineModel(TransformationModel):
    DOF = 6
    MIN_POINTS = 3

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
        Degrees of freedom of the model.
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

        A = np.empty((2*n, 6), dtype=np.float64)
        b = np.empty(2*n, dtype=float)

        for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
            x_src, y_src  = src
            x_dst, y_dst  = dst
            
            A[2*i] = [x_src, y_src, 1, 0, 0, 0]
            A[2*i + 1] = [0, 0, 0, x_src, y_src, 1]

            b[2*i] = x_dst
            b[2*i + 1] = y_dst
        
        solution_vector = np.linalg.solve(a=A.T @ A, b=A.T @ b)
        a_00, a_01, tx, a_10, a_11, ty, = solution_vector

        transformation_matrix = np.array([[a_00, a_01, tx],
                                         [a_10, a_11, ty],
                                         [0, 0, 1]])
        
        return transformation_matrix
    
    def project(
        self,
        M: NDArray,
        pts: NDArray,
    ) -> NDArray:
        """
        Apply transformation matrix M to an array of 2D points.

        Parameters
        ----------
        M   : NDArray  shape (3, 3)
        pts : NDArray  shape (N, 2)

        Returns
        -------
        projected : NDArray  shape (N, 2)
        """
        n = pts.shape[0]
        pts = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])
        pts_projected = pts @ M.T
        return pts_projected[:, :2]

