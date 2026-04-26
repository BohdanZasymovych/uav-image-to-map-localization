from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class ProjectiveModel(TransformationModel):
    DOF = 8
    MIN_POINTS = 4

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
        Estimate the 3x3 transformation matrix from point correspondences using SVD
        with Hartley's data normalization for numerical stability.
        """
        n = src_pts.shape[0]

        if n < self.min_points:
            raise ValueError(f"Need at least {self.min_points} points, got {n}")

        def normalize_points(pts: NDArray) -> tuple[NDArray, NDArray]:
            centroid = np.mean(pts, axis=0)
            shifted = pts - centroid
            mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
            
            scale = np.sqrt(2) / (mean_dist + 1e-16)
            
            T = np.array([
                [scale, 0,     -scale * centroid[0]],
                [0,     scale, -scale * centroid[1]],
                [0,     0,     1                   ]
            ], dtype=np.float64)
            
            pts_norm = shifted * scale
            return pts_norm, T

        src_norm, T_src = normalize_points(src_pts)
        dst_norm, T_dst = normalize_points(dst_pts)

        A = np.zeros((2 * n, 9), dtype=np.float64)
        for i in range(n):
            x, y = src_norm[i]
            X, Y = dst_norm[i]
            A[2 * i]     = [x, y, 1, 0, 0, 0, -X * x, -X * y, -X]
            A[2 * i + 1] = [0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y]

        _, _, Vh = np.linalg.svd(A)
        h = Vh[-1, :]
        H_norm = h.reshape(3, 3)

        H = np.linalg.inv(T_dst) @ H_norm @ T_src

        if np.abs(H[2, 2]) > 1e-10:
            H /= H[2, 2]

        return H

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

        pts_hom = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])
        
        pts_projected_hom = pts_hom @ M.T
        
        w = pts_projected_hom[:, [2]]
        
        return pts_projected_hom[:, :2] / (w + 1e-16)
