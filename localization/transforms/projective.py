from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel


class ProjectiveModel(TransformationModel):
    DOF = 8
    MIN_POINTS = 4

    @property
    def min_points(self) -> int:
        return self.MIN_POINTS

    @property
    def dof(self) -> int:
        return self.DOF

    def estimate(
        self,
        src_pts: NDArray,
        dst_pts: NDArray,
    ) -> NDArray:
        n = src_pts.shape[0]
        if n < self.min_points:
            raise ValueError(f"Need at least {self.min_points} points, got {n}")

        src_norm, t_src = self.__normalize_points(src_pts)
        dst_norm, t_dst = self.__normalize_points(dst_pts)

        a = np.zeros((2 * n, 9), dtype=np.float64)
        for i, (src, dst) in enumerate(zip(src_norm, dst_norm)):
            x, y = float(src[0]), float(src[1])
            u, v = float(dst[0]), float(dst[1])

            a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
            a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

        _, _, vh = np.linalg.svd(a)
        h = vh[-1, :]
        h_norm = h.reshape(3, 3)

        m = np.linalg.inv(t_dst) @ h_norm @ t_src
        if np.isclose(m[2, 2], 0.0):
            raise np.linalg.LinAlgError("Degenerate homography normalization")
        m = m / m[2, 2]
        return m

    def project(
        self,
        M: NDArray,
        pts: NDArray,
    ) -> NDArray:
        n = pts.shape[0]
        pts_h = np.hstack([pts.astype(np.float64), np.ones((n, 1), dtype=np.float64)])
        projected_h = pts_h @ M.T
        w = projected_h[:, 2:3]

        if np.any(np.isclose(w, 0.0)):
            raise np.linalg.LinAlgError("Point projected to infinity")

        projected = projected_h[:, :2] / w
        return projected

    def __normalize_points(self, pts: NDArray) -> tuple[NDArray, NDArray]:
        pts = pts.astype(np.float64)
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        distances = np.linalg.norm(centered, axis=1)
        mean_dist = float(np.mean(distances))
        if np.isclose(mean_dist, 0.0):
            raise np.linalg.LinAlgError("Degenerate point set for homography")

        scale = np.sqrt(2.0) / mean_dist
        t = np.array(
            [
                [scale, 0.0, -scale * centroid[0]],
                [0.0, scale, -scale * centroid[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        pts_norm_h = pts_h @ t.T
        return pts_norm_h[:, :2], t
