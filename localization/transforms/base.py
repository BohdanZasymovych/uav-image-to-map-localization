# =============================================================================
# localization/transforms/base.py
# -----------------------------------------------------------------------------
# OWNER: Person A.
#
# PURPOSE:
#   Abstract base class for all geometric transformation models.
#   This is the most important interface in the project — RANSAC, the
#   pipeline, and the evaluator all depend exclusively on it.
#
# CONTAINS:
#   - TransformationModel (ABC)
#       min_points  (property) -> int
#       dof         (property) -> int
#       estimate(src_pts, dst_pts) -> NDArray  [3x3 matrix]
#       project(M, pts)            -> NDArray  [(N, 2)]
#       reprojection_error(...)    -> NDArray  [(N,)]  NOT abstract,
#                                                  delegates to project()
#
# IMPLEMENTATIONS (each in its own file, same package):
#   localization/transforms/affine.py      — AffineModel     (6-DOF)
#     estimate(): build (2N x 6) system, solve normal equations (A^T A)v = A^T b
#   localization/transforms/projective.py  — ProjectiveModel (8-DOF)
#     estimate(): Direct Linear Transform (DLT) solved via SVD
#   localization/transforms/similarity.py  — SimilarityModel (4-DOF)
#
# IMPLEMENTATION NOTES:
#   - estimate() must raise ValueError if len(src_pts) < min_points.
#   - estimate() must raise np.linalg.LinAlgError on degenerate input.
#   - project() for affine:     drop homogeneous coordinate after multiply.
#   - project() for projective: divide by homogeneous coordinate (w).
#   - reprojection_error() is concrete and must NOT be overridden;
#     RANSAC calls it uniformly across all model types.
#
# CONSUMED BY:
#   localization/ransac.py   — calls min_points, estimate(), reprojection_error()
#   localization/pipeline.py — calls project() for final coordinate projection
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class TransformationModel(ABC):
    """
    Abstracts a geometric transformation between two image coordinate spaces.

    RANSAC depends on this interface exclusively and has no knowledge of
    whether the underlying model is affine, projective, or otherwise.
    """

    @property
    @abstractmethod
    def min_points(self) -> int:
        """
        Minimum correspondences required to call estimate().
        RANSAC uses this as the random sample size per iteration.
        Affine: 3 — Projective: 4 — Similarity: 2.
        """
        ...

    @property
    @abstractmethod
    def dof(self) -> int:
        """
        Degrees of freedom of the model. Informational only.
        Affine: 6 — Projective: 8 — Similarity: 4.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    def reprojection_error(
        self,
        M: NDArray,
        src_pts: NDArray,
        dst_pts: NDArray,
    ) -> NDArray:
        """
        Per-point Euclidean reprojection error.

        Delegates to project() so RANSAC needs no model-specific error logic.

        Returns
        -------
        errors : NDArray  shape (N,)
        """
        projected = self.project(M, src_pts)
        return np.linalg.norm(projected - dst_pts, axis=1)
