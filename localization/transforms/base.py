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

        Returns
        -------
        errors : NDArray  shape (N,)
        """
        projected = self.project(M, src_pts)
        return np.linalg.norm(projected - dst_pts, axis=1)
