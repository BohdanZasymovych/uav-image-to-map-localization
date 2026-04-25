from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

from localization.transforms.base import TransformationModel

logger = logging.getLogger(__name__)

class RANSAC:
    def __init__(
        self,
        model: TransformationModel,
        epsilon: float = 3.0,
        confidence: float = 0.99,
        max_iterations: int = 2000,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.confidence = confidence
        self.max_iterations = max_iterations

    def run(
        self,
        src_pts: NDArray,
        dst_pts: NDArray,
        ) -> tuple[NDArray, NDArray, int]:
        if len(src_pts) != len(dst_pts):
            raise ValueError("src_pts and dst_pts must have the same number of points")

        n_total = len(src_pts)
        min_points = self.model.min_points

        logger.debug("RANSAC: Starting with %d correspondences", n_total)

        if n_total < min_points:
            raise ValueError(
                f"Need at least {min_points} correspondences, got {n_total}"
            )

        best_mask = np.zeros(n_total, dtype=bool)
        best_inlier_count = 0
        n_iters = self.max_iterations
        iterations_done = 0

        rng = np.random.default_rng()

        i = 0
        while i < n_iters:
            iterations_done += 1
            i += 1

            sample_idx = rng.choice(n_total, size=min_points, replace=False)

            try:
                M = self.model.estimate(src_pts[sample_idx], dst_pts[sample_idx])
            except np.linalg.LinAlgError:
                continue

            if not np.all(np.isfinite(M)):
                continue

            errors = self.model.reprojection_error(M, src_pts, dst_pts)

            if errors.shape[0] != n_total:
                raise ValueError("reprojection_error must return one error per point")

            mask = errors < self.epsilon
            n_inliers = int(np.sum(mask))

            if n_inliers > best_inlier_count:
                best_mask = mask.copy()
                best_inlier_count = n_inliers
                old_n_iters = n_iters
                n_iters = min(self.__adaptive_iters(n_inliers, n_total), self.max_iterations)
                if n_iters < old_n_iters:
                    logger.debug("RANSAC iteration %d: found %d inliers, updated max_iters to %d", 
                                 i, n_inliers, n_iters)

        logger.info("RANSAC found %d inliers in %d iterations", best_inlier_count, iterations_done)

        if best_inlier_count < min_points:
            raise RuntimeError("RANSAC failed to find a valid model")

        try:
            best_M = self.model.estimate(src_pts[best_mask], dst_pts[best_mask])
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to re-estimate model on final inliers") from e

        return best_M, best_mask, iterations_done

    def __adaptive_iters(self, n_inliers: int, n_total: int) -> int:
        w = n_inliers / n_total
        if w == 0:
            return self.max_iterations
        p_all_inliers = w ** self.model.min_points
        if p_all_inliers >= 1.0:
            return 1
        denom = np.log(1.0 - p_all_inliers)
        if np.isclose(denom, 0.0):
            return self.max_iterations
        n = int(np.ceil(np.log(1.0 - self.confidence) / denom))
        return min(max(n, 1), self.max_iterations)
       
