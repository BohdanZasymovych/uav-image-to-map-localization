from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray


@dataclass
class MatchResult:
    """
    Intermediate output of the feature matching stage.

    Produced by LocalizationPipeline before RANSAC runs.
    Consumed by the evaluation visualizer and the UI match overlay.
    """

    src_pts: NDArray
    """Matched keypoint coordinates in the UAV image, shape (N, 2)."""

    dst_pts: NDArray
    """Matched keypoint coordinates in the map image, shape (N, 2)."""

    raw_match_image: NDArray
    """
    BGR image produced by cv2.drawMatches showing all matched correspondences
    before RANSAC filtering.
    Shape (H, W, 3). Displayed directly in the UI and saved in evaluation.
    """

    match_image: NDArray
    """
    BGR image produced by cv2.drawMatches showing inlier correspondences
    after RANSAC filtering.
    Shape (H, W, 3). Displayed directly in the UI and saved in evaluation.
    """


@dataclass
class LocalizationResult:
    """
    Final output of a single pipeline run.

    Flows from LocalizationPipeline -> Evaluator -> App.
    """

    transform_matrix: NDArray
    """Estimated 3x3 transformation matrix (affine or projective)."""

    inlier_mask: NDArray
    """Boolean mask of shape (N,) marking RANSAC inliers over all matches."""

    position_px: NDArray
    """
    Estimated UAV center position in map pixel coordinates, shape (2,).
    This is the primary output of the localization pipeline.
    """

    n_raw_matches: int
    """Number of matches after Lowe's ratio test, before RANSAC."""

    n_inliers: int
    """Number of RANSAC inliers used for the final matrix estimation."""

    ransac_iterations: int
    """Actual number of RANSAC iterations executed (adaptive)."""

    runtime_s: float
    """Wall-clock time for the full pipeline.run() call in seconds."""
