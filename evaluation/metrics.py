# =============================================================================
# evaluation/metrics.py
# -----------------------------------------------------------------------------
# OWNER: Person C.
#
# PURPOSE:
#   Runs the localization pipeline over a generated dataset and computes
#   evaluation metrics: RMSE, per-frame pixel errors, and timing statistics.
#
# IMPLEMENT:
#   @dataclass EvaluationReport:
#     rmse              : float
#     per_frame_errors  : NDArray   shape (n_frames,)
#     mean_runtime_s    : float
#     std_runtime_s     : float
#     mean_inlier_ratio : float
#     n_frames          : int
#
#   class Evaluator:
#
#   __init__(self, pipeline: LocalizationPipeline,
#                  generator: DatasetGenerator,
#                  map_img: NDArray):
#
#   run(self, n_frames: int) -> EvaluationReport:
#     1. frames = generator.generate(n_frames)
#     2. For each frame:
#          _, result = pipeline.run(frame.uav_img, map_img)
#          error_i   = np.linalg.norm(result.position_px - frame.ground_truth_px)
#          collect error_i, result.runtime_s, inlier ratio
#     3. Compute:
#          RMSE              = sqrt(mean(error_i^2))
#          mean/std runtime
#          mean inlier ratio = mean(n_inliers / n_raw_matches)
#     4. Return EvaluationReport.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evaluation.base import DatasetGenerator
from localization.pipeline import LocalizationPipeline


@dataclass
class EvaluationReport:
    rmse: float
    per_frame_errors: NDArray
    mean_runtime_s: float
    std_runtime_s: float
    mean_inlier_ratio: float
    n_frames: int


class Evaluator:
    def __init__(
        self,
        pipeline: LocalizationPipeline,
        generator: DatasetGenerator,
        map_img: NDArray,
    ) -> None:
        # TODO: implement
        ...

    def run(self, n_frames: int) -> EvaluationReport:
        # TODO: implement
        ...
