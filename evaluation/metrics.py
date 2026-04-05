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
import logging

import numpy as np
from numpy.typing import NDArray

from evaluation.base import DatasetGenerator
from localization.pipeline import LocalizationPipeline


@dataclass
class EvaluationReport:
    rmse: float
    per_frame_errors: NDArray
    per_frame_runtimes: NDArray
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
        self.pipeline = pipeline
        self.generator = generator
        self.map_img = map_img

    def run(self, n_frames: int) -> EvaluationReport:
        frames = self.generator.generate(n_frames)
        logger = logging.getLogger(__name__)

        errors: list[float] = []
        runtimes: list[float] = []
        inlier_ratios: list[float] = []
        n_failed = 0

        for i, frame in enumerate(frames):
            try:
                _, result = self.pipeline.run(frame.uav_img, self.map_img)
            except Exception as exc:
                n_failed += 1
                logger.warning(
                    "Skipping frame %d/%d due to pipeline error: %s",
                    i + 1,
                    len(frames),
                    exc,
                )
                continue

            error_i = float(np.linalg.norm(result.position_px - frame.ground_truth_px))
            errors.append(error_i)
            runtimes.append(float(result.runtime_s))

            if result.n_raw_matches > 0:
                inlier_ratios.append(float(result.n_inliers / result.n_raw_matches))
            else:
                inlier_ratios.append(0.0)

        per_frame_errors = np.asarray(errors, dtype=np.float64)
        runtimes_arr = np.asarray(runtimes, dtype=np.float64)
        inlier_arr = np.asarray(inlier_ratios, dtype=np.float64)

        if per_frame_errors.size == 0:
            rmse = 0.0
            mean_runtime_s = 0.0
            std_runtime_s = 0.0
            mean_inlier_ratio = 0.0
        else:
            rmse = float(np.sqrt(np.mean(per_frame_errors**2)))
            mean_runtime_s = float(np.mean(runtimes_arr))
            std_runtime_s = float(np.std(runtimes_arr))
            mean_inlier_ratio = float(np.mean(inlier_arr))

        if n_failed > 0:
            logger.warning(
                "Evaluation completed with skipped frames: success=%d failed=%d",
                int(per_frame_errors.size),
                n_failed,
            )

        return EvaluationReport(
            rmse=rmse,
            per_frame_errors=per_frame_errors,
            per_frame_runtimes=runtimes_arr,
            mean_runtime_s=mean_runtime_s,
            std_runtime_s=std_runtime_s,
            mean_inlier_ratio=mean_inlier_ratio,
            n_frames=int(per_frame_errors.size),
        )
