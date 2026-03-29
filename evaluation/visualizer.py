# =============================================================================
# evaluation/visualizer.py
# -----------------------------------------------------------------------------
# OWNER: Person C.
#
# PURPOSE:
#   Produces all plots and visual outputs for the evaluation section.
#   Consumes EvaluationReport and MatchResult only — no pipeline logic here.
#
# IMPLEMENT:
#   class Visualizer:
#
#   plot_error_distribution(report: EvaluationReport) -> matplotlib.figure.Figure:
#     Histogram of per-frame pixel errors with RMSE marked as a vertical line.
#
#   plot_runtime_distribution(report: EvaluationReport) -> matplotlib.figure.Figure:
#     Histogram of per-frame runtimes with mean and std annotations.
#
#   plot_inlier_ratio(report: EvaluationReport) -> matplotlib.figure.Figure:
#     Bar or box plot of inlier ratios across all frames.
#
#   draw_match_overlay(match_result: MatchResult) -> NDArray:
#     Returns match_result.match_image (already computed by pipeline).
#     Optionally adds cv2 text overlays: inlier count, extractor name.
#
#   save_report(report: EvaluationReport, output_dir: str) -> None:
#     Saves all figures as PNG files to output_dir.
#     Prints a summary table to stdout.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from evaluation.metrics import EvaluationReport
from localization.result import MatchResult


class Visualizer:
    # TODO: implement all methods above
    ...
