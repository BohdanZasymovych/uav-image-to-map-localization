from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from evaluation.metrics import EvaluationReport
from localization.result import MatchResult


class Visualizer:
    def plot_error_distribution(self, report: EvaluationReport):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(report.per_frame_errors, bins=20, color="#4C72B0", alpha=0.85)
        ax.axvline(report.rmse, color="#C44E52", linestyle="--", linewidth=2, label=f"RMSE: {report.rmse:.2f}px")
        ax.set_title("Per-frame Pixel Error Distribution")
        ax.set_xlabel("Pixel Error")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_runtime_distribution(self, report: EvaluationReport):
        fig, ax = plt.subplots(figsize=(7, 4))
        runtimes = np.asarray(report.per_frame_runtimes, dtype=np.float64)
        bins = min(20, max(1, runtimes.size))
        ax.hist(runtimes, bins=bins, color="#55A868", alpha=0.85)
        ax.set_title("Runtime Distribution")
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Count")
        ax.text(
            0.98,
            0.95,
            f"mean={report.mean_runtime_s:.4f}s\nstd={report.std_runtime_s:.4f}s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
        )
        fig.tight_layout()
        return fig

    def plot_inlier_ratio(self, report: EvaluationReport):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Mean Inlier Ratio"], [report.mean_inlier_ratio], color="#8172B2")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Ratio")
        ax.set_title("Inlier Ratio")
        ax.text(0, report.mean_inlier_ratio + 0.02, f"{report.mean_inlier_ratio:.3f}", ha="center")
        fig.tight_layout()
        return fig

    def draw_match_overlay(self, match_result: MatchResult) -> NDArray:
        overlay = match_result.match_image.copy()
        n_matches = int(match_result.src_pts.shape[0]) if match_result.src_pts.ndim > 0 else 0
        cv2.putText(
            overlay,
            f"Matches: {n_matches}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def save_report(self, report: EvaluationReport, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig_err = self.plot_error_distribution(report)
        fig_runtime = self.plot_runtime_distribution(report)
        fig_inlier = self.plot_inlier_ratio(report)

        fig_err.savefig(out / "error_distribution.png", dpi=150, bbox_inches="tight")
        fig_runtime.savefig(out / "runtime_distribution.png", dpi=150, bbox_inches="tight")
        fig_inlier.savefig(out / "inlier_ratio.png", dpi=150, bbox_inches="tight")
        plt.close(fig_err)
        plt.close(fig_runtime)
        plt.close(fig_inlier)

        print("Evaluation summary")
        print(f"n_frames          : {report.n_frames}")
        print(f"rmse              : {report.rmse:.4f} px")
        print(f"mean_runtime_s    : {report.mean_runtime_s:.6f} s")
        print(f"std_runtime_s     : {report.std_runtime_s:.6f} s")
        print(f"mean_inlier_ratio : {report.mean_inlier_ratio:.4f}")
