# =============================================================================
# localization/pipeline.py
# -----------------------------------------------------------------------------
# OWNER: Person B.
#
# PURPOSE:
#   Top-level orchestrator that wires all components together into one call.
#   Accepts abstract interfaces — must never import concrete classes directly.
#
# IMPLEMENT:
#   class LocalizationPipeline
#
#   __init__(self, extractor: FeatureExtractor,
#                  model: TransformationModel,
#                  epsilon: float = 3.0,
#                  confidence: float = 0.99,
#                  max_iterations: int = 2000,
#                  georeferencer: Georeferencer | None = None):
#     - Constructs a RANSAC instance internally from model + hyperparameters.
#     - Stores extractor, model, and georeferencer as attributes.
#
#   run(self, uav_img: NDArray,
#             map_img: NDArray) -> tuple[MatchResult, LocalizationResult]:
#
#     Steps (in order):
#       1.  t0 = time.perf_counter()
#       2.  extractor.detect_and_compute(uav_img) -> kp1, desc1
#       3.  extractor.detect_and_compute(map_img) -> kp2, desc2
#       4.  extractor.match(desc1, desc2)          -> good_matches
#       5.  Extract src_pts (N, 2) from kp1 via match.queryIdx
#           Extract dst_pts (N, 2) from kp2 via match.trainIdx
#       6.  ransac.run(src_pts, dst_pts)           -> M, inlier_mask, iters
#       7.  center = np.array([[W/2, H/2]])
#           position_px = model.project(M, center)[0]
#       8.  position_geo = georeferencer.pixel_to_geo(*position_px)
#                          if georeferencer else None
#       9.  match_image = cv2.drawMatches(inlier subset only)
#      10.  runtime_s = time.perf_counter() - t0
#      11.  Construct and return (MatchResult, LocalizationResult).
#
# CONSUMED BY:
#   evaluation/metrics.py — calls run() in a loop over dataset frames.
#   app/ui.py             — calls run() on user-uploaded images.
# =============================================================================

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from localization.features.base import FeatureExtractor
from localization.georeferencing import Georeferencer
from localization.ransac import RANSAC
from localization.result import LocalizationResult, MatchResult
from localization.transforms.base import TransformationModel


class LocalizationPipeline:
    def __init__(
        self,
        extractor: FeatureExtractor,
        model: TransformationModel,
        epsilon: float = 3.0,
        confidence: float = 0.99,
        max_iterations: int = 2000,
        georeferencer: Optional[Georeferencer] = None,
    ) -> None:
        # TODO: implement
        ...

    def run(
        self,
        uav_img: NDArray,
        map_img: NDArray,
    ) -> tuple[MatchResult, LocalizationResult]:
        # TODO: implement
        ...
