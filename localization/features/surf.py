# =============================================================================
# localization/features/surf.py
# -----------------------------------------------------------------------------
# OWNER: Person B.
#
# PURPOSE:
#   SURF-based feature extractor. Faster than SIFT with comparable quality.
#
# IMPLEMENT:
#   class SURFExtractor(FeatureExtractor)
#
#   __init__(self, ratio: float = 0.75, hessian_threshold: float = 400)
#
#   detect_and_compute(img) -> (keypoints, descriptors):
#     - SURF requires opencv-contrib-python (not base opencv-python).
#     - Use cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold).
#     - Raise ImportError with a clear message if contrib is unavailable,
#       so the UI can fall back to SIFT gracefully.
#
#   match(desc1, desc2) -> list[cv2.DMatch]:
#     - Use cv2.BFMatcher(cv2.NORM_L2) — SURF is float-valued like SIFT.
#     - Apply Lowe's ratio test.
#
# NOTE:
#   SURF is patented and disabled in standard OpenCV builds.
#   Wrap the cv2.xfeatures2d import in try/except at the top of the file.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.features.base import FeatureExtractor


class SURFExtractor(FeatureExtractor):
    # TODO: implement
    ...
