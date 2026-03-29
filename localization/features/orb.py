# =============================================================================
# localization/features/orb.py
# -----------------------------------------------------------------------------
# OWNER: Person B.
#
# PURPOSE:
#   ORB-based feature extractor. Fastest option; useful for timing benchmarks.
#
# IMPLEMENT:
#   class ORBExtractor(FeatureExtractor)
#
#   __init__(self, ratio: float = 0.75, n_features: int = 500)
#
#   detect_and_compute(img) -> (keypoints, descriptors):
#     - Convert BGR to grayscale.
#     - Use cv2.ORB_create(nfeatures=n_features).
#
#   match(desc1, desc2) -> list[cv2.DMatch]:
#     - Use cv2.BFMatcher(cv2.NORM_HAMMING).  <-- critical difference vs SIFT
#     - Apply Lowe's ratio test identically to SIFTExtractor.
#
# NOTE:
#   ORB uses binary descriptors (bit strings) — Hamming distance is correct.
#   Using NORM_L2 with ORB will silently produce wrong match distances.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.features.base import FeatureExtractor


class ORBExtractor(FeatureExtractor):
    # TODO: implement
    ...
