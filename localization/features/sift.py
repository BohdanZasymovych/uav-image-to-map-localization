# =============================================================================
# localization/features/sift.py
# -----------------------------------------------------------------------------
# OWNER: Person B.
#
# PURPOSE:
#   SIFT-based feature extractor. The default and primary extractor.
#
# IMPLEMENT:
#   class SIFTExtractor(FeatureExtractor)
#
#   __init__(self, ratio: float = 0.75, n_features: int = 0):
#     ratio      — Lowe's ratio test threshold (0.0–1.0).
#     n_features — maximum keypoints to detect (0 = unlimited).
#
#   detect_and_compute(img) -> (keypoints, descriptors):
#     - Convert BGR to grayscale internally.
#     - Use cv2.SIFT_create(nfeatures=n_features).
#     - Return (kps, descs) from detectAndCompute().
#
#   match(desc1, desc2) -> list[cv2.DMatch]:
#     - Use cv2.BFMatcher(cv2.NORM_L2).
#     - Run knnMatch(k=2) to get two nearest neighbours per descriptor.
#     - Apply Lowe's ratio test: keep m if m.distance < ratio * n.distance.
#     - Return the list of passing DMatch objects.
#
# NOTE:
#   SIFT descriptors are 128-dimensional float vectors -> NORM_L2 is correct.
#   Do NOT use NORM_HAMMING here (that is only for binary descriptors).
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localization.features.base import FeatureExtractor


class SIFTExtractor(FeatureExtractor):
    # TODO: implement
    ...
