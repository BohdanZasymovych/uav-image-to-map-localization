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
import cv2

from localization.features.base import FeatureExtractor


class SIFTExtractor(FeatureExtractor):
    def __init__(self, ratio: float = 0.75, n_features: int = 0) -> None:
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        if n_features < 0:
            raise ValueError(f"n_features must be >= 0, got {n_features}")

        self.ratio = ratio
        self.n_features = n_features
        self.detector = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def detect_and_compute(
        self,
        img: NDArray,
    ) -> tuple[list, NDArray]:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)

        return keypoints, descriptors

    def match(
        self,
        desc1: NDArray,
        desc2: NDArray,
    ) -> list:
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)

        return good_matches
