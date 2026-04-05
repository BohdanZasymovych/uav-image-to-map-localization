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
# pylint: disable=c-extension-no-member


class SIFTExtractor(FeatureExtractor):
    """SIFT-based feature extractor using OpenCV's SIFT implementation."""

    def __init__(self, ratio: float = 0.75, n_features: int = 0) -> None:
        """
        Initialize SIFT feature extractor.

        Parameters:
        ratio : float, default=0.75
            Lowe's ratio test threshold for filtering matches
        n_features : int, default=0
            Maximum number of keypoints to detect (0 = unlimited)
        """
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        if n_features < 0:
            raise ValueError(f"n_features must be >= 0, got {n_features}")

        self.ratio = ratio
        self.n_features = n_features
        self.detector = cv2.SIFT_create(nfeatures=n_features)

        index_params = dict(algorithm=1, trees=5)   # algorithm=1 -- FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params) # -- Flann KDTree
        # self.matcher = cv2.BFMatcher(cv2.NORM_L2) -- Brute Force

    def detect_and_compute(
        self,
        img: NDArray,
    ) -> tuple[list, NDArray]:
        """
        Detect SIFT keypoints and compute descriptors

        Parameters
        img : NDArray
            BGR image array of shape (H, W, 3)

        Returns
        keypoints : list[cv2.KeyPoint]
            List of detected keypoints
        descriptors : NDArray, shape (N, 128)
            SIFT descriptors as float32 array
        """
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
        """
        Match SIFT descriptors using Lowe's ratio test

        Parameters
        desc1 : NDArray, shape (N, 128)
            Descriptors from first image
        desc2 : NDArray, shape (M, 128)
            Descriptors from second image

        Returns
        matches : list[cv2.DMatch]
            Filtered list of good matches
        """
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)

        return [
            pair[0] for pair in knn_matches
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance
        ]
