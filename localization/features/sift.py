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
        max_dim: int = 2048,
    ) -> tuple[list, NDArray]:
        """..."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        h, w = gray.shape
        scale = 1.0

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)

        if scale != 1.0:
            for kp in keypoints:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                kp.size = kp.size / scale

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
