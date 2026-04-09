from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class FeatureExtractor(ABC):
    """
    Abstracts keypoint detection, descriptor computation, and match filtering.

    Implementations are responsible for:
      - Converting images to the format expected by their underlying detector.
      - Choosing the correct distance norm for their descriptor type.
      - Applying ratio-test or other filtering internally inside match().

    All methods receive raw BGR numpy arrays as loaded by OpenCV.
    """

    @abstractmethod
    def detect_and_compute(
        self,
        img: NDArray,
    ) -> tuple[list, NDArray]:
        """
        Detect keypoints and compute descriptors for a single image.

        Parameters
        ----------
        img : NDArray
            BGR image array of shape (H, W, 3).

        Returns
        -------
        keypoints : list[cv2.KeyPoint]
        descriptors : NDArray  shape (N, D)
        """
        ...

    @abstractmethod
    def match(
        self,
        desc1: NDArray,
        desc2: NDArray,
    ) -> list:
        """
        Match descriptors from two images and return filtered correspondences.

        All filtering (ratio test, cross-check, etc.) is encapsulated here.
        Callers receive only matches deemed good.

        Parameters
        ----------
        desc1 : NDArray  shape (N, D)
        desc2 : NDArray  shape (M, D)

        Returns
        -------
        matches : list[cv2.DMatch]
        """
        ...
