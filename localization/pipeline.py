from __future__ import annotations

import numpy as np
from typing import Optional

from time import perf_counter
from numpy.typing import NDArray
import cv2

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
        self.extractor = extractor
        self.model = model
        self.epsilon = epsilon
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.georeferencer = georeferencer
        self.ransac = RANSAC(
            model=self.model,
            epsilon=self.epsilon,
            confidence=self.confidence,
            max_iterations=self.max_iterations,
        )

    def run(
        self,
        uav_img: NDArray,
        map_img: NDArray,
    ) -> tuple[MatchResult, LocalizationResult]:
        time_start = perf_counter()

        uav_keypoints, uav_descriptors = self.extractor.detect_and_compute(uav_img)
        map_keypoints, map_descriptors = self.extractor.detect_and_compute(map_img)

        matches = self.extractor.match(uav_descriptors, map_descriptors)

        src_pts = np.array([uav_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.array([map_keypoints[m.trainIdx].pt for m in matches])

        M, best_inlier_mask, iterations_done = self.ransac.run(
            src_pts=src_pts,
            dst_pts=dst_pts,
        )

        h, w = uav_img.shape[:2]
        uav_img_center = np.array([[w / 2, h / 2]])

        position_px = self.model.project(M, uav_img_center)[0]

        position_geo = None
        if self.georeferencer is not None:
            position_geo = self.georeferencer.pixel_to_geo(*position_px)

        time_end = perf_counter()

        execution_time = time_end - time_start

        raw_match_image = cv2.drawMatches(
            uav_img,
            uav_keypoints,
            map_img,
            map_keypoints,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        inlier_matches = [
            match for match, is_inlier in zip(matches, best_inlier_mask) if is_inlier
        ]

        match_image = cv2.drawMatches(
            uav_img,
            uav_keypoints,
            map_img,
            map_keypoints,
            inlier_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        match_result = MatchResult(
            src_pts=src_pts,
            dst_pts=dst_pts,
            raw_match_image=raw_match_image,
            match_image=match_image,
        )

        localization_result = LocalizationResult(
            transform_matrix=M,
            inlier_mask=best_inlier_mask,
            position_px=position_px,
            position_geo=position_geo,
            n_raw_matches=len(matches),
            n_inliers=int(np.count_nonzero(best_inlier_mask)),
            ransac_iterations=iterations_done,
            runtime_s=execution_time,
        )

        return match_result, localization_result
