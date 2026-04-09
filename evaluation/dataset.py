from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from evaluation.base import DatasetGenerator, SyntheticFrame


class SyntheticDatasetGenerator(DatasetGenerator):
    def __init__(self, map_img: NDArray, seed: int = 42) -> None:
        self.map_img = map_img
        self.rng = np.random.default_rng(seed)
        

    def generate(self, n: int, **kwargs) -> list[SyntheticFrame]:
        frames: list[SyntheticFrame] = []
        h, w = self.map_img.shape[:2]

        scale_range = kwargs.get("scale_range", (0.9, 1.1))
        rotation_range = kwargs.get("rotation_range", (-15, 15))
        shear_range = kwargs.get("shear_range", (-0.03, 0.03))
        crop_size = kwargs.get("crop_size", (100, 100))
        crop_h = max(1, min(int(crop_size[0]), h))
        crop_w = max(1, min(int(crop_size[1]), w))

        for _ in range(n):
            start_x, start_y = self._random_crop_origin(h, w, crop_h, crop_w)
            patch = self.map_img[start_y : start_y + crop_h, start_x : start_x + crop_w]

            M_local = self._random_affine(
                scale_range,
                rotation_range,
                shear_range,
                crop_w,
                crop_h,
            )
            uav_img = cv2.warpAffine(
                patch,
                M_local[:2],
                (crop_w, crop_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            T_origin = np.array(
                [
                    [1.0, 0.0, float(start_x)],
                    [0.0, 1.0, float(start_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            M = T_origin @ M_local
            ground_truth_px = np.array(
                [start_x + crop_w / 2.0, start_y + crop_h / 2.0],
                dtype=np.float64,
            )

            frame = SyntheticFrame(
                uav_img=uav_img,
                ground_truth_px=ground_truth_px,
                transform_matrix=M,
            )
            frames.append(frame)

        return frames

    def _random_crop_origin(
        self,
        h: int,
        w: int,
        crop_h: int,
        crop_w: int,
    ) -> tuple[int, int]:
        max_y = h - crop_h
        max_x = w - crop_w
        start_y = int(self.rng.integers(0, max_y + 1))
        start_x = int(self.rng.integers(0, max_x + 1))
        return start_x, start_y

    def _random_affine(
        self,
        scale_range: tuple,
        rotation_range: tuple,
        shear_range: tuple,
        width: int,
        height: int,
    ) -> NDArray:
        w = float(width)
        h = float(height)

        s = self.rng.uniform(*scale_range)
        theta = np.radians(self.rng.uniform(*rotation_range))
        shx = self.rng.uniform(*shear_range)

        S = np.array([
            [s, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        Sh = np.array([
            [1.0, shx, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        cx = w / 2.0
        cy = h / 2.0
        T_to_center = np.array([
            [1.0, 0.0, -cx],
            [0.0, 1.0, -cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        T_back = np.array([
            [1.0, 0.0, cx],
            [0.0, 1.0, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


        return T_back @ R @ Sh @ S @ T_to_center
