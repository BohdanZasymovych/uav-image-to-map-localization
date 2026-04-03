# =============================================================================
# evaluation/dataset.py
# -----------------------------------------------------------------------------
# OWNER: Person C.
#
# PURPOSE:
#   Concrete dataset generator. Produces synthetic UAV frames by applying
#   random affine transforms to a reference satellite map.
#   _random_affine() is the linear algebra contribution of Person C.
#
# IMPLEMENT:
#   class SyntheticDatasetGenerator(DatasetGenerator)
#
#   __init__(self, map_img: NDArray, seed: int = 42):
#     - Store map_img.
#     - Create self.rng = np.random.default_rng(seed) for reproducibility.
#
#   generate(self, n, scale_range=(0.4, 0.8),
#            rotation_range=(-30, 30),
#            shear_range=(-0.1, 0.1)) -> list[SyntheticFrame]:
#     For each of n frames:
#       1. Call _random_affine() -> M, ground_truth_px.
#       2. Warp map: cv2.warpAffine(map_img, M[:2], (W, H)).
#       3. Append SyntheticFrame(warped, ground_truth_px, M).
#
#   _random_affine(scale_range, rotation_range, shear_range)
#       -> tuple[NDArray, NDArray]:
#     Build 3x3 affine matrix by composing four primitive matrices:
#
#       s     = rng.uniform(*scale_range)
#       theta = radians(rng.uniform(*rotation_range))
#       shx   = rng.uniform(*shear_range)
#       tx,ty = random offsets within map bounds
#
#       S  = [[s,   0,  0],   uniform scale
#             [0,   s,  0],
#             [0,   0,  1]]
#
#       R  = [[cos, -sin, 0],  rotation
#             [sin,  cos, 0],
#             [0,    0,   1]]
#
#       Sh = [[1, shx, 0],    horizontal shear
#             [0,  1,  0],
#             [0,  0,  1]]
#
#       T  = [[1, 0, tx],     translation
#             [0, 1, ty],
#             [0, 0,  1]]
#
#       M  = T @ R @ Sh @ S   <-- ORDER MATTERS
#
#     Compute ground truth:
#       center_h       = [W/2, H/2, 1]
#       ground_truth_h = M @ center_h
#       ground_truth   = ground_truth_h[:2]
#
#     Return (M, ground_truth).
#
# VALIDATION TIP:
#   After implementing, verify that applying M to the UAV image center
#   manually gives ground_truth_px before running the full evaluation.
#   Getting the composition order wrong is the most common bug here.
# =============================================================================

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

        scale_range = kwargs.get("scale_range", (0.4, 0.8))
        rotation_range = kwargs.get("rotation_range", (-30, 30))
        shear_range = kwargs.get("shear_range", (-0.1, 0.1))
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
                borderMode=cv2.BORDER_REFLECT101,
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

        # For cv2.warpAffine, M maps destination coordinates to source coordinates.
        return T_back @ R @ Sh @ S @ T_to_center
