from __future__ import annotations

from typing import Any

import cv2


class MapOverlayRenderer:
    def __init__(self) -> None:
        self.__rect_color = (0, 255, 255)
        self.__center_color = (0, 0, 255)

    def render(
        self,
        map_img: Any,
        position_px_x: float,
        position_px_y: float,
        bbox_width: int,
        bbox_height: int,
    ) -> tuple[Any, tuple[int, int], tuple[int, int]]:
        map_h, map_w = map_img.shape[:2]

        cx = int(round(position_px_x))
        cy = int(round(position_px_y))
        half_w = bbox_width // 2
        half_h = bbox_height // 2

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(map_w - 1, cx + half_w)
        y2 = min(map_h - 1, cy + half_h)

        overlay = map_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.__rect_color, 2)
        cv2.circle(overlay, (cx, cy), 4, self.__center_color, -1)
        cv2.putText(
            overlay,
            f"Estimated center: ({cx}, {cy})",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            self.__rect_color,
            2,
            cv2.LINE_AA,
        )

        return overlay, (x1, y1), (x2, y2)
