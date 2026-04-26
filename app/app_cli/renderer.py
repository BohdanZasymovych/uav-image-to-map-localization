from __future__ import annotations

from typing import Any
import numpy as np
import cv2


class MapOverlayRenderer:
    def __init__(self) -> None:
        self.__rect_color = (0, 255, 255)
        self.__center_color = (0, 0, 255)

    def render(
        self,
        map_img: Any,
        transform_matrix: np.ndarray,
        uav_width: int,
        uav_height: int,
    ) -> tuple[Any, tuple[int, int], tuple[int, int]]:
        """
        Renders the estimated UAV position and bounding box onto the map image.
        
        Uses the transformation matrix to project the four corners of the UAV image
        onto the map coordinate system, accounting for rotation, scale, and perspective.
        """
        map_h, map_w = map_img.shape[:2]

        # Define corners of the UAV image in its own pixel coordinates
        uav_corners = np.array([
            [0, 0],
            [uav_width, 0],
            [uav_width, uav_height],
            [0, uav_height],
            [uav_width // 2, uav_height // 2] # Center point
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Project corners to map coordinates
        map_corners = cv2.perspectiveTransform(uav_corners, transform_matrix)
        map_corners = map_corners.reshape(-1, 2)
        
        # Extract corners and center
        poly_pts = map_corners[:4].astype(np.int32)
        center_pt = map_corners[4]
        cx, cy = int(round(center_pt[0])), int(round(center_pt[1]))

        overlay = map_img.copy()
        
        # Draw projected polygon (bounding box)
        cv2.polylines(overlay, [poly_pts], isClosed=True, color=self.__rect_color, thickness=3)
        
        # Draw center point
        cv2.circle(overlay, (cx, cy), 6, self.__center_color, -1)
        
        # Bounding box for the summary output (axis-aligned)
        x1, y1 = np.min(poly_pts, axis=0)
        x2, y2 = np.max(poly_pts, axis=0)

        cv2.putText(
            overlay,
            f"Estimated center: ({cx}, {cy})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.__rect_color,
            2,
            cv2.LINE_AA,
        )

        return overlay, (int(x1), int(y1)), (int(x2), int(y2))
