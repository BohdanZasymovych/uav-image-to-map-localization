# =============================================================================
# localization/georeferencing.py
# -----------------------------------------------------------------------------
# OWNER: Person A.
#
# PURPOSE:
#   Converts estimated map pixel coordinates to geographic coordinates
#   (latitude, longitude) using the satellite map's known bounding box.
#   Optional component — pipeline works without it; position_geo stays None.
#
# IMPLEMENT:
#   class Georeferencer
#
#   __init__(self, map_shape: tuple[int, int],
#            top_left: tuple[float, float],
#            bottom_right: tuple[float, float]):
#     map_shape     — (H, W) in pixels
#     top_left      — (lat, lon) at pixel (0, 0)         — top-left corner
#     bottom_right  — (lat, lon) at pixel (W-1, H-1)     — bottom-right corner
#
#   pixel_to_geo(self, x_px: float, y_px: float) -> tuple[float, float]:
#     Bilinear interpolation across the bounding box:
#       lat = lat_top  + (y_px / H) * (lat_bottom - lat_top)
#       lon = lon_left + (x_px / W) * (lon_right  - lon_left)
#     Returns (lat, lon).
#
# NOTE:
#   Flat-earth approximation — valid only for small map extents (< ~50 km).
#   For larger areas a proper map projection (e.g. Web Mercator) is needed.
# =============================================================================

from __future__ import annotations


class Georeferencer:
    def __init__(
        self,
        map_shape: tuple[int, int],
        top_left: tuple[float, float],
        bottom_right: tuple[float, float],
    ) -> None:
        # TODO: implement
        ...

    def pixel_to_geo(self, x_px: float, y_px: float) -> tuple[float, float]:
        # TODO: implement
        ...
