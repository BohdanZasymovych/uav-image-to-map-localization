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
