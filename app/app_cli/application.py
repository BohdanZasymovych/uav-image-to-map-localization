from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2

from app.app_cli.args import CliOptions
from app.app_cli.config import YamlConfigLoader
from app.app_cli.factories import LocalizationComponentFactory
from app.app_cli.renderer import MapOverlayRenderer


class LocalizationCliApplication:
    def __init__(self, options: CliOptions) -> None:
        self.__options = options
        self.__logger = logging.getLogger("uav_localization_cli")

    def run(self) -> int:
        self.__prepare_output_dir()
        self.__log_input_paths()
        self.__validate_input_paths()

        config = self.__load_config()
        factory = LocalizationComponentFactory(config)

        epsilon, confidence, max_iterations = factory.ransac_params()
        pipeline = factory.build_pipeline()
        extractor = pipeline.extractor
        model = pipeline.model

        self.__logger.info("Extractor: %s", extractor.__class__.__name__)
        self.__logger.info("Model: %s", model.__class__.__name__)
        self.__logger.info(
            "RANSAC params: epsilon=%s confidence=%s max_iterations=%s",
            epsilon,
            confidence,
            max_iterations,
        )

        map_img, uav_img = self.__read_images()

        match_result, localization_result = pipeline.run(uav_img=uav_img, map_img=map_img)

        self.__logger.info("Pipeline completed successfully")
        self.__logger.info(
            "Matches: raw=%d inliers=%d",
            localization_result.n_raw_matches,
            localization_result.n_inliers,
        )
        self.__logger.info("RANSAC iterations: %d", localization_result.ransac_iterations)
        self.__logger.info(
            "Runtime (without match image rendering): %.6f s",
            localization_result.runtime_s,
        )
        self.__logger.info(
            "Estimated pixel position: [%.3f, %.3f]",
            float(localization_result.position_px[0]),
            float(localization_result.position_px[1]),
        )
        self.__logger.info("Transform matrix:\n%s", localization_result.transform_matrix)

        raw_match_path = self.__options.output_dir / "matches_before_ransac.png"
        inlier_match_path = self.__options.output_dir / "matches_after_ransac.png"
        map_bbox_path = self.__options.output_dir / "map_with_estimated_bbox.png"

        renderer = MapOverlayRenderer()
        uav_h, uav_w = uav_img.shape[:2]
        estimated_x = float(localization_result.position_px[0])
        estimated_y = float(localization_result.position_px[1])
        map_overlay, top_left, bottom_right = renderer.render(
            map_img=map_img,
            position_px_x=estimated_x,
            position_px_y=estimated_y,
            bbox_width=uav_w,
            bbox_height=uav_h,
        )

        cv2.imwrite(str(raw_match_path), match_result.raw_match_image)
        cv2.imwrite(str(inlier_match_path), match_result.match_image)
        cv2.imwrite(str(map_bbox_path), map_overlay)

        self.__logger.info("Saved raw match image: %s", raw_match_path)
        self.__logger.info("Saved inlier match image: %s", inlier_match_path)
        self.__logger.info(
            "Saved map overlay with estimated bbox: %s (top_left=%s bottom_right=%s)",
            map_bbox_path,
            top_left,
            bottom_right,
        )

        summary = self.__build_summary(
            config=config,
            extractor_name=extractor.__class__.__name__,
            model_name=model.__class__.__name__,
            epsilon=epsilon,
            confidence=confidence,
            max_iterations=max_iterations,
            localization_result=localization_result,
            raw_match_path=raw_match_path,
            inlier_match_path=inlier_match_path,
            map_bbox_path=map_bbox_path,
        )

        summary_path = self.__options.output_dir / "localization_summary.json"
        with summary_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2)
        self.__logger.info("Saved JSON summary: %s", summary_path)

        return 0

    def __prepare_output_dir(self) -> None:
        self.__options.output_dir.mkdir(parents=True, exist_ok=True)

    def __log_input_paths(self) -> None:
        self.__logger.info("Starting localization run")
        self.__logger.info("Input map image: %s", self.__options.map_path)
        self.__logger.info("Input UAV image: %s", self.__options.uav_path)
        self.__logger.info("Config path: %s", self.__options.config_path)
        self.__logger.info("Output dir: %s", self.__options.output_dir)

    def __validate_input_paths(self) -> None:
        if not self.__options.map_path.exists():
            raise FileNotFoundError(f"Map image not found: {self.__options.map_path}")
        if not self.__options.uav_path.exists():
            raise FileNotFoundError(f"UAV image not found: {self.__options.uav_path}")
        if not self.__options.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.__options.config_path}")

    def __load_config(self) -> dict[str, Any]:
        config = YamlConfigLoader(self.__options.config_path).load()
        self.__logger.info("Loaded config")
        self.__logger.debug("Config content: %s", config)
        return config

    def __read_images(self) -> tuple[Any, Any]:
        map_img = cv2.imread(str(self.__options.map_path), cv2.IMREAD_COLOR)
        uav_img = cv2.imread(str(self.__options.uav_path), cv2.IMREAD_COLOR)
        if map_img is None:
            raise RuntimeError(f"Failed to read map image: {self.__options.map_path}")
        if uav_img is None:
            raise RuntimeError(f"Failed to read UAV image: {self.__options.uav_path}")

        self.__logger.info("Map image shape: %s dtype=%s", map_img.shape, map_img.dtype)
        self.__logger.info("UAV image shape: %s dtype=%s", uav_img.shape, uav_img.dtype)
        return map_img, uav_img

    def __build_summary(
        self,
        config: dict[str, Any],
        extractor_name: str,
        model_name: str,
        epsilon: float,
        confidence: float,
        max_iterations: int,
        localization_result: Any,
        raw_match_path: Path,
        inlier_match_path: Path,
        map_bbox_path: Path,
    ) -> dict[str, Any]:
        _ = config
        return {
            "map_path": str(self.__options.map_path),
            "uav_path": str(self.__options.uav_path),
            "config_path": str(self.__options.config_path),
            "extractor": extractor_name,
            "model": model_name,
            "ransac": {
                "epsilon": epsilon,
                "confidence": confidence,
                "max_iterations": max_iterations,
            },
            "result": {
                "position_px": [
                    float(localization_result.position_px[0]),
                    float(localization_result.position_px[1]),
                ],
                "position_geo": localization_result.position_geo,
                "n_raw_matches": int(localization_result.n_raw_matches),
                "n_inliers": int(localization_result.n_inliers),
                "ransac_iterations": int(localization_result.ransac_iterations),
                "runtime_s": float(localization_result.runtime_s),
                "transform_matrix": localization_result.transform_matrix.tolist(),
            },
            "artifacts": {
                "raw_match_image": str(raw_match_path),
                "inlier_match_image": str(inlier_match_path),
                "map_with_estimated_bbox": str(map_bbox_path),
            },
        }
