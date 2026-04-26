from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import cv2  # type: ignore[import-not-found]
import yaml  # type: ignore[import-not-found]

__PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(__PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(__PROJECT_ROOT))

from evaluation.base import DatasetGenerator, SyntheticFrame
from evaluation.dataset import SyntheticDatasetGenerator
from evaluation.metrics import Evaluator
from evaluation.visualizer import Visualizer
from localization.features.sift import SIFTExtractor
from localization.pipeline import LocalizationPipeline
from localization.transforms.affine import AffineModel
from localization.transforms.similarity import SimilarityModel


class __ConfiguredSyntheticGenerator(DatasetGenerator):
    def __init__(
        self,
        generator: SyntheticDatasetGenerator,
        scale_range: tuple[float, float],
        rotation_range: tuple[float, float],
        shear_range: tuple[float, float],
        crop_size: tuple[int, int],
    ) -> None:
        self.__generator = generator
        self.__scale_range = scale_range
        self.__rotation_range = rotation_range
        self.__shear_range = shear_range
        self.__crop_size = crop_size

    def generate(self, n: int) -> list[SyntheticFrame]:
        return self.__generator.generate(
            n=n,
            scale_range=self.__scale_range,
            rotation_range=self.__rotation_range,
            shear_range=self.__shear_range,
            crop_size=self.__crop_size,
        )


def __build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic frames and evaluate pipeline using native evaluation classes.",
    )
    parser.add_argument("--map", default="data/satellite02_rescaled.tif", help="Path to map image")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default="outputs/generate_and_evaluate", help="Directory for evaluation outputs")
    parser.add_argument("--n-frames", type=int, default=None, help="Override number of generated frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic generation")
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Crop size for generated UAV frames (height width)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser


def __resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return __PROJECT_ROOT / path


def __load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must contain a top-level mapping")
    return cfg


def __as_pair(value: Any, key: str, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Config key '{key}' must be a pair of values")
    return float(value[0]), float(value[1])


def __build_pipeline(config: dict[str, Any]) -> LocalizationPipeline:
    extractor_name = str(config.get("extractor", "SIFT")).strip().lower()
    if extractor_name != "sift":
        raise NotImplementedError(
            "Only SIFT extractor is supported now because ORB/SURF are not implemented"
        )

    sift_cfg = config.get("sift", {})
    if not isinstance(sift_cfg, dict):
        raise ValueError("Config key 'sift' must be a mapping")
    extractor = SIFTExtractor(
        ratio=float(sift_cfg.get("ratio", 0.75)),
        n_features=int(sift_cfg.get("n_features", 0)),
    )

    model_name = str(config.get("model", "Affine")).strip().lower()
    if model_name == "affine":
        model = AffineModel()
    elif model_name == "similarity":
        model = SimilarityModel()
    else:
        raise NotImplementedError("Only Affine and Similarity models are supported now")

    ransac_cfg = config.get("ransac", {})
    if not isinstance(ransac_cfg, dict):
        raise ValueError("Config key 'ransac' must be a mapping")

    return LocalizationPipeline(
        extractor=extractor,
        model=model,
        epsilon=float(ransac_cfg.get("epsilon", 3.0)),
        confidence=float(ransac_cfg.get("confidence", 0.99)),
        max_iterations=int(ransac_cfg.get("max_iterations", 2000)),
    )


def main() -> int:
    args = __build_parser().parse_args()

    from app.app_cli.logging_utils import LoggingConfigurator
    LoggingConfigurator.configure(log_level=str(args.log_level))
    logger = logging.getLogger("generate_and_evaluate")


    map_path = __resolve_path(str(args.map))
    config_path = __resolve_path(str(args.config))
    output_dir = __resolve_path(str(args.output_dir))

    if not map_path.exists():
        raise FileNotFoundError(f"Map image not found: {map_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    map_img = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if map_img is None:
        raise RuntimeError(f"Failed to read map image: {map_path}")

    config = __load_config(config_path)
    evaluation_cfg = config.get("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        raise ValueError("Config key 'evaluation' must be a mapping")

    n_frames = int(args.n_frames) if args.n_frames is not None else int(evaluation_cfg.get("n_frames", 100))
    scale_range = __as_pair(evaluation_cfg.get("scale_range"), "evaluation.scale_range", (0.4, 0.8))
    rotation_range = __as_pair(evaluation_cfg.get("rotation_range"), "evaluation.rotation_range", (-30.0, 30.0))
    shear_range = __as_pair(evaluation_cfg.get("shear_range"), "evaluation.shear_range", (-0.1, 0.1))

    if args.crop_size is not None:
        crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))
    else:
        crop_size_cfg = evaluation_cfg.get("crop_size", [256, 256])
        if not isinstance(crop_size_cfg, (list, tuple)) or len(crop_size_cfg) != 2:
            raise ValueError("Config key 'evaluation.crop_size' must have two values")
        crop_size = (int(crop_size_cfg[0]), int(crop_size_cfg[1]))

    logger.info("Preparing synthetic generator and evaluator")
    logger.info("n_frames=%d seed=%d crop_size=%s", n_frames, int(args.seed), crop_size)
    logger.info("scale_range=%s rotation_range=%s shear_range=%s", scale_range, rotation_range, shear_range)

    raw_generator = SyntheticDatasetGenerator(map_img=map_img, seed=int(args.seed))
    generator = __ConfiguredSyntheticGenerator(
        generator=raw_generator,
        scale_range=scale_range,
        rotation_range=rotation_range,
        shear_range=shear_range,
        crop_size=crop_size,
    )
    pipeline = __build_pipeline(config)
    evaluator = Evaluator(pipeline=pipeline, generator=generator, map_img=map_img)

    logger.info("Running evaluation")
    report = evaluator.run(n_frames=n_frames)

    visualizer = Visualizer()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    visualizer.save_report(report, str(plots_dir))

    logger.info("RMSE: %.4f px", report.rmse)
    logger.info("Mean runtime: %.6f s", report.mean_runtime_s)
    logger.info("Mean inlier ratio: %.4f", report.mean_inlier_ratio)
    logger.info("Saved evaluation plots directory: %s", plots_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
