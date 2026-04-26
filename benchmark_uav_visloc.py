from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

from localization.features.sift import SIFTExtractor
from localization.pipeline import LocalizationPipeline
from localization.transforms.affine import AffineModel
from localization.transforms.projective import ProjectiveModel
from localization.transforms.similarity import SimilarityModel


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Bounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class FrameSample:
    scene_id: str
    image_path: Path
    map_path: Path
    ground_truth_px: tuple[float, float]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark UAV localization on UAV-VisLoc style datasets for "
            "Affine/Similarity/Projective models and save CSV + plots."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Path to UAV-VisLoc root directory")
    parser.add_argument("--output-dir", required=True, help="Path to save benchmark artifacts")
    parser.add_argument(
        "--models",
        default="affine,similarity,projective",
        help="Comma-separated model names (affine, similarity, projective)",
    )
    parser.add_argument("--max-samples", type=int, default=300, help="Maximum number of frames to benchmark")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Optional scene ids, e.g. 01 03 11")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected frames before truncating")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffling")

    parser.add_argument(
        "--bounds-mode",
        choices=["scene-csv", "satellite-range"],
        default="scene-csv",
        help=(
            "How to convert lat/lon to map pixels: scene-csv uses scene-level min/max lat/lon; "
            "satellite-range uses a global range CSV if available"
        ),
    )
    parser.add_argument(
        "--satellite-range-csv",
        default=None,
        help="Optional path to satellite_coordinates_range.csv for bounds-mode=satellite-range",
    )

    parser.add_argument("--ratio", type=float, default=0.75, help="SIFT Lowe ratio")
    parser.add_argument("--n-features", type=int, default=0, help="SIFT max features (0 = unlimited)")
    parser.add_argument("--epsilon", type=float, default=3.0, help="RANSAC inlier threshold")
    parser.add_argument("--confidence", type=float, default=0.99, help="RANSAC confidence")
    parser.add_argument("--max-iterations", type=int, default=2000, help="RANSAC max iterations")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _normalize_name(name: str) -> str:
    return Path(str(name).strip()).name.lower()


def _discover_scene_dirs(dataset_root: Path, scene_ids: set[str] | None) -> list[Path]:
    scene_dirs: list[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if scene_ids is not None and child.name not in scene_ids:
            continue
        if (child / "drone").exists():
            scene_dirs.append(child)
    return scene_dirs


def _find_map_file(scene_dir: Path) -> Path | None:
    candidates = sorted(scene_dir.glob("satellite*.tif")) + sorted(scene_dir.glob("*.tif"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _find_scene_csv(scene_dir: Path) -> Path | None:
    preferred = scene_dir / f"{scene_dir.name}.csv"
    if preferred.exists():
        return preferred

    for csv_path in sorted(scene_dir.glob("*.csv")):
        if csv_path.name.lower() == "satellite_coordinates_range.csv":
            continue
        if csv_path.is_file():
            return csv_path
    return None


def _load_range_index(range_csv_path: Path) -> dict[str, Bounds]:
    if not range_csv_path.exists():
        raise FileNotFoundError(f"Range CSV not found: {range_csv_path}")

    df = pd.read_csv(range_csv_path)
    if df.empty:
        return {}

    filename_col = _pick_column(df.columns.tolist(), ["filename", "file", "map", "satellite", "name"])
    lat_min_col = _pick_column(df.columns.tolist(), ["lat_min", "min_lat", "latitude_min", "south"])
    lat_max_col = _pick_column(df.columns.tolist(), ["lat_max", "max_lat", "latitude_max", "north"])
    lon_min_col = _pick_column(df.columns.tolist(), ["lon_min", "min_lon", "longitude_min", "west"])
    lon_max_col = _pick_column(df.columns.tolist(), ["lon_max", "max_lon", "longitude_max", "east"])

    if not all([filename_col, lat_min_col, lat_max_col, lon_min_col, lon_max_col]):
        raise ValueError(
            "Range CSV must include filename + lat/lon min/max columns. "
            "Expected names similar to lat_min, lat_max, lon_min, lon_max."
        )

    index: dict[str, Bounds] = {}
    for _, row in df.iterrows():
        filename = _normalize_name(str(row[filename_col]))
        stem = Path(filename).stem.lower()

        bounds = Bounds(
            lat_min=float(row[lat_min_col]),
            lat_max=float(row[lat_max_col]),
            lon_min=float(row[lon_min_col]),
            lon_max=float(row[lon_max_col]),
        )
        index[filename] = bounds
        index[stem] = bounds

    return index


def _bounds_from_scene_csv(meta_df: pd.DataFrame) -> Bounds:
    lat_col = _pick_column(meta_df.columns.tolist(), ["latitude", "lat", "gps_latitude", "center_latitude"])
    lon_col = _pick_column(meta_df.columns.tolist(), ["longitude", "lon", "lng", "gps_longitude", "center_longitude"])
    if lat_col is None or lon_col is None:
        raise ValueError("Scene CSV must include latitude/longitude columns")

    lat_vals = pd.to_numeric(meta_df[lat_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    lon_vals = pd.to_numeric(meta_df[lon_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if lat_vals.size == 0 or lon_vals.size == 0:
        raise ValueError("Latitude/longitude columns are empty")

    lat_min = float(np.min(lat_vals))
    lat_max = float(np.max(lat_vals))
    lon_min = float(np.min(lon_vals))
    lon_max = float(np.max(lon_vals))

    if np.isclose(lat_min, lat_max) or np.isclose(lon_min, lon_max):
        raise ValueError("Degenerate latitude/longitude ranges in scene CSV")

    return Bounds(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def _latlon_to_pixel(lat: float, lon: float, bounds: Bounds, map_shape: tuple[int, int]) -> tuple[float, float]:
    h, w = int(map_shape[0]), int(map_shape[1])

    x = (lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * max(w - 1, 1)
    y = (bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * max(h - 1, 1)
    return float(x), float(y)


def _build_scene_samples(
    scene_dir: Path,
    bounds_mode: str,
    range_index: dict[str, Bounds],
) -> list[FrameSample]:
    scene_id = scene_dir.name
    drone_dir = scene_dir / "drone"
    map_path = _find_map_file(scene_dir)
    csv_path = _find_scene_csv(scene_dir)

    if map_path is None or csv_path is None or not drone_dir.exists():
        return []

    map_img = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if map_img is None:
        return []

    meta_df = pd.read_csv(csv_path)
    if meta_df.empty:
        return []

    file_col = _pick_column(meta_df.columns.tolist(), ["filename", "image", "image_name", "img", "file"])
    lat_col = _pick_column(meta_df.columns.tolist(), ["latitude", "lat", "gps_latitude", "center_latitude"])
    lon_col = _pick_column(meta_df.columns.tolist(), ["longitude", "lon", "lng", "gps_longitude", "center_longitude"])
    if file_col is None or lat_col is None or lon_col is None:
        return []

    if bounds_mode == "satellite-range":
        key1 = _normalize_name(map_path.name)
        key2 = Path(key1).stem
        bounds = range_index.get(key1) or range_index.get(key2)
        if bounds is None:
            bounds = _bounds_from_scene_csv(meta_df)
    else:
        bounds = _bounds_from_scene_csv(meta_df)

    row_index: dict[str, dict[str, Any]] = {}
    for _, row in meta_df.iterrows():
        row_index[_normalize_name(str(row[file_col]))] = dict(row)

    samples: list[FrameSample] = []
    for image_path in sorted(drone_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        row = row_index.get(_normalize_name(image_path.name))
        if row is None:
            continue

        lat = float(row[lat_col])
        lon = float(row[lon_col])
        gt_px = _latlon_to_pixel(lat=lat, lon=lon, bounds=bounds, map_shape=map_img.shape[:2])

        samples.append(
            FrameSample(
                scene_id=scene_id,
                image_path=image_path,
                map_path=map_path,
                ground_truth_px=gt_px,
            )
        )

    return samples


def _load_samples(
    dataset_root: Path,
    max_samples: int,
    scene_ids: set[str] | None,
    shuffle: bool,
    seed: int,
    bounds_mode: str,
    range_csv_path: Path | None,
) -> list[FrameSample]:
    range_index: dict[str, Bounds] = {}
    if bounds_mode == "satellite-range":
        if range_csv_path is None:
            candidate = dataset_root / "satellite_coordinates_range.csv"
            range_csv_path = candidate if candidate.exists() else None
        if range_csv_path is not None:
            range_index = _load_range_index(range_csv_path)

    scene_dirs = _discover_scene_dirs(dataset_root, scene_ids)
    samples: list[FrameSample] = []
    for scene_dir in scene_dirs:
        samples.extend(_build_scene_samples(scene_dir=scene_dir, bounds_mode=bounds_mode, range_index=range_index))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(samples)

    if max_samples > 0:
        samples = samples[:max_samples]

    return samples


def _build_pipeline(model_name: str, args: argparse.Namespace) -> LocalizationPipeline:
    extractor = SIFTExtractor(ratio=float(args.ratio), n_features=int(args.n_features))

    model_name = model_name.strip().lower()
    if model_name == "affine":
        model = AffineModel()
    elif model_name == "similarity":
        model = SimilarityModel()
    elif model_name == "projective":
        model = ProjectiveModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return LocalizationPipeline(
        extractor=extractor,
        model=model,
        epsilon=float(args.epsilon),
        confidence=float(args.confidence),
        max_iterations=int(args.max_iterations),
    )


def _plot_bar_with_std(
    summary_df: pd.DataFrame,
    value_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = np.arange(len(summary_df))
    values = summary_df[value_col].to_numpy(dtype=np.float64)
    stds = summary_df[std_col].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, values, yerr=stds, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"][: len(x)])
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"].tolist())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_success(summary_df: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(summary_df))
    values = summary_df["success_rate"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, values, color=["#8172B2", "#64B5CD", "#CCB974"][: len(x)])
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"].tolist())
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Localization Success Rate")
    ax.set_ylabel("Success Rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logger = logging.getLogger("benchmark_uav_visloc")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = set(args.scene_ids) if args.scene_ids else None
    range_csv = Path(args.satellite_range_csv).expanduser().resolve() if args.satellite_range_csv else None

    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise ValueError("No models selected")

    samples = _load_samples(
        dataset_root=dataset_root,
        max_samples=int(args.max_samples),
        scene_ids=scene_ids,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
        bounds_mode=str(args.bounds_mode),
        range_csv_path=range_csv,
    )

    if not samples:
        raise RuntimeError("No benchmark samples found. Verify dataset layout and CSV columns.")

    logger.info("Selected %d samples from %s", len(samples), dataset_root)
    logger.info("Models: %s", ", ".join(models))

    map_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for model_name in models:
        pipeline = _build_pipeline(model_name=model_name, args=args)
        logger.info("Running model: %s", model_name)

        for i, sample in enumerate(samples, start=1):
            uav_img = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if uav_img is None:
                rows.append(
                    {
                        "scene_id": sample.scene_id,
                        "image": sample.image_path.name,
                        "model": model_name,
                        "success": False,
                        "pixel_error_px": np.nan,
                        "runtime_s": np.nan,
                        "n_raw_matches": 0,
                        "n_inliers": 0,
                        "ransac_iterations": 0,
                        "error": "Failed to read UAV image",
                    }
                )
                continue

            map_key = str(sample.map_path)
            if map_key not in map_cache:
                map_img = cv2.imread(map_key, cv2.IMREAD_COLOR)
                if map_img is None:
                    rows.append(
                        {
                            "scene_id": sample.scene_id,
                            "image": sample.image_path.name,
                            "model": model_name,
                            "success": False,
                            "pixel_error_px": np.nan,
                            "runtime_s": np.nan,
                            "n_raw_matches": 0,
                            "n_inliers": 0,
                            "ransac_iterations": 0,
                            "error": "Failed to read map image",
                        }
                    )
                    continue
                map_cache[map_key] = map_img
            map_img = map_cache[map_key]

            try:
                t0 = perf_counter()
                _, result = pipeline.run(uav_img=uav_img, map_img=map_img)
                elapsed = perf_counter() - t0

                gt = np.array(sample.ground_truth_px, dtype=np.float64)
                pred = np.array(result.position_px, dtype=np.float64)
                pixel_error = float(np.linalg.norm(pred - gt))

                rows.append(
                    {
                        "scene_id": sample.scene_id,
                        "image": sample.image_path.name,
                        "model": model_name,
                        "success": True,
                        "pixel_error_px": pixel_error,
                        "runtime_s": float(result.runtime_s if result.runtime_s > 0 else elapsed),
                        "n_raw_matches": int(result.n_raw_matches),
                        "n_inliers": int(result.n_inliers),
                        "ransac_iterations": int(result.ransac_iterations),
                        "error": "",
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                rows.append(
                    {
                        "scene_id": sample.scene_id,
                        "image": sample.image_path.name,
                        "model": model_name,
                        "success": False,
                        "pixel_error_px": np.nan,
                        "runtime_s": np.nan,
                        "n_raw_matches": 0,
                        "n_inliers": 0,
                        "ransac_iterations": 0,
                        "error": str(exc),
                    }
                )

            if i % 25 == 0:
                logger.info("%s: processed %d/%d", model_name, i, len(samples))

    result_df = pd.DataFrame(rows)
    per_sample_path = output_dir / "per_sample_results.csv"
    result_df.to_csv(per_sample_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for model_name in models:
        model_df = result_df[result_df["model"] == model_name]
        success_df = model_df[model_df["success"]]

        n_total = int(model_df.shape[0])
        n_success = int(success_df.shape[0])
        success_rate = float(n_success / n_total) if n_total > 0 else 0.0

        summary_rows.append(
            {
                "model": model_name,
                "n_total": n_total,
                "n_success": n_success,
                "success_rate": success_rate,
                "mean_runtime_s": float(success_df["runtime_s"].mean()) if n_success > 0 else np.nan,
                "std_runtime_s": float(success_df["runtime_s"].std(ddof=0)) if n_success > 0 else np.nan,
                "mean_pixel_error_px": float(success_df["pixel_error_px"].mean()) if n_success > 0 else np.nan,
                "std_pixel_error_px": float(success_df["pixel_error_px"].std(ddof=0)) if n_success > 0 else np.nan,
                "median_pixel_error_px": float(success_df["pixel_error_px"].median()) if n_success > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_by_model.csv"
    summary_df.to_csv(summary_path, index=False)

    _plot_bar_with_std(
        summary_df=summary_df,
        value_col="mean_runtime_s",
        std_col="std_runtime_s",
        title="Mean Runtime by Model",
        ylabel="Runtime (s)",
        output_path=output_dir / "runtime_by_model.png",
    )
    _plot_bar_with_std(
        summary_df=summary_df,
        value_col="mean_pixel_error_px",
        std_col="std_pixel_error_px",
        title="Mean Pixel Error by Model",
        ylabel="Pixel Error (px)",
        output_path=output_dir / "pixel_error_by_model.png",
    )
    _plot_success(summary_df=summary_df, output_path=output_dir / "success_rate_by_model.png")

    logger.info("Saved per-sample CSV: %s", per_sample_path)
    logger.info("Saved summary CSV: %s", summary_path)
    logger.info("Saved plots in: %s", output_dir)

    print("\nBenchmark summary")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
