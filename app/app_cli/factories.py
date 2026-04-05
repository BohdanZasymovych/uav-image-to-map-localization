from __future__ import annotations

from typing import Any

from localization.features.base import FeatureExtractor
from localization.features.sift import SIFTExtractor
from localization.pipeline import LocalizationPipeline
from localization.transforms.affine import AffineModel
from localization.transforms.base import TransformationModel
from localization.transforms.similarity import SimilarityModel


class LocalizationComponentFactory:
    def __init__(self, config: dict[str, Any]) -> None:
        self.__config = config

    def build_extractor(self) -> FeatureExtractor:
        extractor_name = str(self.__config.get("extractor", "SIFT")).strip().lower()

        if extractor_name == "sift":
            sift_cfg = self.__mapping(self.__config.get("sift", {}), "sift")
            ratio = float(sift_cfg.get("ratio", 0.75))
            n_features = int(sift_cfg.get("n_features", 0))
            return SIFTExtractor(ratio=ratio, n_features=n_features)

        if extractor_name in {"orb", "surf"}:
            raise NotImplementedError(
                f"Extractor '{extractor_name}' is not implemented in this repository"
            )

        raise ValueError(f"Unsupported extractor: {extractor_name}")

    def build_model(self) -> TransformationModel:
        model_name = str(self.__config.get("model", "Affine")).strip().lower()

        if model_name == "affine":
            return AffineModel()
        if model_name == "similarity":
            return SimilarityModel()
        if model_name == "projective":
            raise NotImplementedError("Projective model is not implemented in this repository")

        raise ValueError(f"Unsupported model: {model_name}")

    def build_pipeline(self) -> LocalizationPipeline:
        extractor = self.build_extractor()
        model = self.build_model()
        epsilon, confidence, max_iterations = self.ransac_params()

        return LocalizationPipeline(
            extractor=extractor,
            model=model,
            epsilon=epsilon,
            confidence=confidence,
            max_iterations=max_iterations,
            georeferencer=None,
        )

    def ransac_params(self) -> tuple[float, float, int]:
        ransac_cfg = self.__mapping(self.__config.get("ransac", {}), "ransac")
        epsilon = float(ransac_cfg.get("epsilon", 3.0))
        confidence = float(ransac_cfg.get("confidence", 0.99))
        max_iterations = int(ransac_cfg.get("max_iterations", 2000))
        return epsilon, confidence, max_iterations

    def __mapping(self, value: Any, key: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"Config key '{key}' must be a mapping")
        return value
