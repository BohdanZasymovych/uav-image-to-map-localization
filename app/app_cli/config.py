from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-not-found]


class YamlConfigLoader:
    def __init__(self, config_path: Path) -> None:
        self.__config_path = config_path

    def load(self) -> dict[str, Any]:
        with self.__config_path.open("r", encoding="utf-8") as file_obj:
            config = yaml.safe_load(file_obj)
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a top-level mapping")
        return config
