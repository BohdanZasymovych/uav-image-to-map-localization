from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CliOptions:
    map_path: Path
    uav_path: Path
    config_path: Path
    output_dir: Path
    log_file: Path | None
    log_level: str


class CliArgumentParser:
    def __init__(self) -> None:
        self.__parser = self.__build_parser()

    def parse(self, argv: Sequence[str] | None = None) -> CliOptions:
        namespace = self.__parser.parse_args(argv)
        return self.__namespace_to_options(namespace)

    def __build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run UAV-to-map localization for one image pair using YAML config.",
        )
        parser.add_argument("--map", dest="map_path", required=True, help="Path to map (reference) image")
        parser.add_argument("--uav", dest="uav_path", required=True, help="Path to UAV image")
        parser.add_argument("--config", dest="config_path", required=True, help="Path to YAML config file")
        parser.add_argument("--output-dir", default="outputs", help="Directory for output images and JSON summary")
        parser.add_argument("--log-file", default=None, help="Optional path to a log file")
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging verbosity",
        )
        return parser

    def __namespace_to_options(self, namespace: argparse.Namespace) -> CliOptions:
        log_file = Path(namespace.log_file) if namespace.log_file else None
        return CliOptions(
            map_path=Path(namespace.map_path),
            uav_path=Path(namespace.uav_path),
            config_path=Path(namespace.config_path),
            output_dir=Path(namespace.output_dir),
            log_file=log_file,
            log_level=str(namespace.log_level),
        )
