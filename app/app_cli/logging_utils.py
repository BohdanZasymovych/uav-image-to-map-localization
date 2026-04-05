from __future__ import annotations

import logging
from pathlib import Path


class LoggingConfigurator:
    @staticmethod
    def configure(log_level: str, log_file: Path | None) -> None:
        handlers = LoggingConfigurator.__build_handlers(log_file)
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=handlers,
        )

    @staticmethod
    def __build_handlers(log_file: Path | None) -> list[logging.Handler]:
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
        return handlers
