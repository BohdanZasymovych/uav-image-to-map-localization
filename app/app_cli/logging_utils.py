from __future__ import annotations

import logging
from pathlib import Path


class LoggingConfigurator:
    @staticmethod
    def configure(
        log_level: str = "INFO", 
        log_file: Path | None = None, 
        extra_handlers: list[logging.Handler] | None = None
    ) -> None:
        """
        Configures the root logger with standard formatting and optional file/custom handlers.
        """
        # Ensure we don't add duplicate handlers if called multiple times
        root_logger = logging.getLogger()
        
        # Remove existing handlers to avoid duplicates on re-config
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        handlers = LoggingConfigurator.__build_handlers(log_file)
        if extra_handlers:
            handlers.extend(extra_handlers)

        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        
        for handler in handlers:
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Ensure specific sub-loggers are also set to the correct level
        logging.getLogger("localization").setLevel(getattr(logging, log_level.upper()))
        logging.getLogger("app").setLevel(getattr(logging, log_level.upper()))

    @staticmethod
    def __build_handlers(log_file: Path | None) -> list[logging.Handler]:
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_file is not None:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
            except Exception as e:
                # Fallback if file logging fails
                logging.warning(f"Failed to setup file logging: {e}")
        return handlers
