from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_cli.application import LocalizationCliApplication
from app.app_cli.args import CliArgumentParser
from app.app_cli.logging_utils import LoggingConfigurator


def main() -> int:
    parser = CliArgumentParser()
    options = parser.parse()
    LoggingConfigurator.configure(options.log_level, options.log_file)

    try:
        return LocalizationCliApplication(options).run()
    except Exception:
        logging.getLogger("uav_localization_cli").exception("Localization run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
