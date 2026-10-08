from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config.loader import load_config
from .decision.controller import VisionController
from .platform.macos.input import MacOSInputController
from .capture.test_source import TestImageSource
from .platform.macos.capture import MacOSWindowSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MapleStory macOS vision loop")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--window-title", default=None)
    parser.add_argument("--test-image", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only propose actions; never emit keyboard input.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(Path(args.config))
    if args.window_title:
        config["game_window"]["title_contains"] = args.window_title
    if args.test_image:
        config["capture"]["test_image"] = args.test_image
    if args.dry_run:
        config["system"]["dry_run"] = True

    image_path = config["capture"].get("test_image", "")
    if image_path:
        source = TestImageSource(Path(image_path))
    else:
        source = MacOSWindowSource(config["game_window"]["title_contains"])

    controller = VisionController(
        source=source,
        input_controller=MacOSInputController(
            dry_run=bool(config["system"].get("dry_run", True))
        ),
        config=config,
    )
    controller.run()
    return 0
