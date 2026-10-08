from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config.loader import load_config
from .platform.macos.input import MacOSInputController
from .capture.test_source import TestImageSource
from .platform.macos.capture import MacOSWindowSource, list_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MapleStory macOS vision loop")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--window-title", default=None)
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List visible macOS windows and exit.",
    )
    parser.add_argument(
        "--bind-window",
        action="store_true",
        help="Bind the matching macOS window, print its metadata, and exit.",
    )
    parser.add_argument("--window-id", type=int, default=None)
    parser.add_argument("--test-image", default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture and process one frame, then exit.",
    )
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

    title_contains = config["game_window"]["title_contains"]
    if args.list_windows:
        for window in list_windows():
            print(
                f"id={window.window_id} owner={window.owner!r} "
                f"title={window.title!r} bounds={window.bounds} "
                f"layer={window.layer}"
            )
        return 0

    image_path = config["capture"].get("test_image", "")
    if image_path:
        source = TestImageSource(Path(image_path))
    else:
        source = MacOSWindowSource(title_contains, window_id=args.window_id)

    info = getattr(source, "window_info", None)
    if info is not None:
        print(
            f"bound window: id={info.window_id} owner={info.owner!r} "
            f"title={info.title!r} bounds={info.bounds}"
        )
    if args.bind_window:
        source.close()
        return 0

    from .decision.controller import VisionController

    controller = VisionController(
        source=source,
        input_controller=MacOSInputController(
            dry_run=bool(config["system"].get("dry_run", True))
        ),
        config=config,
    )
    controller.run(max_frames=1 if args.once else None)
    return 0
