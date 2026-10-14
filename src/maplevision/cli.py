from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .config.loader import load_config
from .capture.test_source import TestImageSource
from .gui_common import application_root

if sys.platform == "win32":
    from .platform.windows.capture import WindowsWindowSource, list_windows
    from .platform.windows.input import WindowsInputController
else:
    from .platform.macos.capture import MacOSWindowSource, list_windows
    from .platform.macos.input import MacOSInputController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MapleStory vision loop")
    parser.add_argument(
        "--config",
        default=None,
        help="Configuration path (defaults to config/default.yaml beside the app).",
    )
    parser.add_argument("--window-title", default=None)
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List visible game windows and exit.",
    )
    parser.add_argument(
        "--bind-window",
        action="store_true",
        help="Bind the matching game window, print its metadata, and exit.",
    )
    parser.add_argument("--window-id", type=int, default=None)
    parser.add_argument("--test-image", default=None)
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Write the annotated frame to this path after processing.",
    )
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
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the platform-specific window binding and runtime interface.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("post_message", "foreground"),
        default=None,
        help="Windows keyboard delivery mode; ignored on macOS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else application_root() / "config" / "default.yaml"
    )
    config = load_config(config_path)
    if args.gui:
        from .gui import run_gui

        run_gui(config_path)
        return 0

    if args.window_title:
        config["game_window"]["title_contains"] = args.window_title
    if args.test_image:
        config["capture"]["test_image"] = args.test_image
    if args.debug_output:
        config["system"]["debug_output"] = args.debug_output
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
        source = _window_source(
            title_contains,
            args.window_id,
            config.get("capture", {}).get("output_size"),
        )

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
        input_controller=_input_controller(config, source, args),
        config=config,
    )
    controller.run(max_frames=1 if args.once else None)
    return 0


def _window_source(title_contains: str, window_id: int | None, output_size):
    if sys.platform == "win32":
        return WindowsWindowSource(
            title_contains,
            window_id=window_id,
            output_size=output_size,
        )
    return MacOSWindowSource(
        title_contains,
        window_id=window_id,
        output_size=output_size,
    )


def _input_controller(config: dict, source, args):
    dry_run = bool(config["system"].get("dry_run", True))
    info = getattr(source, "window_info", None)
    if sys.platform == "win32":
        mode = args.input_mode or str(
            config["system"].get("windows_input_mode", "post_message")
        )
        return WindowsInputController(
            dry_run=dry_run,
            target_hwnd=getattr(info, "window_id", None),
            mode=mode,
        )
    return MacOSInputController(
        dry_run=dry_run,
        target_pid=getattr(info, "owner_pid", None),
    )
