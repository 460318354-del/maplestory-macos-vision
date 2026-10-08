from __future__ import annotations

import logging
import time
from typing import Any

from ..capture.base import FrameSource
from ..platform.macos.input import MacOSInputController
from ..vision.player import find_color_point

logger = logging.getLogger(__name__)


class VisionController:
    def __init__(
        self,
        source: FrameSource,
        input_controller: MacOSInputController,
        config: dict[str, Any],
    ):
        self.source = source
        self.input = input_controller
        self.config = config

    def run(self) -> None:
        fps = max(1, int(self.config["system"].get("fps", 8)))
        delay = 1.0 / fps
        try:
            while True:
                started = time.monotonic()
                frame = self.source.next_frame()
                if frame is None:
                    return
                action = self._propose_action(frame)
                self.input.emit(action or "none")
                elapsed = time.monotonic() - started
                if elapsed < delay:
                    time.sleep(delay - elapsed)
        finally:
            self.source.close()

    def _propose_action(self, frame) -> str | None:
        vision = self.config["vision"]
        lower = tuple(vision["player_color_hsv"]["lower"])
        upper = tuple(vision["player_color_hsv"]["upper"])
        point = find_color_point(
            frame,
            tuple(vision["player_region"]),
            lower,
            upper,
        )
        if point is None:
            logger.warning("player point not found")
            return "none"
        logger.info("player point: (%s, %s)", point.x, point.y)
        return "observe"
