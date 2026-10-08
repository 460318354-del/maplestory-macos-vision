from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..capture.base import FrameSource
from ..platform.macos.input import MacOSInputController
from ..vision.minimap import observe_minimap, save_debug_frame

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

    def run(self, max_frames: int | None = None) -> None:
        fps = max(1, int(self.config["system"].get("fps", 8)))
        delay = 1.0 / fps
        frames = 0
        try:
            while True:
                started = time.monotonic()
                frame = self.source.next_frame()
                if frame is None:
                    return
                action = self._propose_action(frame)
                self.input.emit(action or "none")
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    return
                elapsed = time.monotonic() - started
                if elapsed < delay:
                    time.sleep(delay - elapsed)
        finally:
            self.source.close()

    def _propose_action(self, frame) -> str | None:
        minimap = self.config["minimap"]
        region = tuple(minimap["region"])
        lower = tuple(minimap["player_color_hsv"]["lower"])
        upper = tuple(minimap["player_color_hsv"]["upper"])
        observation = observe_minimap(
            frame,
            region,
            lower,
            upper,
        )
        if observation.player_global is None:
            logger.warning("player point not found")
            return "none"
        point = observation.player_global
        logger.info(
            "minimap player: local=(%s, %s) global=(%s, %s)",
            observation.player_local.x,
            observation.player_local.y,
            point.x,
            point.y,
        )
        debug_output = self.config["system"].get("debug_output", "")
        if debug_output:
            save_debug_frame(frame, observation, Path(debug_output))
        return "observe"
