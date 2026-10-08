from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..capture.base import FrameSource
from ..platform.macos.input import MacOSInputController
from ..vision.combat import observe_combat, save_combat_debug_frame
from ..vision.minimap import observe_minimap
from ..vision.monster import MonsterTracker

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
        monster_config = config.get("combat", {}).get("monsters", {})
        self.monster_tracker = MonsterTracker(
            confirm_frames=int(monster_config.get("confirm_frames", 2)),
            max_misses=int(monster_config.get("max_misses", 2)),
            max_distance=int(monster_config.get("max_distance", 120)),
        )

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
        else:
            logger.info(
                "minimap player: local=(%s, %s) global=(%s, %s)",
                observation.player_local.x,
                observation.player_local.y,
                observation.player_global.x,
                observation.player_global.y,
            )

        combat = observe_combat(
            frame,
            self.config.get("combat", {}),
            observation,
            self.monster_tracker,
        )
        if combat.health.ratio is None:
            logger.info("hp: not-found")
        else:
            logger.info("hp: %.1f%% healthy=%s", combat.health.ratio * 100, combat.health.healthy)
        if combat.target is None:
            logger.info("combat: targets=%s selected=none reason=%s", len(combat.targets), combat.reason)
        else:
            logger.info(
                "combat: targets=%s selected=%s score=%.3f action=%s reason=%s",
                len(combat.targets),
                combat.target.box,
                combat.target.score,
                combat.action,
                combat.reason,
            )

        debug_output = self.config["system"].get("debug_output", "")
        if debug_output:
            save_combat_debug_frame(frame, observation, combat, Path(debug_output))
        return combat.action
