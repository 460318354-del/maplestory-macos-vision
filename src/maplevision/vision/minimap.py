from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .player import Point, find_compact_color_point


@dataclass(frozen=True)
class MinimapObservation:
    region: tuple[int, int, int, int]
    player_local: Point | None
    player_global: Point | None

    @property
    def horizontal_ratio(self) -> float | None:
        if self.player_local is None or self.region[2] <= 1:
            return None
        return self.player_local.x / max(1, self.region[2] - 1)

    @property
    def player_target(self) -> Point | None:
        """Return the player's position in minimap-local coordinates."""
        return self.player_local


def observe_minimap(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> MinimapObservation:
    x, y, width, height = region
    # The minimap contains yellow terrain and decorations. The player marker
    # is a compact component; selecting the largest yellow component often
    # locks onto a tree or platform and produces a false direction.
    player_global = find_compact_color_point(
        frame_bgr,
        region,
        lower_hsv,
        upper_hsv,
    )
    player_local = None
    if player_global is not None:
        player_local = Point(player_global.x - x, player_global.y - y)
    return MinimapObservation(
        region=region,
        player_local=player_local,
        player_global=player_global,
    )


def render_debug_frame(
    frame_bgr: np.ndarray,
    observation: MinimapObservation,
) -> np.ndarray:
    output = frame_bgr.copy()
    x, y, width, height = observation.region
    cv2.rectangle(output, (x, y), (x + width, y + height), (255, 0, 0), 2)
    label = "minimap"
    if observation.player_local is None:
        label += " player=not-found"
    else:
        point = observation.player_global
        assert point is not None
        cv2.circle(output, (point.x, point.y), 8, (0, 255, 255), 2)
        cv2.drawMarker(
            output,
            (point.x, point.y),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        label += (
            f" player=({observation.player_local.x},"
            f"{observation.player_local.y})"
        )
    cv2.putText(
        output,
        label,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def save_debug_frame(
    frame_bgr: np.ndarray,
    observation: MinimapObservation,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), render_debug_frame(frame_bgr, observation)):
        raise RuntimeError(f"Unable to write debug image: {path}")
