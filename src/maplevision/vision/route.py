from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .player import Point


def load_route(path: Path) -> np.ndarray:
    route = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if route is None:
        raise FileNotFoundError(f"Unable to read route image: {path}")
    return cv2.cvtColor(route, cv2.COLOR_BGR2RGB)


def nearest_action(
    route_rgb: np.ndarray,
    player: Point,
    colors: dict[str, str],
    radius: int,
) -> str | None:
    height, width = route_rgb.shape[:2]
    x0 = max(0, player.x - radius)
    x1 = min(width, player.x + radius + 1)
    y0 = max(0, player.y - radius)
    y1 = min(height, player.y + radius + 1)

    best: tuple[int, str] | None = None
    for y in range(y0, y1):
        for x in range(x0, x1):
            key = ",".join(str(value) for value in route_rgb[y, x])
            action = colors.get(key)
            if action is None:
                continue
            distance = abs(x - player.x) + abs(y - player.y)
            if best is None or distance < best[0]:
                best = (distance, action)
    return best[1] if best else None
