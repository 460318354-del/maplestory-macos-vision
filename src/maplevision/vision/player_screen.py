from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PlayerScreenDetection:
    name: str
    score: float
    name_box: tuple[int, int, int, int]
    anchor: tuple[int, int]


def detect_player_screen(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
) -> PlayerScreenDetection | None:
    if not config.get("enabled", True):
        return None

    template_path = Path(str(config.get("template", "")))
    if not str(template_path) or not template_path.is_file():
        return None
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        return None

    x, y, width, height = tuple(config.get("search_region") or ())
    frame_height, frame_width = frame_bgr.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_width, int(x + width))
    y1 = min(frame_height, int(y + height))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0 or template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
        return None

    result = cv2.matchTemplate(
        _match_image(roi),
        _match_image(template),
        cv2.TM_CCOEFF_NORMED,
    )
    _, score, _, location = cv2.minMaxLoc(result)
    if score < float(config.get("threshold", 0.75)):
        return None

    match_x = x0 + int(location[0])
    match_y = y0 + int(location[1])
    template_height, template_width = template.shape[:2]
    offset_x, offset_y = tuple(config.get("anchor_offset", [template_width // 2, 0]))
    return PlayerScreenDetection(
        name=str(config.get("name", "player")),
        score=float(score),
        name_box=(match_x, match_y, int(template_width), int(template_height)),
        anchor=(match_x + int(offset_x), match_y + int(offset_y)),
    )


def _match_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)
