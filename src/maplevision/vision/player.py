from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Point:
    x: int
    y: int


def find_color_point(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> Point | None:
    x, y, width, height = region
    roi = frame_bgr[y : y + height, x : x + width]
    if roi.size == 0:
        return None
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), lower_hsv, upper_hsv)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if component_count <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    moments = cv2.moments(component)
    if moments["m00"] == 0:
        return None
    center_x = int(moments["m10"] / moments["m00"]) + x
    center_y = int(moments["m01"] / moments["m00"]) + y
    return Point(center_x, center_y)


def find_compact_color_point(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    *,
    min_area: int = 4,
    max_area: int = 80,
    max_width: int = 12,
    max_height: int = 12,
) -> Point | None:
    """Find a small color marker while ignoring large background regions."""
    x, y, width, height = region
    roi = frame_bgr[y : y + height, x : x + width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask
    )
    candidates: list[tuple[float, int]] = []
    for label in range(1, component_count):
        _, _, component_width, component_height, area = stats[label]
        if not (
            min_area <= area <= max_area
            and component_width <= max_width
            and component_height <= max_height
        ):
            continue
        box_area = max(1, int(component_width) * int(component_height))
        compactness = float(area) / box_area
        if compactness < 0.45:
            continue
        # Prefer a filled marker over tiny anti-aliased fragments.
        score = compactness * float(area)
        candidates.append((score, label))

    if not candidates:
        return None
    _, selected_label = max(candidates)
    center_x, center_y = centroids[selected_label]
    return Point(int(center_x) + x, int(center_y) + y)
