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
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    center_x = int(moments["m10"] / moments["m00"]) + x
    center_y = int(moments["m01"] / moments["m00"]) + y
    return Point(center_x, center_y)
