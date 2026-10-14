from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class EnemyHealthBarObservation:
    present: bool
    score: float
    region: tuple[int, int, int, int] | None
    fill_ratio: float | None
    search_region: tuple[int, int, int, int] | None = None
    row_scores: tuple[float, ...] = ()


class EnemyHealthBarDetector:
    """Detect the fixed black/white health-bar frame near a locked monster."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def detect(
        self,
        frame_bgr: np.ndarray,
        target_box: tuple[int, int, int, int],
    ) -> EnemyHealthBarObservation:
        if not self.config.get("enabled", True):
            return EnemyHealthBarObservation(False, 0.0, None, None)

        frame_height, frame_width = frame_bgr.shape[:2]
        target_x, target_y, target_width, _ = target_box
        target_center_x = target_x + target_width // 2
        widths = self._widths()
        nominal_height = max(8, int(self.config.get("height", 10)))
        height_tolerance = max(
            0,
            int(self.config.get("height_tolerance", 2)),
        )
        heights = tuple(
            sorted(
                {
                    max(8, nominal_height + offset)
                    for offset in range(
                        -height_tolerance,
                        height_tolerance + 1,
                    )
                }
            )
        )
        x_padding = max(0, int(self.config.get("search_x_padding", 24)))
        top_offset = int(self.config.get("search_top_offset", -60))
        bottom_offset = int(self.config.get("search_bottom_offset", 10))
        search_region = (
            max(0, target_center_x - max(widths, default=52) // 2 - x_padding),
            max(0, target_y + top_offset),
            min(
                frame_width,
                target_center_x
                + max(widths, default=52) // 2
                + x_padding,
            ),
            min(frame_height, target_y + bottom_offset + max(heights)),
        )
        best: (
            tuple[float, tuple[int, int, int, int], float | None, tuple[float, ...]]
            | None
        ) = None

        for width in widths:
            base_x = target_center_x - width // 2
            for dx in range(-x_padding, x_padding + 1):
                left = base_x + dx
                if left < 0 or left + width > frame_width:
                    continue
                for relative_y in range(top_offset, bottom_offset + 1):
                    top = target_y + relative_y
                    for height in heights:
                        if top < 0 or top + height > frame_height:
                            continue
                        crop = frame_bgr[top : top + height, left : left + width]
                        score, fill_ratio, row_scores = _structure_score(crop)
                        prefer_candidate = (
                            best is None
                            or score > best[0] + 1e-9
                            or (
                                abs(score - best[0]) <= 1e-9
                                and abs(height - nominal_height)
                                < abs(best[1][3] - nominal_height)
                            )
                        )
                        if prefer_candidate:
                            best = (
                                score,
                                (left, top, width, height),
                                fill_ratio,
                                row_scores,
                            )

        if best is None:
            return EnemyHealthBarObservation(
                False,
                0.0,
                None,
                None,
                search_region=search_region,
            )

        score, region, fill_ratio, row_scores = best
        present = score >= float(
            self.config.get("structure_threshold", 0.52)
        )
        return EnemyHealthBarObservation(
            present=present,
            score=score,
            region=region if present else None,
            fill_ratio=fill_ratio if present else None,
            search_region=search_region,
            row_scores=row_scores,
        )

    def _widths(self) -> tuple[int, ...]:
        raw_widths = self.config.get("widths")
        if raw_widths is None:
            raw_widths = [self.config.get("width", 52)]
        if isinstance(raw_widths, (str, int, float)):
            raw_widths = [raw_widths]
        widths = sorted(
            {
                max(20, int(float(value)))
                for value in raw_widths
                if str(value).strip()
            }
        )
        return tuple(widths or [52])


def _structure_score(
    crop_bgr: np.ndarray,
) -> tuple[float, float | None, tuple[float, ...]]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    height, width = value.shape
    if height < 8 or width < 20:
        return 0.0, None, ()

    black = value <= 90
    white = (saturation <= 90) & (value >= 145)
    usable_left = 1
    usable_right = width - 1

    # The bar is normally black/white/black at both edges, but capture
    # scaling and antialiasing can move a boundary by a pixel. Score each
    # expected row against its small neighborhood instead of requiring one
    # exact row to contain every edge pixel.
    row_scores = (
        _best_row_ratio(black, 0, usable_left, usable_right),
        _best_row_ratio(white, 1, usable_left, usable_right),
        _best_row_ratio(black, 2, usable_left, usable_right),
        _best_row_ratio(black, height - 3, usable_left, usable_right),
        _best_row_ratio(white, height - 2, usable_left, usable_right),
        _best_row_ratio(black, height - 1, usable_left, usable_right),
    )
    side_score = min(
        _column_ratio(black[:, 0]),
        _column_ratio(black[:, width - 1]),
    )

    fill = (
        ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95))
        | (hsv[:, :, 0] <= 12)
        | (hsv[:, :, 0] >= 168)
    ) & (saturation >= 65) & (value >= 45)
    fill_area = fill[3 : max(3, height - 3), 2 : max(2, width - 2)]
    fill_columns = np.count_nonzero(fill_area, axis=0)
    fill_ratio = (
        float(np.count_nonzero(fill_columns >= 1) / max(1, fill_columns.size))
        if fill_columns.size
        else None
    )
    # Require a colored fill column, but allow a nearly empty bar. This
    # distinguishes the frame from white UI/background lines while still
    # detecting a target whose health is almost depleted.
    fill_score = min(1.0, (fill_ratio or 0.0) / 0.08)
    structure_score = float(
        (sum(row_scores) / len(row_scores)) * 0.72
        + side_score * 0.13
        + fill_score * 0.15
    )
    return structure_score, fill_ratio, row_scores


def _row_ratio(row: np.ndarray) -> float:
    return float(np.count_nonzero(row) / max(1, row.size))


def _best_row_ratio(
    mask: np.ndarray,
    row: int,
    left: int,
    right: int,
    radius: int = 1,
) -> float:
    start = max(0, row - radius)
    stop = min(mask.shape[0], row + radius + 1)
    return max(
        (
            _row_ratio(mask[index, left:right])
            for index in range(start, stop)
        ),
        default=0.0,
    )


def _column_ratio(column: np.ndarray) -> float:
    return float(np.count_nonzero(column) / max(1, column.size))
