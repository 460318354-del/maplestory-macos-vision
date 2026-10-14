from __future__ import annotations

import numpy as np

from maplevision.vision.enemy_health_bar import EnemyHealthBarDetector


def _frame_with_bar(fill_width: int | None = 24):
    frame = np.full((120, 180, 3), 180, dtype=np.uint8)
    x, y, width, height = 64, 35, 52, 10
    frame[y : y + height, x : x + width] = (0, 0, 0)
    frame[y + 1, x + 1 : x + width - 1] = (255, 255, 255)
    frame[y + height - 2, x + 1 : x + width - 1] = (255, 255, 255)
    frame[y + 2, x + 1 : x + width - 1] = (0, 0, 0)
    frame[y + height - 3, x + 1 : x + width - 1] = (0, 0, 0)
    if fill_width:
        frame[
            y + 3 : y + height - 3,
            x + 2 : x + 2 + fill_width,
        ] = (0, 220, 0)
    return frame


def _detector():
    return EnemyHealthBarDetector(
        {
            "enabled": True,
            "widths": [52],
            "height": 10,
            "search_x_padding": 2,
            "search_top_offset": -30,
            "search_bottom_offset": 5,
            "structure_threshold": 0.60,
        }
    )


def test_enemy_health_bar_detects_fixed_black_white_structure():
    frame = _frame_with_bar()

    observation = _detector().detect(frame, (70, 50, 40, 30))

    assert observation.present is True
    assert observation.region == (64, 35, 52, 10)
    assert observation.score >= 0.60
    assert observation.fill_ratio is not None
    assert observation.search_region is not None
    assert len(observation.row_scores) == 6


def test_enemy_health_bar_does_not_require_full_green_fill():
    frame = _frame_with_bar(fill_width=4)

    observation = _detector().detect(frame, (70, 50, 40, 30))

    assert observation.present is True
    assert observation.fill_ratio is not None


def test_enemy_health_bar_rejects_plain_background():
    frame = np.full((120, 180, 3), 180, dtype=np.uint8)

    observation = _detector().detect(frame, (70, 50, 40, 30))

    assert observation.present is False


def test_enemy_health_bar_tolerates_one_pixel_vertical_boundary_shift():
    frame = _frame_with_bar()
    shifted = np.full_like(frame, 180)
    shifted[34:44, 64:116] = frame[35:45, 64:116]

    observation = _detector().detect(shifted, (70, 50, 40, 30))

    assert observation.present is True
    assert observation.region is not None
    assert observation.region[1] == 34
