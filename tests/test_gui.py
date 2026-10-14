import numpy as np

from maplevision.gui import (
    crop_region,
    minimap_frame_point,
    minimap_local_point,
    player_search_region,
    safe_template_category,
)
from maplevision.vision.height_band import screen_region_from_band


def test_crop_region_clips_to_frame_bounds():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    frame[2:7, 4:9] = [1, 2, 3]

    cropped = crop_region(frame, (2, 1, 8, 8))

    assert cropped.shape == (8, 8, 3)
    assert tuple(cropped[1, 2]) == (1, 2, 3)


def test_crop_region_returns_empty_for_region_outside_frame():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)

    cropped = crop_region(frame, (20, 20, 4, 4))

    assert cropped.size == 0


def test_direct_screen_region_uses_frame_y_without_projection():
    assert screen_region_from_band((240, 510), 1600, 900) == (
        0,
        240,
        1600,
        270,
    )


def test_direct_screen_region_clamps_and_sorts_frame_y():
    assert screen_region_from_band((1100, -40), 1600, 900) == (
        0,
        0,
        1600,
        900,
    )


def test_player_search_region_uses_marked_region_when_available():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert player_search_region(frame, (0, 220, 800, 160)) == [0, 220, 800, 160]


def test_player_search_region_falls_back_to_full_frame():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert player_search_region(frame, None) == [0, 0, 800, 600]


def test_patrol_and_fixed_points_use_independent_minimap_local_coordinates():
    region = (5, 72, 157, 85)

    patrol_point = minimap_local_point((20, 84), region)
    fixed_point = minimap_local_point((130, 140), region)

    assert patrol_point == (15, 12)
    assert fixed_point == (125, 68)
    assert patrol_point != fixed_point
    assert minimap_frame_point(patrol_point, region) == (20, 84)
    assert minimap_frame_point(fixed_point, region) == (130, 140)


def test_minimap_local_point_rejects_points_outside_top_left_minimap():
    region = (5, 72, 157, 85)

    assert minimap_local_point((200, 84), region) is None
    assert minimap_frame_point((10, 20), ()) == (10, 20)


def test_safe_template_category_keeps_role_label_and_removes_path_separators():
    assert safe_template_category("角色") == "角色"
    assert safe_template_category("new/boss") == "new_boss"
