from maplevision.pathing import (
    current_point_name,
    minimap_local_rect,
    minimap_scaled_point_to_local,
    minimap_scaled_rect_to_local,
    path_name,
    point_in_region,
    step_label,
)


def test_minimap_local_rect_clamps_frame_selection_to_map():
    assert minimap_local_rect((3, 8, 20, 20), (5, 10, 30, 15)) == (
        0,
        0,
        18,
        15,
    )


def test_scaled_minimap_rect_is_converted_back_to_local_pixels():
    assert minimap_scaled_rect_to_local((75, 102, 123, 33), 3) == (
        25,
        34,
        41,
        11,
    )


def test_scaled_minimap_point_is_converted_back_to_local_pixels():
    assert minimap_scaled_point_to_local((123, 204), 3) == (41, 68)


def test_point_region_and_current_point_name_use_tolerance():
    points = [
        {"name": "点1", "region": [10, 20, 12, 8]},
        {"name": "点2", "region": [40, 20, 12, 8]},
    ]

    assert point_in_region((9, 24), (10, 20, 12, 8), tolerance=1)
    assert current_point_name((9, 24), points, tolerance=1) == "点1"
    assert current_point_name((30, 24), points, tolerance=1) is None


def test_path_name_and_step_label_are_stable():
    assert path_name("点1", "点2") == "点1 -> 点2"
    assert step_label(
        {
            "type": "climb",
            "ladder": "梯子1",
            "key": "up",
            "duration_ms": 1800,
        }
    ) == "向上爬梯 梯子1 1800ms"
