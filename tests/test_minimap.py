import numpy as np

from maplevision.vision.minimap import observe_minimap


def test_minimap_player_point_is_reported_in_local_and_global_coordinates():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[8:10, 13:15] = [0, 255, 255]

    observation = observe_minimap(
        frame,
        region=(10, 5, 12, 10),
        lower_hsv=(20, 120, 120),
        upper_hsv=(40, 255, 255),
    )

    assert observation.player_local is not None
    assert observation.player_global is not None
    assert (observation.player_local.x, observation.player_local.y) == (3, 3)
    assert (observation.player_global.x, observation.player_global.y) == (13, 8)
    assert observation.horizontal_ratio == 3 / 11


def test_minimap_uses_the_compact_marker_instead_of_large_yellow_background():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[2:3, 2:3] = [0, 255, 255]
    frame[8:14, 13:19] = [0, 255, 255]
    frame[5:10, 22:29] = [0, 180, 180]

    observation = observe_minimap(
        frame,
        region=(0, 0, 30, 20),
        lower_hsv=(20, 120, 120),
        upper_hsv=(40, 255, 255),
    )

    assert observation.player_global is not None
    assert (observation.player_global.x, observation.player_global.y) == (15, 10)
