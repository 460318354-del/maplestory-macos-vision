import numpy as np

from maplevision.vision.minimap import observe_minimap


def test_minimap_player_point_is_reported_in_local_and_global_coordinates():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    # BGR value (0, 255, 255) is yellow and falls in the configured HSV range.
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


def test_minimap_uses_the_largest_color_component():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[2:3, 2:3] = [0, 255, 255]
    frame[8:11, 13:16] = [0, 255, 255]
    observation = observe_minimap(
        frame,
        region=(0, 0, 30, 20),
        lower_hsv=(20, 120, 120),
        upper_hsv=(40, 255, 255),
    )
    assert observation.player_global is not None
    assert (observation.player_global.x, observation.player_global.y) == (14, 9)
