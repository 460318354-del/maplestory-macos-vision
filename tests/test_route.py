import numpy as np

from maplevision.vision.player import Point
from maplevision.vision.route import nearest_action


def test_nearest_action_uses_color_key():
    route = np.zeros((5, 5, 3), dtype=np.uint8)
    route[2, 3] = [255, 0, 0]
    action = nearest_action(
        route,
        Point(2, 2),
        {"255,0,0": "left"},
        radius=2,
    )
    assert action == "left"
