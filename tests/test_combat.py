import numpy as np

from maplevision.vision.combat import (
    observe_combat,
    observe_health,
)
from maplevision.vision.minimap import MinimapObservation
from maplevision.vision.player import Point


def test_health_observation_reads_red_fill_ratio():
    frame = np.zeros((20, 120, 3), dtype=np.uint8)
    frame[6:14, 0:60] = [0, 0, 255]
    health = observe_health(
        frame,
        {
            "enabled": True,
            "region": [0, 0, 100, 20],
            "expected_width": 100,
            "low_threshold": 0.7,
            "color_hsv": {
                "ranges": [
                    {"lower": [0, 100, 80], "upper": [12, 255, 255]},
                    {"lower": [168, 100, 80], "upper": [179, 255, 255]},
                ]
            },
        },
    )
    assert health.ratio == 0.6
    assert health.healthy is False


def test_combat_prioritizes_heal_over_attack():
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[10:14, 45:49] = [0, 0, 255]
    frame[42:46, 0:20] = [0, 0, 255]
    minimap = MinimapObservation(
        region=(0, 0, 100, 40),
        player_local=Point(40, 12),
        player_global=Point(40, 12),
    )
    combat = observe_combat(
        frame,
        {
            "heal_action": "heal",
            "attack_action": "attack",
            "observe_action": "observe",
            "screen_targets": {
                "enabled": False,
                "region": [0, 0, 100, 40],
                "color_hsv": {
                    "ranges": [
                        {"lower": [0, 100, 80], "upper": [12, 255, 255]},
                    ]
                },
                "min_area": 4,
                "max_area": 100,
            },
            "health": {
                "enabled": True,
                "region": [0, 40, 100, 10],
                "expected_width": 100,
                "low_threshold": 0.5,
                "color_hsv": {
                    "ranges": [
                        {"lower": [0, 100, 80], "upper": [12, 255, 255]},
                        {"lower": [168, 100, 80], "upper": [179, 255, 255]},
                    ]
                },
            },
        },
        minimap,
    )
    assert combat.action == "heal"
    assert combat.reason == "hp-low"
