import numpy as np

import maplevision.vision.combat as combat_module
from maplevision.vision.combat import (
    HealthObservation,
    observe_combat,
    observe_health,
)
from maplevision.vision.monster import MonsterDetection
from maplevision.vision.minimap import MinimapObservation
from maplevision.vision.player_screen import PlayerScreenDetection
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


def test_combat_requires_screen_player_before_attack(monkeypatch):
    target = _target(200, 100)
    monkeypatch.setattr(combat_module, "observe_health", lambda *_: _healthy())
    monkeypatch.setattr(combat_module, "detect_monsters", lambda *_: (target,))
    monkeypatch.setattr(combat_module, "detect_player_screen", lambda *_: None)

    combat = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat_config(),
        _minimap(),
    )

    assert combat.action == "observe"
    assert combat.target is None
    assert combat.reason == "player-not-found"


def test_combat_attacks_nearest_confirmed_target_inside_configured_range(monkeypatch):
    farther = _target(240, 100)
    nearer = _target(180, 100)
    player = PlayerScreenDetection(
        name="Conquerork",
        score=0.99,
        name_box=(100, 100, 40, 20),
        anchor=(150, 110),
    )
    monkeypatch.setattr(combat_module, "observe_health", lambda *_: _healthy())
    monkeypatch.setattr(combat_module, "detect_monsters", lambda *_: (farther, nearer))
    monkeypatch.setattr(combat_module, "detect_player_screen", lambda *_: player)

    combat = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat_config(),
        _minimap(),
    )

    assert combat.action == "attack"
    assert combat.target == nearer
    assert combat.in_range_targets == (farther, nearer)
    assert combat.reason == "confirmed-monster-in-attack-range"


def test_combat_observes_when_confirmed_monster_is_out_of_range(monkeypatch):
    target = _target(260, 100)
    player = PlayerScreenDetection(
        name="Conquerork",
        score=0.99,
        name_box=(100, 100, 40, 20),
        anchor=(150, 110),
    )
    monkeypatch.setattr(combat_module, "observe_health", lambda *_: _healthy())
    monkeypatch.setattr(combat_module, "detect_monsters", lambda *_: (target,))
    monkeypatch.setattr(combat_module, "detect_player_screen", lambda *_: player)

    combat = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat_config(),
        _minimap(),
    )

    assert combat.action == "observe"
    assert combat.target is None
    assert combat.reason == "no-monster-in-attack-range"


def _healthy():
    return HealthObservation(region=None, ratio=1.0, healthy=True)


def _target(x, y):
    return MonsterDetection(
        label="stump",
        score=0.95,
        box=(x, y, 20, 20),
    )


def _combat_config():
    return {
        "attack_action": "attack",
        "observe_action": "observe",
        "player": {"enabled": True},
        "attack_range": {"x": 100, "y": 50},
    }


def _minimap():
    return MinimapObservation(
        region=(0, 0, 10, 10),
        player_local=None,
        player_global=None,
    )
