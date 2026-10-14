import numpy as np

import maplevision.vision.combat as combat_module
from maplevision.vision.combat import (
    HealthObservation,
    observe_combat,
    observe_mana,
    observe_health,
)
from maplevision.vision.monster import MonsterDetection
from maplevision.vision.player_screen import PlayerScreenDetection


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
    )
    assert combat.action == "heal"
    assert combat.reason == "hp-low"


def test_mana_observation_reads_fixed_blue_bar_ratio():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[583:597, 328:380] = [255, 0, 0]
    mana = observe_mana(
        frame,
        {
            "enabled": True,
            "region": [328, 583, 105, 14],
            "expected_width": 105,
            "low_threshold": 0.60,
            "color_hsv": {
                "ranges": [
                    {"lower": [95, 80, 80], "upper": [125, 255, 255]},
                ]
            },
        },
    )

    assert mana.ratio == 52 / 105
    assert mana.healthy is False


def test_combat_prioritizes_heal_then_mana_before_monsters(monkeypatch):
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(
        combat_module,
        "observe_health",
        lambda *_: HealthObservation(None, 0.8, True),
    )
    monkeypatch.setattr(
        combat_module,
        "observe_mana",
        lambda *_: HealthObservation((328, 583, 105, 14), 0.2, False),
    )

    combat = observe_combat(
        frame,
        {
            "heal_action": "heal(1)",
            "mana_action": "mana(2)",
            "attack_action": "attack",
            "observe_action": "observe",
        },
    )

    assert combat.action == "mana(2)"
    assert combat.reason == "mp-low"


def test_combat_requires_screen_player_before_attack(monkeypatch):
    target = _target(200, 100)
    monkeypatch.setattr(combat_module, "observe_health", lambda *_: _healthy())
    monkeypatch.setattr(combat_module, "detect_monsters", lambda *_: (target,))
    monkeypatch.setattr(combat_module, "detect_player_screen", lambda *_: None)

    combat = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat_config(),
    )

    assert combat.action == "observe"
    assert combat.target is None
    assert combat.reason == "player-not-found"


def test_combat_skips_player_detection_when_no_monster(monkeypatch):
    called = False

    def fail_if_called(*_):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(combat_module, "observe_health", lambda *_: _healthy())
    monkeypatch.setattr(combat_module, "detect_monsters", lambda *_: ())
    monkeypatch.setattr(combat_module, "detect_player_screen", fail_if_called)

    combat = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat_config(),
    )

    assert combat.reason == "no-target"
    assert combat.player is None
    assert called is False


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
    )

    assert combat.action == "attack"
    assert combat.target == nearer
    assert combat.in_range_targets == (farther, nearer)
    assert combat.reason == "confirmed-monster-in-attack-range"
    assert combat.target_direction == "right"


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
    )

    assert combat.target is None
    assert combat.action == "move_right"
    assert combat.reason == "nearest-monster-out-of-attack-range"
    assert combat.target_direction == "right"


def test_combat_moves_toward_nearest_target_and_attacks_at_distance(monkeypatch):
    farther = _target(260, 100)
    nearer = _target(190, 100)
    player = PlayerScreenDetection(
        name="Conquerork",
        score=0.99,
        name_box=(100, 100, 40, 20),
        anchor=(150, 110),
    )
    monkeypatch.setattr(combat_module, "observe_health", lambda *_: _healthy())
    monkeypatch.setattr(
        combat_module,
        "detect_monsters",
        lambda *_: (farther, nearer),
    )
    monkeypatch.setattr(combat_module, "detect_player_screen", lambda *_: player)

    moving = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        {
            "move_left_action": "left",
            "move_right_action": "right",
            "attack_action": "ctrl",
            "observe_action": "observe",
            "player": {"enabled": True},
            "attack_range": {"distance": 20, "y": 50},
        },
    )
    assert moving.action == "right"
    assert moving.nearest_target == nearer
    assert moving.target_direction == "right"

    monkeypatch.setattr(
        combat_module,
        "detect_monsters",
        lambda *_: (_target(165, 100), farther),
    )
    attacking = observe_combat(
        np.zeros((240, 320, 3), dtype=np.uint8),
        {
            "move_left_action": "left",
            "move_right_action": "right",
            "attack_action": "ctrl",
            "observe_action": "observe",
            "player": {"enabled": True},
            "attack_range": {"distance": 25, "y": 50},
        },
    )
    assert attacking.action == "ctrl"
    assert attacking.reason == "confirmed-monster-in-attack-range"


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
