from dataclasses import replace

import numpy as np

import maplevision.route_test as route_test_module
from maplevision.route_test import (
    RouteTestController,
)
from maplevision.vision.minimap import MinimapObservation
from maplevision.vision.player import Point
from maplevision.vision.player_screen import PlayerScreenDetection


def _config():
    return {
        "route_combat": {
            "point_tolerance": 2,
            "ladder_tolerance": 2,
            "jump_key": "alt",
            "jump_point_tolerance": 8,
            "climb_minimap_rise_px": 1,
            "points": [
                {"name": "点1", "region": [0, 0, 10, 10]},
                {"name": "点2", "region": [10, 0, 10, 10]},
            ],
            "ladders": [
                {
                    "name": "梯子1",
                    "point": [5, 5],
                    "climb_duration_ms": 1800,
                }
            ],
        },
        "minimap": {
            "region": [0, 0, 30, 20],
            "player_color_hsv": {
                "lower": [20, 100, 100],
                "upper": [40, 255, 255],
            },
        },
        "combat": {
            "monsters": {"region": [0, 0, 800, 600]},
            "player": {"enabled": True},
        },
    }


def test_route_move_jumps_when_yellow_point_stops(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(route_test_module.random, "randint", lambda _low, _high: 500)
    config = _config()
    config["route_combat"].update(
        {
            "obstacle_check_interval_ms": 300,
            "obstacle_jump_delay_min_ms": 500,
            "obstacle_jump_delay_max_ms": 700,
        }
    )
    config["route_combat"]["ladders"][0]["point"] = [15, 5]
    controller = RouteTestController(
        config,
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [{"type": "move_to_ladder", "ladder": "梯子1"}],
        },
    )
    current = [
        MinimapObservation(
            region=(0, 0, 30, 20),
            player_local=Point(0, 5),
            player_global=Point(0, 5),
        )
    ]
    controller._observe_minimap = lambda _frame: current[0]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert controller.process_frame(frame).action == "move_right(right)"
    clock[0] = 0.30
    jumped = controller.process_frame(frame)
    assert jumped.action == "jump_right(right,alt)"
    assert "obstacle-no-progress:0px" in jumped.reason
    clock[0] = 0.80
    retry = controller.process_frame(frame)
    assert retry.action == "move_right(right)"
    assert "retry-direction" in retry.reason
    clock[0] = 0.95
    assert controller.process_frame(frame).action == "jump_right(right,alt)"

    current[0] = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(3, 5),
        player_global=Point(3, 5),
    )
    clock[0] = 1.45
    progress = controller.process_frame(frame)
    assert progress.action == "move_right(right)"
    assert "obstacle-jump-progress:+3px" in progress.reason


def test_directional_move_stops_at_target_point_fixed_marker(monkeypatch):
    config = _config()
    config["route_combat"]["point_profiles"] = {
        "点2": {
            "fixed_point": [18, 7],
            "fixed_point_enabled": True,
        }
    }
    controller = RouteTestController(
        config,
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [{"type": "move_right"}],
        },
    )
    current = [
        MinimapObservation(
            region=(0, 0, 30, 20),
            player_local=Point(5, 5),
            player_global=Point(5, 5),
        )
    ]
    controller._observe_minimap = lambda _frame: current[0]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    moving = controller.process_frame(frame)
    assert moving.action == "move_right(right)"
    assert controller.step_index == 0

    current[0] = replace(
        current[0],
        player_local=Point(18, 7),
        player_global=Point(18, 7),
    )
    arrived = controller.process_frame(frame)
    assert arrived.action == "observe"
    assert arrived.state == "complete"
    assert arrived.reason == "route-complete"


def test_move_to_point_uses_fixed_marker_instead_of_region(monkeypatch):
    config = _config()
    config["route_combat"]["point_profiles"] = {
        "点2": {
            "fixed_point": [18, 7],
            "fixed_point_enabled": True,
        }
    }
    controller = RouteTestController(
        config,
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [{"type": "move_to_point", "point": "点2"}],
        },
    )
    current = [
        MinimapObservation(
            region=(0, 0, 30, 20),
            player_local=Point(12, 5),
            player_global=Point(12, 5),
        )
    ]
    controller._observe_minimap = lambda _frame: current[0]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    moving = controller.process_frame(frame)
    assert moving.action == "move_right(right)"
    assert controller.step_index == 0

    current[0] = replace(
        current[0],
        player_local=Point(18, 7),
        player_global=Point(18, 7),
    )
    arrived = controller.process_frame(frame)
    assert arrived.state == "complete"


def test_route_test_climb_reaches_target_point(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    controller = RouteTestController(
        _config(),
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [
                {"type": "move_to_ladder", "ladder": "梯子1"},
                {
                    "type": "climb",
                    "ladder": "梯子1",
                    "key": "up",
                    "duration_ms": 1800,
                },
            ],
        },
    )
    grounded_minimap = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 5),
        player_global=Point(5, 5),
    )
    lifted_minimap = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 3),
        player_global=Point(5, 3),
    )
    grounded_player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(100, 500, 40, 10),
        anchor=(120, 534),
    )
    lifted_player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(100, 100, 40, 10),
        anchor=(120, 134),
    )
    current_minimap = [grounded_minimap]
    controller._observe_minimap = lambda _frame: current_minimap[0]
    controller._detect_player = lambda _frame: grounded_player
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    first = controller.process_frame(frame)
    assert first.action == "tap_alt"
    assert first.reason == "route-climb-alt"

    clock[0] = 0.35
    second = controller.process_frame(frame)
    assert second.action == "observe"

    clock[0] = 0.60
    third = controller.process_frame(frame)
    assert third.action == "hold_up"
    assert controller.phase == "climb-lift-wait"

    current_minimap[0] = lifted_minimap
    clock[0] = 1.60
    fourth = controller.process_frame(frame)
    assert fourth.action == "hold_up"
    assert controller.phase == "climb-hold"

    clock[0] = 2.41
    fifth = controller.process_frame(frame)
    assert fifth.action == "observe"
    assert fifth.reason == "route-climb-release"

    target_minimap = replace(
        lifted_minimap,
        player_local=Point(15, 5),
        player_global=Point(15, 5),
    )
    controller._observe_minimap = lambda _frame: target_minimap
    clock[0] = 2.50
    sixth = controller.process_frame(frame)
    assert sixth.state == "complete"
    assert sixth.reason == "route-complete"


def test_route_test_climb_continues_to_next_step_before_target_check(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    config = _config()
    config["route_combat"]["point_profiles"] = {
        "点2": {
            "fixed_point": [18, 3],
            "fixed_point_enabled": True,
        }
    }
    controller = RouteTestController(
        config,
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [
                {"type": "move_to_ladder", "ladder": "梯子1"},
                {"type": "climb", "ladder": "梯子1", "duration_ms": 1800},
                {"type": "move_to_point", "point": "点2"},
            ],
        },
    )
    current = [
        MinimapObservation(
            region=(0, 0, 30, 20),
            player_local=Point(5, 5),
            player_global=Point(5, 5),
        )
    ]
    controller._observe_minimap = lambda _frame: current[0]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert controller.process_frame(frame).reason == "route-climb-alt"
    clock[0] = 0.60
    assert controller.process_frame(frame).reason == "route-climb-up"
    current[0] = replace(
        current[0],
        player_local=Point(5, 3),
        player_global=Point(5, 3),
    )
    clock[0] = 1.60
    assert controller.process_frame(frame).reason == "route-climb-hold"
    clock[0] = 2.41
    assert controller.process_frame(frame).reason == "route-climb-release"
    clock[0] = 2.50
    moving = controller.process_frame(frame)

    assert moving.action == "move_right(right)"
    assert moving.reason == "route-move_to_point;obstacle-move-start"
    assert controller.step_index == 2


def test_route_test_retries_when_minimap_yellow_point_did_not_rise(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    controller = RouteTestController(
        _config(),
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [
                {"type": "move_to_ladder", "ladder": "梯子1"},
                {"type": "climb", "ladder": "梯子1", "duration_ms": 1800},
            ],
        },
    )
    minimap = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 5),
        player_global=Point(5, 5),
    )
    grounded_player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(100, 500, 40, 10),
        anchor=(120, 534),
    )
    controller._observe_minimap = lambda _frame: minimap
    controller._detect_player = lambda _frame: None
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert controller.process_frame(frame).action == "tap_alt"
    clock[0] = 0.35
    assert controller.process_frame(frame).action == "observe"
    clock[0] = 0.60
    assert controller.process_frame(frame).action == "hold_up"
    clock[0] = 1.60
    retry = controller.process_frame(frame)

    assert retry.action == "observe"
    assert retry.reason == "route-climb-not-lifted"
    assert controller.climb_attempts == 2
    assert controller.phase == "reposition-ladder"

    repositioned = controller.process_frame(frame)
    assert repositioned.action == "observe"
    assert repositioned.reason == "route-ladder-repositioned"

    next_attempt = controller.process_frame(frame)
    assert next_attempt.action == "tap_alt"
    assert next_attempt.reason == "route-climb-alt"


def test_route_test_uses_configured_alt_delay(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    config = _config()
    config["route_combat"]["alt_delay_ms"] = 800
    controller = RouteTestController(
        config,
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [
                {"type": "move_to_ladder", "ladder": "梯子1"},
                {"type": "climb", "ladder": "梯子1", "duration_ms": 1800},
            ],
        },
    )
    minimap = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 5),
        player_global=Point(5, 5),
    )
    controller._observe_minimap = lambda _frame: minimap
    controller._detect_player = lambda _frame: PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(100, 100, 40, 10),
        anchor=(120, 134),
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert controller.process_frame(frame).action == "tap_alt"
    clock[0] = 0.79
    assert controller.process_frame(frame).action == "observe"
    clock[0] = 0.80
    assert controller.process_frame(frame).action == "hold_up"


def test_route_test_jump_holds_direction_and_taps_alt_once(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    controller = RouteTestController(
        _config(),
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [
                {"type": "jump_right", "duration_ms": 500},
            ],
        },
    )
    grounded = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 10),
        player_global=Point(5, 10),
    )
    lifted = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 8),
        player_global=Point(5, 8),
    )
    current_minimap = [grounded]
    controller._observe_minimap = lambda _frame: current_minimap[0]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    first = controller.process_frame(frame)
    assert first.action == "jump_right(right,alt)"
    assert first.reason == "route-jump-start"
    clock[0] = 0.20
    second = controller.process_frame(frame)
    assert second.action == "jump_right(right,alt)"
    assert second.reason == "route-jump-hold"
    current_minimap[0] = lifted
    clock[0] = 0.60
    third = controller.process_frame(frame)
    assert third.action == "observe"
    assert third.reason == "route-jump-release"
    clock[0] = 0.75
    fourth = controller.process_frame(frame)
    assert fourth.reason == "route-jump-success"
    assert fourth.state == "running"
    clock[0] = 0.90
    fifth = controller.process_frame(frame)
    assert fifth.state == "complete"


def test_route_test_retries_jump_when_minimap_y_does_not_rise(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    controller = RouteTestController(
        _config(),
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [
                {"type": "move_to_ladder", "ladder": "梯子1"},
                {"type": "jump_right", "duration_ms": 500},
            ],
        },
    )
    minimap = MinimapObservation(
        region=(0, 0, 30, 20),
        player_local=Point(5, 5),
        player_global=Point(5, 5),
    )
    controller._observe_minimap = lambda _frame: minimap
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert controller.process_frame(frame).action == "jump_right(right,alt)"
    clock[0] = 0.60
    assert controller.process_frame(frame).reason == "route-jump-release"
    clock[0] = 0.75
    failed = controller.process_frame(frame)
    assert failed.reason == "route-jump-not-raised"
    assert controller.phase == "jump-reposition"
    clock[0] = 0.90
    repositioned = controller.process_frame(frame)
    assert repositioned.reason == "route-jump-point-repositioned"
    clock[0] = 1.05
    retried = controller.process_frame(frame)
    assert retried.action == "jump_right(right,alt)"
    assert retried.reason == "route-jump-start"


def test_route_test_uses_configured_jump_key(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route_test_module.time, "monotonic", lambda: clock[0])
    config = _config()
    config["route_combat"]["jump_key"] = "space"
    controller = RouteTestController(
        config,
        {
            "name": "点1 -> 点2",
            "source": "点1",
            "target": "点2",
            "steps": [{"type": "jump_right", "duration_ms": 500}],
        },
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    result = controller.process_frame(frame)

    assert result.action == "jump_right(right,space)"
