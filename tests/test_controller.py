from dataclasses import replace
import time

import numpy as np

import maplevision.decision.controller as controller_module
from maplevision.decision.controller import VisionController
from maplevision.decision.controller import _log_monster_sides
from maplevision.platform.macos.input import MacOSInputController
from maplevision.vision.combat import CombatObservation, HealthObservation
from maplevision.vision.monster import MonsterDetection
from maplevision.vision.minimap import MinimapObservation
from maplevision.vision.player import Point
from maplevision.vision.player_screen import PlayerScreenDetection


def test_controller_scans_coins_when_no_monster_target():
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=_config(pickup_enabled=True),
    )
    coin = MonsterDetection("金币", 0.95, (102, 120, 18, 18))
    second_coin = MonsterDetection("金币", 0.94, (280, 120, 18, 18))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )
    controller._detect_coins = lambda _frame: (coin, second_coin)
    controller._detect_player_for_auxiliary_action = lambda _frame: player

    combat = _combat(
        reason="no-target",
    )

    result = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        combat,
    )

    assert result.reason == "pickup-coins"
    assert result.action == "pickup(z,3)"
    assert result.player == player


def test_log_monster_sides_reports_both_directions(caplog):
    player = PlayerScreenDetection(
        name="角色",
        score=0.95,
        name_box=(130, 80, 30, 12),
        anchor=(150, 120),
    )
    left = MonsterDetection("stump", 0.95, (110, 110, 20, 20))
    right = MonsterDetection("red_snail", 0.94, (180, 110, 20, 20))
    combat = _combat(
        reason="confirmed-monster-in-attack-range",
        player=player,
        targets=(left, right),
        confirmed_targets=(left, right),
    )

    with caplog.at_level("INFO"):
        _log_monster_sides(combat)

    messages = [record.getMessage() for record in caplog.records]
    assert any("发现左边怪物" in message for message in messages)
    assert any("发现右边怪物" in message for message in messages)


def test_controller_prioritizes_monster_target_over_coins():
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=_config(pickup_enabled=True),
    )
    target = MonsterDetection("stump", 0.95, (90, 110, 20, 20))
    controller._detect_coins = lambda _frame: (
        (_ for _ in ()).throw(AssertionError("coins must not be scanned"))
    )
    combat = _combat(
        player=None,
        targets=(target,),
        confirmed_targets=(target,),
        reason="confirmed-monster-in-attack-range",
    )

    result = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        combat,
    )

    assert result == combat


def test_coin_pickup_waits_for_configured_interval(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "maplevision.decision.controller.time.monotonic",
        lambda: now[0],
    )
    config = _config(pickup_enabled=True)
    config["combat"]["pickup"].update(
        {"interval_seconds": 10.0, "scan_seconds": 1.0}
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller._detect_coins = lambda _frame: (
        (_ for _ in ()).throw(AssertionError("must not scan before interval"))
    )

    result = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )

    assert result.reason == "no-target"
    assert controller.loot_phase == "waiting"
    now[0] = 110.0


def test_coin_pickup_cycle_handles_one_coin_then_restarts_timer(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "maplevision.decision.controller.time.monotonic",
        lambda: now[0],
    )
    config = _config(pickup_enabled=True)
    config["combat"]["pickup"].update(
        {"interval_seconds": 10.0, "scan_seconds": 1.0, "min_coins": 1}
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.next_pickup_at = now[0]
    far_coin = MonsterDetection("金币", 0.95, (130, 120, 18, 18))
    near_coin = MonsterDetection("金币", 0.95, (105, 120, 18, 18))
    other_coin = MonsterDetection("金币", 0.95, (280, 120, 18, 18))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )
    detected = [
        (far_coin, other_coin),
        (near_coin, other_coin),
        (near_coin, other_coin),
    ]
    controller._detect_coins = lambda _frame: detected.pop(0)
    controller._detect_player_for_auxiliary_action = lambda _frame: player

    first = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )
    assert first.reason.startswith("move-to-coins:")
    assert controller.loot_phase == "picking"
    assert controller.pickup_target == far_coin

    now[0] = 100.5
    second = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )
    assert second.reason == "pickup-coins"
    assert controller.loot_phase == "picking"
    assert controller.pickup_target == near_coin
    assert controller.pickup_action_sent is True

    now[0] = 101.6
    third = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )
    assert third.reason == "no-target"
    assert controller.loot_phase == "waiting"
    assert controller.next_pickup_at == 111.6
    assert controller.pickup_target is None


def test_coin_move_survives_one_frame_without_coin_detection(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "maplevision.decision.controller.time.monotonic",
        lambda: now[0],
    )
    config = _config(pickup_enabled=True)
    config["combat"]["pickup"].update(
        {
            "interval_seconds": 0.0,
            "scan_seconds": 1.0,
            "min_coins": 1,
            "target_missing_grace_seconds": 0.45,
        }
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    far_coin = MonsterDetection("金币", 0.95, (200, 120, 18, 18))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )
    detected = [(far_coin,), ()]
    controller._detect_coins = lambda _frame: detected.pop(0)
    controller._detect_player_for_auxiliary_action = lambda _frame: player
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    first = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))
    assert first.reason.startswith("move-to-coins:")
    assert controller.pickup_move_direction == "right"

    now[0] = 100.2
    second = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))
    assert second.reason.startswith("move-to-coins-target-lost:")


def test_coin_pickup_uses_obstacle_jump_when_moving_to_coin(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "maplevision.decision.controller.time.monotonic",
        lambda: now[0],
    )
    config = _config(pickup_enabled=True)
    config["combat"]["pickup"].update(
        {"interval_seconds": 0.0, "scan_seconds": 1.0, "min_coins": 1}
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    far_coin = MonsterDetection("金币", 0.95, (200, 120, 18, 18))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )

    class FakePickupMoveController:
        def process_combat_approach(self, _frame, direction):
            assert direction == "right"
            return type(
                "RouteResult",
                (),
                {
                    "action": "jump_right(right,alt)",
                    "state": "running",
                    "reason": "route-combat-approach;obstacle-jump",
                    "player": None,
                },
            )()

        def reset_obstacle_state(self):
            return None

    controller.pickup_move_controller = FakePickupMoveController()
    controller._detect_coins = lambda _frame: (far_coin,)
    controller._detect_player_for_auxiliary_action = lambda _frame: player

    result = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )

    assert result.action == "jump_right(right,alt)"
    assert result.reason.startswith("move-to-coins:")


def test_coin_pickup_preempts_active_attack_when_cycle_starts(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "maplevision.decision.controller.time.monotonic",
        lambda: now[0],
    )
    config = _config(pickup_enabled=True)
    config["combat"]["pickup"].update(
        {"interval_seconds": 0.0, "scan_seconds": 1.0, "min_coins": 2}
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    coins = (
        MonsterDetection("金币", 0.95, (102, 120, 18, 18)),
        MonsterDetection("金币", 0.94, (180, 120, 18, 18)),
    )
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )
    controller._detect_coins = lambda _frame: coins
    controller._detect_player_for_auxiliary_action = lambda _frame: player
    target = MonsterDetection("stump", 0.95, (90, 110, 20, 20))

    result, active = controller._apply_pickup_priority(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(
            player=player,
            targets=(target,),
            confirmed_targets=(target,),
            reason="confirmed-monster-in-attack-range",
        ),
    )

    assert active is True
    assert result.reason == "pickup-coins"
    assert result.action == "pickup(z,3)"


def test_path_combat_uses_profile_for_current_minimap_point():
    config = _config(pickup_enabled=False)
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "point_tolerance": 1,
        "points": [
            {"name": "点1", "region": [0, 0, 10, 10]},
            {"name": "点2", "region": [30, 0, 10, 10]},
        ],
        "point_profiles": {
            "点1": {
                "fixed_point": [4, 5],
                "fixed_point_enabled": True,
                "monster_region": [0, 100, 400, 80],
            },
            "点2": {
                "fixed_point": [40, 6],
                "fixed_point_enabled": True,
                "monster_region": [0, 200, 400, 90],
            },
        },
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(32, 4),
            player_global=Point(32, 4),
        )
    )

    controller._activate_route_combat_profile(np.zeros((240, 320, 3), dtype=np.uint8))

    assert controller.route_active_point == "点2"
    assert controller.config["combat"]["monsters"]["region"] == [0, 200, 400, 90]
    assert controller.config["combat"]["fixed_point"]["point"] == [40, 6]
    assert controller.config["combat"]["fixed_point"]["enabled"] is True


def test_path_combat_limits_current_point_detection_to_active_route_group():
    config = _config(pickup_enabled=False)
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "active_group": "猪",
        "point_tolerance": 1,
        "points": [
            {
                "name": "8蘑菇点1",
                "group": "8蘑菇",
                "region": [10, 10, 40, 12],
            },
            {"name": "猪1", "group": "猪", "region": [10, 10, 40, 12]},
        ],
        "point_profiles": {
            "8蘑菇点1": {"monster_region": [0, 100, 400, 80]},
            "猪1": {"monster_region": [0, 200, 400, 90]},
        },
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(20, 12),
            player_global=Point(20, 12),
        )
    )

    controller._activate_route_combat_profile(np.zeros((240, 320, 3), dtype=np.uint8))

    assert controller.route_active_point == "猪1"
    assert controller.config["combat"]["monsters"]["region"] == [0, 200, 400, 90]


def test_path_combat_falls_back_to_common_config_when_point_is_unknown():
    config = _config(pickup_enabled=False)
    config["combat"]["monsters"]["region"] = [1, 2, 300, 80]
    config["combat"]["fixed_point"] = {
        "enabled": False,
        "point": [8, 9],
    }
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "points": [{"name": "点1", "region": [0, 0, 10, 10]}],
        "point_profiles": {
            "点1": {
                "fixed_point": [4, 5],
                "monster_region": [0, 100, 400, 80],
            }
        },
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(70, 4),
            player_global=Point(70, 4),
        )
    )

    controller._activate_route_combat_profile(np.zeros((240, 320, 3), dtype=np.uint8))

    assert controller.route_active_point is None
    assert controller.config["combat"]["monsters"]["region"] == []
    assert controller.config["combat"]["fixed_point"]["enabled"] is False


def test_path_combat_uses_actual_point_profile_before_route_target():
    config = _config(pickup_enabled=False)
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "point_tolerance": 1,
        "points": [
            {"name": "点1", "region": [0, 0, 10, 10]},
            {"name": "点2", "region": [30, 0, 10, 10]},
        ],
        "point_profiles": {
            "点1": {
                "fixed_point": [4, 5],
                "fixed_point_enabled": True,
                "monster_region": [0, 100, 400, 80],
            },
            "点2": {
                "fixed_point": [40, 6],
                "fixed_point_enabled": True,
                "monster_region": [0, 200, 400, 90],
            },
        },
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.route_target_point = "点2"
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(4, 4),
            player_global=Point(4, 4),
        )
    )

    controller._activate_route_combat_profile(
        np.zeros((240, 320, 3), dtype=np.uint8)
    )

    assert controller.route_active_point == "点1"
    assert controller.route_profile_point == "点1"
    assert controller.config["combat"]["monsters"]["region"] == [
        0,
        100,
        400,
        80,
    ]
    assert controller.config["combat"]["fixed_point"]["point"] == [4, 5]


def test_path_combat_does_not_inherit_previous_point_region_when_profile_is_empty():
    config = _config(pickup_enabled=False)
    config["combat"]["monsters"]["region"] = [1, 2, 300, 80]
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "points": [{"name": "点1", "region": [0, 0, 10, 10]}],
        "point_profiles": {
            "点1": {
                "fixed_point": [4, 5],
                "fixed_point_enabled": True,
                "monster_region": None,
            }
        },
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(4, 4),
            player_global=Point(4, 4),
        )
    )

    controller._activate_route_combat_profile(
        np.zeros((240, 320, 3), dtype=np.uint8)
    )

    assert controller.route_active_point == "点1"
    assert controller.config["combat"]["monsters"]["region"] == []


def test_path_combat_selects_next_path_after_no_monster_timeout(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    config = _config(pickup_enabled=False)
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "no_monster_seconds": 1.0,
        "points": [
            {"name": "点1", "region": [0, 0, 10, 10]},
            {"name": "点2", "region": [30, 0, 10, 10]},
        ],
        "point_profiles": {"点1": {}, "点2": {}},
        "paths": [
            {
                "name": "点1 -> 点2",
                "source": "点1",
                "target": "点2",
                "steps": [{"type": "move_right", "duration_ms": 500}],
            }
        ],
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(4, 4),
            player_global=Point(4, 4),
        )
    )

    class FakeRouteController:
        def __init__(self, config, path):
            self.path = path

        def process_frame(self, _frame):
            return type(
                "RouteResult",
                (),
                {
                    "state": "running",
                    "action": "move_right(right)",
                    "reason": "route-move_right",
                    "player": None,
                },
            )()

    monkeypatch.setattr(
        controller_module,
        "RouteTestController",
        FakeRouteController,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    controller._activate_route_combat_profile(frame)
    no_target = _combat(reason="no-target")

    assert controller._maybe_start_route(frame, no_target, no_target) is None
    clock[0] = 1.01
    result = controller._maybe_start_route(frame, no_target, no_target)

    assert result is not None
    assert result.action == "move_right(right)"
    assert result.reason == "route-move_right"
    assert controller.route_path_name == "点1 -> 点2"
    assert controller.route_controller is not None


def test_path_combat_returns_to_assigned_point_after_unexpected_fall(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {"enabled": True, "region": [0, 0, 100, 80]}
    config["route_combat"] = {
        "mode": "path_combat",
        "no_monster_seconds": 20.0,
        "points": [
            {"name": "点1", "region": [0, 0, 10, 10]},
            {"name": "点2", "region": [30, 0, 10, 10]},
            {"name": "点3", "region": [60, 0, 10, 10]},
        ],
        "point_profiles": {
            "点1": {"monster_region": [1, 1, 10, 10]},
            "点2": {"monster_region": [2, 2, 10, 10]},
            "点3": {"monster_region": [3, 3, 10, 10]},
        },
        "paths": [
            {
                "name": "点1 -> 点2",
                "source": "点1",
                "target": "点2",
                "steps": [{"type": "move_right", "duration_ms": 500}],
            },
            {
                "name": "点3 -> 点2",
                "source": "点3",
                "target": "点2",
                "steps": [{"type": "move_left", "duration_ms": 500}],
            },
        ],
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    actual_point = ["点3"]
    controller._observe_route_combat_minimap = lambda _frame: (
        MinimapObservation(
            region=(0, 0, 100, 80),
            player_local=Point(
                {"点1": 4, "点2": 34, "点3": 64}[actual_point[0]],
                4,
            ),
            player_global=Point(
                {"点1": 4, "点2": 34, "点3": 64}[actual_point[0]],
                4,
            ),
        )
    )

    class FakeRouteController:
        def __init__(self, config, path):
            self.path = path

        def process_frame(self, _frame):
            return type(
                "RouteResult",
                (),
                {
                    "state": "running",
                    "action": "move_left(left)",
                    "reason": "route-return-to-target",
                    "player": None,
                },
            )()

    monkeypatch.setattr(
        controller_module,
        "RouteTestController",
        FakeRouteController,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    controller.route_target_point = "点2"
    controller._activate_route_combat_profile(frame)
    no_target = _combat(reason="no-target")

    result = controller._maybe_start_route(frame, no_target, no_target)

    assert result is not None
    assert result.reason == "route-return-to-target"
    assert result.action == "move_left(left)"
    assert controller.route_path_name == "点3 -> 点2"
    assert controller.route_target_point == "点2"


def test_path_combat_recovery_uses_first_hop_for_multi_segment_route():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {
        "mode": "path_combat",
        "paths": [
            {
                "name": "点1 -> 点2",
                "source": "点1",
                "target": "点2",
                "steps": [{"type": "move_right", "duration_ms": 500}],
            },
            {
                "name": "点2 -> 点3",
                "source": "点2",
                "target": "点3",
                "steps": [{"type": "move_right", "duration_ms": 500}],
            },
        ],
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )

    first_hop = controller._route_path_between("点1", "点3")

    assert first_hop is not None
    assert first_hop["name"] == "点1 -> 点2"
    assert controller._route_path_between("点2", "点3")["name"] == "点2 -> 点3"


def test_path_combat_recovery_finds_route_through_multiple_points():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {
        "mode": "path_combat",
        "paths": [
            {
                "name": "点1 -> 点2",
                "source": "点1",
                "target": "点2",
                "steps": [{"type": "move_right", "duration_ms": 500}],
            },
            {
                "name": "点2 -> 点3",
                "source": "点2",
                "target": "点3",
                "steps": [{"type": "move_right", "duration_ms": 500}],
            },
            {
                "name": "点3 -> 点1",
                "source": "点3",
                "target": "点1",
                "steps": [{"type": "move_left", "duration_ms": 500}],
            },
        ],
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )

    assert controller._route_path_between("点3", "点2")["name"] == "点3 -> 点1"


def test_path_combat_returns_to_fighting_after_route_completion(monkeypatch):
    config = _config(pickup_enabled=False)
    config["route_combat"] = {"mode": "path_combat"}
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )

    class FakeRouteController:
        path = {"name": "点1 -> 点2", "target": "点2"}

        def process_frame(self, _frame):
            return type(
                "RouteResult",
                (),
                {
                    "state": "complete",
                    "action": "observe",
                    "reason": "route-complete",
                    "player": None,
                },
            )()

    controller.route_controller = FakeRouteController()
    controller.route_path_name = "点1 -> 点2"
    combat = _combat(reason="no-target")

    result = controller._process_route_navigation(
        np.zeros((240, 320, 3), dtype=np.uint8),
        combat,
    )

    assert result.reason == "route-arrived"
    assert controller.route_controller is None
    assert controller.route_path_name is None


def test_path_navigation_attacks_visible_monster_before_route_step():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {"mode": "path_combat"}
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (180, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )

    class FakeRouteController:
        def __init__(self):
            self.reset_calls = 0
            self.route_calls = 0
            self.path = {"name": "点1 -> 点2", "target": "点2"}

        def reset_obstacle_state(self):
            self.reset_calls += 1

        def process_combat_approach(self, _frame, _direction):
            raise AssertionError("in-range target should attack")

        def process_frame(self, _frame):
            self.route_calls += 1
            return type(
                "RouteResult",
                (),
                {
                    "state": "running",
                    "action": "move_right(right)",
                    "reason": "route-resumed",
                    "player": None,
                },
            )()

    route = FakeRouteController()
    controller.route_controller = route
    controller.config["combat"]["enemy_health_bar"] = {"enabled": False}
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="tap_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="right",
    )

    result = controller._process_route_navigation(frame, combat)

    assert result.action == "tap_attack(ctrl,right)"
    assert result.reason == "attack-target-present"
    assert route.reset_calls == 1
    assert route.route_calls == 0


def test_path_navigation_approaches_out_of_range_monster_before_route_step():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {"mode": "path_combat"}
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (40, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )

    class FakeRouteController:
        path = {"name": "点1 -> 点2", "target": "点2"}

        def process_combat_approach(self, _frame, direction):
            assert direction == "left"
            return type(
                "RouteResult",
                (),
                {
                    "action": "jump_left(left,alt)",
                    "state": "running",
                    "reason": "route-combat-approach;obstacle-jump",
                    "player": None,
                },
            )()

        def process_frame(self, _frame):
            raise AssertionError("route step must pause while approaching")

    controller.route_controller = FakeRouteController()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(),
        target=None,
        nearest_target=target,
        action="move_left(left)",
        reason="nearest-monster-out-of-attack-range",
        target_direction="left",
    )

    result = controller._process_route_navigation(frame, combat)

    assert result.action == "jump_left(left,alt)"
    assert result.reason.startswith("route-combat-approach:")


def test_path_navigation_ignores_monsters_while_climbing():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {"mode": "path_combat"}
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (180, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )

    class FakeRouteController:
        path = {"name": "点1 -> 点2", "target": "点2"}
        steps = [{"type": "climb", "ladder": "梯子1"}]
        step_index = 0

        def process_combat_approach(self, _frame, _direction):
            raise AssertionError("climb step must not approach monsters")

        def reset_obstacle_state(self):
            raise AssertionError("climb step must not enter combat branch")

        def process_frame(self, _frame):
            return type(
                "RouteResult",
                (),
                {
                    "state": "running",
                    "action": "hold_up",
                    "reason": "route-climb-hold",
                    "player": None,
                },
            )()

    controller.route_controller = FakeRouteController()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="tap_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="right",
    )

    result = controller._process_route_navigation(frame, combat)

    assert result.action == "hold_up"
    assert result.reason == "route-climb-hold"


def test_path_combat_point_approach_uses_obstacle_jump_when_monster_out_of_range():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {"mode": "path_combat"}
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (40, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(80, 80, 30, 12),
        anchor=(100, 130),
    )

    class FakeApproachController:
        reset_calls = 0

        def process_combat_approach(self, _frame, direction):
            assert direction == "left"
            return type(
                "RouteResult",
                (),
                {
                    "action": "jump_left(left,alt)",
                    "state": "running",
                    "reason": "route-combat-approach;obstacle-jump",
                    "player": None,
                },
            )()

        def reset_obstacle_state(self):
            self.reset_calls += 1

    fake = FakeApproachController()
    controller.combat_approach_controller = fake
    combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(),
        target=None,
        nearest_target=target,
        action="move_left(left)",
        reason="nearest-monster-out-of-attack-range",
        target_direction="left",
    )

    result = controller._apply_loot_and_patrol(
        np.zeros((240, 320, 3), dtype=np.uint8),
        combat,
    )

    assert result.action == "jump_left(left,alt)"
    assert result.reason.startswith("combat-approach:")
    assert fake.reset_calls == 0


def test_path_navigation_resumes_route_after_three_missing_monster_frames():
    config = _config(pickup_enabled=False)
    config["route_combat"] = {"mode": "path_combat"}
    config["combat"]["enemy_health_bar"] = {"enabled": False}
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (180, 110, 20, 20))

    class FakeRouteController:
        path = {"name": "点1 -> 点2", "target": "点2"}

        def reset_obstacle_state(self):
            return None

        def process_frame(self, _frame):
            return type(
                "RouteResult",
                (),
                {
                    "state": "running",
                    "action": "move_right(right)",
                    "reason": "route-resumed",
                    "player": None,
                },
            )()

    controller.route_controller = FakeRouteController()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    attacking = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=None,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="tap_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
    )
    assert controller._process_route_navigation(frame, attacking).action == (
        "tap_attack(ctrl)"
    )

    missing = _combat(reason="no-target")
    assert controller._process_route_navigation(frame, missing).action == (
        "tap_attack(ctrl)"
    )
    assert controller._process_route_navigation(frame, missing).action == (
        "tap_attack(ctrl)"
    )
    resumed = controller._process_route_navigation(frame, missing)
    assert resumed.action == "move_right(right)"
    assert resumed.reason == "route-resumed"


def test_controller_attacks_by_target_presence_when_health_bar_is_disabled():
    config = _config(pickup_enabled=False)
    config["combat"].update(
        {
            "attack_action": "hold_attack(ctrl)",
            "attack_target_missing_frames": 1,
            "enemy_health_bar": {"enabled": False},
        }
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.enemy_health_bar_detector.detect = lambda *_args: (
        (_ for _ in ()).throw(AssertionError("health-bar detector must be off"))
    )
    target = MonsterDetection("stump", 0.95, (90, 110, 20, 20))
    combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=None,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="hold_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
    )

    attacking = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        combat,
    )

    assert attacking.action == "hold_attack(ctrl)"
    assert attacking.reason == "attack-target-present"
    assert controller.attack_phase == "attacking"
    assert controller.last_enemy_health_bar.present is False

    lost = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )

    assert lost.reason == "attack-target-lost"
    assert controller.attack_phase == "search"


def test_controller_keeps_attack_for_three_consecutive_missing_targets():
    config = _config(pickup_enabled=False)
    config["combat"].update(
        {
            "attack_action": "hold_attack(ctrl)",
            "attack_target_missing_frames": 3,
            "enemy_health_bar": {"enabled": False},
        }
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (90, 110, 20, 20))
    combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=None,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="hold_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
    )
    controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        combat,
    )

    missing = _combat(reason="no-target")
    first = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        missing,
    )
    second = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        missing,
    )
    third = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        missing,
    )

    assert first.reason == "attack-target-temporarily-lost"
    assert second.reason == "attack-target-temporarily-lost"
    assert first.action == "hold_attack(ctrl)"
    assert second.action == "hold_attack(ctrl)"
    assert third.reason == "attack-target-lost"
    assert third.action == "observe"
    assert controller.attack_phase == "search"


def test_controller_moves_toward_locked_target_when_it_leaves_attack_range():
    config = _config(pickup_enabled=False)
    config["combat"].update(
        {
            "attack_action": "hold_attack(ctrl)",
            "enemy_health_bar": {"enabled": False},
        }
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    target = MonsterDetection("stump", 0.95, (220, 110, 20, 20))
    in_range = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=None,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="hold_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="right",
    )
    controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        in_range,
    )

    out_of_range = replace(
        in_range,
        in_range_targets=(),
        target=None,
        action="move_right(right)",
        reason="nearest-monster-out-of-attack-range",
    )
    result = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        out_of_range,
    )

    assert result.action == "move_right(right)"
    assert result.reason == "nearest-monster-out-of-attack-range"
    assert controller.attack_phase == "search"


def test_controller_switches_to_opposite_side_target_after_locked_target_disappears():
    config = _config(pickup_enabled=False)
    config["combat"].update(
        {
            "attack_action": "hold_attack(ctrl)",
            "enemy_health_bar": {"enabled": False},
            "attack_target_missing_frames": 3,
        }
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    player = PlayerScreenDetection(
        name="角色",
        score=0.95,
        name_box=(130, 80, 30, 12),
        anchor=(150, 120),
    )
    first = MonsterDetection("stump", 0.95, (110, 110, 20, 20))
    second = MonsterDetection("red_snail", 0.94, (185, 110, 20, 20))
    first_combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(first,),
        confirmed_targets=(first,),
        in_range_targets=(first,),
        target=first,
        nearest_target=first,
        action="hold_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="left",
    )
    controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        first_combat,
    )

    second_combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(second,),
        confirmed_targets=(second,),
        in_range_targets=(second,),
        target=second,
        nearest_target=second,
        action="hold_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="right",
    )
    result = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        second_combat,
    )

    assert result.action == "hold_attack(ctrl,right)"
    assert result.target == second
    assert result.nearest_target == second
    assert result.target_direction == "right"
    assert controller.attack_locked_target == second


def test_controller_switches_to_nearer_opposite_target_while_both_are_visible():
    config = _config(pickup_enabled=False)
    config["combat"].update(
        {
            "attack_action": "hold_attack(ctrl)",
            "enemy_health_bar": {"enabled": False},
        }
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    player = PlayerScreenDetection(
        name="角色",
        score=0.95,
        name_box=(130, 80, 30, 12),
        anchor=(150, 120),
    )
    first = MonsterDetection("stump", 0.95, (110, 110, 20, 20))
    second = MonsterDetection("red_snail", 0.94, (165, 110, 20, 20))
    first_combat = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(first,),
        confirmed_targets=(first,),
        in_range_targets=(first,),
        target=first,
        nearest_target=first,
        action="hold_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="left",
    )
    controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        first_combat,
    )

    both_combat = replace(
        first_combat,
        targets=(first, second),
        confirmed_targets=(first, second),
        in_range_targets=(first, second),
        target=second,
        nearest_target=second,
        target_direction="right",
    )
    result = controller._apply_attack_state(
        np.zeros((240, 320, 3), dtype=np.uint8),
        both_combat,
    )

    assert result.action == "hold_attack(ctrl,right)"
    assert result.target == second
    assert result.target_direction == "right"
    assert controller.attack_locked_target == second


def test_controller_patrols_right_when_minimap_player_is_at_left_edge():
    config = _config(pickup_enabled=False)
    config["combat"]["patrol"] = {"move_duration_ms": [3000, 6000]}
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
        "edge_threshold": 0.25,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 1:3] = [0, 255, 255]

    result = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))

    assert result.reason == "patrol-no-target"
    assert result.action == "move_right(right)"


def test_controller_does_not_patrol_when_patrol_is_disabled():
    config = _config(pickup_enabled=False)
    config["combat"]["patrol"] = {
        "enabled": False,
        "move_duration_ms": [3000, 6000],
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0

    result = controller._apply_loot_and_patrol(
        np.zeros((20, 30, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )

    assert result.reason == "no-target"
    assert result.action == "observe"


def test_controller_recovers_left_when_minimap_player_is_at_right_edge():
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
        "edge_threshold": 0.25,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 18:20] = [0, 255, 255]

    result = controller._apply_player_recovery(
        frame,
        _combat(reason="player-not-found"),
    )

    assert result.reason == "player-not-found-recovery"
    assert result.action == "move_left(left)"
    assert controller.player_recovery_direction == "left"


def test_controller_keeps_recovery_direction_after_initial_minimap_choice():
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
        "edge_threshold": 0.25,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    first_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    first_frame[3:5, 18:20] = [0, 255, 255]
    result = controller._apply_player_recovery(
        first_frame,
        _combat(reason="player-not-found"),
    )

    second_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    second_frame[3:5, 1:3] = [0, 255, 255]
    continued = controller._apply_player_recovery(
        second_frame,
        _combat(reason="no-target"),
    )

    assert result.action == "move_left(left)"
    assert continued.action == "move_left(left)"


def test_controller_keeps_patrol_direction_until_hold_expires(monkeypatch):
    config = _config(pickup_enabled=False)
    config["combat"]["patrol"] = {"move_duration_ms": [3000, 6000]}
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
        "edge_threshold": 0.25,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0
    calls = []

    def fake_uniform(low, high):
        calls.append((low, high))
        return 5.0

    monkeypatch.setattr(
        "maplevision.decision.controller.random.uniform",
        fake_uniform,
    )
    left_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    left_frame[3:5, 1:3] = [0, 255, 255]
    first = controller._apply_loot_and_patrol(
        left_frame,
        _combat(reason="no-target"),
    )

    right_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    right_frame[3:5, 18:20] = [0, 255, 255]
    second = controller._apply_loot_and_patrol(
        right_frame,
        _combat(reason="no-target"),
    )
    controller.patrol_until = time.monotonic() - 1
    release = controller._apply_loot_and_patrol(
        right_frame,
        _combat(reason="no-target"),
    )

    assert first.action == "move_right(right)"
    assert second.action == "move_right(right)"
    assert release.reason == "patrol-hold-complete"
    assert release.action == "observe"
    assert controller.patrol_direction is None
    assert calls[-1] == (3.0, 6.0)


def test_controller_uses_configured_patrol_hold_range(monkeypatch):
    config = _config(pickup_enabled=False)
    config["combat"]["patrol"] = {
        "no_target_seconds": 2.0,
        "move_duration_ms": [3000, 8000],
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0
    calls = []

    def fake_uniform(low, high):
        calls.append((low, high))
        return 4.0

    monkeypatch.setattr(
        "maplevision.decision.controller.random.uniform",
        fake_uniform,
    )

    result = controller._apply_loot_and_patrol(
        np.zeros((20, 30, 3), dtype=np.uint8),
        _combat(reason="no-target"),
    )

    assert result.reason == "patrol-no-target"
    assert calls[-1] == (3.0, 8.0)


def test_controller_patrols_toward_configured_minimap_target(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
        "edge_threshold": 0.25,
    }
    config["combat"]["patrol"] = {
        "no_target_seconds": 2.0,
        "move_duration_ms": [3000, 8000],
        "mode": "minimap_target",
        "minimap_target": [18, 4],
        "target_tolerance": 1,
    }
    monkeypatch.setattr(
        "maplevision.decision.controller.random.uniform",
        lambda _low, _high: 4.0,
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 1:3] = [0, 255, 255]

    result = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))

    assert result.reason == "patrol-no-target"
    assert result.action == "move_right(right)"
    assert controller.patrol_direction == "right"


def test_controller_stays_when_minimap_target_is_reached(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
        "edge_threshold": 0.25,
    }
    config["combat"]["patrol"] = {
        "no_target_seconds": 2.0,
        "move_duration_ms": [3000, 8000],
        "mode": "minimap_target",
        "minimap_target": [2, 4],
        "target_tolerance": 2,
    }
    monkeypatch.setattr(
        "maplevision.decision.controller.random.uniform",
        lambda _low, _high: 4.0,
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 1:3] = [0, 255, 255]

    result = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))

    assert result.reason == "no-target"
    assert result.action == "observe"
    assert controller.patrol_direction == "stay"


def test_patrol_target_uses_minimap_local_coordinates(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [10, 5, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
    }
    config["combat"]["patrol"] = {
        "no_target_seconds": 2.0,
        "move_duration_ms": [3000, 8000],
        "mode": "minimap_target",
        "minimap_target": [8, 4],
        "target_tolerance": 1,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    controller.no_target_since = time.monotonic() - 3.0
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    frame[8:10, 13:15] = [0, 255, 255]

    result = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))

    assert result.action == "move_right(right)"
    assert controller.last_minimap.player_local.x == 3


def test_fixed_point_returns_when_distance_exceeds_threshold(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
    }
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [15, 4],
        "return_distance_px": 5,
        "target_tolerance": 1,
        "roam_enabled": False,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 2:4] = [0, 255, 255]
    combat = _combat(reason="no-target")

    returning = controller._apply_loot_and_patrol(frame, combat)
    assert returning.reason.startswith("fixed-point-return:")
    assert returning.action == "move_right(right)"

    frame[3:5, 2:4] = 0
    frame[3:5, 15:17] = [0, 255, 255]
    reached = controller._apply_loot_and_patrol(frame, combat)
    assert reached.reason == "fixed-point-hold"
    assert reached.action == "observe"
    assert controller.fixed_point_returning is False


def test_fixed_point_return_has_priority_over_visible_monster():
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
    }
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [15, 4],
        "return_distance_px": 5,
        "target_tolerance": 1,
        "roam_enabled": False,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 2:4] = [0, 255, 255]
    monster = MonsterDetection("stump", 0.95, (80, 80, 20, 20))

    result = controller._apply_loot_and_patrol(
        frame,
        _combat(
            reason="confirmed-monster-in-attack-range",
            confirmed_targets=(monster,),
        ),
    )

    assert result.reason.startswith("fixed-point-return:")
    assert result.action == "move_right(right)"


def test_fixed_point_stays_when_distance_is_within_threshold(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
    }
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [10, 4],
        "return_distance_px": 5,
        "target_tolerance": 1,
        "roam_enabled": False,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 10:12] = [0, 255, 255]
    combat = _combat(reason="no-target")

    initial = controller._apply_loot_and_patrol(frame, combat)
    assert initial.reason == "fixed-point-hold"

    frame[3:5, 10:12] = 0
    frame[3:5, 7:9] = [0, 255, 255]
    waiting = controller._apply_loot_and_patrol(frame, combat)
    assert waiting.reason == "fixed-point-hold"

    frame[3:5, 7:9] = 0
    frame[3:5, 2:4] = [0, 255, 255]
    returning = controller._apply_loot_and_patrol(frame, combat)
    assert returning.reason.startswith("fixed-point-return:")
    assert returning.action == "move_right(right)"


def test_fixed_point_uses_configured_return_distance(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
    }
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [15, 4],
        "return_distance_px": 15,
        "target_tolerance": 1,
        "roam_enabled": False,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 2:4] = [0, 255, 255]

    initial = controller._apply_loot_and_patrol(
        frame,
        _combat(reason="no-target"),
    )
    assert initial.reason == "fixed-point-hold"
    waiting = controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))
    assert waiting.reason == "fixed-point-hold"

    config["combat"]["fixed_point"]["return_distance_px"] = 5
    returning = controller._apply_loot_and_patrol(
        frame,
        _combat(reason="no-target"),
    )
    assert returning.reason.startswith("fixed-point-return:")
    assert returning.action == "move_right(right)"


def test_fixed_point_return_uses_obstacle_jump(monkeypatch):
    config = _config(pickup_enabled=False)
    config["minimap"] = {
        "enabled": True,
        "region": [0, 0, 20, 10],
        "player_color_hsv": {
            "lower": [20, 120, 120],
            "upper": [40, 255, 255],
        },
    }
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [15, 4],
        "return_distance_px": 5,
        "target_tolerance": 1,
        "roam_enabled": False,
    }
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )

    class FakeReturnController:
        def process_combat_approach(self, _frame, direction):
            assert direction == "right"
            return type(
                "RouteResult",
                (),
                {
                    "action": "jump_right(right,alt)",
                    "state": "running",
                    "reason": "route-combat-approach;obstacle-jump",
                    "player": None,
                },
            )()

        def reset_obstacle_state(self):
            return None

    controller.fixed_point_return_controller = FakeReturnController()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[3:5, 2:4] = [0, 255, 255]

    controller._apply_loot_and_patrol(frame, _combat(reason="no-target"))
    returning = controller._apply_loot_and_patrol(
        frame,
        _combat(reason="no-target"),
    )

    assert returning.action == "jump_right(right,alt)"
    assert returning.reason.startswith("fixed-point-return:")


def test_fixed_point_moves_toward_monster_on_the_right_when_out_of_range(
    monkeypatch,
):
    config = _config(pickup_enabled=False)
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [10, 4],
        "check_interval_ms": [30000, 30000],
    }
    config["combat"]["enemy_health_bar"] = {"enabled": False}
    target = MonsterDetection("stump", 0.95, (240, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(100, 80, 30, 12),
        anchor=(150, 120),
    )
    observed = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(),
        target=None,
        nearest_target=target,
        action="move_right(right)",
        reason="nearest-monster-out-of-attack-range",
        target_direction="right",
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    monkeypatch.setattr(
        "maplevision.decision.controller.observe_combat",
        lambda *_args: observed,
    )
    result = controller.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result.target is None
    assert result.nearest_target == target
    assert result.target_direction == "right"
    assert result.reason == "nearest-monster-out-of-attack-range"
    assert result.action == "move_right(right)"


def test_fixed_point_moves_toward_monster_on_the_left_when_out_of_range(
    monkeypatch,
):
    config = _config(pickup_enabled=False)
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [10, 4],
        "check_interval_ms": [30000, 30000],
    }
    config["combat"]["enemy_health_bar"] = {"enabled": False}
    target = MonsterDetection("stump", 0.95, (60, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(180, 80, 30, 12),
        anchor=(230, 120),
    )
    observed = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(),
        target=None,
        nearest_target=target,
        action="move_left(left)",
        reason="nearest-monster-out-of-attack-range",
        target_direction="left",
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    monkeypatch.setattr(
        "maplevision.decision.controller.observe_combat",
        lambda *_args: observed,
    )

    result = controller.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result.reason == "nearest-monster-out-of-attack-range"
    assert result.action == "move_left(left)"
    assert result.target_direction == "left"


def test_fixed_point_attacks_monster_after_it_enters_attack_range(monkeypatch):
    config = _config(pickup_enabled=False)
    config["combat"]["fixed_point"] = {
        "enabled": True,
        "point": [10, 4],
        "check_interval_ms": [30000, 30000],
    }
    config["combat"]["enemy_health_bar"] = {"enabled": False}
    target = MonsterDetection("stump", 0.95, (185, 110, 20, 20))
    player = PlayerScreenDetection(
        name="角色",
        score=0.9,
        name_box=(100, 80, 30, 12),
        anchor=(150, 120),
    )
    observed = CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=(target,),
        confirmed_targets=(target,),
        in_range_targets=(target,),
        target=target,
        nearest_target=target,
        action="tap_attack(ctrl)",
        reason="confirmed-monster-in-attack-range",
        target_direction="right",
    )
    controller = VisionController(
        source=_Source(),
        input_controller=MacOSInputController(dry_run=True),
        config=config,
    )
    monkeypatch.setattr(
        "maplevision.decision.controller.observe_combat",
        lambda *_args: observed,
    )

    result = controller.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result.target == target
    assert result.reason == "attack-target-present"
    assert result.action == "tap_attack(ctrl,right)"
    assert result.target_direction == "right"


def test_controller_taps_periodic_key_without_changing_action(monkeypatch):
    class FakeInput(MacOSInputController):
        def __init__(self):
            super().__init__(dry_run=True)
            self.taps = []

        def tap_key(self, key: str) -> None:
            self.taps.append(key)

    config = _config(pickup_enabled=False)
    config["combat"]["periodic_key"] = {
        "enabled": True,
        "interval_seconds": 3.0,
        "key": "3",
    }
    fake_input = FakeInput()
    controller = VisionController(
        source=_Source(),
        input_controller=fake_input,
        config=config,
    )
    now = [10.0]
    monkeypatch.setattr(
        "maplevision.decision.controller.time.monotonic",
        lambda: now[0],
    )

    controller._emit_periodic_key()
    now[0] = 12.9
    controller._emit_periodic_key()
    now[0] = 13.0
    controller._emit_periodic_key()

    assert fake_input.taps == ["3"]


class _Source:
    def next_frame(self):
        return None

    def close(self):
        return None


def _config(pickup_enabled: bool) -> dict:
    return {
        "system": {"debug_output": ""},
        "minimap": {"enabled": False},
        "combat": {
            "move_left_action": "move_left(left)",
            "move_right_action": "move_right(right)",
            "observe_action": "observe",
            "player": {"enabled": False},
            "monsters": {
                "enabled": False,
                "template_groups": {},
            },
            "pickup": {
                "enabled": pickup_enabled,
                "min_coins": 2,
                "pickup_distance": 28,
                "action": "pickup(z,3)",
                "coins": {
                    "enabled": pickup_enabled,
                    "template_groups": {},
                },
            },
        },
    }


def _combat(
    *,
    reason: str,
    player=None,
    targets=(),
    confirmed_targets=(),
) -> CombatObservation:
    return CombatObservation(
        health=HealthObservation(None, 1.0, True),
        mana=HealthObservation(None, 1.0, True),
        player=player,
        targets=targets,
        confirmed_targets=confirmed_targets,
        in_range_targets=(),
        target=None,
        nearest_target=None,
        action="observe",
        reason=reason,
    )
