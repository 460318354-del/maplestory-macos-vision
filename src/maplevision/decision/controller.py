from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..capture.base import FrameSource
from ..route_test import RouteTestController
from ..platform.macos.input import MacOSInputController
from ..vision.combat import observe_combat, save_combat_debug_frame
from ..vision.enemy_health_bar import (
    EnemyHealthBarDetector,
    EnemyHealthBarObservation,
)
from ..vision.minimap import MinimapObservation, observe_minimap
from ..vision.monster import (
    MonsterDetection,
    MonsterDetector,
    MonsterTracker,
    detect_monsters,
)
from ..vision.player_screen import PlayerScreenDetector, detect_player_screen
from ..pathing import (
    active_route_group,
    current_point_name,
    route_path_group,
    route_point_group,
)

logger = logging.getLogger(__name__)


def pickup_config_value(
    config: dict[str, Any],
    key: str,
    default: float,
) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def patrol_delay_seconds(config: dict[str, Any]) -> float:
    try:
        return max(
            0.1,
            float(
                config.get("combat", {})
                .get("patrol", {})
                .get("no_target_seconds", 2.0)
            ),
        )
    except (TypeError, ValueError):
        return 2.0


def _seconds_range(
    config: dict[str, Any],
    milliseconds_key: str,
    default_seconds: tuple[float, float],
    legacy_seconds_key: str | None = None,
) -> tuple[float, float]:
    value = config.get(milliseconds_key)
    scale = 0.001
    if value is None and legacy_seconds_key:
        value = config.get(legacy_seconds_key)
        scale = 1.0
    try:
        if value is None:
            low, high = default_seconds
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = float(value[0]) * scale, float(value[1]) * scale
        else:
            low = high = float(value) * scale
        low = max(0.001, min(low, high))
        high = max(low, high)
        return low, high
    except (TypeError, ValueError):
        return default_seconds


def _fixed_point_return_distance(config: dict[str, Any]) -> float:
    """Return the minimap distance that triggers fixed-point return."""
    value = config.get("return_distance_px")
    if value is None:
        # Older saved configurations only have target_tolerance. Keep them
        # usable by treating that value as the return threshold.
        value = config.get("target_tolerance", 2)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 10.0


class VisionController:
    def __init__(
        self,
        source: FrameSource,
        input_controller: MacOSInputController,
        config: dict[str, Any],
    ):
        self.source = source
        self.input = input_controller
        self.config = config
        self.base_combat_config = deepcopy(config.get("combat", {}))
        self.route_active_point: str | None = None
        self.route_target_point: str | None = None
        self.route_profile_point: str | None = None
        self.route_active_profile: dict[str, Any] | None = None
        self.route_controller: RouteTestController | None = None
        self.route_path_name: str | None = None
        self.route_no_target_since: float | None = None
        self.route_failed_path: str | None = None
        self.combat_approach_controller: RouteTestController | None = None
        self.fixed_point_return_controller: RouteTestController | None = None
        self.pickup_move_controller: RouteTestController | None = None
        monster_config = self.base_combat_config.get("monsters", {})
        self.monster_tracker = MonsterTracker(
            confirm_frames=int(monster_config.get("confirm_frames", 2)),
            max_misses=int(monster_config.get("max_misses", 2)),
            max_distance=int(monster_config.get("max_distance", 120)),
        )
        self.monster_detector = MonsterDetector(monster_config)
        self.enemy_health_bar_detector = EnemyHealthBarDetector(
            config.get("combat", {}).get("enemy_health_bar", {})
        )
        self.attack_phase = "search"
        self.attack_locked_target = None
        self.attack_wait_frames = 0
        self.attack_missing_frames = 0
        self.attack_cooldown_frames = 0
        self.player_recovery_until: float | None = None
        self.player_recovery_direction: str | None = None
        self.no_target_since: float | None = None
        self.patrol_direction: str | None = None
        self.patrol_until: float | None = None
        self.loot_scan_until: float | None = None
        self.pickup_cycle_until: float | None = None
        self.pickup_target: MonsterDetection | None = None
        self.pickup_action_sent = False
        self.pickup_move_direction: str | None = None
        self.pickup_target_lost_since: float | None = None
        self.pickup_interval_seconds = max(
            0.0,
            pickup_config_value(
                config.get("combat", {}).get("pickup", {}),
                "interval_seconds",
                0.0,
            ),
        )
        self.pickup_scan_seconds = max(
            0.1,
            pickup_config_value(
                config.get("combat", {}).get("pickup", {}),
                "scan_seconds",
                1.0,
            ),
        )
        self.next_pickup_at: float | None = None
        self.loot_phase = "waiting"
        self.fixed_point_check_at: float | None = None
        self.fixed_point_roam_at: float | None = None
        self.fixed_point_roam_until: float | None = None
        self.fixed_point_returning = False
        self.fixed_point_release_pending = False
        self.patrol_release_pending = False
        self.last_coin_detections: tuple[MonsterDetection, ...] = ()
        self.last_minimap: MinimapObservation | None = None
        self.last_enemy_health_bar = EnemyHealthBarObservation(
            present=False,
            score=0.0,
            region=None,
            fill_ratio=None,
        )
        player_config = config.get("combat", {}).get("player", {})
        self.player_detector = (
            PlayerScreenDetector(player_config)
            if _has_player_templates(player_config)
            else None
        )
        pickup_config = config.get("combat", {}).get("pickup", {})
        self.pickup_config = pickup_config
        # Keep compatibility with hand-written legacy configs that explicitly
        # disabled pickup. The GUI and shipped default always write this as
        # true; absence of the key also means enabled.
        self.pickup_enabled = bool(pickup_config.get("enabled", True))
        coin_config = dict(pickup_config.get("coins", {}))
        if "template_groups" not in coin_config:
            coin_config["template_groups"] = {
                "金币": ["assets/templates/monsters/金币"]
            }
        # Coin pickup is always enabled. The GUI no longer exposes an on/off
        # switch; the interval controls when the pickup state machine starts.
        coin_config["enabled"] = True
        coin_config.setdefault("allowed_labels", ["金币"])
        self.coin_config = coin_config
        self.coin_detector = MonsterDetector(coin_config)
        self.last_combat = None
        self.periodic_key_config = config.get("combat", {}).get(
            "periodic_key",
            {},
        )
        self.next_periodic_key_at: float | None = None

    def run(self, max_frames: int | None = None) -> None:
        fps = max(1, int(self.config["system"].get("fps", 8)))
        delay = 1.0 / fps
        frames = 0
        try:
            while True:
                started = time.monotonic()
                frame = self.source.next_frame()
                if frame is None:
                    return
                combat = self.process_frame(frame)
                self.input.emit(combat.action or "none")
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    return
                elapsed = time.monotonic() - started
                if elapsed < delay:
                    time.sleep(delay - elapsed)
        finally:
            self.input.release_all()
            self.source.close()

    def _propose_action(self, frame) -> str | None:
        return self.process_frame(frame).action

    def process_frame(self, frame):
        self._activate_route_combat_profile(frame)
        observed_combat = observe_combat(
            frame,
            self.config.get("combat", {}),
            self.monster_tracker,
            self.monster_detector,
            self.player_detector,
        )
        route_combat = self._process_route_navigation(frame, observed_combat)
        if route_combat is not None:
            combat = route_combat
        else:
            pickup_combat, pickup_active = self._apply_pickup_priority(
                frame,
                observed_combat,
            )
            if pickup_active:
                combat = pickup_combat
            else:
                combat = self._apply_attack_state(frame, observed_combat)
                combat = self._apply_player_recovery(frame, combat)
                combat = self._apply_loot_and_patrol(
                    frame,
                    combat,
                    pickup_checked=True,
                )
            if self._path_combat_enabled() and not pickup_active:
                route_combat = self._maybe_start_route(
                    frame,
                    observed_combat,
                    combat,
                )
                if route_combat is not None:
                    combat = route_combat
        self._emit_periodic_key()
        self.last_combat = combat
        if combat.health.ratio is None:
            logger.info("hp: not-found")
        else:
            logger.info("hp: %.1f%% healthy=%s", combat.health.ratio * 100, combat.health.healthy)
        if combat.mana.ratio is None:
            logger.info("mp: not-found")
        else:
            logger.info("mp: %.1f%% healthy=%s", combat.mana.ratio * 100, combat.mana.healthy)
        if combat.target is None:
            logger.info(
                "combat: targets=%s labels=%s selected=none nearest=%s "
                "action=%s reason=%s",
                len(combat.targets),
                _target_labels(combat.targets),
                combat.nearest_target.box if combat.nearest_target else None,
                combat.action,
                combat.reason,
            )
        else:
            logger.info(
                "combat: targets=%s labels=%s selected=%s label=%s "
                "score=%.3f action=%s reason=%s",
                len(combat.targets),
                _target_labels(combat.targets),
                combat.target.box,
                combat.target.label,
                combat.target.score,
                combat.action,
                combat.reason,
            )
        if combat.player is None:
            logger.info("screen player: not-found")
        else:
            logger.info(
                "screen player: name=%s anchor=%s score=%.3f in_range=%s",
                combat.player.name,
                combat.player.anchor,
                combat.player.score,
                len(combat.in_range_targets),
            )
            _log_monster_sides(combat)
        if combat.target_direction is not None:
            logger.info("combat target direction: %s", combat.target_direction)
        stats = self.monster_detector.last_stats
        logger.info(
            "vision timing: monsters=%.1fms templates=%s matches=%s roi=%sx%s",
            stats.elapsed_ms,
            stats.template_count,
            stats.match_calls,
            stats.roi_shape[0],
            stats.roi_shape[1],
        )
        if combat.confirmed_targets and self.player_detector is not None:
            logger.info(
                "vision timing: player=%.1fms",
                self.player_detector.last_elapsed_ms,
            )
        logger.info(
            "enemy health bar: phase=%s present=%s score=%.3f fill=%s "
            "region=%s search=%s rows=%s",
            self.attack_phase,
            self.last_enemy_health_bar.present,
            self.last_enemy_health_bar.score,
            (
                f"{self.last_enemy_health_bar.fill_ratio:.0%}"
                if self.last_enemy_health_bar.fill_ratio is not None
                else "-"
            ),
            self.last_enemy_health_bar.region,
            self.last_enemy_health_bar.search_region,
            tuple(
                round(value, 2)
                for value in self.last_enemy_health_bar.row_scores
            ),
        )
        logger.info(
            "coins: count=%s phase=%s next_in=%.2fs",
            len(self.last_coin_detections),
            self.loot_phase,
            self._pickup_seconds_until_next(),
        )
        if self.last_minimap is not None:
            logger.info(
                "minimap: player=%s ratio=%s patrol=%s",
                self.last_minimap.player_global,
                (
                    f"{self.last_minimap.horizontal_ratio:.2f}"
                    if self.last_minimap.horizontal_ratio is not None
                    else "-"
                ),
                self.patrol_direction or "-",
            )

        debug_output = self.config["system"].get("debug_output", "")
        if debug_output:
            save_combat_debug_frame(frame, combat, Path(debug_output))
        return combat

    def _process_route_navigation(self, frame, combat):
        if not self._path_combat_enabled() or self.route_controller is None:
            return None
        if combat.reason in {"hp-low", "mp-low"}:
            self._reset_attack_state()
            return combat

        if self._route_current_step_type() == "climb":
            self._reset_attack_state()
            self._reset_loot_state()
            result = self.route_controller.process_frame(frame)
            return self._route_result_to_combat(combat, result)

        attack_action = str(
            self.config.get("combat", {}).get(
                "attack_action",
                "tap_attack(ctrl)",
            )
        )

        # A monster found while travelling temporarily owns the input stream.
        # Keep the route controller and its current step intact; after the
        # target is gone, the next route tick resumes that same step.
        if self.attack_locked_target is not None:
            attacked = self._apply_attack_state(frame, combat)
            if _same_action_kind(attacked.action, attack_action):
                self.route_controller.reset_obstacle_state()
                return attacked
            combat = attacked

        if combat.confirmed_targets:
            if combat.target is not None:
                self.route_controller.reset_obstacle_state()
                return self._apply_attack_state(frame, combat)
            if combat.player is None:
                self._reset_attack_state()
                return replace(
                    combat,
                    action=str(
                        self.config.get("combat", {}).get(
                            "observe_action",
                            "observe",
                        )
                    ),
                    reason="route-combat-player-not-found",
                )
            if combat.target_direction is not None:
                self._reset_attack_state()
                approach = self.route_controller.process_combat_approach(
                    frame,
                    combat.target_direction,
                )
                return replace(
                    combat,
                    action=approach.action,
                    reason=f"route-combat-approach:{approach.reason}",
                )

        result = self.route_controller.process_frame(frame)
        return self._route_result_to_combat(combat, result)

    def _route_current_step_type(self) -> str | None:
        if self.route_controller is None:
            return None
        steps = getattr(self.route_controller, "steps", None)
        step_index = getattr(self.route_controller, "step_index", None)
        if not isinstance(steps, list) or not isinstance(step_index, int):
            return None
        if step_index < 0 or step_index >= len(steps):
            return None
        step = steps[step_index]
        if not isinstance(step, dict):
            return None
        return str(step.get("type", "")).strip() or None

    def _route_result_to_combat(self, combat, result):
        if result.state == "complete":
            logger.info(
                "route combat arrived at target: path=%s target=%s",
                self.route_path_name,
                self.route_controller.path.get("target"),
            )
            self.route_controller = None
            self.route_path_name = None
            self.route_no_target_since = None
            self.route_failed_path = None
            self._reset_attack_state()
            self._reset_loot_state()
            return replace(
                combat,
                action=str(
                    self.config.get("combat", {}).get(
                        "observe_action",
                        "observe",
                    )
                ),
                reason="route-arrived",
            )
        if result.state == "failed":
            self.route_failed_path = self.route_path_name
            logger.error(
                "route combat path failed: path=%s reason=%s",
                self.route_path_name,
                result.reason,
            )
            self.route_controller = None
            self.route_path_name = None
            self.route_no_target_since = time.monotonic()
            self._reset_attack_state()
            return replace(
                combat,
                action=str(
                    self.config.get("combat", {}).get(
                        "observe_action",
                        "observe",
                    )
                ),
                reason=f"route-failed:{result.reason}",
            )

        self._reset_loot_state()
        return replace(
            combat,
            player=result.player or combat.player,
            action=result.action,
            reason=result.reason,
        )

    def _maybe_start_route(self, frame, observed_combat, combat):
        if self.route_controller is not None:
            return None
        if self.route_target_point is None:
            if self.route_active_point is not None:
                self.route_target_point = self.route_active_point
            else:
                self.route_no_target_since = None
                return None

        if self.route_active_point is None:
            self.route_no_target_since = None
            return None

        # The combat target is authoritative. If the character was knocked
        # onto another platform, return to the intended point first instead
        # of switching combat profiles to the accidental landing point.
        if self.route_active_point != self.route_target_point:
            path = self._route_path_between(
                self.route_active_point,
                self.route_target_point,
            )
            if path is None:
                logger.warning(
                    "no recovery route from current point=%s to target point=%s",
                    self.route_active_point,
                    self.route_target_point,
                )
                return replace(
                    combat,
                    action=str(
                        self.config.get("combat", {}).get(
                            "observe_action",
                            "observe",
                        )
                    ),
                    reason="route-target-recovery-path-not-found",
                )
            return self._start_route_path(frame, combat, path)

        if observed_combat.confirmed_targets:
            self.route_no_target_since = None
            self.route_failed_path = None
            return None
        if combat.reason in {
            "pickup-coins",
            "move-to-coins",
            "coin-player-not-found",
        }:
            return None
        if observed_combat.reason != "no-target":
            return None

        now = time.monotonic()
        if self.route_no_target_since is None:
            self.route_no_target_since = now
        no_monster_seconds = self._route_no_monster_seconds()
        if now - self.route_no_target_since < no_monster_seconds:
            return None

        path = self._next_route_path(self.route_target_point)
        if path is None:
            path_name = self.route_target_point
            if self.route_failed_path != path_name:
                logger.warning(
                    "no route path configured from current point=%s",
                    path_name,
                )
                self.route_failed_path = path_name
            return None
        if not path.get("steps"):
            logger.warning("route path has no steps: %s", path.get("name"))
            self.route_failed_path = str(path.get("name", ""))
            return None

        self.route_target_point = str(path.get("target", "")).strip() or None
        return self._start_route_path(frame, combat, path)

    def _start_route_path(self, frame, combat, path):
        self.route_controller = RouteTestController(
            config=self.config,
            path=path,
        )
        self.route_path_name = str(path.get("name", "")).strip() or None
        self.route_no_target_since = None
        self.route_failed_path = None
        logger.info(
            "route combat starting: current=%s path=%s target=%s",
            self.route_active_point,
            self.route_path_name,
            path.get("target"),
        )
        return self._process_route_navigation(frame, combat)

    def _next_route_path(self, source: str) -> dict[str, Any] | None:
        return self._route_path_between(source, None)

    def _route_path_between(
        self,
        source: str,
        target: str | None,
    ) -> dict[str, Any] | None:
        """Return the first path segment needed to reach ``target``.

        Route records describe one hop between combat points. Recovery can
        start from an unexpected point, so a direct source/target lookup is
        insufficient when the configured route is multi-segment. For a
        specific target, search the directed route graph and return the first
        hop on the shortest configured path. With no target, preserve the
        normal patrol behavior and return the first outgoing path.
        """
        route_config = self.config.get("route_combat", {})
        paths = self._active_route_paths(route_config)
        source_name = str(source).strip()
        target_name = None if target is None else str(target).strip()
        if not source_name:
            return None

        outgoing: dict[str, list[dict[str, Any]]] = {}
        for path in paths:
            path_source = str(path.get("source", "")).strip()
            path_target = str(path.get("target", "")).strip()
            if not path_source or not path_target:
                continue
            outgoing.setdefault(path_source, []).append(path)

        direct = outgoing.get(source_name, [])
        if target_name is None:
            return direct[0] if direct else None
        if not target_name or source_name == target_name:
            return None

        # Each queue item carries the first hop from the requested source.
        # Keeping the hop with the node avoids reconstructing or mutating the
        # saved path records while traversing the graph.
        queue: deque[tuple[str, dict[str, Any]]] = deque()
        visited = {source_name}
        for path in direct:
            next_point = str(path.get("target", "")).strip()
            if next_point and next_point not in visited:
                visited.add(next_point)
                queue.append((next_point, path))

        while queue:
            point_name, first_hop = queue.popleft()
            if point_name == target_name:
                return first_hop
            for path in outgoing.get(point_name, []):
                next_point = str(path.get("target", "")).strip()
                if next_point and next_point not in visited:
                    visited.add(next_point)
                    queue.append((next_point, first_hop))
        return None

    def _active_route_group(self, route_config: dict[str, Any]) -> str:
        return active_route_group(route_config)

    def _active_route_points(
        self,
        route_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        group = self._active_route_group(route_config)
        return [
            dict(point)
            for point in (route_config.get("points") or [])
            if isinstance(point, dict)
            and route_point_group(point) == group
        ]

    def _active_route_paths(
        self,
        route_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        group = self._active_route_group(route_config)
        return [
            dict(path)
            for path in (route_config.get("paths") or [])
            if isinstance(path, dict)
            and route_path_group(path) == group
        ]

    def _route_no_monster_seconds(self) -> float:
        try:
            return max(
                0.1,
                float(
                    self.config.get("route_combat", {}).get(
                        "no_monster_seconds",
                        20.0,
                    )
                ),
            )
        except (TypeError, ValueError):
            return 20.0

    def _path_combat_enabled(self) -> bool:
        return (
            str(
                self.config.get("route_combat", {}).get(
                    "mode",
                    "normal",
                )
            ).strip().lower()
            in {"path_combat", "path", "route"}
        )

    def _activate_route_combat_profile(self, frame) -> None:
        """Apply the profile belonging to the current minimap combat point."""
        if not self._path_combat_enabled():
            self.config["combat"] = deepcopy(self.base_combat_config)
            self.route_active_point = None
            self.route_active_profile = None
            return

        route_config = self.config.get("route_combat", {})
        points = self._active_route_points(route_config)
        profiles = route_config.get("point_profiles") or {}
        if not points or not isinstance(profiles, dict):
            active_combat = deepcopy(self.base_combat_config)
            active_combat.setdefault("monsters", {})["region"] = []
            active_combat.setdefault("fixed_point", {})["enabled"] = False
            self.config["combat"] = active_combat
            self.route_active_point = None
            self.route_active_profile = None
            self.route_profile_point = None
            return

        observation = self._observe_route_combat_minimap(frame)
        player_local = (
            (observation.player_local.x, observation.player_local.y)
            if observation is not None and observation.player_local is not None
            else None
        )
        tolerance = max(0, int(route_config.get("point_tolerance", 3)))
        point_name = current_point_name(player_local, points, tolerance)
        if self.route_target_point is None and point_name is not None:
            self.route_target_point = point_name
        # Combat vision must follow the platform where the character is
        # actually standing. The route target is only for navigation; using
        # its profile while the character is elsewhere makes the detector
        # search the wrong platform after a fall or knockback.
        profile_name = point_name
        profile = profiles.get(profile_name) if profile_name else None

        active_combat = deepcopy(self.base_combat_config)
        # A path profile is authoritative. Do not fall back to the common
        # monster ROI while the initial minimap point is unknown or a point
        # has not been configured yet.
        active_combat.setdefault("monsters", {})["region"] = []
        active_combat.setdefault("fixed_point", {})["enabled"] = False
        previous_profile_point = self.route_profile_point
        if isinstance(profile, dict):
            if "monster_region" in profile:
                monster_region = profile.get("monster_region")
                if (
                    isinstance(monster_region, (list, tuple))
                    and len(monster_region) == 4
                ):
                    active_combat.setdefault("monsters", {})["region"] = [
                        int(value) for value in monster_region
                    ]
                else:
                    # An explicitly empty profile must not inherit the
                    # previous point's screen ROI.
                    active_combat.setdefault("monsters", {})["region"] = []
                    logger.warning(
                        "route combat point %s has no monster region",
                        profile_name,
                    )

            fixed_config = active_combat.setdefault("fixed_point", {})
            fixed_point = profile.get("fixed_point")
            if isinstance(fixed_point, dict):
                point_value = fixed_point.get("point")
                if isinstance(point_value, (list, tuple)) and len(point_value) == 2:
                    fixed_config["point"] = [
                        int(point_value[0]),
                        int(point_value[1]),
                    ]
                elif point_value is None:
                    fixed_config["point"] = None
                    fixed_config["enabled"] = False
                if "enabled" in fixed_point:
                    fixed_config["enabled"] = bool(fixed_point["enabled"])
            elif isinstance(fixed_point, (list, tuple)) and len(fixed_point) == 2:
                fixed_config["point"] = [
                    int(fixed_point[0]),
                    int(fixed_point[1]),
                ]
                fixed_config["enabled"] = bool(
                    profile.get("fixed_point_enabled", True)
                )
            elif "fixed_point_enabled" in profile:
                fixed_config["enabled"] = bool(profile["fixed_point_enabled"])

        self.config["combat"] = active_combat
        if previous_profile_point != profile_name:
            self.monster_tracker = MonsterTracker(
                confirm_frames=self.monster_tracker.confirm_frames,
                max_misses=self.monster_tracker.max_misses,
                max_distance=self.monster_tracker.max_distance,
            )
            self._reset_attack_state()
            self.fixed_point_check_at = None
            self.fixed_point_roam_at = None
            self.fixed_point_roam_until = None
            self.fixed_point_returning = False
            self.fixed_point_release_pending = False
        self.route_active_point = point_name
        self.route_profile_point = profile_name
        self.route_active_profile = (
            dict(profile) if isinstance(profile, dict) else None
        )
        if point_name or profile_name:
            logger.info(
                "route combat point: actual=%s target=%s profile=%s "
                "monster_region=%s fixed_point=%s",
                point_name,
                self.route_target_point,
                "loaded" if isinstance(profile, dict) else "missing",
                active_combat.get("monsters", {}).get("region"),
                active_combat.get("fixed_point", {}).get("point"),
            )

    def _observe_route_combat_minimap(self, frame):
        minimap_config = self.config.get("minimap", {})
        if not minimap_config.get("enabled", True):
            return None
        region = tuple(minimap_config.get("region") or ())
        if len(region) != 4:
            return None
        lower = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("lower", [24, 145, 150])
        )
        upper = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("upper", [32, 255, 255])
        )
        self.last_minimap = observe_minimap(frame, region, lower, upper)
        return self.last_minimap

    def _apply_pickup_priority(self, frame, combat):
        """Give an already-started coin cycle priority over combat actions."""
        if combat.reason in {
            "hp-low",
            "mp-low",
            "player-not-found-recovery",
            "player-not-found",
        }:
            self._reset_loot_state()
            return combat, False

        now = time.monotonic()
        action = self._pickup_state_action(frame, combat, now)
        if action is not None:
            return action, True
        if self.loot_phase == "scanning":
            # The scan window itself is an active pickup state. Keep combat
            # from taking over until the one-second empty-window check ends.
            return replace(
                combat,
                action=str(
                    self.config.get("combat", {}).get(
                        "observe_action",
                        "observe",
                    )
                ),
                reason="pickup-scanning",
            ), True
        return combat, False

    def _apply_loot_and_patrol(self, frame, combat, pickup_checked=False):
        pickup_config = self.pickup_config
        if combat.reason in {"hp-low", "mp-low", "player-not-found-recovery"} or (
            combat.reason == "player-not-found"
            and not (
                self._fixed_point_enabled()
                and combat.confirmed_targets
            )
        ):
            self._reset_loot_state()
            return combat

        now = time.monotonic()
        # Fixed-point distance correction has priority over combat. A
        # knockback or movement drift must return to the marker even when a
        # monster is still visible.
        if combat.confirmed_targets and self._fixed_point_enabled():
            fixed_point_action = self._fixed_point_return_if_needed(
                frame,
                combat,
            )
            if fixed_point_action is not None:
                return fixed_point_action

        if combat.confirmed_targets:
            self.no_target_since = None
            self.patrol_direction = None
            self.patrol_until = None
            self.patrol_release_pending = False
            # Keep an active loot cycle paused while combat has priority. It
            # resumes after the target disappears instead of restarting the
            # interval on every combat frame.
            if self.loot_phase in {"scanning", "picking"}:
                self.loot_phase = "paused-combat"
                self._reset_pickup_motion()
            if (
                self._path_combat_enabled()
                and combat.reason == "nearest-monster-out-of-attack-range"
                and combat.target_direction in {"left", "right"}
            ):
                return self._apply_combat_obstacle_approach(frame, combat)
            self._reset_combat_approach_obstacle_state()
            self._reset_fixed_point_return_obstacle_state()
            return combat

        if combat.reason == "no-target" and self.no_target_since is None:
            self.no_target_since = now
        if not combat.confirmed_targets:
            self._reset_combat_approach_obstacle_state()

        if not pickup_checked:
            loot_action = self._pickup_state_action(frame, combat, now)
            if loot_action is not None:
                return loot_action
            if self.loot_phase == "scanning":
                # The one-second scan window owns the no-target decision. Do
                # not begin patrol movement until that window has closed.
                return combat

        if self._fixed_point_enabled():
            return self._apply_fixed_point(frame, combat, now)

        if self._path_combat_enabled():
            return combat

        patrol_config = self.config.get("combat", {}).get("patrol", {})
        if not bool(patrol_config.get("enabled", True)):
            return combat

        if self.no_target_since is None:
            return combat
        patrol_delay = patrol_delay_seconds(self.config)
        if now - self.no_target_since < patrol_delay:
            return combat

        if (
            self.patrol_direction is None
            or self.patrol_until is None
            or now >= self.patrol_until
        ):
            if (
                self.patrol_direction is not None
                and self.patrol_until is not None
                and now >= self.patrol_until
                and not self.patrol_release_pending
            ):
                self.patrol_release_pending = True
                self.patrol_direction = None
                self.patrol_until = None
                self.loot_phase = "scanning"
                return replace(
                    combat,
                    action=str(
                        self.config.get("combat", {}).get(
                            "observe_action",
                            "observe",
                        )
                    ),
                    reason="patrol-hold-complete",
                )
            self.patrol_release_pending = False
            self.patrol_direction = self._patrol_direction(frame)
            patrol_config = self.config.get("combat", {}).get("patrol", {})
            minimum, maximum = _seconds_range(
                patrol_config,
                "move_duration_ms",
                (3.0, 8.0),
                legacy_seconds_key=None,
            )
            self.patrol_until = now + random.uniform(minimum, maximum)
            logger.info(
                "patrol hold: direction=%s duration=%.0fms",
                self.patrol_direction,
                (self.patrol_until - now) * 1000,
            )
        direction = self.patrol_direction
        if direction not in {"left", "right"}:
            self.loot_phase = "scanning"
            return combat
        self.loot_phase = "patrolling"
        action_key = (
            "move_left_action"
            if direction == "left"
            else "move_right_action"
        )
        return replace(
            combat,
            action=str(
                self.config.get("combat", {}).get(
                    action_key,
                    f"move_{direction}({direction})",
                )
            ),
            reason="patrol-no-target",
        )

    def _pickup_state_action(self, frame, combat, now: float):
        """Run the periodic coin-pickup state machine.

        A cycle is started only when its configured interval expires. The
        first detected coin is locked as the target for that cycle. The
        controller may move toward that target and keep emitting the single
        ``pickup(z,3)`` action until the input layer finishes its three taps,
        but it never switches to another coin in the same cycle.
        """
        if not self.pickup_enabled:
            self.last_coin_detections = ()
            self.loot_phase = "disabled"
            return None

        interval = max(0.0, self.pickup_interval_seconds)
        if self.loot_phase == "paused-combat":
            self.loot_phase = "scanning"
            self.loot_scan_until = now + self.pickup_scan_seconds
            self.pickup_target = None
            self.pickup_action_sent = False
            logger.info(
                "coin pickup cycle resumed after combat: scan_seconds=%.2f",
                self.pickup_scan_seconds,
            )

        if self.loot_phase not in {"scanning", "picking"}:
            if self.next_pickup_at is None:
                self.next_pickup_at = now + interval
            if now < self.next_pickup_at:
                self.last_coin_detections = ()
                self.loot_phase = "waiting"
                return None
            self.loot_phase = "scanning"
            self.loot_scan_until = now + self.pickup_scan_seconds
            self.pickup_target = None
            self.pickup_action_sent = False
            logger.info(
                "coin pickup cycle started: scan_seconds=%.2f",
                self.pickup_scan_seconds,
            )

        coins = self._detect_coins(frame)
        self.last_coin_detections = coins

        if self.loot_phase == "picking":
            if self.pickup_action_sent:
                if (
                    self.pickup_cycle_until is not None
                    and now < self.pickup_cycle_until
                ):
                    return replace(
                        combat,
                        action=str(
                            self.pickup_config.get(
                                "action",
                                "pickup(z,3)",
                            )
                        ),
                        reason="pickup-coins",
                    )
                self._finish_pickup_cycle(now)
                return None

            if coins:
                action = self._coin_pickup_action(frame, combat, coins)
                if action is not None:
                    return action
            if self.pickup_target is not None:
                action = self._continue_coin_move(frame, combat, now)
                if action is not None:
                    return action
                # The locked target was not reacquired within the grace
                # period. End this one-target cycle instead of switching to
                # another coin that happens to be visible.
                self._finish_pickup_cycle(now)
                return None

        min_coins = max(1, int(self.pickup_config.get("min_coins", 1)))
        if len(coins) >= min_coins:
            self.loot_scan_until = now + self.pickup_scan_seconds
            action = self._coin_pickup_action(frame, combat, coins)
            if action is not None:
                self.loot_phase = "picking"
                return action
        elif self.loot_phase == "picking":
            action = self._continue_coin_move(frame, combat, now)
            if action is not None:
                return action

        if self.loot_scan_until is not None and now < self.loot_scan_until:
            self.loot_phase = "scanning"
            return None

        self._finish_pickup_cycle(now)
        return None

    def _finish_pickup_cycle(self, now: float) -> None:
        self.loot_phase = "waiting"
        self.loot_scan_until = None
        self.pickup_cycle_until = None
        self.pickup_target = None
        self.pickup_action_sent = False
        self.last_coin_detections = ()
        self._reset_pickup_motion()
        self.next_pickup_at = now + max(0.0, self.pickup_interval_seconds)
        logger.info(
            "coin pickup cycle finished; next cycle in %.2fs",
            max(0.0, self.pickup_interval_seconds),
        )

    def _pickup_seconds_until_next(self) -> float:
        if self.next_pickup_at is None:
            return max(0.0, self.pickup_interval_seconds)
        return max(0.0, self.next_pickup_at - time.monotonic())

    def _fixed_point_enabled(self) -> bool:
        return bool(
            self.config.get("combat", {})
            .get("fixed_point", {})
            .get("enabled", False)
        )

    def _apply_fixed_point(self, frame, combat, now: float):
        fixed_config = self.config.get("combat", {}).get("fixed_point", {})
        target = fixed_config.get("point")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            logger.warning("fixed-point mode is enabled without a minimap point")
            return self._fixed_point_idle_action(combat)

        fixed_point_action = self._fixed_point_return_if_needed(frame, combat)
        if fixed_point_action is not None:
            return fixed_point_action

        attack_action = str(
            self.config.get("combat", {}).get(
                "attack_action",
                "tap_attack(ctrl)",
            )
        )
        # When the marker distance is within the configured threshold, combat
        # movement and attacks retain their normal priority.
        if combat.reason == "nearest-monster-out-of-attack-range":
            return combat
        if combat.target is not None and _same_action_kind(
            combat.action,
            attack_action,
        ):
            return combat

        observation = self.last_minimap
        if observation is None or observation.player_global is None:
            logger.info("fixed-point minimap player not found; holding position")
            self._reset_fixed_point_return_obstacle_state()
            return self._fixed_point_idle_action(
                combat,
                "fixed-point-player-not-found",
            )

        self.fixed_point_returning = False
        self._reset_fixed_point_return_obstacle_state()
        return self._fixed_point_idle_action(combat, "fixed-point-hold")

    def _fixed_point_return_if_needed(self, frame, combat):
        """Return a correction action when the minimap marker is too far away."""
        fixed_config = self.config.get("combat", {}).get("fixed_point", {})
        target = fixed_config.get("point")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return None

        observation = self._observe_fixed_point_minimap(frame)
        if observation is None or observation.player_global is None:
            return None

        target_x = int(target[0])
        target_y = int(target[1])
        player_local = observation.player_target
        if player_local is None:
            return None
        player_x = player_local.x
        player_y = player_local.y
        distance = math.hypot(target_x - player_x, target_y - player_y)
        return_distance = _fixed_point_return_distance(fixed_config)
        if distance > return_distance:
            self.fixed_point_returning = True
            direction = "right" if target_x > player_x else "left"
            logger.info(
                "fixed-point return: player=(%s,%s) target=(%s,%s) "
                "distance=%.1fpx threshold=%.1fpx direction=%s",
                player_x,
                player_y,
                target_x,
                target_y,
                distance,
                return_distance,
                direction,
            )
            result = self._fixed_point_obstacle_return(frame, combat, direction)
            return replace(
                combat,
                action=result.action,
                reason=f"fixed-point-return:{result.reason}",
            )

        self.fixed_point_returning = False
        self._reset_fixed_point_return_obstacle_state()
        return None

    def _observe_fixed_point_minimap(self, frame):
        minimap_config = self.config.get("minimap", {})
        if not minimap_config.get("enabled", True):
            return None
        region = tuple(minimap_config.get("region") or ())
        if len(region) != 4:
            return None
        lower = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("lower", [24, 145, 150])
        )
        upper = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("upper", [32, 255, 255])
        )
        self.last_minimap = observe_minimap(frame, region, lower, upper)
        return self.last_minimap

    def _fixed_point_move(self, combat, direction: str, reason: str):
        if direction not in {"left", "right"}:
            return self._fixed_point_idle_action(combat)
        action_key = "move_left_action" if direction == "left" else "move_right_action"
        return replace(
            combat,
            action=str(
                self.config.get("combat", {}).get(
                    action_key,
                    f"move_{direction}({direction})",
                )
            ),
            reason=reason,
        )

    def _fixed_point_obstacle_return(self, frame, combat, direction: str):
        if direction not in {"left", "right"}:
            return self._fixed_point_idle_action(combat)
        if self.fixed_point_return_controller is None:
            self.fixed_point_return_controller = RouteTestController(
                config=self.config,
                path={
                    "name": "fixed-point-return",
                    "source": self.route_active_point or "",
                    "target": self.route_active_point or "",
                    "steps": [],
                },
            )
        result = self.fixed_point_return_controller.process_combat_approach(
            frame,
            direction,
        )
        return replace(
            combat,
            action=result.action,
            reason=f"fixed-point-return:{result.reason}",
        )

    def _reset_fixed_point_return_obstacle_state(self) -> None:
        if self.fixed_point_return_controller is not None:
            self.fixed_point_return_controller.reset_obstacle_state()

    def _fixed_point_idle_action(self, combat, reason: str = "fixed-point-hold"):
        if combat.action and "attack" in combat.action.lower():
            return combat
        return replace(
            combat,
            action=str(
                self.config.get("combat", {}).get("observe_action", "observe")
            ),
            reason=reason,
        )

    def _coin_pickup_action(self, frame, combat, coins):
        player = combat.player
        if player is None:
            player = self._detect_player_for_auxiliary_action(frame)
        if player is None:
            continued = self._continue_coin_move(frame, combat, time.monotonic())
            if continued is not None:
                return continued
            return replace(
                combat,
                action=str(
                    self.config.get("combat", {}).get(
                        "observe_action",
                        "observe",
                    )
                ),
                reason="coin-player-not-found",
            )

        if self.pickup_target is None:
            self.pickup_target = min(
                coins,
                key=lambda coin: abs(coin.center[0] - player.anchor[0]),
            )
        else:
            target_center = self.pickup_target.center
            nearest = min(
                coins,
                key=lambda coin: math.hypot(
                    coin.center[0] - target_center[0],
                    coin.center[1] - target_center[1],
                ),
            )
            target_distance = math.hypot(
                nearest.center[0] - target_center[0],
                nearest.center[1] - target_center[1],
            )
            target_width = max(self.pickup_target.box[2], 1)
            target_height = max(self.pickup_target.box[3], 1)
            if target_distance > max(32.0, target_width * 3, target_height * 3):
                return None
            self.pickup_target = nearest

        nearest = self.pickup_target
        distance = abs(nearest.center[0] - player.anchor[0])
        pickup_distance = max(
            1,
            int(self.pickup_config.get("pickup_distance", 28)),
        )
        if distance <= pickup_distance:
            self._reset_pickup_motion()
            now = time.monotonic()
            if not self.pickup_action_sent:
                self.pickup_cycle_until = now + max(
                    0.8,
                    float(
                        pickup_config_value(
                            self.pickup_config,
                            "cycle_seconds",
                            1.0,
                        )
                    ),
                )
                self.pickup_action_sent = True
            return replace(
                combat,
                player=player,
                action=str(self.pickup_config.get("action", "pickup(z,3)")),
                reason="pickup-coins",
            )
        self.pickup_cycle_until = None
        direction = self.pickup_move_direction
        if nearest.center[0] < player.anchor[0]:
            direction = "left"
        elif nearest.center[0] > player.anchor[0]:
            direction = "right"
        self.pickup_move_direction = direction
        self.pickup_target_lost_since = None
        return self._coin_move_action(
            frame,
            combat,
            player,
            direction,
            "move-to-coins",
        )

    def _continue_coin_move(self, frame, combat, now: float):
        """Keep a coin approach alive through a short detection gap."""
        direction = self.pickup_move_direction
        if direction not in {"left", "right"}:
            return None
        if self.pickup_target_lost_since is None:
            self.pickup_target_lost_since = now
        grace = max(
            0.15,
            float(
                pickup_config_value(
                    self.pickup_config,
                    "target_missing_grace_seconds",
                    0.45,
                )
            ),
        )
        if now - self.pickup_target_lost_since > grace:
            self._reset_pickup_motion()
            return None
        return self._coin_move_action(
            frame,
            combat,
            combat.player,
            direction,
            "move-to-coins-target-lost",
        )

    def _coin_move_action(self, frame, combat, player, direction: str, reason: str):
        if direction not in {"left", "right"}:
            self._reset_pickup_motion()
            return replace(
                combat,
                player=player,
                action=str(
                    self.config.get("combat", {}).get(
                        "observe_action",
                        "observe",
                    )
                ),
                reason=reason,
            )
        if self.pickup_move_controller is None:
            self.pickup_move_controller = RouteTestController(
                config=self.config,
                path={
                    "name": "pickup-move",
                    "source": self.route_active_point or "",
                    "target": self.route_active_point or "",
                    "steps": [],
                },
            )
        result = self.pickup_move_controller.process_combat_approach(
            frame,
            direction,
        )
        return replace(
            combat,
            player=player,
            action=result.action,
            reason=f"{reason}:{result.reason}",
        )

    def _detect_coins(self, frame) -> tuple[MonsterDetection, ...]:
        if not self.coin_config.get("enabled", True):
            return ()
        region = tuple(self.coin_config.get("region") or ())
        if len(region) != 4:
            height, width = frame.shape[:2]
            region = (0, 0, width, height)
        active_config = dict(self.coin_config)
        active_config["region"] = list(region)
        return detect_monsters(frame, active_config, self.coin_detector)

    def _detect_player_for_auxiliary_action(self, frame):
        player_config = self.config.get("combat", {}).get("player", {})
        if self.player_detector is not None:
            return self.player_detector.detect(frame, player_config)
        return detect_player_screen(frame, player_config)

    def _patrol_direction(self, frame) -> str | None:
        self.patrol_direction = self._direction_from_patrol_config(frame)
        return self.patrol_direction

    def _direction_from_patrol_config(self, frame) -> str | None:
        patrol_config = self.config.get("combat", {}).get("patrol", {})
        if patrol_config.get("mode") != "minimap_target":
            return self._direction_from_minimap(frame)
        direction = self._direction_to_minimap_target(frame, patrol_config)
        return direction if direction is not None else self._direction_from_minimap(frame)

    def _direction_to_minimap_target(
        self,
        frame,
        patrol_config: dict[str, Any],
    ) -> str | None:
        target = patrol_config.get("minimap_target")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return None
        minimap_config = self.config.get("minimap", {})
        if not minimap_config.get("enabled", True):
            return None
        region = tuple(minimap_config.get("region") or ())
        if len(region) != 4:
            return None
        lower = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("lower", [24, 145, 150])
        )
        upper = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("upper", [32, 255, 255])
        )
        self.last_minimap = observe_minimap(frame, region, lower, upper)
        if self.last_minimap.player_global is None:
            return None
        tolerance = max(0, int(patrol_config.get("target_tolerance", 2)))
        target_x = int(target[0])
        player_local = self.last_minimap.player_target
        if player_local is None:
            return None
        player_x = player_local.x
        if abs(target_x - player_x) <= tolerance:
            logger.info(
                "patrol target reached: player_x=%s target_x=%s",
                player_x,
                target_x,
            )
            return "stay"
        direction = "right" if target_x > player_x else "left"
        logger.info(
            "patrol target direction=%s player_x=%s target_x=%s",
            direction,
            player_x,
            target_x,
        )
        return direction

    def _direction_from_minimap(self, frame) -> str:
        minimap_config = self.config.get("minimap", {})
        if not minimap_config.get("enabled", True):
            return random.choice(("left", "right"))
        region = tuple(minimap_config.get("region") or ())
        if len(region) != 4:
            return random.choice(("left", "right"))
        lower = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("lower", [24, 145, 150])
        )
        upper = tuple(
            minimap_config.get("player_color_hsv", {})
            .get("upper", [32, 255, 255])
        )
        self.last_minimap = observe_minimap(frame, region, lower, upper)
        ratio = self.last_minimap.horizontal_ratio
        edge_threshold = float(
            minimap_config.get("edge_threshold", 0.25)
        )
        if ratio is None:
            direction = random.choice(("left", "right"))
        elif ratio <= edge_threshold:
            direction = "right"
        elif ratio >= 1.0 - edge_threshold:
            direction = "left"
        else:
            direction = random.choice(("left", "right"))
        logger.info(
            "patrol direction=%s minimap_ratio=%s",
            direction,
            f"{ratio:.2f}" if ratio is not None else "not-found",
        )
        return direction

    def _reset_loot_state(self) -> None:
        self.loot_phase = "waiting" if self.pickup_enabled else "disabled"
        self.loot_scan_until = None
        self.pickup_target = None
        self.pickup_action_sent = False
        self.last_coin_detections = ()
        self.pickup_cycle_until = None
        self._reset_pickup_motion()
        if self.pickup_enabled and self.next_pickup_at is None:
            self.next_pickup_at = time.monotonic() + max(
                0.0,
                self.pickup_interval_seconds,
            )

    def _reset_pickup_motion(self) -> None:
        self.pickup_move_direction = None
        self.pickup_target_lost_since = None
        if self.pickup_move_controller is not None:
            self.pickup_move_controller.reset_obstacle_state()

    def _apply_player_recovery(self, frame, combat):
        """Move left for a bounded random interval when the player is missing."""
        combat_config = self.config.get("combat", {})
        if self._fixed_point_enabled() or self._path_combat_enabled():
            self.player_recovery_until = None
            self.player_recovery_direction = None
            return combat
        if combat.reason in {"hp-low", "mp-low"}:
            self.player_recovery_until = None
            self.player_recovery_direction = None
            return combat

        now = time.monotonic()
        observe_action = str(combat_config.get("observe_action", "observe"))

        if self.player_recovery_until is not None:
            if combat.player is not None:
                logger.info("player recovery complete: player detected")
                self.player_recovery_until = None
                self.player_recovery_direction = None
                return combat
            if now < self.player_recovery_until:
                return replace(
                    combat,
                    action=self._movement_action(
                        self.player_recovery_direction or "left"
                    ),
                    reason="player-not-found-recovery",
                )
            logger.info("player recovery complete: timeout reached")
            self.player_recovery_until = None
            self.player_recovery_direction = None
            return replace(
                combat,
                action=observe_action,
                reason="player-recovery-timeout",
            )

        if combat.reason != "player-not-found":
            return combat

        recovery_config = combat_config.get("player_recovery", {})
        minimum = max(
            0.1,
            float(recovery_config.get("min_seconds", 3.0)),
        )
        maximum = max(
            minimum,
            float(recovery_config.get("max_seconds", 10.0)),
        )
        duration = random.uniform(minimum, maximum)
        self.player_recovery_until = now + duration
        self.player_recovery_direction = self._direction_from_minimap(frame)
        logger.warning(
            "player not found: holding %s for %.2fs",
            self.player_recovery_direction,
            duration,
        )
        return replace(
            combat,
            action=self._movement_action(self.player_recovery_direction),
            reason="player-not-found-recovery",
        )

    def _movement_action(self, direction: str) -> str:
        action_key = (
            "move_left_action"
            if direction == "left"
            else "move_right_action"
        )
        fallback = (
            "move_left(left)"
            if direction == "left"
            else "move_right(right)"
        )
        return str(self.config.get("combat", {}).get(action_key, fallback))

    def _apply_combat_obstacle_approach(self, frame, combat):
        if self.combat_approach_controller is None:
            self.combat_approach_controller = RouteTestController(
                config=self.config,
                path={
                    "name": "combat-approach",
                    "source": self.route_active_point or "",
                    "target": self.route_active_point or "",
                    "steps": [],
                },
            )
        result = self.combat_approach_controller.process_combat_approach(
            frame,
            str(combat.target_direction),
        )
        return replace(
            combat,
            action=result.action,
            reason=f"combat-approach:{result.reason}",
        )

    def _reset_combat_approach_obstacle_state(self) -> None:
        if self.combat_approach_controller is not None:
            self.combat_approach_controller.reset_obstacle_state()

    def _apply_attack_state(self, frame, combat):
        combat_config = self.config.get("combat", {})
        attack_action = str(
            combat_config.get("attack_action", "tap_attack(ctrl)")
        )
        observe_action = str(combat_config.get("observe_action", "observe"))
        bar_config = combat_config.get("enemy_health_bar", {})
        health_bar_enabled = bool(bar_config.get("enabled", True))
        target_missing_confirm_frames = max(
            1,
            int(combat_config.get("attack_target_missing_frames", 3)),
        )
        appear_confirm_frames = max(
            1,
            int(bar_config.get("appear_confirm_frames", 1)),
        )
        missing_confirm_frames = max(
            1,
            int(bar_config.get("missing_confirm_frames", 2)),
        )
        acquire_timeout_frames = max(
            1,
            int(bar_config.get("acquire_timeout_frames", 3)),
        )

        if combat.reason in {"hp-low", "mp-low"} or (
            combat.reason == "player-not-found"
            and not (
                self._fixed_point_enabled()
                and combat.confirmed_targets
            )
        ):
            self._reset_attack_state()
            self.last_enemy_health_bar = EnemyHealthBarObservation(
                present=False,
                score=0.0,
                region=None,
                fill_ratio=None,
            )
            return combat

        if self.attack_cooldown_frames > 0:
            self.attack_cooldown_frames -= 1
            self._reset_attack_state(keep_cooldown=True)
            self.last_enemy_health_bar = EnemyHealthBarObservation(
                present=False,
                score=0.0,
                region=None,
                fill_ratio=None,
            )
            return replace(
                combat,
                action=observe_action,
                reason="attack-target-cooldown",
            )

        if self.attack_locked_target is None:
            if combat.target is None or combat.action != attack_action:
                self.attack_phase = "search"
                self.last_enemy_health_bar = EnemyHealthBarObservation(
                    present=False,
                    score=0.0,
                    region=None,
                    fill_ratio=None,
                )
                return combat
            self.attack_locked_target = combat.target
            self.attack_phase = (
                "awaiting-bar"
                if health_bar_enabled
                else "attacking"
            )
            self.attack_wait_frames = 0
            self.attack_missing_frames = 0

        locked_target = self._find_locked_target(combat)
        if locked_target is None:
            # The previously locked monster is gone. Prefer a target from the
            # current frame immediately so an opposite-side monster can take
            # over without spending the missing-target grace frames attacking
            # the old direction.
            replacement = combat.target
            if replacement is not None and replacement in combat.in_range_targets:
                self.attack_locked_target = replacement
                self.attack_missing_frames = 0
                self.attack_wait_frames = 0
                self.attack_phase = (
                    "awaiting-bar"
                    if health_bar_enabled
                    else "attacking"
                )
                locked_target = replacement
            else:
                self.attack_missing_frames += 1
                if (
                    self.attack_missing_frames
                    < target_missing_confirm_frames
                    and self.attack_locked_target is not None
                ):
                    return replace(
                        combat,
                        target=self.attack_locked_target,
                        nearest_target=self.attack_locked_target,
                        target_direction=self._target_direction(
                            combat,
                            self.attack_locked_target,
                        ),
                        action=_oriented_attack_action(
                            attack_action,
                            self._target_direction(
                                combat,
                                self.attack_locked_target,
                            ),
                        ),
                        reason="attack-target-temporarily-lost",
                    )
                self._reset_attack_state()
                return replace(
                    combat,
                    action=observe_action,
                    reason="attack-target-lost",
                )

        # observe_combat selects the nearest in-range monster. If a second
        # monster becomes the nearer target on the opposite side, follow that
        # current-frame decision instead of continuing to attack the old side.
        preferred_target = combat.target
        if (
            preferred_target is not None
            and preferred_target in combat.in_range_targets
            and preferred_target != locked_target
        ):
            self.attack_locked_target = preferred_target
            self.attack_wait_frames = 0
            self.attack_missing_frames = 0
            self.attack_phase = (
                "awaiting-bar"
                if health_bar_enabled
                else "attacking"
            )
            locked_target = preferred_target

        # A locked target can move out of range while it is still visible.
        # Release the attack immediately and let observe_combat's movement
        # proposal steer back toward the target.
        if locked_target not in combat.in_range_targets:
            self._reset_attack_state()
            return combat

        self.attack_missing_frames = 0
        if not health_bar_enabled:
            self.last_enemy_health_bar = EnemyHealthBarObservation(
                present=False,
                score=0.0,
                region=None,
                fill_ratio=None,
            )
            return replace(
                combat,
                target=locked_target,
                nearest_target=locked_target,
                target_direction=self._target_direction(combat, locked_target),
                action=_oriented_attack_action(
                    attack_action,
                    self._target_direction(combat, locked_target),
                ),
                reason="attack-target-present",
            )

        bar = self.enemy_health_bar_detector.detect(
            frame,
            locked_target.box,
        )
        self.last_enemy_health_bar = bar
        if bar.present:
            self.attack_wait_frames += 1
            self.attack_missing_frames = 0
            if self.attack_wait_frames >= appear_confirm_frames:
                self.attack_phase = "attacking"
            return replace(
                combat,
                target=locked_target,
                nearest_target=locked_target,
                target_direction=self._target_direction(combat, locked_target),
                action=_oriented_attack_action(
                    attack_action,
                    self._target_direction(combat, locked_target),
                ),
                reason="enemy-health-bar-present",
            )

        self.attack_missing_frames += 1
        if self.attack_phase == "awaiting-bar":
            self.attack_wait_frames += 1
            if self.attack_wait_frames < acquire_timeout_frames:
                return replace(
                    combat,
                    target=locked_target,
                    nearest_target=locked_target,
                    target_direction=self._target_direction(
                        combat,
                        locked_target,
                    ),
                    action=_oriented_attack_action(
                        attack_action,
                        self._target_direction(combat, locked_target),
                    ),
                    reason="awaiting-enemy-health-bar",
                )
            self._reset_attack_state(cooldown_frames=1)
            return replace(
                combat,
                action=observe_action,
                reason="enemy-health-bar-not-found",
            )

        if self.attack_missing_frames < missing_confirm_frames:
            return replace(
                combat,
                target=locked_target,
                nearest_target=locked_target,
                target_direction=self._target_direction(combat, locked_target),
                action=_oriented_attack_action(
                    attack_action,
                    self._target_direction(combat, locked_target),
                ),
                reason="enemy-health-bar-temporarily-missing",
            )

        self._reset_attack_state(cooldown_frames=1)
        return replace(
            combat,
            action=observe_action,
            reason="enemy-health-bar-disappeared",
        )

    def _find_locked_target(self, combat):
        candidates = combat.confirmed_targets or combat.targets
        if not candidates or self.attack_locked_target is None:
            return None
        label = self.attack_locked_target.label
        old_x, old_y = self.attack_locked_target.center
        matching = [target for target in candidates if target.label == label]
        if not matching:
            return None
        return min(
            matching,
            key=lambda target: (
                (target.center[0] - old_x) ** 2
                + (target.center[1] - old_y) ** 2
            ),
        )

    @staticmethod
    def _target_direction(combat, target) -> str | None:
        if combat.player is None or target is None:
            return None
        player_x = combat.player.anchor[0]
        target_x = target.center[0]
        if target_x < player_x:
            return "left"
        if target_x > player_x:
            return "right"
        return None

    def _reset_attack_state(
        self,
        cooldown_frames: int = 0,
        keep_cooldown: bool = False,
    ) -> None:
        self.input.release_key("ctrl")
        self.attack_phase = "search"
        self.attack_locked_target = None
        self.attack_wait_frames = 0
        self.attack_missing_frames = 0
        if not keep_cooldown:
            self.attack_cooldown_frames = max(0, cooldown_frames)

    def _emit_periodic_key(self) -> None:
        config = self.periodic_key_config
        if not config.get("enabled", False):
            self.next_periodic_key_at = None
            return
        try:
            interval = float(config.get("interval_seconds", 0))
        except (TypeError, ValueError):
            self.next_periodic_key_at = None
            return
        if interval <= 0:
            self.next_periodic_key_at = None
            return

        now = time.monotonic()
        if self.next_periodic_key_at is None:
            self.next_periodic_key_at = now + interval
            return
        if now < self.next_periodic_key_at:
            return

        key = str(config.get("key", "3")).strip() or "3"
        self.input.tap_key(key)
        self.next_periodic_key_at = now + interval


def _target_labels(targets) -> str:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.label] = counts.get(target.label, 0) + 1
    return ",".join(
        f"{label}:{count}" for label, count in sorted(counts.items())
    ) or "-"


def _same_action_kind(first: str, second: str) -> bool:
    return str(first or "").split("(", 1)[0].strip().lower() == str(
        second or ""
    ).split("(", 1)[0].strip().lower()


def _oriented_attack_action(action: str, direction: str | None) -> str:
    """Append a one-shot facing direction to supported attack actions."""
    if direction not in {"left", "right"}:
        return action
    prefix, separator, arguments = str(action).partition("(")
    if not separator or prefix.strip().lower() not in {"tap_attack", "hold_attack"}:
        return action
    arguments = arguments.rsplit(")", 1)[0]
    values = [value.strip() for value in arguments.split(",") if value.strip()]
    if len(values) > 1 and values[1] in {"left", "right"}:
        return action
    return f"{prefix}({arguments},{direction})"


def _log_monster_sides(combat) -> None:
    """Log every confirmed monster side relative to the detected player."""
    targets = combat.confirmed_targets or combat.targets
    if not targets or combat.player is None:
        return

    player_x = combat.player.anchor[0]
    left = [target for target in targets if target.center[0] < player_x]
    right = [target for target in targets if target.center[0] > player_x]
    center = [target for target in targets if target.center[0] == player_x]

    if left:
        logger.info("发现左边怪物：%s", _target_labels(left))
    if right:
        logger.info("发现右边怪物：%s", _target_labels(right))
    if center:
        logger.info("发现正前方怪物：%s", _target_labels(center))


def _has_player_templates(config: dict[str, Any]) -> bool:
    paths = []
    if config.get("template"):
        paths.append(config["template"])
    raw_templates = config.get("templates", [])
    paths.extend(
        [raw_templates]
        if isinstance(raw_templates, (str, Path))
        else list(raw_templates or [])
    )
    for value in paths:
        path = Path(str(value))
        if path.is_file():
            return True
        if path.is_dir() and any(path.glob("*.png")):
            return True
    return False
