from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import time
from typing import Any

import numpy as np

from .pathing import active_route_group, point_in_region, route_point_group
from .vision.minimap import MinimapObservation, observe_minimap
from .vision.player_screen import PlayerScreenDetection, detect_player_screen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteTestResult:
    action: str
    state: str
    reason: str
    minimap: MinimapObservation | None
    player: PlayerScreenDetection | None


class RouteTestController:
    """Standalone route executor used by the route page test button.

    This controller intentionally has no monster or combat dependency. It
    executes one saved path and can be promoted into the main controller after
    the route behavior is validated.
    """

    def __init__(
        self,
        config: dict[str, Any],
        path: dict[str, Any],
    ):
        self.config = config
        self.route_config = dict(config.get("route_combat", {}))
        self.route_group = active_route_group(self.route_config)
        self.path = dict(path)
        self.points = [
            dict(point)
            for point in (self.route_config.get("points") or [])
            if isinstance(point, dict)
            and route_point_group(point) == self.route_group
        ]
        self.ladders = [
            dict(ladder)
            for ladder in (self.route_config.get("ladders") or [])
            if isinstance(ladder, dict)
        ]
        self.steps = [
            dict(step)
            for step in (self.path.get("steps") or [])
            if isinstance(step, dict)
        ]
        self.step_index = 0
        self.state = "running"
        self.phase = "step"
        self.phase_until: float | None = None
        self.climb_started_at: float | None = None
        self.climb_duration_ms = 0
        self.climb_ladder_name: str | None = None
        self.climb_attempts = 0
        self.climb_ground_minimap_y: int | None = None
        self.jump_started_at: float | None = None
        self.jump_duration_ms = 0
        self.jump_ladder_name: str | None = None
        self.jump_attempts = 0
        self.jump_ground_minimap_y: int | None = None
        self.obstacle_phase = "moving"
        self.obstacle_phase_until: float | None = None
        self.obstacle_reference_x: int | None = None
        self.obstacle_jump_reference_x: int | None = None
        self.obstacle_direction: str | None = None
        self.obstacle_next_check_at: float | None = None
        self.obstacle_jump_attempts = 0
        self.last_minimap: MinimapObservation | None = None
        self.last_player: PlayerScreenDetection | None = None

    def process_frame(self, frame: np.ndarray) -> RouteTestResult:
        now = time.monotonic()
        observation = self._observe_minimap(frame)
        player = self._detect_player(frame)
        self.last_minimap = observation
        self.last_player = player

        if self.state == "complete":
            return self._result("observe", "complete", "route-complete")
        if self.state == "failed":
            return self._result("observe", "failed", "route-failed")

        for _ in range(len(self.steps) + 2):
            result = self._process_phase(
                frame,
                now,
                observation,
                player,
            )
            if result is not None:
                return result
        self.state = "failed"
        logger.error("route test exceeded phase transition limit")
        return self._result("observe", "failed", "route-phase-loop")

    def stop(self) -> RouteTestResult:
        self.state = "stopped"
        self.phase = "stopped"
        return self._result("observe", "stopped", "route-stopped")

    def _process_phase(
        self,
        frame: np.ndarray,
        now: float,
        observation: MinimapObservation | None,
        player: PlayerScreenDetection | None,
    ) -> RouteTestResult | None:
        if self.step_index >= len(self.steps):
            self.state = "complete"
            self.phase = "complete"
            logger.info("route test complete: path=%s", self.path.get("name"))
            return self._result("observe", "complete", "route-complete")

        step = self.steps[self.step_index]
        step_type = str(step.get("type", "wait"))

        if step_type == "climb":
            return self._process_climb(
                frame,
                now,
                observation,
                player,
                step,
            )

        if step_type in {"jump_left", "jump_right"}:
            return self._process_jump(
                now,
                observation,
                step,
            )

        if step_type in {"move_to_point", "move_to_ladder"}:
            target_reached = self._target_reached(
                observation,
                step_type,
                step,
            )
            if target_reached:
                self._advance_step()
                return None
            direction = self._direction_to_target(observation, step_type, step)
            if direction is None:
                return self._result(
                    "observe",
                    "running",
                    "route-target-position-not-found",
                )
            return self._obstacle_aware_move(
                now,
                observation,
                direction,
                f"route-{step_type}",
            )

        if step_type in {
            "move_left",
            "move_right",
        }:
            direction = {
                "move_left": "left",
                "move_right": "right",
            }[step_type]
            if self._target_reached(
                observation,
                step_type,
                step,
            ):
                self._advance_step()
                return None
            return self._obstacle_aware_move(
                now,
                observation,
                direction,
                f"route-{step_type}",
            )

        if step_type == "wait":
            duration_ms = self._duration_ms(step)
            if self.phase != "timed":
                self.phase = "timed"
                self.phase_until = now + duration_ms / 1000.0
            if self.phase_until is not None and now < self.phase_until:
                return self._result("observe", "running", "route-wait")
            self._advance_step()
            return None

        logger.warning("unsupported route step: %s", step)
        self._advance_step()
        return None

    def _timed_action(self, step_type: str, direction: str) -> str:
        if step_type in {"jump_left", "jump_right"}:
            jump_key = str(
                self.route_config.get("jump_key", "alt")
            ).strip().lower() or "alt"
            return f"jump_{direction}({direction},{jump_key})"
        return f"move_{direction}({direction})"

    def process_combat_approach(
        self,
        frame: np.ndarray,
        direction: str,
    ) -> RouteTestResult:
        """Move toward a visible monster without advancing the route step."""
        observation = self._observe_minimap(frame)
        result = self._obstacle_aware_move(
            time.monotonic(),
            observation,
            direction,
            "route-combat-approach",
        )
        self.last_minimap = observation
        self.last_player = None
        return result

    def reset_obstacle_state(self) -> None:
        """Forget horizontal progress after an attack or other interruption."""
        self.obstacle_phase = "moving"
        self.obstacle_phase_until = None
        self.obstacle_reference_x = None
        self.obstacle_jump_reference_x = None
        self.obstacle_direction = None
        self.obstacle_next_check_at = None

    def _obstacle_aware_move(
        self,
        now: float,
        observation: MinimapObservation | None,
        direction: str,
        reason: str,
    ) -> RouteTestResult:
        """Hold a direction and jump repeatedly when the yellow point stalls."""
        direction = "left" if direction == "left" else "right"
        jump_key = (
            str(self.route_config.get("jump_key", "alt")).strip().lower()
            or "alt"
        )
        base_action = f"move_{direction}({direction})"
        jump_action = f"jump_{direction}({direction},{jump_key})"
        check_interval_ms = self._positive_int(
            self.route_config.get("obstacle_check_interval_ms", 300),
            300,
        )
        min_progress_px = self._positive_int(
            self.route_config.get("obstacle_min_progress_px", 1),
            1,
        )
        jump_delay_min_ms = self._positive_int(
            self.route_config.get("obstacle_jump_delay_min_ms", 500),
            500,
        )
        jump_delay_max_ms = self._positive_int(
            self.route_config.get("obstacle_jump_delay_max_ms", 700),
            700,
        )
        if jump_delay_max_ms < jump_delay_min_ms:
            jump_delay_min_ms, jump_delay_max_ms = (
                jump_delay_max_ms,
                jump_delay_min_ms,
            )

        if self.obstacle_direction != direction:
            self.reset_obstacle_state()
            self.obstacle_direction = direction

        current_x = self._player_x(observation)
        if self.obstacle_phase == "jump-wait":
            if (
                self.obstacle_phase_until is not None
                and now < self.obstacle_phase_until
            ):
                return self._result(
                    jump_action,
                    "running",
                    f"{reason};obstacle-jump-wait",
                )
            progress = self._directional_progress(
                self.obstacle_jump_reference_x,
                current_x,
                direction,
            )
            if progress >= min_progress_px:
                self.obstacle_phase = "moving"
                self.obstacle_phase_until = None
                self.obstacle_reference_x = current_x
                self.obstacle_next_check_at = (
                    now + check_interval_ms / 1000.0
                )
                return self._result(
                    base_action,
                    "running",
                    f"{reason};obstacle-jump-progress:+{progress}px",
                )
            self.obstacle_phase = "moving"
            self.obstacle_phase_until = None
            self.obstacle_reference_x = current_x
            self.obstacle_next_check_at = now
            return self._result(
                base_action,
                "running",
                f"{reason};obstacle-jump-no-progress;"
                "retry-direction",
            )

        if current_x is None:
            return self._start_obstacle_jump(
                now,
                None,
                direction,
                jump_action,
                reason + ";obstacle-yellow-point-missing",
                jump_delay_min_ms,
                jump_delay_max_ms,
            )
        if self.obstacle_reference_x is None:
            self.obstacle_reference_x = current_x
            self.obstacle_next_check_at = now + check_interval_ms / 1000.0
            return self._result(
                base_action,
                "running",
                reason + ";obstacle-move-start",
            )
        if (
            self.obstacle_next_check_at is not None
            and now < self.obstacle_next_check_at
        ):
            return self._result(
                base_action,
                "running",
                reason + ";obstacle-move-hold",
            )

        previous_x = self.obstacle_reference_x
        self.obstacle_reference_x = current_x
        self.obstacle_next_check_at = now + check_interval_ms / 1000.0
        progress = self._directional_progress(
            previous_x,
            current_x,
            direction,
        )
        if progress >= min_progress_px:
            return self._result(
                base_action,
                "running",
                f"{reason};obstacle-progress:+{progress}px",
            )
        return self._start_obstacle_jump(
            now,
            current_x,
            direction,
            jump_action,
            f"{reason};obstacle-no-progress:{progress}px",
            jump_delay_min_ms,
            jump_delay_max_ms,
        )

    def _start_obstacle_jump(
        self,
        now: float,
        reference_x: int | None,
        direction: str,
        jump_action: str,
        reason: str,
        jump_delay_min_ms: int,
        jump_delay_max_ms: int,
    ) -> RouteTestResult:
        delay_ms = random.randint(jump_delay_min_ms, jump_delay_max_ms)
        self.obstacle_phase = "jump-wait"
        self.obstacle_phase_until = now + delay_ms / 1000.0
        self.obstacle_jump_reference_x = reference_x
        self.obstacle_jump_attempts += 1
        logger.info(
            "route obstacle jump attempt=%s direction=%s delay=%sms "
            "reference_x=%s reason=%s",
            self.obstacle_jump_attempts,
            direction,
            delay_ms,
            reference_x,
            reason,
        )
        return self._result(
            jump_action,
            "running",
            f"{reason};jump-delay={delay_ms}ms",
        )

    @staticmethod
    def _directional_progress(
        previous_x: int | None,
        current_x: int | None,
        direction: str,
    ) -> int:
        if previous_x is None or current_x is None:
            return 0
        delta = int(current_x) - int(previous_x)
        return delta if direction == "right" else -delta

    @staticmethod
    def _player_x(
        observation: MinimapObservation | None,
    ) -> int | None:
        if observation is None or observation.player_local is None:
            return None
        return int(observation.player_local.x)

    @staticmethod
    def _positive_int(value: Any, fallback: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return fallback

    def _process_jump(
        self,
        now: float,
        observation: MinimapObservation | None,
        step: dict[str, Any],
    ) -> RouteTestResult:
        step_type = str(step.get("type", "jump_right"))
        direction = "left" if step_type == "jump_left" else "right"
        action = self._timed_action(step_type, direction)

        if self.phase == "step":
            self.jump_attempts += 1
            self.jump_duration_ms = self._duration_ms(step)
            self.jump_ladder_name = self._jump_source_ladder()
            self.jump_ground_minimap_y = (
                int(observation.player_local.y)
                if observation is not None and observation.player_local is not None
                else None
            )
            self.jump_started_at = now
            self.phase = "jump-timed"
            self.phase_until = now + self.jump_duration_ms / 1000.0
            logger.info(
                "route jump attempt=%s direction=%s duration=%sms "
                "ground_minimap_y=%s",
                self.jump_attempts,
                direction,
                self.jump_duration_ms,
                self.jump_ground_minimap_y,
            )
            return self._result(action, "running", "route-jump-start")

        if self.phase == "jump-timed":
            if self.phase_until is not None and now < self.phase_until:
                return self._result(action, "running", "route-jump-hold")
            self.phase = "jump-check"
            self.phase_until = None
            return self._result("observe", "running", "route-jump-release")

        if self.phase == "jump-check":
            if self._minimap_y_raised(
                observation,
                self.jump_ground_minimap_y,
                "jump",
            ):
                logger.info(
                    "route jump succeeded direction=%s after %s attempts",
                    direction,
                    self.jump_attempts,
                )
                self._advance_step()
                return self._result("observe", "running", "route-jump-success")
            self.phase = "jump-reposition"
            self.phase_until = None
            logger.warning(
                "route jump failed direction=%s attempt=%s; "
                "reposition jump point before retry",
                direction,
                self.jump_attempts,
            )
            return self._result("observe", "running", "route-jump-not-raised")

        if self.phase == "jump-reposition":
            ladder_name = self.jump_ladder_name
            if not ladder_name or self._target_reached(
                observation,
                "move_to_ladder",
                {"ladder": ladder_name},
                allow_vertical_drift=True,
            ):
                self.phase = "step"
                self.phase_until = None
                return self._result(
                    "observe",
                    "running",
                    "route-jump-point-repositioned",
                )
            direction_to_point = self._direction_to_target(
                observation,
                "move_to_ladder",
                {"ladder": ladder_name},
            )
            if direction_to_point is None:
                return self._result(
                    "observe",
                    "running",
                    "route-jump-point-position-not-found",
                )
            return self._result(
                self._obstacle_aware_move(
                    now,
                    observation,
                    direction_to_point,
                    "route-reposition-jump-point",
                ).action,
                "running",
                "route-reposition-jump-point",
            )

        self.phase = "step"
        return self._result(action, "running", "route-jump-reset")

    def _jump_source_ladder(self) -> str | None:
        previous_index = self.step_index - 1
        if previous_index < 0:
            return None
        previous = self.steps[previous_index]
        if str(previous.get("type", "")) != "move_to_ladder":
            return None
        name = str(previous.get("ladder", "")).strip()
        return name or None

    def _process_climb(
        self,
        frame: np.ndarray,
        now: float,
        observation: MinimapObservation | None,
        player: PlayerScreenDetection | None,
        step: dict[str, Any],
    ) -> RouteTestResult:
        ladder_name = str(step.get("ladder", "")).strip()
        ladder = self._find_ladder(ladder_name)
        if ladder is None:
            self.state = "failed"
            return self._result(
                "observe",
                "failed",
                f"route-ladder-not-found:{ladder_name}",
            )

        if self.phase == "step":
            self.climb_ladder_name = ladder_name
            self.climb_duration_ms = max(
                1,
                int(step.get("duration_ms", ladder.get("climb_duration_ms", 1800))),
            )
            self.climb_attempts += 1
            self.climb_ground_minimap_y = (
                int(observation.player_local.y)
                if observation is not None and observation.player_local is not None
                else None
            )
            self.phase = "climb-alt-wait"
            alt_delay_ms = self._alt_delay_ms()
            self.phase_until = now + alt_delay_ms / 1000.0
            logger.info(
                "route climb attempt=%s ladder=%s: tap alt, wait %sms "
                "ground_minimap_y=%s",
                self.climb_attempts,
                ladder_name,
                alt_delay_ms,
                self.climb_ground_minimap_y,
            )
            return self._result("tap_alt", "running", "route-climb-alt")

        if self.phase == "climb-alt-wait":
            if self.phase_until is not None and now < self.phase_until:
                return self._result("observe", "running", "route-climb-alt-wait")
            self.phase = "climb-lift-wait"
            self.phase_until = now + 1.000
            self.climb_started_at = now
            logger.info(
                "route climb ladder=%s: hold up, wait 1000ms before checking",
                ladder_name,
            )
            return self._result("hold_up", "running", "route-climb-up")

        if self.phase == "climb-lift-wait":
            if self.phase_until is not None and now < self.phase_until:
                return self._result("hold_up", "running", "route-climb-up-wait")
            if not self._minimap_player_raised(observation):
                return self._retry_climb(now, ladder_name, "route-climb-not-lifted")
            self.phase = "climb-hold"
            started = self.climb_started_at or now
            self.phase_until = started + self.climb_duration_ms / 1000.0
            logger.info(
                "route climb ladder=%s: nameplate lifted, hold up for %sms",
                ladder_name,
                self.climb_duration_ms,
            )
            return self._result("hold_up", "running", "route-climb-hold")

        if self.phase == "climb-hold":
            if self.phase_until is not None and now < self.phase_until:
                return self._result("hold_up", "running", "route-climb-hold")
            self.phase = "climb-target-check"
            self.phase_until = None
            return self._result("observe", "running", "route-climb-release")

        if self.phase == "climb-target-check":
            if self.step_index + 1 < len(self.steps):
                logger.info(
                    "route climb finished ladder=%s; continuing to next step",
                    ladder_name,
                )
                self._advance_step()
                return None
            target_name = str(self.path.get("target", "")).strip()
            target_reached = self._point_target_reached(
                observation,
                target_name,
            )
            if target_reached:
                logger.info(
                    "route climb reached target point=%s after %s attempts",
                    target_name,
                    self.climb_attempts,
                )
                self._advance_step()
                return None
            return self._retry_climb(
                now,
                ladder_name,
                "route-climb-target-not-reached",
            )

        if self.phase == "reposition-ladder":
            if self._target_reached(
                observation,
                "move_to_ladder",
                {"ladder": ladder_name},
            ):
                self.phase = "step"
                self.phase_until = None
                return self._result(
                    "observe",
                    "running",
                    "route-ladder-repositioned",
                )
            direction = self._direction_to_target(
                observation,
                "move_to_ladder",
                {"ladder": ladder_name},
            )
            if direction is None:
                return self._result(
                    "observe",
                    "running",
                    "route-ladder-position-not-found",
                )
            return self._result(
                self._obstacle_aware_move(
                    now,
                    observation,
                    direction,
                    "route-reposition-ladder",
                ).action,
                "running",
                "route-reposition-ladder",
            )

        self.phase = "step"
        return None

    def _retry_climb(
        self,
        now: float,
        ladder_name: str,
        reason: str,
    ) -> RouteTestResult:
        self.climb_attempts += 1
        self.phase = "reposition-ladder"
        self.phase_until = None
        logger.warning(
            "%s ladder=%s attempt=%s; reposition ladder before retry",
            reason,
            ladder_name,
            self.climb_attempts,
        )
        return self._result("observe", "running", reason)

    def _target_reached(
        self,
        observation: MinimapObservation | None,
        step_type: str,
        step: dict[str, Any],
        allow_vertical_drift: bool = False,
    ) -> bool:
        if observation is None or observation.player_local is None:
            return False
        if step_type == "move_to_point":
            point_name = str(step.get("point", "")).strip()
            if self._fixed_point_for_name(point_name) is not None:
                return self._point_target_reached(observation, point_name)
            point = self._find_point(point_name)
            if point is None:
                return False
            return point_in_region(
                (
                    observation.player_local.x,
                    observation.player_local.y,
                ),
                tuple(point.get("region", ())),
                self._point_tolerance(),
            )
        if step_type in {"move_left", "move_right"}:
            target_type, target_step = self._directional_target(step)
            return self._target_reached(
                observation,
                target_type,
                target_step,
            )
        if step_type == "move_to_fixed_point":
            target = self._fixed_point_for_name(str(step.get("point", "")))
            if target is None:
                return False
            return (
                abs(observation.player_local.x - target[0])
                <= self._point_tolerance()
                and abs(observation.player_local.y - target[1])
                <= self._point_tolerance()
            )
        ladder = self._find_ladder(str(step.get("ladder", "")))
        if ladder is None:
            return False
        target = ladder.get("point")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return False
        if allow_vertical_drift or self._next_step_is_jump():
            return (
                abs(observation.player_local.x - int(target[0]))
                <= self._jump_point_tolerance()
            )
        return (
            abs(observation.player_local.x - int(target[0]))
            <= self._ladder_tolerance()
            and abs(observation.player_local.y - int(target[1]))
            <= self._ladder_tolerance()
        )

    def _directional_target(
        self,
        step: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a directional move to the next saved map marker."""
        explicit_type = str(step.get("target_type", "")).strip()
        explicit_name = str(step.get("target", "")).strip()
        if explicit_type in {"move_to_point", "move_to_ladder"} and explicit_name:
            return explicit_type, (
                {"point": explicit_name}
                if explicit_type == "move_to_point"
                else {"ladder": explicit_name}
            )

        for candidate in self.steps[self.step_index + 1 :]:
            candidate_type = str(candidate.get("type", "")).strip()
            if candidate_type == "move_to_point":
                point_name = str(candidate.get("point", "")).strip()
                if self._fixed_point_for_name(point_name) is not None:
                    return "move_to_fixed_point", {"point": point_name}
                return candidate_type, candidate
            if candidate_type in {"move_to_ladder", "climb"}:
                ladder_name = str(candidate.get("ladder", "")).strip()
                if ladder_name:
                    return "move_to_ladder", {"ladder": ladder_name}

        target_name = str(self.path.get("target", "")).strip()
        if target_name:
            if self._fixed_point_for_name(target_name) is not None:
                return "move_to_fixed_point", {"point": target_name}
            return "move_to_point", {"point": target_name}
        return "move_to_point", {"point": ""}

    def _fixed_point_for_name(self, name: str) -> tuple[int, int] | None:
        profiles = self.route_config.get("point_profiles") or {}
        profile = profiles.get(str(name).strip())
        if not isinstance(profile, dict):
            return None
        fixed = profile.get("fixed_point")
        if isinstance(fixed, dict):
            if fixed.get("enabled") is False:
                return None
            fixed = fixed.get("point")
        elif profile.get("fixed_point_enabled") is False:
            return None
        if not isinstance(fixed, (list, tuple)) or len(fixed) != 2:
            return None
        try:
            return int(fixed[0]), int(fixed[1])
        except (TypeError, ValueError):
            return None

    def _direction_to_target(
        self,
        observation: MinimapObservation | None,
        step_type: str,
        step: dict[str, Any],
    ) -> str | None:
        if observation is None or observation.player_local is None:
            return None
        if step_type == "move_to_point":
            point_name = str(step.get("point", "")).strip()
            fixed_target = self._fixed_point_for_name(point_name)
            if fixed_target is not None:
                target_x = fixed_target[0]
                if target_x == observation.player_local.x:
                    return None
                return (
                    "right"
                    if target_x > observation.player_local.x
                    else "left"
                )
            point = self._find_point(point_name)
            region = point.get("region") if point else None
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                return None
            target_x = int(region[0]) + int(region[2]) // 2
        else:
            ladder = self._find_ladder(str(step.get("ladder", "")))
            target = ladder.get("point") if ladder else None
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                return None
            target_x = int(target[0])
        if target_x == observation.player_local.x:
            return None
        return "right" if target_x > observation.player_local.x else "left"

    def _point_target_reached(
        self,
        observation: MinimapObservation | None,
        point_name: str,
    ) -> bool:
        if observation is None or observation.player_local is None:
            return False
        fixed_target = self._fixed_point_for_name(point_name)
        if fixed_target is not None:
            return (
                abs(observation.player_local.x - fixed_target[0])
                <= self._point_tolerance()
                and abs(observation.player_local.y - fixed_target[1])
                <= self._point_tolerance()
            )
        point = self._find_point(point_name)
        if point is None:
            return False
        return point_in_region(
            (
                observation.player_local.x,
                observation.player_local.y,
            ),
            tuple(point.get("region", ())),
            self._point_tolerance(),
        )

    def _next_step_is_jump(self) -> bool:
        next_index = self.step_index + 1
        if next_index >= len(self.steps):
            return False
        return str(self.steps[next_index].get("type", "")) in {
            "jump_left",
            "jump_right",
        }

    def _jump_point_tolerance(self) -> int:
        try:
            return max(
                self._ladder_tolerance(),
                int(self.route_config.get("jump_point_tolerance", 8)),
            )
        except (TypeError, ValueError):
            return max(self._ladder_tolerance(), 8)

    def _minimap_player_raised(
        self,
        observation: MinimapObservation | None,
    ) -> bool:
        return self._minimap_y_raised(
            observation,
            self.climb_ground_minimap_y,
            "climb",
        )

    def _minimap_y_raised(
        self,
        observation: MinimapObservation | None,
        baseline_y: int | None,
        action_name: str,
    ) -> bool:
        if observation is None or observation.player_local is None:
            logger.warning(
                "route %s lift check: minimap yellow point not found",
                action_name,
            )
            return False
        if baseline_y is None:
            logger.warning(
                "route %s lift check: no pre-action minimap baseline",
                action_name,
            )
            return False
        current_y = int(observation.player_local.y)
        minimum_rise = self._minimap_rise_threshold()
        lifted = current_y <= baseline_y - minimum_rise
        logger.info(
            "route %s lift check: current_minimap_y=%s "
            "ground_minimap_y=%s minimum_rise=%s lifted=%s",
            action_name,
            current_y,
            baseline_y,
            minimum_rise,
            lifted,
        )
        return lifted

    def _observe_minimap(self, frame) -> MinimapObservation | None:
        minimap = self.config.get("minimap", {})
        region = tuple(minimap.get("region") or ())
        if len(region) != 4:
            return None
        hsv = minimap.get("player_color_hsv", {})
        lower = tuple(int(value) for value in hsv.get("lower", [24, 145, 150]))
        upper = tuple(int(value) for value in hsv.get("upper", [32, 255, 255]))
        return observe_minimap(frame, region, lower, upper)

    def _detect_player(self, frame) -> PlayerScreenDetection | None:
        player_config = dict(self.config.get("combat", {}).get("player", {}))
        if not player_config:
            return None
        return detect_player_screen(frame, player_config)

    def _find_point(self, name: str) -> dict[str, Any] | None:
        return next(
            (
                point
                for point in self.points
                if str(point.get("name", "")).strip() == name
            ),
            None,
        )

    def _find_ladder(self, name: str) -> dict[str, Any] | None:
        return next(
            (
                ladder
                for ladder in self.ladders
                if str(ladder.get("name", "")).strip() == name
            ),
            None,
        )

    def _advance_step(self) -> None:
        self.step_index += 1
        self.phase = "step"
        self.phase_until = None
        self.reset_obstacle_state()
        self.climb_started_at = None
        self.climb_ground_minimap_y = None
        self.jump_started_at = None
        self.jump_ground_minimap_y = None

    def _minimap_rise_threshold(self) -> int:
        try:
            return max(
                1,
                int(self.route_config.get("climb_minimap_rise_px", 1)),
            )
        except (TypeError, ValueError):
            return 1

    def _duration_ms(self, step: dict[str, Any]) -> int:
        try:
            return max(1, int(step.get("duration_ms", 1000)))
        except (TypeError, ValueError):
            return 1000

    def _point_tolerance(self) -> int:
        try:
            return max(0, int(self.route_config.get("point_tolerance", 3)))
        except (TypeError, ValueError):
            return 3

    def _ladder_tolerance(self) -> int:
        try:
            return max(1, int(self.route_config.get("ladder_tolerance", 3)))
        except (TypeError, ValueError):
            return 3

    def _alt_delay_ms(self) -> int:
        try:
            return max(
                1,
                int(self.route_config.get("alt_delay_ms", 600)),
            )
        except (TypeError, ValueError):
            return 600

    def _result(
        self,
        action: str,
        state: str,
        reason: str,
    ) -> RouteTestResult:
        return RouteTestResult(
            action=action,
            state=state,
            reason=reason,
            minimap=self.last_minimap,
            player=self.last_player,
        )
