from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .monster import MonsterDetection, MonsterDetector, MonsterTracker, detect_monsters
from .player_screen import (
    PlayerScreenDetection,
    PlayerScreenDetector,
    detect_player_screen,
)


@dataclass(frozen=True)
class HealthObservation:
    region: tuple[int, int, int, int] | None
    ratio: float | None
    healthy: bool | None


@dataclass(frozen=True)
class CombatObservation:
    health: HealthObservation
    mana: HealthObservation
    player: PlayerScreenDetection | None
    targets: tuple[MonsterDetection, ...]
    confirmed_targets: tuple[MonsterDetection, ...]
    in_range_targets: tuple[MonsterDetection, ...]
    target: MonsterDetection | None
    nearest_target: MonsterDetection | None
    action: str
    reason: str
    target_direction: str | None = None


def _hsv_ranges(config: Any) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    if isinstance(config, dict) and "ranges" in config:
        return [
            (tuple(item["lower"]), tuple(item["upper"]))
            for item in config.get("ranges", [])
        ]
    if isinstance(config, dict) and "lower" in config and "upper" in config:
        return [(tuple(config["lower"]), tuple(config["upper"]))]
    return []


def _color_mask(
    frame_bgr: np.ndarray,
    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def observe_health(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
) -> HealthObservation:
    if not config.get("enabled", True):
        return HealthObservation(region=None, ratio=None, healthy=None)

    hsv_ranges = _hsv_ranges(config.get("color_hsv", {}))
    if not hsv_ranges:
        return HealthObservation(region=None, ratio=None, healthy=None)

    threshold = float(config.get("low_threshold", 0.35))
    expected_width = int(config.get("expected_width", 210))
    region = tuple(config.get("region") or ())
    if len(region) == 4:
        ratio = _health_ratio_in_region(frame_bgr, region, hsv_ranges, expected_width)
        return HealthObservation(
            region=region,
            ratio=ratio,
            healthy=ratio > threshold,
        )

    auto_region, fill_width = _find_health_component(frame_bgr, hsv_ranges)
    if auto_region is None:
        return HealthObservation(region=None, ratio=None, healthy=None)

    ratio = min(1.0, fill_width / max(1, expected_width))
    x, y, _, height = auto_region
    bar_region = (x, y, expected_width, height)
    return HealthObservation(
        region=bar_region,
        ratio=ratio,
        healthy=ratio > threshold,
    )


def observe_mana(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
) -> HealthObservation:
    return observe_health(frame_bgr, config)


def _health_ratio_in_region(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    hsv_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    expected_width: int,
) -> float:
    x, y, width, height = region
    roi = frame_bgr[y : y + height, x : x + width]
    if roi.size == 0:
        return 0.0
    mask = _color_mask(roi, hsv_ranges)
    min_pixels = max(2, int(height * 0.25))
    columns = np.where(np.count_nonzero(mask, axis=0) >= min_pixels)[0]
    if columns.size == 0:
        return 0.0
    fill_width = int(columns.max()) + 1
    denominator = expected_width if expected_width > 0 else width
    return min(1.0, fill_width / max(1, denominator))


def _find_health_component(
    frame_bgr: np.ndarray,
    hsv_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> tuple[tuple[int, int, int, int] | None, int]:
    height, width = frame_bgr.shape[:2]
    start_y = int(height * 0.78)
    search = frame_bgr[start_y:, : int(width * 0.55)]
    mask = _color_mask(search, hsv_ranges)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, comp_width, comp_height, area = stats[label]
        if area < 400:
            continue
        if comp_width < 60 or comp_width > 260:
            continue
        if comp_height < 8 or comp_height > 50:
            continue
        candidates.append(
            (
                int(y + comp_height),
                int(area),
                int(x),
                int(y + start_y),
                int(comp_width),
                int(comp_height),
            )
        )
    if not candidates:
        return None, 0
    _, _, x, y, comp_width, comp_height = max(candidates)
    return (x, y, comp_width, comp_height), comp_width


def observe_combat(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    monster_tracker: MonsterTracker | None = None,
    monster_detector: MonsterDetector | None = None,
    player_detector: PlayerScreenDetector | None = None,
) -> CombatObservation:
    health = observe_health(frame_bgr, config.get("health", {}))
    mana = observe_mana(frame_bgr, config.get("mana", {}))
    heal_action = str(config.get("heal_action", "heal"))
    mana_action = str(config.get("mana_action", "mana"))
    attack_action = str(config.get("attack_action", "attack"))
    observe_action = str(config.get("observe_action", "observe"))

    if health.healthy is False:
        return CombatObservation(
            health=health,
            mana=mana,
            player=None,
            targets=(),
            confirmed_targets=(),
            in_range_targets=(),
            target=None,
            nearest_target=None,
            action=heal_action,
            reason="hp-low",
        )
    if mana.healthy is False:
        return CombatObservation(
            health=health,
            mana=mana,
            player=None,
            targets=(),
            confirmed_targets=(),
            in_range_targets=(),
            target=None,
            nearest_target=None,
            action=mana_action,
            reason="mp-low",
        )

    monster_config = config.get("monsters", {})
    if monster_detector is None:
        detected_targets = detect_monsters(frame_bgr, monster_config)
    else:
        detected_targets = detect_monsters(
            frame_bgr,
            monster_config,
            monster_detector,
        )
    allowed_labels = {
        str(label)
        for label in monster_config.get("allowed_labels", [])
        if str(label)
    }
    targets = (
        tuple(
            target
            for target in detected_targets
            if target.label in allowed_labels
        )
        if allowed_labels
        else detected_targets
    )
    confirmed_targets = (
        monster_tracker.update(targets)
        if monster_tracker is not None
        else targets
    )
    if not confirmed_targets:
        return CombatObservation(
            health=health,
            mana=mana,
            player=None,
            targets=targets,
            confirmed_targets=(),
            in_range_targets=(),
            target=None,
            nearest_target=None,
            action=observe_action,
            reason="no-target",
        )

    player_config = config.get("player", {})
    player = (
        player_detector.detect(frame_bgr, player_config)
        if player_detector is not None
        else detect_player_screen(frame_bgr, player_config)
    )
    attack_range = config.get("attack_range", {})
    range_x = max(
        0,
        int(attack_range.get("distance", attack_range.get("x", 0))),
    )
    range_y = max(0, int(attack_range.get("y", 0)))
    in_range_targets = ()
    if player is not None:
        px, py = player.anchor
        in_range_targets = tuple(
            target
            for target in confirmed_targets
            if abs(target.center[0] - px) <= range_x
            and abs(target.center[1] - py) <= range_y
        )
    nearest_target = None
    target_direction = None
    if player is not None:
        nearest_target = min(
            confirmed_targets,
            key=lambda item: abs(item.center[0] - player.anchor[0]),
            default=None,
        )
        if nearest_target is not None:
            target_direction = _horizontal_direction(
                nearest_target.center[0],
                player.anchor[0],
            )
    target = None
    if nearest_target is not None and nearest_target in in_range_targets:
        target = nearest_target

    if target is not None:
        return CombatObservation(
            health=health,
            mana=mana,
            player=player,
            targets=targets,
            confirmed_targets=confirmed_targets,
            in_range_targets=in_range_targets,
            target=target,
            nearest_target=nearest_target,
            action=attack_action,
            reason="confirmed-monster-in-attack-range",
            target_direction=target_direction,
        )

    if player is None:
        reason = "player-not-found"
    elif confirmed_targets:
        reason = "nearest-monster-out-of-attack-range"
    else:
        reason = "no-target"
    move_action = observe_action
    if player is not None and nearest_target is not None:
        move_action = (
            str(config.get("move_left_action", "move_left"))
            if nearest_target.center[0] < player.anchor[0]
            else str(config.get("move_right_action", "move_right"))
        )
    return CombatObservation(
        health=health,
        mana=mana,
        player=player,
        targets=targets,
        confirmed_targets=confirmed_targets,
        in_range_targets=in_range_targets,
        target=None,
        nearest_target=nearest_target,
        action=move_action,
        reason=reason,
        target_direction=target_direction,
    )


def _horizontal_direction(target_x: int, player_x: int) -> str | None:
    if target_x < player_x:
        return "left"
    if target_x > player_x:
        return "right"
    return None


def render_combat_debug_frame(
    frame_bgr: np.ndarray,
    combat: CombatObservation,
) -> np.ndarray:
    output = frame_bgr.copy()
    if combat.player is not None:
        x, y, width, height = combat.player.name_box
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.drawMarker(
            output,
            combat.player.anchor,
            (255, 255, 0),
            cv2.MARKER_CROSS,
            20,
            2,
        )
        cv2.putText(
            output,
            f"{combat.player.name} {combat.player.score:.2f}",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    for target in combat.targets:
        color = (0, 0, 255)
        thickness = 1
        if target in combat.confirmed_targets:
            color = (0, 200, 255)
            thickness = 2
        if target in combat.in_range_targets:
            color = (0, 255, 0)
            thickness = 2
        x, y, width, height = target.box
        cv2.rectangle(output, (x, y), (x + width, y + height), color, thickness)
        cv2.putText(
            output,
            f"{target.label} {target.score:.2f}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if combat.health.region is not None:
        x, y, width, height = combat.health.region
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 2)
        if combat.health.ratio is not None:
            cv2.putText(
                output,
                f"hp={combat.health.ratio:.0%}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    if combat.mana.region is not None:
        x, y, width, height = combat.mana.region
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 0, 0), 2)
        if combat.mana.ratio is not None:
            cv2.putText(
                output,
                f"mp={combat.mana.ratio:.0%}",
                (x, min(output.shape[0] - 8, y + height + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        output,
        f"combat action={combat.action} reason={combat.reason}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if combat.nearest_target is not None:
        x, y, width, height = combat.nearest_target.box
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 0, 255),
            3,
        )
    return output


def save_combat_debug_frame(
    frame_bgr: np.ndarray,
    combat: CombatObservation,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = render_combat_debug_frame(frame_bgr, combat)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Unable to write debug image: {path}")
