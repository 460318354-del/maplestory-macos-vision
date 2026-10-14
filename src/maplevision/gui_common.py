from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def application_root() -> Path:
    """Return the writable app directory or the source project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_app_path(value: str | Path, root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def percentage_threshold(value: Any, label: str) -> float:
    try:
        percent = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是 0 到 100 的百分比") from exc
    if not 0 <= percent <= 100:
        raise ValueError(f"{label}必须在 0 到 100 之间")
    return percent / 100.0


def positive_int(value: Any, label: str) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是正整数") from exc
    if number <= 0:
        raise ValueError(f"{label}必须大于 0")
    return number


def ensure_editor_data_layout(
    config: dict[str, Any],
    project_root: Path,
) -> None:
    """Create template and work folders used by both desktop editors."""
    template_root = project_root / "assets" / "templates" / "monsters"
    categories = {
        "yellow_mushroom",
        "red_snail",
        "small_mushroom",
        "角色",
        "血条",
        "金币",
    }
    groups = (
        config.get("combat", {})
        .get("monsters", {})
        .get("template_groups", {})
    )
    categories.update(str(label) for label in groups)
    for category in categories:
        (template_root / safe_template_category(category)).mkdir(
            parents=True,
            exist_ok=True,
        )
    (project_root / "assets" / "templates" / "player").mkdir(
        parents=True,
        exist_ok=True,
    )
    (project_root / "work" / "templates").mkdir(parents=True, exist_ok=True)
    (project_root / "work" / "captures").mkdir(parents=True, exist_ok=True)


def crop_region(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, width, height = clip_region_to_frame(frame_bgr, region)
    x1 = x0 + width
    y1 = y0 + height
    return frame_bgr[y0:y1, x0:x1].copy()


def clip_region_to_frame(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return a region clipped to the actual captured frame bounds."""
    x, y, width, height = (int(value) for value in region)
    frame_height, frame_width = frame_bgr.shape[:2]
    x0 = max(0, min(frame_width, x))
    y0 = max(0, min(frame_height, y))
    x1 = max(x0, min(frame_width, x + width))
    y1 = max(y0, min(frame_height, y + height))
    return x0, y0, x1 - x0, y1 - y0


def player_search_region(
    frame_bgr: np.ndarray,
    marked_region: tuple[int, int, int, int] | None,
) -> list[int]:
    if marked_region is not None:
        return list(marked_region)
    frame_height, frame_width = frame_bgr.shape[:2]
    return [0, 0, frame_width, frame_height]


def minimap_local_point(
    point: tuple[int, int],
    region: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if len(region) != 4:
        return None
    x, y, width, height = (int(value) for value in region)
    px, py = (int(value) for value in point)
    if not (x <= px <= x + width and y <= py <= y + height):
        return None
    return px - x, py - y


def minimap_frame_point(
    point: tuple[int, int] | None,
    region: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if point is None:
        return None
    if len(region) != 4:
        return int(point[0]), int(point[1])
    return int(region[0]) + int(point[0]), int(region[1]) + int(point[1])


def safe_template_category(value: str) -> str:
    cleaned = re.sub(r"[\\/:\0]+", "_", str(value).strip()).strip(" ._")
    return cleaned or "custom"


def runtime_event_messages(
    combat: Any,
    controller: Any,
    state: dict[str, Any],
) -> list[str]:
    """Return only meaningful runtime state transitions for the UI log."""
    messages: list[str] = []
    route_name = str(getattr(controller, "route_path_name", "") or "").strip()
    previous_route = str(state.get("route_name", "") or "").strip()
    if route_name != previous_route:
        if route_name:
            messages.append(f"寻路 {route_name} 启动")
        elif previous_route:
            messages.append("寻路完成")
        state["route_name"] = route_name

    monster_present = bool(
        getattr(combat, "confirmed_targets", ())
        or getattr(combat, "targets", ())
    )
    if monster_present and not state.get("monster_present", False):
        messages.append("发现怪物")
    elif not monster_present and state.get("monster_present", False):
        if route_name:
            messages.append("怪物消失，恢复寻路")
        else:
            messages.append("怪物消失")
    state["monster_present"] = monster_present

    coin_present = bool(getattr(controller, "last_coin_detections", ()))
    if coin_present and not state.get("coin_present", False):
        messages.append("发现金币")
    elif not coin_present and state.get("coin_present", False):
        messages.append("金币区域已清空")
    state["coin_present"] = coin_present

    reason = str(getattr(combat, "reason", "") or "")
    action = str(getattr(combat, "action", "") or "")
    obstacle_active = (
        "obstacle-" in reason
        and ("jump" in reason or action.startswith("jump_"))
    )
    if obstacle_active and not state.get("obstacle_active", False):
        messages.append("有障碍，启动跳跃")
    state["obstacle_active"] = obstacle_active

    fixed_point_returning = reason.startswith("fixed-point-return:")
    if fixed_point_returning and not state.get("fixed_point_returning", False):
        messages.append("偏离固定点，开始回标点")
    elif not fixed_point_returning and state.get("fixed_point_returning", False):
        messages.append("已回到固定点")
    state["fixed_point_returning"] = fixed_point_returning

    attack_action = str(
        getattr(controller, "config", {})
        .get("combat", {})
        .get("attack_action", "tap_attack(ctrl)")
    )
    attacking = action == attack_action
    if attacking and not state.get("attacking", False):
        messages.append("开始攻击")
    state["attacking"] = attacking

    if "route-failed:" in reason:
        route_failure = reason
        if route_failure != state.get("route_failure"):
            messages.append(f"寻路失败：{route_failure.split(':', 1)[1]}")
            state["route_failure"] = route_failure
    else:
        state["route_failure"] = None

    return messages
