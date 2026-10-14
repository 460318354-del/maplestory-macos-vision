from __future__ import annotations

from typing import Any


ROUTE_STEP_TYPES = (
    "move_to_point",
    "move_to_ladder",
    "climb",
    "move_left",
    "move_right",
    "jump_left",
    "jump_right",
    "wait",
)

DEFAULT_ROUTE_GROUP = "默认"


def infer_route_group(name: str | None) -> str:
    """Infer a route group for older configs that do not store one yet."""
    value = str(name or "").strip()
    if value.startswith("猪"):
        return "猪"
    if "蘑菇" in value:
        return "8蘑菇"
    return DEFAULT_ROUTE_GROUP


def route_point_group(point: dict[str, Any]) -> str:
    group = str(point.get("group", "")).strip()
    if group:
        return group
    return infer_route_group(str(point.get("name", "")))


def route_path_group(path: dict[str, Any]) -> str:
    group = str(path.get("group", "")).strip()
    if group:
        return group
    source_group = infer_route_group(str(path.get("source", "")))
    target_group = infer_route_group(str(path.get("target", "")))
    if source_group == target_group:
        return source_group
    return source_group if source_group != DEFAULT_ROUTE_GROUP else target_group


def route_ladder_group(ladder: dict[str, Any]) -> str:
    group = str(ladder.get("group", "")).strip()
    if group:
        return group
    return infer_route_group(str(ladder.get("name", "")))


def route_group_names(route_config: dict[str, Any]) -> list[str]:
    names = {
        route_point_group(point)
        for point in (route_config.get("points") or [])
        if isinstance(point, dict)
    }
    names.update(
        route_path_group(path)
        for path in (route_config.get("paths") or [])
        if isinstance(path, dict)
    )
    active = str(route_config.get("active_group", "")).strip()
    if active:
        names.add(active)
    return sorted(names) or [DEFAULT_ROUTE_GROUP]


def active_route_group(route_config: dict[str, Any]) -> str:
    active = str(route_config.get("active_group", "")).strip()
    if active:
        return active
    return route_group_names(route_config)[0]


def normalize_rect(rect: Any) -> tuple[int, int, int, int] | None:
    """Return a non-negative x/y/width/height rectangle."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        x, y, width, height = (int(value) for value in rect)
    except (TypeError, ValueError):
        return None
    return max(0, x), max(0, y), max(0, width), max(0, height)


def minimap_local_rect(
    frame_rect: tuple[int, int, int, int],
    minimap_region: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Convert a frame rectangle to a clamped minimap-local rectangle."""
    selected = normalize_rect(frame_rect)
    minimap = normalize_rect(minimap_region)
    if selected is None or minimap is None:
        return None

    frame_x, frame_y, frame_width, frame_height = selected
    map_x, map_y, map_width, map_height = minimap
    left = max(frame_x, map_x)
    top = max(frame_y, map_y)
    right = min(frame_x + frame_width, map_x + map_width)
    bottom = min(frame_y + frame_height, map_y + map_height)
    if right <= left or bottom <= top:
        return None
    return left - map_x, top - map_y, right - left, bottom - top


def minimap_scaled_rect_to_local(
    rect: tuple[int, int, int, int],
    scale: int,
) -> tuple[int, int, int, int] | None:
    """Convert a rectangle from an enlarged minimap to local map pixels."""
    if scale <= 0 or not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        x, y, width, height = (int(value) for value in rect)
    except (TypeError, ValueError):
        return None
    return (
        max(0, int(round(x / scale))),
        max(0, int(round(y / scale))),
        max(0, int(round(width / scale))),
        max(0, int(round(height / scale))),
    )


def minimap_scaled_point_to_local(
    point: tuple[int, int],
    scale: int,
) -> tuple[int, int] | None:
    """Convert a point from an enlarged minimap to local map pixels."""
    if scale <= 0 or not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x, y = (int(value) for value in point)
    except (TypeError, ValueError):
        return None
    return int(round(x / scale)), int(round(y / scale))


def point_in_region(
    point: tuple[int, int],
    region: tuple[int, int, int, int],
    tolerance: int = 0,
) -> bool:
    x, y, width, height = region
    px, py = point
    padding = max(0, int(tolerance))
    return (
        x - padding <= px <= x + width + padding
        and y - padding <= py <= y + height + padding
    )


def current_point_name(
    player_local: tuple[int, int] | None,
    points: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    tolerance: int = 0,
) -> str | None:
    """Return the first configured combat point containing the player."""
    if player_local is None:
        return None
    for point in points:
        name = str(point.get("name", "")).strip()
        region = normalize_rect(point.get("region"))
        if name and region and point_in_region(player_local, region, tolerance):
            return name
    return None


def path_name(source: str, target: str) -> str:
    return f"{str(source).strip()} -> {str(target).strip()}"


def step_label(step: dict[str, Any]) -> str:
    step_type = str(step.get("type", "wait"))
    if step_type == "move_to_point":
        return f"移动到挂机点 {step.get('point', '')}"
    if step_type == "move_to_ladder":
        return f"移动到梯子 {step.get('ladder', '')}"
    if step_type == "climb":
        key = "上" if str(step.get("key", "up")) == "up" else "下"
        return f"向{key}爬梯 {step.get('ladder', '')} {step.get('duration_ms', 0)}ms"
    if step_type == "move_left":
        return "向左移动到目标固定点"
    if step_type == "move_right":
        return "向右移动到目标固定点"
    if step_type == "jump_left":
        return f"向左跳跃 {step.get('duration_ms', 0)}ms"
    if step_type == "jump_right":
        return f"向右跳跃 {step.get('duration_ms', 0)}ms"
    if step_type == "wait":
        return f"等待 {step.get('duration_ms', 0)}ms"
    return step_type
