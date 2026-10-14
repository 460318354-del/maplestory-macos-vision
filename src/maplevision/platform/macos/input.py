from __future__ import annotations

import logging
import random
import re
import time

logger = logging.getLogger(__name__)


KEY_CODES = {
    "ctrl": 59,
    "control": 59,
    "alt": 58,
    "option": 58,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
    "space": 49,
    "z": 6,
    "1": 18,
    "2": 19,
    "3": 20,
}


class MacOSInputController:
    """Input boundary for dry-run logs and macOS keyboard events."""

    def __init__(
        self,
        dry_run: bool = True,
        target_pid: int | None = None,
    ):
        self.dry_run = dry_run
        self.target_pid = int(target_pid) if target_pid else None
        self._held_keys: set[str] = set()
        self._attack_key: str | None = None
        self._attack_direction: str | None = None
        self._next_attack_at: float | None = None
        self._jump_action: tuple[str, str] | None = None
        self._next_z_at: float | None = None
        self._pickup_key: str | None = None
        self._pickup_remaining = 0
        self._pickup_completed = False
        self._next_pickup_at: float | None = None
        self._quartz = None

    def emit(self, action: str) -> None:
        if self.dry_run:
            logger.info("dry-run action: %s", action)
            return

        command = _parse_action(action)
        if command.kind == "hold":
            self._hold_only(command.key)
            return
        if command.kind == "jump":
            self._jump_only(command.key, command.jump_key)
            return
        if command.kind == "attack":
            self._attack_only(command.key, command.direction)
            return
        if command.kind == "pickup":
            self._pickup_only(command.key, command.count)
            return
        if command.kind == "tap":
            self.release_all()
            self._tap(command.key)
            return
        self.release_all()

    def release_all(self) -> None:
        for key in tuple(self._held_keys):
            self._key_event(key, False)
            self._held_keys.discard(key)
        self._reset_attack()
        self._reset_jump()
        self._next_z_at = None
        self._reset_pickup()

    def release_key(self, key: str) -> None:
        normalized = _normalize_key(key)
        if normalized not in self._held_keys:
            return
        self._key_event(normalized, False)
        self._held_keys.discard(normalized)
        if normalized in {"left", "right"}:
            self._reset_jump()
        if normalized in {"left", "right", "ctrl", "control"}:
            self._next_z_at = None

    def tap_key(self, key: str) -> None:
        normalized = _normalize_key(key)
        if self.dry_run:
            logger.info("dry-run tap key: %s", normalized)
            return
        self._tap(normalized)

    def _hold_only(self, key: str) -> None:
        if self._attack_key is not None:
            self._reset_attack()
        self._reset_jump()
        self._reset_pickup()
        now = time.monotonic()
        z_enabled_key = key in {"left", "right", "ctrl", "control"}
        previous_hold_key = next(iter(self._held_keys), None)
        for held_key in tuple(self._held_keys):
            if held_key != key:
                self._key_event(held_key, False)
                self._held_keys.discard(held_key)
        if key not in self._held_keys:
            self._key_event(key, True)
            self._held_keys.add(key)
        if not z_enabled_key:
            self._next_z_at = None
            return
        if previous_hold_key != key or self._next_z_at is None:
            interval = (0.2, 0.35) if key in {"ctrl", "control"} else (0.2, 0.3)
            self._next_z_at = now + random.uniform(*interval)
            return
        if now >= self._next_z_at:
            self._tap("z")
            interval = (0.2, 0.35) if key in {"ctrl", "control"} else (0.2, 0.3)
            self._next_z_at = now + random.uniform(*interval)

    def _jump_only(self, direction: str, jump_key: str) -> None:
        if self._attack_key is not None:
            self._reset_attack()
        self._reset_pickup()

        direction = _normalize_key(direction)
        jump_key = _normalize_key(jump_key or "alt")
        action = (direction, jump_key)
        if self._jump_action != action:
            for held_key in tuple(self._held_keys):
                if held_key != direction:
                    self._key_event(held_key, False)
                    self._held_keys.discard(held_key)
            # Re-arm the direction key so every jump has the explicit
            # sequence: direction down -> jump tap.
            if direction in self._held_keys:
                self._key_event(direction, False)
                self._held_keys.discard(direction)
            self._key_event(direction, True)
            self._held_keys.add(direction)
            self._jump_action = action
            self._tap(jump_key)
            return

        if direction not in self._held_keys:
            self._key_event(direction, True)
            self._held_keys.add(direction)
        self._next_z_at = None

    def _attack_only(self, key: str, direction: str | None = None) -> None:
        self._reset_jump()
        self._reset_pickup()
        now = time.monotonic()
        for held_key in tuple(self._held_keys):
            self._key_event(held_key, False)
            self._held_keys.discard(held_key)

        direction = _normalize_key(direction) if direction else None
        direction_changed = (
            direction in {"left", "right"}
            and direction != self._attack_direction
        )
        if direction_changed:
            self._tap(direction)
            self._next_attack_at = now
        self._attack_direction = direction

        if self._attack_key != key:
            self._attack_key = key
            self._next_attack_at = now
            self._next_z_at = now + random.uniform(0.2, 0.35)

        if self._next_attack_at is not None and now >= self._next_attack_at:
            self._tap(key)
            self._next_attack_at = now + random.uniform(0.3, 0.45)

        if self._next_z_at is not None and now >= self._next_z_at:
            self._tap("z")
            self._next_z_at = now + random.uniform(0.2, 0.35)

    def _pickup_only(self, key: str, count: int) -> None:
        self._reset_jump()
        for held_key in tuple(self._held_keys):
            self._key_event(held_key, False)
            self._held_keys.discard(held_key)
        self._next_z_at = None

        now = time.monotonic()
        if self._pickup_key != key:
            self._pickup_key = key
            self._pickup_remaining = max(1, int(count))
            self._pickup_completed = False
            self._next_pickup_at = now
        if self._pickup_completed or self._next_pickup_at is None:
            return
        if now < self._next_pickup_at:
            return

        self._tap(key)
        self._pickup_remaining -= 1
        if self._pickup_remaining <= 0:
            self._pickup_remaining = 0
            self._pickup_completed = True
            self._next_pickup_at = None
            return
        self._next_pickup_at = now + random.uniform(0.2, 0.35)

    def _reset_pickup(self) -> None:
        self._pickup_key = None
        self._pickup_remaining = 0
        self._pickup_completed = False
        self._next_pickup_at = None

    def _reset_attack(self) -> None:
        self._attack_key = None
        self._attack_direction = None
        self._next_attack_at = None

    def _reset_jump(self) -> None:
        self._jump_action = None

    def _tap(self, key: str) -> None:
        self._key_event(key, True)
        time.sleep(0.03)
        self._key_event(key, False)

    def _key_event(self, key: str, down: bool) -> None:
        key_code = KEY_CODES.get(key)
        if key_code is None:
            logger.warning("unknown key for action: %s", key)
            return
        quartz = self._load_quartz()
        event = quartz.CGEventCreateKeyboardEvent(None, key_code, down)
        if self.target_pid is not None:
            post_to_pid = getattr(quartz, "CGEventPostToPid", None)
            if post_to_pid is not None:
                post_to_pid(self.target_pid, event)
            else:
                logger.warning(
                    "CGEventPostToPid unavailable; falling back to foreground event"
                )
                quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        else:
            quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        logger.info("keyboard %s: %s", "down" if down else "up", key)

    def _load_quartz(self):
        if self._quartz is None:
            try:
                import Quartz
            except ImportError as exc:  # pragma: no cover - platform guard
                raise RuntimeError(
                    "Quartz support is unavailable; install pyobjc-framework-Quartz"
                ) from exc
            self._quartz = Quartz
        return self._quartz


class _Action:
    def __init__(
        self,
        kind: str,
        key: str | None = None,
        count: int = 1,
        jump_key: str | None = None,
        direction: str | None = None,
    ):
        self.kind = kind
        self.key = key or ""
        self.count = max(1, int(count))
        self.jump_key = jump_key or "alt"
        self.direction = direction


def _parse_action(action: str) -> _Action:
    text = str(action or "").strip()
    lowered = text.lower()
    if lowered.startswith("pickup"):
        key_match = re.search(r"\(([^)]+)\)", lowered)
        values = [
            value.strip()
            for value in (key_match.group(1).split(",") if key_match else [])
            if value.strip()
        ]
        key = _normalize_key(values[0] if values else "z")
        try:
            count = int(values[1]) if len(values) > 1 else 3
        except ValueError:
            count = 3
        return _Action("pickup", key, count)
    key_match = re.search(r"\(([^)]+)\)", lowered)
    values = [
        value.strip()
        for value in (key_match.group(1).split(",") if key_match else [])
        if value.strip()
    ]
    key = values[0] if values else lowered

    if lowered.startswith("jump_left"):
        return _Action(
            "jump",
            _normalize_key(key or "left"),
            jump_key=_normalize_key(values[1] if len(values) > 1 else "alt"),
        )
    if lowered.startswith("jump_right"):
        return _Action(
            "jump",
            _normalize_key(key or "right"),
            jump_key=_normalize_key(values[1] if len(values) > 1 else "alt"),
        )

    if lowered.startswith("hold_attack"):
        return _Action(
            "hold",
            _normalize_key(key or "ctrl"),
            direction=_normalize_key(values[1]) if len(values) > 1 else None,
        )
    if lowered.startswith("tap_attack"):
        return _Action(
            "attack",
            _normalize_key(key or "ctrl"),
            direction=_normalize_key(values[1]) if len(values) > 1 else None,
        )
    if lowered.startswith("move_left"):
        return _Action("hold", _normalize_key(key or "left"))
    if lowered.startswith("move_right"):
        return _Action("hold", _normalize_key(key or "right"))
    if lowered.startswith("hold_up"):
        return _Action("hold", "up")
    if lowered.startswith("hold_down"):
        return _Action("hold", "down")
    if lowered.startswith("tap_alt"):
        return _Action("tap", "alt")
    if lowered.startswith("heal"):
        return _Action("tap", _normalize_key(key or "1"))
    if lowered.startswith("mana"):
        return _Action("tap", _normalize_key(key or "2"))

    if lowered in {
        "ctrl",
        "control",
        "alt",
        "option",
        "up",
        "down",
        "left",
        "right",
        "space",
    }:
        return _Action("hold", _normalize_key(lowered))
    if lowered in {"1", "2"}:
        return _Action("tap", lowered)
    return _Action("release")


def _normalize_key(key: str) -> str:
    lowered = key.strip().lower()
    return {
        "arrowleft": "left",
        "left_arrow": "left",
        "arrowright": "right",
        "right_arrow": "right",
        "option": "alt",
    }.get(lowered, lowered)
