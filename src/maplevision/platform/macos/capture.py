from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ...capture.base import FrameSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    owner: str
    title: str
    bounds: tuple[int, int, int, int]
    layer: int
    on_screen: bool

    @property
    def display_name(self) -> str:
        return self.title or self.owner or f"window-{self.window_id}"


def _load_quartz():
    try:
        import Quartz
    except ImportError as exc:
        raise RuntimeError(
            "Quartz support is unavailable; install "
            "pyobjc-framework-Quartz in the project virtual environment"
        ) from exc
    return Quartz


def _window_info_from_record(quartz: Any, window: dict[str, Any]) -> WindowInfo | None:
    window_id = window.get(quartz.kCGWindowNumber)
    if window_id is None:
        return None

    bounds = window.get(quartz.kCGWindowBounds) or {}
    return WindowInfo(
        window_id=int(window_id),
        owner=str(window.get(quartz.kCGWindowOwnerName, "") or ""),
        title=str(window.get(quartz.kCGWindowName, "") or ""),
        bounds=(
            int(bounds.get("X", 0)),
            int(bounds.get("Y", 0)),
            int(bounds.get("Width", 0)),
            int(bounds.get("Height", 0)),
        ),
        layer=int(window.get(quartz.kCGWindowLayer, 0) or 0),
        on_screen=bool(window.get(quartz.kCGWindowIsOnscreen, False)),
    )


def list_windows(title_contains: str | None = None) -> list[WindowInfo]:
    """Return visible application windows, optionally filtered by owner/title."""
    quartz = _load_quartz()
    options = quartz.kCGWindowListOptionOnScreenOnly
    options |= quartz.kCGWindowListExcludeDesktopElements
    windows = quartz.CGWindowListCopyWindowInfo(
        options,
        quartz.kCGNullWindowID,
    )
    needle = title_contains.lower() if title_contains else None
    result = []
    for record in windows or []:
        info = _window_info_from_record(quartz, record)
        if info is None:
            continue
        haystack = f"{info.owner} {info.title}".lower()
        if needle and needle not in haystack:
            continue
        result.append(info)
    return result


class MacOSWindowSource(FrameSource):
    """Capture one bound on-screen macOS window."""

    def __init__(self, title_contains: str, window_id: int | None = None):
        if not title_contains:
            raise ValueError("title_contains must not be empty")
        self._quartz = _load_quartz()
        self._title_contains = title_contains
        self._window_info = self._bind_window(window_id)
        self._window_id = self._window_info.window_id

    @property
    def window_info(self) -> WindowInfo:
        return self._window_info

    def _bind_window(self, window_id: int | None) -> WindowInfo:
        if window_id is not None:
            candidates = [
                info for info in list_windows()
                if info.window_id == window_id
            ]
        else:
            candidates = list_windows(self._title_contains)
        if candidates:
            info = candidates[0]
            logger.info(
                "bound game window id=%s owner=%r title=%r bounds=%s",
                info.window_id,
                info.owner,
                info.title,
                info.bounds,
            )
            return info
        raise RuntimeError(
            f"No visible macOS window matched: {self._title_contains!r}. "
            "Run with --list-windows to inspect available windows."
        )

    def next_frame(self) -> np.ndarray | None:
        import numpy as np

        quartz = self._quartz
        image = quartz.CGWindowListCreateImage(
            quartz.CGRectInfinite,
            quartz.kCGWindowListOptionIncludingWindow,
            self._window_id,
            quartz.kCGWindowImageDefault,
        )
        if image is None:
            return None

        width = quartz.CGImageGetWidth(image)
        height = quartz.CGImageGetHeight(image)
        provider = quartz.CGImageGetDataProvider(image)
        data = quartz.CGDataProviderCopyData(provider)
        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        return frame[:, :, :3].copy()

    def close(self) -> None:
        return None
