from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

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
    owner_pid: int | None = None

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
        owner_pid=_owner_pid(quartz, window),
    )


def _owner_pid(quartz: Any, window: dict[str, Any]) -> int | None:
    key = getattr(quartz, "kCGWindowOwnerPID", None)
    if key is None:
        return None
    value = window.get(key)
    return int(value) if value is not None else None


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

    def __init__(
        self,
        title_contains: str,
        window_id: int | None = None,
        output_size: tuple[int, int] | list[int] | None = None,
    ):
        if not title_contains:
            raise ValueError("title_contains must not be empty")
        self._quartz = _load_quartz()
        self._title_contains = title_contains
        self._window_info = self._bind_window(window_id)
        self._window_id = self._window_info.window_id
        self._output_size = _normalize_output_size(output_size)

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
                "bound game window id=%s pid=%s owner=%r title=%r bounds=%s",
                info.window_id,
                info.owner_pid,
                info.owner,
                info.title,
                info.bounds,
            )
            return info
        raise RuntimeError(
            f"No visible macOS window matched: {self._title_contains!r}. "
            "Run with --list-windows to inspect available windows."
        )

    def _refresh_window_info(self) -> None:
        for info in list_windows():
            if info.window_id == self._window_id:
                self._window_info = info
                return
        raise RuntimeError(
            f"Bound macOS window is no longer visible: id={self._window_id}"
        )

    def next_frame(self) -> np.ndarray | None:
        quartz = self._quartz
        self._refresh_window_info()
        x, y, width, height = self._window_info.bounds
        capture_rect = quartz.CGRectMake(
            float(x),
            float(y),
            float(width),
            float(height),
        )
        image = quartz.CGWindowListCreateImage(
            capture_rect,
            quartz.kCGWindowListOptionIncludingWindow,
            self._window_id,
            quartz.kCGWindowImageDefault,
        )
        if image is None:
            return None

        width = quartz.CGImageGetWidth(image)
        height = quartz.CGImageGetHeight(image)
        bytes_per_row = quartz.CGImageGetBytesPerRow(image)
        provider = quartz.CGImageGetDataProvider(image)
        data = quartz.CGDataProviderCopyData(provider)
        expected_size = height * bytes_per_row
        raw = np.frombuffer(data, dtype=np.uint8)
        if raw.size < expected_size:
            raise RuntimeError(
                "Captured window image buffer is smaller than its declared "
                f"stride: {raw.size} < {expected_size}"
            )
        # Quartz rows may include alignment padding, especially on Retina
        # displays. Read using the declared stride, then remove the padding.
        frame = raw[:expected_size].reshape((height, bytes_per_row))
        frame = frame[:, : width * 4].reshape((height, width, 4))
        frame_bgr = frame[:, :, :3].copy()
        if self._output_size is None:
            return frame_bgr
        return normalize_window_frame(
            frame_bgr,
            self._window_info.bounds,
            self._output_size,
        )

    def close(self) -> None:
        return None


def _normalize_output_size(
    output_size: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    if output_size is None:
        return None
    if len(output_size) != 2:
        raise ValueError("capture output_size must contain [width, height]")
    width, height = (int(value) for value in output_size)
    if width <= 0 or height <= 0:
        raise ValueError("capture output_size must be positive")
    return width, height


def normalize_window_frame(
    frame_bgr: np.ndarray,
    window_bounds: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Remove the title bar and normalize Retina pixels to logical pixels."""
    output_width, output_height = output_size
    frame_height, frame_width = frame_bgr.shape[:2]
    if (frame_width, frame_height) == output_size:
        return frame_bgr.copy()

    _, _, window_width, window_height = window_bounds
    if window_width <= 0 or window_height <= 0:
        return cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_AREA)

    scale_x = frame_width / float(window_width)
    scale_y = frame_height / float(window_height)
    content_width = int(round(output_width * scale_x))
    content_height = int(round(output_height * scale_y))
    if (
        content_width <= 0
        or content_height <= 0
        or content_width > frame_width
        or content_height > frame_height
    ):
        return cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_AREA)

    # The game content is anchored to the bottom of the window below the
    # title bar. This keeps the returned coordinates in logical game pixels.
    crop_x = max(0, (frame_width - content_width) // 2)
    crop_y = max(0, frame_height - content_height)
    crop = frame_bgr[
        crop_y : crop_y + content_height,
        crop_x : crop_x + content_width,
    ]
    if crop.shape[1] != content_width or crop.shape[0] != content_height:
        return cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_AREA)
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
