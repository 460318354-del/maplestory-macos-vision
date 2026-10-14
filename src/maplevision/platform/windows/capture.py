from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import cv2
import numpy as np

from ...capture.base import FrameSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowInfo:
    """Metadata for a Windows top-level window client area."""

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


def _load_win32():
    try:
        import win32gui
        import win32process
    except ImportError as exc:
        raise RuntimeError(
            "Windows support requires pywin32. "
            "Install it with: python -m pip install pywin32"
        ) from exc
    return win32gui, win32process


def _load_mss():
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError(
            "Windows capture requires mss. "
            "Install it with: python -m pip install mss"
        ) from exc
    return mss


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


def _client_bounds(win32gui: Any, hwnd: int) -> tuple[int, int, int, int] | None:
    if not win32gui.IsWindow(hwnd):
        return None
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    except Exception:
        return None
    width = int(right - left)
    height = int(bottom - top)
    if width <= 0 or height <= 0:
        return None
    return int(screen_left), int(screen_top), width, height


def _window_info(
    win32gui: Any,
    win32process: Any,
    hwnd: int,
) -> WindowInfo | None:
    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return None
    bounds = _client_bounds(win32gui, hwnd)
    if bounds is None:
        return None
    try:
        title = str(win32gui.GetWindowText(hwnd) or "")
        owner = str(win32gui.GetClassName(hwnd) or "")
        _, owner_pid = win32process.GetWindowThreadProcessId(hwnd)
    except Exception:
        return None
    return WindowInfo(
        window_id=int(hwnd),
        owner=owner,
        title=title,
        bounds=bounds,
        layer=0,
        on_screen=True,
        owner_pid=int(owner_pid) if owner_pid else None,
    )


def list_windows(title_contains: str | None = None) -> list[WindowInfo]:
    """List visible, non-minimized Windows top-level windows."""
    win32gui, win32process = _load_win32()
    needle = str(title_contains).lower() if title_contains else None
    result: list[WindowInfo] = []

    def callback(hwnd: int, _extra: Any) -> None:
        info = _window_info(win32gui, win32process, hwnd)
        if info is None:
            return
        haystack = f"{info.owner} {info.title}".lower()
        if needle and needle not in haystack:
            return
        result.append(info)

    win32gui.EnumWindows(callback, None)
    return result


class WindowsWindowSource(FrameSource):
    """Capture the client area of one bound Windows window with mss."""

    def __init__(
        self,
        title_contains: str,
        window_id: int | None = None,
        output_size: tuple[int, int] | list[int] | None = None,
    ):
        if not title_contains and window_id is None:
            raise ValueError("title_contains or window_id must be provided")
        self._title_contains = title_contains
        self._window_id = int(window_id) if window_id is not None else None
        self._output_size = _normalize_output_size(output_size)
        self._mss = _load_mss().mss()
        self._window_info = self._bind_window()

    @property
    def window_info(self) -> WindowInfo:
        return self._window_info

    def _bind_window(self) -> WindowInfo:
        if self._window_id is not None:
            candidates = [
                info
                for info in list_windows()
                if info.window_id == self._window_id
            ]
        else:
            candidates = list_windows(self._title_contains)
        if not candidates:
            query = (
                f"hwnd={self._window_id}"
                if self._window_id is not None
                else repr(self._title_contains)
            )
            raise RuntimeError(
                f"No visible Windows window matched {query}. "
                "Run with --list-windows to inspect available windows."
            )
        info = candidates[0]
        logger.info(
            "bound Windows window hwnd=%s pid=%s class=%r title=%r bounds=%s",
            info.window_id,
            info.owner_pid,
            info.owner,
            info.title,
            info.bounds,
        )
        return info

    def _refresh_window_info(self) -> None:
        win32gui, win32process = _load_win32()
        info = _window_info(win32gui, win32process, self._window_info.window_id)
        if info is None:
            raise RuntimeError(
                "Bound Windows window is no longer visible: "
                f"hwnd={self._window_info.window_id}"
            )
        self._window_info = info

    def next_frame(self) -> np.ndarray | None:
        self._refresh_window_info()
        x, y, width, height = self._window_info.bounds
        screenshot = self._mss.grab(
            {
                "left": int(x),
                "top": int(y),
                "width": int(width),
                "height": int(height),
            }
        )
        frame_bgra = np.asarray(screenshot, dtype=np.uint8)
        if frame_bgra.size == 0:
            return None
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        if self._output_size is None:
            return frame_bgr
        output_width, output_height = self._output_size
        if (frame_bgr.shape[1], frame_bgr.shape[0]) == (
            output_width,
            output_height,
        ):
            return frame_bgr
        return cv2.resize(
            frame_bgr,
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        )

    def close(self) -> None:
        close = getattr(self._mss, "close", None)
        if close is not None:
            close()
