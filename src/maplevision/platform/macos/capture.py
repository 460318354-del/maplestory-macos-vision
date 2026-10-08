from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...capture.base import FrameSource

logger = logging.getLogger(__name__)


class MacOSWindowSource(FrameSource):
    """Capture the first on-screen macOS window matching a title fragment."""

    def __init__(self, title_contains: str):
        if not title_contains:
            raise ValueError("title_contains must not be empty")
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError(
                "Quartz support is unavailable; install the macOS dependencies"
            ) from exc

        self._quartz = Quartz
        self._title_contains = title_contains
        self._window_id = self._find_window_id()

    def _find_window_id(self) -> int:
        quartz = self._quartz
        options = quartz.kCGWindowListOptionOnScreenOnly
        options |= quartz.kCGWindowListExcludeDesktopElements
        windows = quartz.CGWindowListCopyWindowInfo(options, quartz.kCGNullWindowID)
        for window in windows or []:
            title = window.get(quartz.kCGWindowName, "") or ""
            owner = window.get(quartz.kCGWindowOwnerName, "") or ""
            if self._title_contains.lower() in f"{owner} {title}".lower():
                window_id = window.get(quartz.kCGWindowNumber)
                if window_id is not None:
                    logger.info("capturing window %s (%s)", window_id, title or owner)
                    return int(window_id)
        raise RuntimeError(
            f"No visible macOS window matched: {self._title_contains!r}"
        )

    def next_frame(self) -> np.ndarray | None:
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
