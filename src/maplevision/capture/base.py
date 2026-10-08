from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class FrameSource(Protocol):
    def next_frame(self) -> Any:
        """Return a BGR frame or None when the source is exhausted."""

    def close(self) -> None:
        """Release resources held by the source."""


def load_bgr_image(path: Path) -> Any:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image
