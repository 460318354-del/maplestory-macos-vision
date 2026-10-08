from __future__ import annotations

from pathlib import Path

from .base import load_bgr_image


class TestImageSource:
    def __init__(self, path: Path):
        self._frame = load_bgr_image(path)
        self._returned = False

    def next_frame(self) -> np.ndarray | None:
        if self._returned:
            return None
        self._returned = True
        return self._frame.copy()

    def close(self) -> None:
        self._frame = None
