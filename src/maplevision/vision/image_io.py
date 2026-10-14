from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read images through Python I/O so Unicode Windows paths are supported."""
    try:
        data = np.fromfile(Path(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(
    path: str | Path,
    image: np.ndarray,
    extension: str | None = None,
) -> bool:
    """Write images through Python I/O so Unicode Windows paths are supported."""
    target = Path(path)
    suffix = extension or target.suffix or ".png"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True
