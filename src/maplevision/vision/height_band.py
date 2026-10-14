from __future__ import annotations

def screen_region_from_band(
    band_y: tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Convert a directly marked frame y-band into a full-width region."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    screen_min, screen_max = sorted(
        max(0, min(frame_height, int(value))) for value in band_y
    )
    return (0, screen_min, int(frame_width), screen_max - screen_min)
