from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MonsterTemplate:
    name: str
    image_bgr: np.ndarray
    match_image: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])


@dataclass(frozen=True)
class MonsterDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.box
        return x + width // 2, y + height // 2


@dataclass
class _MonsterTrack:
    detection: MonsterDetection
    hits: int = 1
    misses: int = 0


def load_templates(paths: list[str | Path]) -> tuple[MonsterTemplate, ...]:
    templates: list[MonsterTemplate] = []
    for path_value in paths:
        path = Path(path_value)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read monster template: {path}")
        templates.append(
            MonsterTemplate(
                name=path.stem,
                image_bgr=image,
                match_image=_match_image(image),
            )
        )
    return tuple(templates)


def detect_monsters(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
) -> tuple[MonsterDetection, ...]:
    if not config.get("enabled", True):
        return ()
    region = tuple(config.get("region") or ())
    if len(region) != 4:
        return ()
    template_paths = [str(path) for path in config.get("templates", [])]
    if not template_paths:
        return ()
    templates = load_templates(template_paths)
    return detect_monsters_with_templates(frame_bgr, region, templates, config)


def detect_monsters_with_templates(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    templates: tuple[MonsterTemplate, ...],
    config: dict[str, Any],
) -> tuple[MonsterDetection, ...]:
    x, y, width, height = region
    roi = frame_bgr[y : y + height, x : x + width]
    if roi.size == 0:
        return ()

    threshold = float(config.get("threshold", 0.62))
    max_results = int(config.get("max_results", 12))
    overlap_threshold = float(config.get("overlap_threshold", 0.65))

    roi_match = _match_image(roi)
    candidates: list[MonsterDetection] = []
    for template in templates:
        if template.width > roi.shape[1] or template.height > roi.shape[0]:
            continue
        result = cv2.matchTemplate(
            roi_match,
            template.match_image,
            cv2.TM_CCOEFF_NORMED,
        )
        peak_window = int(config.get("peak_window", 25))
        local_peaks = cv2.dilate(
            result,
            np.ones((peak_window, peak_window), dtype=np.uint8),
        )
        ys, xs = np.where((result >= threshold) & (result == local_peaks))
        for local_x, local_y in zip(xs, ys):
            score = float(result[local_y, local_x])
            candidates.append(
                MonsterDetection(
                    label=template.name,
                    score=score,
                    box=(x + int(local_x), y + int(local_y), template.width, template.height),
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[MonsterDetection] = []
    for candidate in candidates:
        if any(_intersection_over_union(candidate.box, item.box) > overlap_threshold for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_results:
            break
    return tuple(sorted(selected, key=lambda item: item.center[0]))


def _match_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


class MonsterTracker:
    """Confirm moving candidates across frames before exposing them to combat."""

    def __init__(
        self,
        confirm_frames: int = 2,
        max_misses: int = 2,
        max_distance: int = 120,
    ):
        self.confirm_frames = max(1, confirm_frames)
        self.max_misses = max(0, max_misses)
        self.max_distance = max(1, max_distance)
        self._tracks: list[_MonsterTrack] = []

    def update(
        self,
        detections: tuple[MonsterDetection, ...],
    ) -> tuple[MonsterDetection, ...]:
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        pairs: list[tuple[float, int, int]] = []

        for track_index, track in enumerate(self._tracks):
            tx, ty = track.detection.center
            for detection_index, detection in enumerate(detections):
                dx, dy = detection.center
                distance = float(np.hypot(dx - tx, dy - ty))
                if distance <= self.max_distance:
                    pairs.append((distance, track_index, detection_index))

        for _, track_index, detection_index in sorted(pairs):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks[track_index]
            track.detection = detections[detection_index]
            track.hits += 1
            track.misses = 0
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.misses += 1

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._tracks.append(_MonsterTrack(detection=detection))

        self._tracks = [
            track for track in self._tracks if track.misses <= self.max_misses
        ]
        return tuple(
            track.detection
            for track in self._tracks
            if track.hits >= self.confirm_frames
        )


def _intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0
