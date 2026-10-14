from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .image_io import read_image


@dataclass(frozen=True)
class MonsterTemplate:
    name: str
    image_bgr: np.ndarray
    match_image: np.ndarray
    label: str | None = None
    path: str | None = None

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])


@dataclass(frozen=True)
class PreparedMonsterTemplate:
    source: MonsterTemplate
    match_image: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class MonsterDetectionStats:
    elapsed_ms: float
    template_count: int
    match_calls: int
    roi_shape: tuple[int, int]


@dataclass(frozen=True)
class MonsterDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]
    template_path: str | None = None

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.box
        return x + width // 2, y + height // 2


@dataclass
class _MonsterTrack:
    detection: MonsterDetection
    hits: int = 1
    misses: int = 0


class MonsterDetector:
    """Load and prepare monster templates once for a runtime session."""

    def __init__(self, config: dict[str, Any]):
        self._config = config
        groups = config.get("template_groups") or {}
        if groups:
            templates = load_template_groups(groups)
        else:
            template_paths = [str(path) for path in config.get("templates", [])]
            templates = load_templates(template_paths) if template_paths else ()

        self.templates = tuple(templates)
        self.prepared_templates = tuple(
            prepared
            for template in self.templates
            for prepared in (_prepare_template(template),)
        )
        self.last_stats = MonsterDetectionStats(
            elapsed_ms=0.0,
            template_count=len(self.prepared_templates),
            match_calls=0,
            roi_shape=(0, 0),
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        region: tuple[int, int, int, int],
        config: dict[str, Any] | None = None,
    ) -> tuple[MonsterDetection, ...]:
        active_config = config or self._config
        started = time.perf_counter()
        allowed_labels = {
            str(label)
            for label in active_config.get("allowed_labels", [])
            if str(label)
        }
        templates = (
            tuple(
                template
                for template in self.prepared_templates
                if (template.source.label or template.source.name) in allowed_labels
            )
            if allowed_labels
            else self.prepared_templates
        )
        detections, match_calls, roi_shape = _detect_prepared_templates(
            frame_bgr,
            region,
            templates,
            active_config,
        )
        self.last_stats = MonsterDetectionStats(
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            template_count=len(templates),
            match_calls=match_calls,
            roi_shape=roi_shape,
        )
        return detections


def load_templates(paths: list[str | Path]) -> tuple[MonsterTemplate, ...]:
    templates: list[MonsterTemplate] = []
    for path in _template_files(paths):
        image = read_image(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read monster template: {path}")
        templates.append(
            MonsterTemplate(
                name=path.stem,
                image_bgr=image,
                match_image=_match_image(image),
                path=str(path),
            )
        )
    return tuple(templates)


def load_template_groups(
    groups: dict[str, list[str | Path] | str | Path],
) -> tuple[MonsterTemplate, ...]:
    templates: list[MonsterTemplate] = []
    for label, group_paths in groups.items():
        paths = (
            [group_paths]
            if isinstance(group_paths, (str, Path))
            else list(group_paths)
        )
        for path in _template_files(paths):
            image = read_image(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read monster template: {path}")
            templates.append(
                MonsterTemplate(
                    name=path.stem,
                    image_bgr=image,
                    match_image=_match_image(image),
                    label=str(label),
                    path=str(path),
                )
            )
    return tuple(templates)


def _template_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for path_value in paths:
        path = Path(path_value)
        if path.is_dir():
            files.extend(sorted(path.glob("*.png")))
        elif path.is_file():
            files.append(path)
    return files


def detect_monsters(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    detector: MonsterDetector | None = None,
) -> tuple[MonsterDetection, ...]:
    if not config.get("enabled", True):
        return ()
    region = tuple(config.get("region") or ())
    if len(region) != 4:
        return ()
    if detector is None:
        detector = MonsterDetector(config)
    return detector.detect(frame_bgr, region, config)


def detect_monsters_with_templates(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    templates: tuple[MonsterTemplate, ...],
    config: dict[str, Any],
) -> tuple[MonsterDetection, ...]:
    prepared_templates = tuple(
        _prepare_template(template)
        for template in templates
    )
    detections, _, _ = _detect_prepared_templates(
        frame_bgr,
        region,
        prepared_templates,
        config,
    )
    return detections


def _detect_prepared_templates(
    frame_bgr: np.ndarray,
    region: tuple[int, int, int, int],
    templates: tuple[PreparedMonsterTemplate, ...],
    config: dict[str, Any],
) -> tuple[tuple[MonsterDetection, ...], int, tuple[int, int]]:
    x, y, width, height = region
    frame_height, frame_width = frame_bgr.shape[:2]
    left = max(0, min(frame_width, x))
    top = max(0, min(frame_height, y))
    right = max(left, min(frame_width, x + width))
    bottom = max(top, min(frame_height, y + height))
    roi = frame_bgr[top:bottom, left:right]
    if roi.size == 0:
        return (), 0, (0, 0)

    threshold = float(config.get("threshold", 0.62))
    max_results = int(config.get("max_results", 12))
    overlap_threshold = float(config.get("overlap_threshold", 0.65))

    roi_match = _match_image(roi)
    candidates: list[MonsterDetection] = []
    match_calls = 0
    peak_window = max(3, int(config.get("peak_window", 25)))
    peak_kernel = np.ones((peak_window, peak_window), dtype=np.uint8)
    for template in templates:
        if (
            template.width > roi.shape[1]
            or template.height > roi.shape[0]
            or template.width < 4
            or template.height < 4
        ):
            continue
        result = cv2.matchTemplate(
            roi_match,
            template.match_image,
            cv2.TM_CCOEFF_NORMED,
        )
        match_calls += 1
        local_peaks = cv2.dilate(result, peak_kernel)
        ys, xs = np.where((result >= threshold) & (result == local_peaks))
        for local_x, local_y in zip(xs, ys):
            score = float(result[local_y, local_x])
            candidates.append(
                MonsterDetection(
                    label=template.source.label or template.source.name,
                    score=score,
                    box=(
                        left + int(local_x),
                        top + int(local_y),
                        template.width,
                        template.height,
                    ),
                    template_path=template.source.path,
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
    return (
        tuple(sorted(selected, key=lambda item: item.center[0])),
        match_calls,
        (int(roi.shape[1]), int(roi.shape[0])),
    )


def _prepare_template(template: MonsterTemplate) -> PreparedMonsterTemplate:
    return PreparedMonsterTemplate(
        source=template,
        match_image=template.match_image,
        width=template.width,
        height=template.height,
    )


def render_monster_debug_frame(
    frame_bgr: np.ndarray,
    detections: tuple[MonsterDetection, ...],
) -> np.ndarray:
    output = frame_bgr.copy()
    colors = {
        "yellow_mushroom": (0, 200, 255),
        "red_snail": (0, 0, 255),
        "small_mushroom": (255, 200, 0),
    }
    for detection in detections:
        x, y, width, height = detection.box
        color = colors.get(detection.label, (0, 255, 0))
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            3,
        )
        cv2.putText(
            output,
            f"{detection.label} {detection.score:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


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
            if track.hits >= self.confirm_frames and track.misses == 0
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
