from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PlayerScreenDetection:
    name: str
    score: float
    name_box: tuple[int, int, int, int]
    anchor: tuple[int, int]


@dataclass(frozen=True)
class PreparedPlayerTemplate:
    path: Path
    image_bgr: np.ndarray
    scaled_templates: tuple[tuple[float, np.ndarray, np.ndarray], ...]


class PlayerScreenDetector:
    """Cache the player name template and its processed image per runtime."""

    def __init__(self, config: dict[str, Any]):
        self._template_paths = tuple(_player_template_files(config))
        if not self._template_paths:
            raise FileNotFoundError(
                "Unable to read player template: no templates configured"
            )
        self._prepared_templates = tuple(
            _prepare_player_template(path, config)
            for path in self._template_paths
        )
        self.last_elapsed_ms = 0.0

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: dict[str, Any],
    ) -> PlayerScreenDetection | None:
        started = time.perf_counter()
        try:
            if not config.get("enabled", True):
                return None
            template_paths = tuple(_player_template_files(config))
            if template_paths != self._template_paths:
                return detect_player_screen(frame_bgr, config)
            return _detect_player_with_templates(
                frame_bgr,
                config,
                self._prepared_templates,
            )
        finally:
            self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0


def player_template_candidates(
    player_id: str,
    config: dict[str, Any],
) -> list[Path]:
    player_id = str(player_id).strip()
    if not player_id:
        return []

    configured_id = str(config.get("id", config.get("name", ""))).strip()
    configured_template = Path(str(config.get("template", "")))
    candidates: list[Path] = []

    requested_path = Path(player_id).expanduser()
    if requested_path.is_file():
        candidates.append(requested_path)

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", player_id).strip("._")
    if safe_id:
        candidates.extend(
            [
                Path("assets/templates/player") / f"name_{safe_id.lower()}.png",
                Path("assets/templates/player") / f"{safe_id.lower()}.png",
            ]
        )
    if player_id.casefold() == configured_id.casefold() and str(configured_template):
        candidates.append(configured_template)
    return candidates


def resolve_player_config(
    player_id: str,
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    player_id = str(player_id).strip()
    if not player_id:
        return None

    project_root = project_root or Path.cwd()
    resolved_template = None
    for candidate in player_template_candidates(player_id, config):
        if candidate.is_absolute():
            if candidate.is_file():
                resolved_template = candidate
                break
            continue
        project_candidate = project_root / candidate
        if project_candidate.is_file():
            resolved_template = project_candidate
            break
        if candidate.is_file():
            resolved_template = candidate
            break

    if resolved_template is None:
        return None

    resolved = dict(config)
    resolved["id"] = player_id
    resolved["name"] = player_id
    resolved["template"] = str(resolved_template.resolve())
    return resolved


def detect_player_screen(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
) -> PlayerScreenDetection | None:
    if not config.get("enabled", True):
        return None

    prepared_templates = tuple(
        _prepare_player_template(path, config)
        for path in _player_template_files(config)
    )
    if not prepared_templates:
        return None
    return _detect_player_with_templates(
        frame_bgr,
        config,
        prepared_templates,
    )


def _detect_player_with_templates(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    templates: tuple[PreparedPlayerTemplate, ...],
) -> PlayerScreenDetection | None:
    x, y, width, height = tuple(config.get("search_region") or ())
    frame_height, frame_width = frame_bgr.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_width, int(x + width))
    y1 = min(frame_height, int(y + height))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    roi_match = _match_image(roi)
    best: tuple[float, tuple[int, int], float, np.ndarray] | None = None
    for template in templates:
        for scale, candidate, candidate_match in template.scaled_templates:
            if (
                candidate.shape[0] > roi.shape[0]
                or candidate.shape[1] > roi.shape[1]
                or candidate.shape[0] < 4
                or candidate.shape[1] < 4
            ):
                continue
            result = cv2.matchTemplate(
                roi_match,
                candidate_match,
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(result)
            if best is None or score > best[0]:
                best = (
                    float(score),
                    (int(location[0]), int(location[1])),
                    scale,
                    candidate,
                )

    if best is None:
        return None

    score, location, scale, matched_template = best
    if score < float(config.get("threshold", 0.45)):
        return None

    match_x = x0 + int(location[0])
    match_y = y0 + int(location[1])
    template_height, template_width = matched_template.shape[:2]
    if "anchor_offset" in config:
        offset_x, offset_y = tuple(config["anchor_offset"])
    else:
        below_nameplate = int(config.get("anchor_below_nameplate", 0))
        offset_x, offset_y = (
            template_width // 2,
            template_height + below_nameplate,
        )
    if scale != 1.0:
        offset_x = int(round(float(offset_x) * scale))
        offset_y = int(round(float(offset_y) * scale))
    return PlayerScreenDetection(
        name=str(config.get("name", "player")),
        score=float(score),
        name_box=(match_x, match_y, int(template_width), int(template_height)),
        anchor=(match_x + int(offset_x), match_y + int(offset_y)),
    )


def _prepare_player_template(
    path: Path,
    config: dict[str, Any],
) -> PreparedPlayerTemplate:
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Unable to read player template: {path}")
    return PreparedPlayerTemplate(
        path=path,
        image_bgr=template,
        scaled_templates=_scaled_template_matches(template, config),
    )


def _player_template_files(config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    raw_templates = config.get("templates", [])
    values = (
        [raw_templates]
        if isinstance(raw_templates, (str, Path))
        else list(raw_templates or [])
    )
    if config.get("template"):
        values.append(config["template"])

    seen: set[Path] = set()
    for value in values:
        path = Path(str(value))
        files = sorted(path.glob("*.png")) if path.is_dir() else [path]
        for file_path in files:
            if not file_path.is_file() or file_path in seen:
                continue
            paths.append(file_path)
            seen.add(file_path)
    return paths


def _scaled_template_matches(
    template: np.ndarray,
    config: dict[str, Any],
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    raw_scales = config.get("scales", [1.0])
    scales: list[float] = []
    for raw_scale in raw_scales:
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError):
            continue
        if scale <= 0 or scale in scales:
            continue
        scales.append(scale)

    if not scales:
        scales = [1.0]

    prepared: list[tuple[float, np.ndarray, np.ndarray]] = []
    height, width = template.shape[:2]
    for scale in scales:
        if scale == 1.0:
            scaled = template
        else:
            scaled_width = max(4, int(round(width * scale)))
            scaled_height = max(4, int(round(height * scale)))
            scaled = cv2.resize(
                template,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
        prepared.append((scale, scaled, _match_image(scaled)))
    return tuple(prepared)


def _match_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)
