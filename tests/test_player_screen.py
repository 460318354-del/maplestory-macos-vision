from pathlib import Path

import cv2
import numpy as np

from maplevision.vision.player_screen import (
    PlayerScreenDetector,
    detect_player_screen,
    resolve_player_config,
)


def test_player_name_template_returns_screen_box_and_offset_anchor(tmp_path):
    template = np.zeros((20, 48, 3), dtype=np.uint8)
    template[3:17, 5:43] = [20, 180, 240]
    template_path = tmp_path / "player-name.png"
    assert cv2.imwrite(str(template_path), template)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[45:65, 70:118] = template
    detection = detect_player_screen(
        frame,
        {
            "enabled": True,
            "name": "Conquerork",
            "template": str(template_path),
            "search_region": [20, 20, 120, 80],
            "threshold": 0.95,
            "anchor_offset": [24, -4],
        },
    )

    assert detection is not None
    assert detection.name == "Conquerork"
    assert detection.name_box == (70, 45, 48, 20)
    assert detection.anchor == (94, 41)
    assert detection.score > 0.99


def test_player_detection_returns_none_below_threshold(tmp_path):
    template = np.zeros((20, 48, 3), dtype=np.uint8)
    template[3:17, 5:43] = [20, 180, 240]
    template_path = tmp_path / "player-name.png"
    assert cv2.imwrite(str(template_path), template)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = detect_player_screen(
        frame,
        {
            "enabled": True,
            "template": str(template_path),
            "search_region": [0, 0, 160, 120],
            "threshold": 0.8,
        },
    )

    assert detection is None


def test_player_detection_uses_configured_id_as_result_name(tmp_path):
    template = np.zeros((12, 24, 3), dtype=np.uint8)
    template[2:10, 4:20] = [30, 160, 230]
    template_path = tmp_path / "role-id.png"
    assert cv2.imwrite(str(template_path), template)

    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[30:42, 50:74] = template
    detection = detect_player_screen(
        frame,
        {
            "enabled": True,
            "id": "role-42",
            "name": "role-42",
            "template": str(template_path),
            "search_region": [0, 0, 100, 80],
            "threshold": 0.95,
            "anchor_offset": [12, -3],
        },
    )

    assert detection is not None
    assert detection.name == "role-42"
    assert detection.anchor == (62, 27)


def test_player_detector_caches_template_and_matches_same_result(tmp_path):
    template = np.zeros((16, 32, 3), dtype=np.uint8)
    template[3:13, 5:27] = [40, 170, 220]
    template_path = tmp_path / "role.png"
    assert cv2.imwrite(str(template_path), template)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[30:46, 50:82] = template
    config = {
        "enabled": True,
        "template": str(template_path),
        "search_region": [0, 0, 120, 80],
        "threshold": 0.95,
        "anchor_offset": [16, 0],
    }
    detector = PlayerScreenDetector(config)

    detection = detector.detect(frame, config)

    assert detection is not None
    assert detection.name_box == (50, 30, 32, 16)
    assert detection.anchor == (66, 30)
    assert detector.last_elapsed_ms >= 0


def test_player_detection_can_match_scaled_name_template(tmp_path):
    template = np.zeros((20, 48, 3), dtype=np.uint8)
    template[3:17, 5:43] = [20, 180, 240]
    template_path = tmp_path / "player-name.png"
    assert cv2.imwrite(str(template_path), template)

    scaled = cv2.resize(template, (24, 10), interpolation=cv2.INTER_AREA)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[30:40, 50:74] = scaled

    detection = detect_player_screen(
        frame,
        {
            "enabled": True,
            "template": str(template_path),
            "search_region": [0, 0, 100, 80],
            "threshold": 0.80,
            "scales": [0.50, 1.0],
            "anchor_offset": [24, 20],
        },
    )

    assert detection is not None
    assert detection.name_box == (50, 30, 24, 10)
    assert detection.anchor == (62, 40)


def test_player_detection_uses_best_match_from_template_directory(tmp_path):
    template_dir = tmp_path / "角色"
    template_dir.mkdir()
    weak = np.zeros((16, 28, 3), dtype=np.uint8)
    weak[4:12, 6:22] = [80, 80, 80]
    strong = np.zeros((12, 24, 3), dtype=np.uint8)
    strong[2:10, 4:20] = [30, 160, 230]
    assert cv2.imwrite(str(template_dir / "template_01.png"), weak)
    assert cv2.imwrite(str(template_dir / "template_02.png"), strong)

    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[30:42, 50:74] = strong

    detection = detect_player_screen(
        frame,
        {
            "enabled": True,
            "name": "player",
            "templates": [str(template_dir)],
            "search_region": [0, 0, 100, 80],
            "threshold": 0.95,
            "anchor_below_nameplate": 24,
        },
    )

    assert detection is not None
    assert detection.name_box == (50, 30, 24, 12)
    assert detection.anchor == (62, 66)


def test_resolve_player_config_uses_role_id_template_filename(tmp_path):
    template_dir = tmp_path / "assets" / "templates" / "player"
    template_dir.mkdir(parents=True)
    template_path = template_dir / "name_conquerork.png"
    template_path.write_bytes(b"png")

    resolved = resolve_player_config(
        "Conquerork",
        {
            "enabled": True,
            "search_region": [0, 0, 100, 100],
            "threshold": 0.7,
        },
        tmp_path,
    )

    assert resolved is not None
    assert resolved["id"] == "Conquerork"
    assert resolved["name"] == "Conquerork"
    assert Path(resolved["template"]) == template_path.resolve()


def test_resolve_player_config_falls_back_to_config_template(tmp_path):
    template_path = tmp_path / "custom.png"
    template_path.write_bytes(b"png")

    resolved = resolve_player_config(
        "MainRole",
        {
            "id": "MainRole",
            "template": str(template_path),
        },
        tmp_path,
    )

    assert resolved is not None
    assert Path(resolved["template"]) == template_path.resolve()
