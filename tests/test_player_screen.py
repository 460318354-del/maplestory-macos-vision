import cv2
import numpy as np

from maplevision.vision.player_screen import detect_player_screen


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
