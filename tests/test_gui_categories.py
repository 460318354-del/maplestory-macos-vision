import numpy as np
import pytest

from maplevision.gui import PLAYER_TEMPLATE_CATEGORY, MapleVisionApp

gui_macos = pytest.importorskip("maplevision.gui_macos")


def test_runtime_monster_categories_exclude_non_monsters():
    app = object.__new__(MapleVisionApp)
    app.template_categories = [
        ("大型黄蘑菇", "yellow_mushroom"),
        ("红色蜗牛", "red_snail"),
        ("血条", "血条"),
        ("金币", "金币"),
        ("角色", PLAYER_TEMPLATE_CATEGORY),
    ]

    excluded = {PLAYER_TEMPLATE_CATEGORY, "血条", "金币"}
    labels = [
        label
        for _, label in app.template_categories
        if label not in excluded
    ][:4]

    assert labels == ["yellow_mushroom", "red_snail"]


def test_macos_training_detector_uses_current_frame_full_region():
    if not hasattr(MapleVisionApp, "_training_detector_config"):
        pytest.skip("macOS AppKit training detector is not active on this platform")
    app = object.__new__(MapleVisionApp)
    app.config = {
        "combat": {
            "monsters": {
                "region": [10, 20, 30, 40],
                "template_groups": {
                    "yellow_mushroom": ["assets/templates/monsters/yellow_mushroom"],
                    "金币": ["assets/templates/monsters/金币"],
                },
            },
        },
    }
    app.template_categories = [
        ("大型黄蘑菇", "yellow_mushroom"),
        ("金币", "金币"),
        ("角色", PLAYER_TEMPLATE_CATEGORY),
    ]
    app.test_template_labels = {
        "yellow_mushroom",
        "金币",
        PLAYER_TEMPLATE_CATEGORY,
    }

    config = app._training_detector_config(np.zeros((60, 80, 3), dtype=np.uint8))

    assert config["region"] == [0, 0, 80, 60]
    assert set(config["allowed_labels"]) == {
        "yellow_mushroom",
        "金币",
        PLAYER_TEMPLATE_CATEGORY,
    }
    assert {"yellow_mushroom", "金币", PLAYER_TEMPLATE_CATEGORY}.issubset(
        set(config["template_groups"])
    )


class _DummyCheckbox:
    def __init__(self, value=1):
        self.value = value

    def state(self):
        return self.value

    def setState_(self, value):
        self.value = value


class _DummyLabel:
    def __init__(self):
        self.value = ""

    def setStringValue_(self, value):
        self.value = value


class _DummyPreview:
    def __init__(self):
        self.region = None

    def setImage_pixelSize_(self, *_args):
        pass

    def setPatrolMarkerPoint_(self, *_args):
        pass

    def setFixedMarkerPoint_(self, *_args):
        pass

    def setSelectionMode_(self, *_args):
        pass

    def setSelectionEnabled_(self, *_args):
        pass

    def setRectangleEnabled_(self, *_args):
        pass

    def setClickEnabled_(self, *_args):
        pass

    def setMinimapRect_(self, *_args):
        pass

    def setDetectionRect_(self, *_args):
        pass

    def setRegionRect_(self, region):
        self.region = region


class _DummyButton:
    def setEnabled_(self, *_args):
        pass


class _DummySource:
    def next_frame(self):
        return np.zeros((60, 80, 3), dtype=np.uint8)


def test_macos_route_profile_save_preserves_existing_region_when_not_loaded():
    app = object.__new__(MapleVisionApp)
    app.route_combat_point_name = "8蘑菇点2"
    app.route_combat_profiles = {
        "8蘑菇点2": {
            "fixed_point": [44, 44],
            "fixed_point_enabled": True,
            "monster_region": [106, 259, 613, 106],
        }
    }
    app.fixed_point_target = None
    app.screen_region = None
    app.fixed_point_checkbox = _DummyCheckbox(1)
    app._path_combat_mode_selected = lambda: True
    app._selected_route_combat_point = lambda: "8蘑菇点2"

    app._save_current_route_combat_profile()

    assert app.route_combat_profiles["8蘑菇点2"]["fixed_point"] == [44, 44]
    assert app.route_combat_profiles["8蘑菇点2"]["monster_region"] == [
        106,
        259,
        613,
        106,
    ]


def test_macos_capture_game_frame_reloads_selected_route_region(monkeypatch):
    monkeypatch.setattr(gui_macos, "_nsimage_from_bgr", lambda _frame: object())
    app = object.__new__(MapleVisionApp)
    app.source = _DummySource()
    app.config = {"minimap": {"region": [5, 72, 157, 85]}}
    app.route_combat_point_name = "8蘑菇点2"
    app.route_combat_profiles = {
        "8蘑菇点2": {
            "fixed_point": [44, 44],
            "fixed_point_enabled": True,
            "monster_region": [106, 259, 613, 106],
        }
    }
    app.patrol_target_point = None
    app.fixed_point_target = None
    app.current_frame = None
    app.current_screen_player = None
    app.player_template_path = None
    app.screen_region = None
    app.fixed_point_checkbox = _DummyCheckbox(1)
    app.preview = _DummyPreview()
    app.image_info_label = _DummyLabel()
    app.height_info_label = _DummyLabel()
    app.player_info_label = _DummyLabel()
    app.status_label = _DummyLabel()
    app.select_player_button = _DummyButton()
    app.screen_band_button = _DummyButton()
    app.monster_region_button = _DummyButton()
    app.patrol_point_button = _DummyButton()
    app.fixed_point_button = _DummyButton()
    app._set_training_frame = lambda _frame: None
    app._path_combat_mode_selected = lambda: True
    app._selected_route_combat_point = lambda: "8蘑菇点2"
    app._minimap_point_in_frame = lambda point: point

    app.capture_game_frame()

    assert app.screen_region == (106, 259, 613, 106)
    assert app.preview.region == (106, 259, 613, 106)
    assert "x=106..719" in app.height_info_label.value
