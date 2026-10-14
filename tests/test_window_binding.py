import numpy as np

from maplevision.platform.macos.capture import (
    MacOSWindowSource,
    WindowInfo,
    _window_info_from_record,
    normalize_window_frame,
)


class FakeQuartz:
    kCGWindowNumber = "number"
    kCGWindowOwnerName = "owner"
    kCGWindowName = "name"
    kCGWindowBounds = "bounds"
    kCGWindowLayer = "layer"
    kCGWindowIsOnscreen = "onscreen"
    kCGWindowOwnerPID = "owner_pid"


def test_window_info_extracts_binding_metadata():
    info = _window_info_from_record(
        FakeQuartz,
        {
            "number": 42,
            "owner": "MapleStory",
            "name": "MapleStory Client",
            "bounds": {"X": 10, "Y": 20, "Width": 1280, "Height": 720},
            "layer": 0,
            "onscreen": True,
            "owner_pid": 777,
        },
    )
    assert info is not None
    assert info.window_id == 42
    assert info.display_name == "MapleStory Client"
    assert info.bounds == (10, 20, 1280, 720)
    assert info.owner_pid == 777


def test_normalize_window_frame_removes_title_bar_and_retina_scale():
    frame = np.zeros((1256, 1600, 3), dtype=np.uint8)
    frame[:56] = [10, 20, 30]
    frame[56:] = [40, 50, 60]

    normalized = normalize_window_frame(
        frame,
        (0, 0, 800, 628),
        (800, 600),
    )

    assert normalized.shape == (600, 800, 3)
    assert tuple(normalized[0, 0]) == (40, 50, 60)
    assert tuple(normalized[-1, -1]) == (40, 50, 60)


def test_window_source_refreshes_bound_window_position(monkeypatch):
    source = object.__new__(MacOSWindowSource)
    source._window_id = 42
    source._window_info = WindowInfo(
        window_id=42,
        owner="MapleStory",
        title="MapleStory",
        bounds=(10, 20, 800, 628),
        layer=0,
        on_screen=True,
    )
    refreshed = WindowInfo(
        window_id=42,
        owner="MapleStory",
        title="MapleStory",
        bounds=(2, 33, 800, 628),
        layer=0,
        on_screen=True,
    )
    monkeypatch.setattr(
        "maplevision.platform.macos.capture.list_windows",
        lambda title_contains=None: [refreshed],
    )

    source._refresh_window_info()

    assert source.window_info.bounds == (2, 33, 800, 628)
