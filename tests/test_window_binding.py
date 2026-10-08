from maplevision.platform.macos.capture import _window_info_from_record


class FakeQuartz:
    kCGWindowNumber = "number"
    kCGWindowOwnerName = "owner"
    kCGWindowName = "name"
    kCGWindowBounds = "bounds"
    kCGWindowLayer = "layer"
    kCGWindowIsOnscreen = "onscreen"


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
        },
    )
    assert info is not None
    assert info.window_id == 42
    assert info.display_name == "MapleStory Client"
    assert info.bounds == (10, 20, 1280, 720)
