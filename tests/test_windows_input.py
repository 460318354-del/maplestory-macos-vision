from maplevision.platform.windows.input import (
    VK_CODES,
    WindowsInputController,
    _message_lparam,
)


class FakeWin32Api:
    def __init__(self):
        self.events = []

    def keybd_event(self, key_code, _scan, flags, _extra):
        self.events.append(("foreground", key_code, flags))


class FakeWin32Con:
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    KEYEVENTF_KEYUP = 0x0002


class FakeWin32Gui:
    def __init__(self):
        self.messages = []

    def PostMessage(self, hwnd, message, key_code, lparam):
        self.messages.append((hwnd, message, key_code, lparam))


def test_windows_input_posts_key_messages_to_bound_window():
    controller = WindowsInputController(
        dry_run=False,
        target_hwnd=1234,
        mode="post_message",
    )
    api = FakeWin32Api()
    con = FakeWin32Con()
    gui = FakeWin32Gui()
    controller._win32 = api, con, gui

    controller._key_event("right", True)
    controller._key_event("right", False)

    assert gui.messages == [
        (1234, con.WM_KEYDOWN, VK_CODES["right"], _message_lparam(0x27, True)),
        (1234, con.WM_KEYUP, VK_CODES["right"], _message_lparam(0x27, False)),
    ]
    assert api.events == []


def test_windows_input_uses_system_messages_for_alt():
    controller = WindowsInputController(
        dry_run=False,
        target_hwnd=1234,
        mode="post_message",
    )
    api = FakeWin32Api()
    con = FakeWin32Con()
    gui = FakeWin32Gui()
    controller._win32 = api, con, gui

    controller._key_event("alt", True)
    controller._key_event("alt", False)

    assert [message[1] for message in gui.messages] == [
        con.WM_SYSKEYDOWN,
        con.WM_SYSKEYUP,
    ]


def test_windows_input_foreground_mode_uses_keybd_event():
    controller = WindowsInputController(
        dry_run=False,
        target_hwnd=1234,
        mode="foreground",
    )
    api = FakeWin32Api()
    con = FakeWin32Con()
    gui = FakeWin32Gui()
    controller._win32 = api, con, gui

    controller._key_event("ctrl", True)
    controller._key_event("ctrl", False)

    assert api.events == [
        ("foreground", VK_CODES["ctrl"], 0),
        ("foreground", VK_CODES["ctrl"], con.KEYEVENTF_KEYUP),
    ]
    assert gui.messages == []
