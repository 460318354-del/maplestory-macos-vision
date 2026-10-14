from __future__ import annotations

import logging

from ..macos.input import MacOSInputController, _normalize_key

logger = logging.getLogger(__name__)


VK_CODES = {
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "space": 0x20,
    "z": 0x5A,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
}


class WindowsInputController(MacOSInputController):
    """Reuse the action scheduler with Win32 keyboard delivery.

    ``post_message`` sends WM_KEY* messages to the bound window and normally
    leaves the user's foreground application untouched. Some games read raw
    input instead of the window message queue; for those clients use
    ``mode=foreground`` and keep the game window focused.
    """

    def __init__(
        self,
        dry_run: bool = True,
        target_hwnd: int | None = None,
        mode: str = "post_message",
    ):
        super().__init__(dry_run=dry_run, target_pid=None)
        self.target_hwnd = int(target_hwnd) if target_hwnd else None
        normalized_mode = str(mode or "post_message").strip().lower()
        self.mode = (
            normalized_mode
            if normalized_mode in {"post_message", "foreground"}
            else "post_message"
        )
        self._win32 = None

    def _load_win32(self):
        if self._win32 is None:
            try:
                import win32api
                import win32con
                import win32gui
            except ImportError as exc:
                raise RuntimeError(
                    "Windows input requires pywin32. "
                    "Install it with: python -m pip install pywin32"
                ) from exc
            self._win32 = win32api, win32con, win32gui
        return self._win32

    def _key_event(self, key: str, down: bool) -> None:
        normalized = _normalize_key(key)
        key_code = VK_CODES.get(normalized)
        if key_code is None:
            logger.warning("unknown Windows key for action: %s", key)
            return

        win32api, win32con, win32gui = self._load_win32()
        if self.mode == "post_message" and self.target_hwnd:
            if normalized in {"alt", "option"}:
                message = (
                    win32con.WM_SYSKEYDOWN
                    if down
                    else win32con.WM_SYSKEYUP
                )
            else:
                message = (
                    win32con.WM_KEYDOWN
                    if down
                    else win32con.WM_KEYUP
                )
            win32gui.PostMessage(
                self.target_hwnd,
                message,
                key_code,
                _message_lparam(key_code, down),
            )
            logger.info(
                "Windows keyboard %s hwnd=%s: %s",
                "down" if down else "up",
                self.target_hwnd,
                normalized,
            )
            return

        flags = 0 if down else win32con.KEYEVENTF_KEYUP
        win32api.keybd_event(key_code, 0, flags, 0)
        logger.info(
            "Windows foreground keyboard %s: %s",
            "down" if down else "up",
            normalized,
        )


def _message_lparam(key_code: int, down: bool) -> int:
    """Build a minimal Win32 keyboard message lParam."""
    scan_code = (key_code & 0xFF) << 16
    if down:
        return 1 | scan_code
    return 1 | scan_code | (1 << 30) | (1 << 31)
