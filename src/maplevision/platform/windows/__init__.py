"""Windows integrations for window capture and keyboard input."""

from .capture import WindowInfo, WindowsWindowSource, list_windows
from .input import WindowsInputController

__all__ = [
    "WindowInfo",
    "WindowsInputController",
    "WindowsWindowSource",
    "list_windows",
]
