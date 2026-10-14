"""Platform-neutral GUI entry point.

The existing AppKit editor remains the macOS implementation. Windows uses a
Tkinter editor so importing the package does not require AppKit or PyObjC.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from .gui_common import (
        crop_region,
        minimap_frame_point,
        minimap_local_point,
        player_search_region,
        safe_template_category,
    )
    from .windows_gui import WindowsMapleVisionApp, run_gui

    MapleVisionApp = WindowsMapleVisionApp
    PLAYER_TEMPLATE_CATEGORY = "角色"
else:
    from .gui_macos import *  # noqa: F401,F403
