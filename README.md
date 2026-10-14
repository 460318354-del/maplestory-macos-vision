# MapleStory Vision

Cross-platform visual automation research project for a MapleStory client.

The first milestone focuses on a read-only perception loop:

```text
game window -> frame capture -> visual observations -> action proposal
```

Keyboard control is intentionally disabled by default. The project is
structured with a platform boundary so a Windows backend can be added later
without changing the perception and decision layers.

## Scope

- macOS window capture through Quartz
- Windows client-area capture through `mss` and `pywin32`
- Windows keyboard delivery through window messages or foreground input
- configurable frame and region processing
- category-based monster recognition using multiple visual templates
- multi-frame confirmation for moving monster candidates
- HP-bar observation and dry-run heal proposals
- route/action proposal based on a color-coded route image
- dry-run debug output
- platform interface reserved for a future Windows backend

This repository is a new implementation and does not copy source files from
the upstream MapleStoryAuto project.

Use only in environments where automation is permitted. Game rules and
service terms may prohibit automation.

## Requirements

- Python 3.11+
- macOS: Screen Recording permission for the terminal or Python interpreter
- macOS: Accessibility permission when real input is enabled
- Windows: install `mss` and `pywin32` (included by the Windows extra markers)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows, the platform-specific dependencies can also be installed
explicitly:

```powershell
python -m pip install mss pywin32
```

## Run the read-only loop

```bash
python -m maplevision --window-title "MapleStory" --dry-run
```

Open the desktop interface to bind the game window, mark the game frame, train
monster templates, and run the dry-run recognition loop:

```bash
PYTHONPATH=src python -m maplevision --gui
```

On macOS and Windows the interface has three tabs: `运行识别`,
`怪物模板训练`, and `路径`. The Windows implementation uses Tkinter and
supports window binding, game-frame capture, full-width height-band marking,
continuous template box selection, template testing and deletion, minimap
combat-point and ladder recording, path-step editing, path saving, and
standalone path testing. The same YAML templates, monster categories, combat
logic, and path-combat state machine are shared.

Windows input defaults to `post_message`, which sends `WM_KEYDOWN` and
`WM_KEYUP` to the bound game window without taking focus. Some clients read
raw input and ignore window messages; choose `foreground` in the Windows
interface or set `system.windows_input_mode` to use foreground keyboard
events.

On the `运行识别` tab, select `路径打怪` to enable per-point combat profiles.
The `路径挂机点` dropdown selects a saved minimap combat point. For each point,
mark and save its own fixed calibration point and monster search region. Attack
distance, selected monster categories, healing, mana, pickup, and timing remain
shared settings. At runtime, the yellow minimap marker determines the current
point; the matching profile is applied before monster and fixed-point detection.
If the current point cannot be identified, the common combat configuration is
kept as a fallback.

The `路径` tab is a complete editor. Capture a frame, create named combat
points by boxing rectangular areas inside the fixed minimap, then select each
point and save its independent fixed calibration point and monster search
region. You can also save named ladder/jump markers from the current minimap
yellow point. A path is defined independently for each pair of combat points
and consists of ordered steps such as moving to a point, moving to a ladder,
climbing, walking, jumping, or waiting. In
`路径打怪` mode, the main controller identifies the current combat point from
the minimap yellow marker, waits for the configured no-monster timeout, selects
the first saved path whose `source` matches that point, and executes it until
the target point is confirmed. It then loads the target point's combat profile.
Use `测试当前路径` to run only the selected path in the standalone route test
controller; `停止测试` releases all route-test keys. The ladder test sends
`alt`, waits the configured `Alt后延迟 ms` (default 150 ms), holds `up`,
records the minimap yellow-point `y` coordinate before the climb, waits
1000 ms, and verifies that the yellow point moved upward by at least
`黄点上移 px` (default 1 pixel). It then holds `up` for the configured
duration and verifies the minimap point. The nameplate is not used to decide
whether the character left the ground.
Jump steps use the same yellow-point check: the minimap `y` coordinate is
recorded before `方向键 + Alt`, the direction is held for the step duration,
and the jump is accepted only when the yellow point moves upward by the same
threshold. A failed jump returns to the recorded jump point and retries.

The primary calibration flow uses the full captured game frame. At an 800x600
game resolution, the default player-search and monster regions are 800x600.
On Retina macOS displays, the window capture is cropped below the title bar
and normalized to this logical 800x600 frame before matching, so marking and
template coordinates use the same pixels shown in the GUI.
Click `获取游戏画面`,
then use `标注画面高度` and drag vertically over the frame. The resulting
`y` values are clamped to the captured frame and used directly as the
full-width recognition region; no coordinate conversion is involved.
The marked frame region is used directly for monster matching. The window
capture remains full-frame so health observation and player localization can
use their required areas; image matching itself is limited to the marked
region.

The GUI also accepts a `角色 ID`. It resolves `Conquerork` to
`assets/templates/player/name_conquerork.png` by convention, or accepts a
template path directly. For a more reliable per-session match, click
`框选本人` on the full game frame and drag a rectangle around the player.
The crop is saved under `work/templates/` and used as the current matching
template. Click `识别角色` to capture one frame and show the matched box and
score.

Use the `怪物模板训练` tab to select `大型黄蘑菇`, `红色蜗牛`, or
`小型蘑菇`, capture a frame, and box one complete monster. Each sample is
saved under `assets/templates/monsters/<category>/`. Collect several samples
for different positions, animation states, and lighting conditions. The
At runtime, templates are loaded and preprocessed once, only selected monster
categories are matched, and templates are matched at their original size
without multi-scale resizing. Moving detections are confirmed across multiple
frames. Player matching runs only after a monster candidate is confirmed.

List visible game windows before binding:

```bash
python -m maplevision --list-windows
```

Bind the target window and print its metadata without entering the loop:

```bash
python -m maplevision --bind-window --window-title "MapleStory"
```

Capture and process exactly one frame after binding:

```bash
python -m maplevision --window-title "MapleStory" --once --dry-run
```

Write an annotated frame showing player and monster detections:

```bash
python -m maplevision \
  --window-title "MapleStory" \
  --once \
  --dry-run \
  --debug-output work/captures/combat-debug.png
```

The current combat milestone uses the main-screen `Conquerork` name tag for
the character anchor, the marked game-frame region, category-based monster
templates, multi-frame tracking, and the HP bar. Attack proposals require a
confirmed monster and the screen-space player anchor to be within the
configured rectangular attack range. If the name tag is not found, the
controller proposes `observe` rather than attack. Actions remain proposals
only while `dry-run` is enabled.

Coin pickup is always enabled in the desktop interface. The run page exposes
only the pickup interval in seconds. When the interval expires, the controller
starts a coin-pickup state machine and scans the configured gameplay ROI for
one second. A detected coin refreshes that one-second window and causes the
character to move toward the nearest coin or tap `z` three times when close.
The state machine ends after the ROI remains empty for the scan window, then
waits for the next configured interval. Confirmed monsters still have priority
over coin pickup.

For deterministic local testing with an image:

```bash
python -m maplevision --test-image path/to/frame.png --dry-run
```

The test-image mode does not require a game window or macOS capture
permissions.

## Project layout

```text
src/maplevision/
  capture/       Frame source abstractions and test image capture
  config/        YAML configuration loading
  decision/      State machine and action proposals
  platform/      macOS and Windows capture/input implementations
  vision/        Player, route, and target observations
  gui_common.py  Platform-neutral GUI geometry helpers
  gui_macos.py   macOS AppKit editor
  windows_gui.py Windows Tkinter runtime editor
  cli.py         Command-line entry point
tests/           Offline tests for configuration and route logic
config/          Example configuration
```

## Development

```bash
python -m pytest
python -m compileall -q src tests
```
