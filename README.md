# MapleStory macOS Vision

macOS-first visual automation research project for a MapleStory client.

The first milestone focuses on a read-only perception loop:

```text
game window -> frame capture -> visual observations -> action proposal
```

Keyboard control is intentionally disabled by default. The project is
structured with a platform boundary so a Windows backend can be added later
without changing the perception and decision layers.

## Scope

- macOS window capture through Quartz
- configurable frame and region processing
- route/action proposal based on a color-coded route image
- dry-run debug output
- platform interface reserved for a future Windows backend

This repository is a new implementation and does not copy source files from
the upstream MapleStoryAuto project.

Use only in environments where automation is permitted. Game rules and
service terms may prohibit automation.

## Requirements

- macOS
- Python 3.11+
- Screen Recording permission for the terminal or Python interpreter
- Accessibility permission if input control is enabled in a future milestone

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Run the read-only loop

```bash
python -m maplevision --window-title "MapleStory" --dry-run
```

List visible macOS windows before binding:

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

Write an annotated frame showing the minimap region and detected player point:

```bash
python -m maplevision \
  --window-title "MapleStory" \
  --once \
  --dry-run \
  --debug-output work/captures/minimap-debug.png
```

For deterministic local testing with an image:

```bash
python -m maplevision --test-image path/to/frame.png --dry-run
```

The test-image mode does not require a game window or macOS capture
permissions.

## Project layout

```text
src/maplevision/
  capture/       Frame source abstractions and macOS capture
  config/        YAML configuration loading
  decision/      State machine and action proposals
  platform/      macOS implementation boundary
  vision/        Player, route, and target observations
  cli.py         Command-line entry point
tests/           Offline tests for configuration and route logic
config/          Example configuration
```

## Development

```bash
python -m pytest
python -m compileall -q src tests
```
