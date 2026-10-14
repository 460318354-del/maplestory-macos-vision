from __future__ import annotations

import sys

from maplevision.cli import main


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    if "--gui" not in arguments:
        arguments.append("--gui")
    raise SystemExit(main(arguments))
