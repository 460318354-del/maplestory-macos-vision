from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MacOSInputController:
    """Input boundary; dry-run is the only supported mode in v0.1."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run

    def emit(self, action: str) -> None:
        if self.dry_run:
            logger.info("dry-run action: %s", action)
            return
        raise NotImplementedError(
            "Live keyboard control is intentionally not enabled in v0.1"
        )
