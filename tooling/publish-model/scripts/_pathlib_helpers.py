#!/usr/bin/env python3
"""Path helpers shared by the publish-model scripts."""
from __future__ import annotations

from pathlib import Path


def repo_root(start: Path) -> Path:
    """Return the nearest parent containing .git, or start if none is found."""
    resolved = start.resolve()
    for path in [resolved, *resolved.parents]:
        if (path / ".git").exists():
            return path
    return resolved
