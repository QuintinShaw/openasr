#!/usr/bin/env python3
"""Retry GitHub Release downloads. Large draft assets often fail with EOF."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def download_asset(
    tag: str,
    name: str,
    dest_dir: Path,
    *,
    repository: str | None = None,
    attempts: int = 6,
) -> None:
    if Path(name).name != name:
        raise ValueError(f"unsafe release asset name: {name!r}")
    command = [
        "gh",
        "release",
        "download",
        tag,
        "-p",
        name,
        "-D",
        str(dest_dir),
        "--clobber",
    ]
    if repository:
        command.extend(["--repo", repository])
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(min(2 ** attempt, 16))
    assert last_error is not None
    raise last_error
