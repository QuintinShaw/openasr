from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path


def posix_script_command(script: Path, *args: str) -> list[str]:
    """Return a subprocess command for a repo POSIX script on every host."""
    if os.name == "nt":
        shell = os.environ.get("OPENASR_TEST_POSIX_SHELL") or shutil.which("bash") or shutil.which("sh")
        if not shell:
            raise unittest.SkipTest(f"{script.name} requires a POSIX shell")
        return [shell, str(script), *args]
    return [str(script), *args]


def posix_path(path: Path) -> str:
    """Render a native path for a POSIX shell hosted by Git for Windows."""
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/{drive}/{tail}"


def native_path(path: str) -> Path:
    """Map an MSYS /c/... path back to a native Windows pathlib value."""
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == "/":
        return Path(f"{path[1]}:/{path[3:]}")
    return Path(path)
