#!/usr/bin/env python3
"""Small file-loading and atomic-write helpers for publish-model scripts."""
from __future__ import annotations

import json
import tempfile
import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_required_json(path: Path) -> Any:
    try:
        return load_json(path)
    except FileNotFoundError:
        raise SystemExit(f"required file missing: {path}") from None
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {path}: {error}") from None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    compact: bool = False,
) -> None:
    separators = (",", ":") if compact else None
    rendered = json.dumps(
        data,
        indent=None if compact else indent,
        sort_keys=sort_keys,
        separators=separators,
    ) + "\n"
    atomic_write_text(path, rendered)
