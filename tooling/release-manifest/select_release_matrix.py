#!/usr/bin/env python3
"""Filter the release-binaries matrix before any build job is instantiated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MATRIX_PATH = Path(__file__).resolve().parent / "release_binaries_matrix.json"


def load_matrix(path: Path = MATRIX_PATH) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path} must contain a JSON array")
    return rows


def known_targets(rows: list[dict[str, Any]]) -> list[str]:
    targets: list[str] = []
    for row in rows:
        target = row.get("target")
        if isinstance(target, str):
            targets.append(target)
    return targets


def select_matrix(rows: list[dict[str, Any]], only_target: str) -> list[dict[str, Any]]:
    if only_target == "":
        return rows
    selected = [row for row in rows if row.get("target") == only_target]
    if not selected:
        expected = ", ".join(known_targets(rows))
        raise SystemExit(
            f"unknown only_target {only_target!r}; expected one of: {expected}"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-target", default="")
    args = parser.parse_args()
    selected = select_matrix(load_matrix(), args.only_target)
    sys.stdout.write(json.dumps(selected, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
