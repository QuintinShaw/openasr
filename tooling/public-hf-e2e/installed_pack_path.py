#!/usr/bin/env python3
"""Validate the installed pack path reported by ``openasr pull``."""

from __future__ import annotations

import re
import sys
from pathlib import Path


SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validate_installed_pack_path(home_arg: str, pack_arg: str, sha256: str) -> None:
    if SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"installed pack has an invalid sha256: {sha256}")

    home = Path(home_arg).resolve(strict=True)
    pack = Path(pack_arg).resolve(strict=True)
    if not pack.is_file():
        raise ValueError(f"installed pack is not a file: {pack}")

    try:
        relative_pack = pack.relative_to(home)
    except ValueError as error:
        raise ValueError(f"installed pack is outside OPENASR_HOME: {pack}") from error

    # Legacy model-store layouts name the installed file itself with the
    # package extension. The current store names immutable objects by their
    # digest and keeps the GGUF payload in a file named ``content``. Accept the
    # latter only at its exact canonical location, with the directory digest
    # matching the bytes the caller just hashed.
    if pack.suffix == ".oasr":
        return

    expected_parts = ("models", "objects", "sha256", sha256, "content")
    if relative_pack.parts != expected_parts:
        raise ValueError(
            "installed pack is neither a .oasr file nor the canonical "
            f"content-addressed object for sha256 {sha256}: {pack}"
        )


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: installed_pack_path.py OPENASR_HOME PACK_PATH SHA256",
            file=sys.stderr,
        )
        return 2

    try:
        validate_installed_pack_path(argv[1], argv[2], argv[3])
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
