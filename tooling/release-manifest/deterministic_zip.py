#!/usr/bin/env python3
"""Create a byte-deterministic zip archive from a directory's contents.

  deterministic_zip.py create <out.zip> <src-dir>

Used by `.github/workflows/release-binaries.yml` to package content-addressed
runtime payloads for target-scoped Windows backend packs. Stable archive bytes
ensure that unchanged runtime inputs retain the same content identity.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_UNIX_MODE = 0o644


class DeterministicZipError(Exception):
    pass


def iter_sorted_files(src_dir: Path) -> list[Path]:
    """Every regular file under `src_dir`, sorted by its POSIX-style relative path."""
    files = [path for path in src_dir.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: path.relative_to(src_dir).as_posix())


def create_deterministic_zip(out_path: Path, src_dir: Path) -> None:
    if not src_dir.is_dir():
        raise DeterministicZipError(f"source directory not found: {src_dir}")
    files = iter_sorted_files(src_dir)
    if not files:
        raise DeterministicZipError(f"source directory is empty: {src_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            arcname = path.relative_to(src_dir).as_posix()
            info = zipfile.ZipInfo(filename=arcname, date_time=FIXED_DATE_TIME)
            info.external_attr = FIXED_UNIX_MODE << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as handle:
                archive.writestr(info, handle.read())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="Create a deterministic zip from a directory")
    create.add_argument("out", type=Path, help="Output .zip path")
    create.add_argument("src_dir", type=Path, help="Directory whose files become the zip's entries")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command == "create":
        try:
            create_deterministic_zip(args.out, args.src_dir)
        except DeterministicZipError as error:
            print(f"deterministic_zip.py: {error}", file=sys.stderr)
            return 1
        print(args.out)
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
