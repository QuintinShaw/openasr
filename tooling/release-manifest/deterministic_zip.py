#!/usr/bin/env python3
"""Create a byte-deterministic zip archive from a directory's contents.

  deterministic_zip.py create <out.zip> <src-dir>

Used by `.github/workflows/release-binaries.yml`'s Windows CUDA/HIP legs to
package the large, content-addressed `vendor_layers` archive (NVIDIA cudart/
cuBLAS, AMD rocBLAS/hipBLAS + their `library` Tensile subtrees) -- NOT for the
small per-release "sidecar" archive (`openasr.exe` + docs), which keeps using
PowerShell's `Compress-Archive` exactly as before (its non-determinism does
not matter there: that archive is named after the immutable release version,
not its own content hash).

PowerShell's `Compress-Archive` embeds each entry's real filesystem mtime and
does not guarantee a stable entry order, so zipping byte-identical input
twice (even in the same CI run) is NOT guaranteed to produce byte-identical
output. That breaks the whole point of a content-addressed vendor archive
(`core/vendor/<sha256>/<asset>`, deduped by sha256 across every release that
pins a compatible toolchain, per `docs/backend-kernels.md`): a spurious hash
change on a re-run with unchanged inputs would silently stop the dedup from
firing (a new upload for content that was already there) -- not a correctness
bug, but it defeats the mechanism's purpose.

This script fixes that by controlling the two sources of non-determinism
`zipfile` itself does not hide by default:
  - entries are added in a FIXED (sorted, POSIX-separator) order regardless
    of the filesystem's own directory-listing order;
  - every entry's `date_time` (zipfile has no seconds-since-epoch mtime
    field; it stores a 6-tuple) is pinned to a fixed constant instead of the
    real filesystem mtime, and every entry's Unix permission bits (encoded in
    the zip's `external_attr` high 16 bits) are pinned to a fixed constant
    (0o644 files, 0o755 the implicit directories zipfile does not need since
    we only ever add files) instead of whatever the staging step happened to
    leave on disk.

Compression itself (`zipfile.ZIP_DEFLATED`, default compresslevel) is left at
Python's default -- it is deterministic for identical input bytes on a fixed
Python/zlib build, which is exactly the "same CI runner, same run" scope this
guards. It intentionally does NOT attempt cross-platform/cross-zlib-version
byte identity (that would require vendoring a specific zlib), since nothing
in this pipeline compares hashes computed on different machines/runs before
this content-addressed archive itself is content-addressed by its own output.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# 1980-01-01 00:00:00 -- the earliest date the ZIP format's DOS-style
# date_time field can represent, and a conventional "no real timestamp"
# sentinel (the same one `SOURCE_DATE_EPOCH`-aware reproducible-build tooling
# commonly uses for zip entries).
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

# Unix permission bits packed into the zip's external_attr high 16 bits
# (`external_attr = (mode << 16) | ...`), matching what a `chmod 644` file
# looks like to an `unzip`/`ditto` on macOS/Linux. Windows itself ignores
# these bits entirely, so pinning them cannot break anything on the platform
# this script actually runs on (release-binaries.yml's Windows GPU legs).
FIXED_UNIX_MODE = 0o644


class DeterministicZipError(Exception):
    pass


def iter_sorted_files(src_dir: Path) -> list[Path]:
    """Every regular file under `src_dir`, sorted by its POSIX-style relative
    path -- fixed regardless of the OS/filesystem's own listing order."""
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
            info.external_attr = (FIXED_UNIX_MODE << 16)
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
