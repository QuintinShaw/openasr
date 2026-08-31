#!/usr/bin/env python3
"""Single source for GitHub Release completeness: required vs optional assets.

The formal release completeness gate must see draft releases (contents:write)
and must derive qualification filenames from ``qualification_manifest.artifact_cell``,
the same helper that compiles the inert manifests. Experimental matrix rows may
appear on the release when they succeed; they are optional, never required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import qualification_manifest

MATRIX_PATH = Path(__file__).resolve().parent / "release_binaries_matrix.json"
LOCK_ASSET_SUFFIX = "-qualification-mutation.lock"


class CompletenessError(ValueError):
    pass


def load_matrix(path: Path = MATRIX_PATH) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise CompletenessError(f"{path} must contain a JSON array")
    return rows


def _archive_name(version: str, row: dict[str, Any]) -> str:
    asset = row.get("asset")
    ext = row.get("archive_ext")
    if not isinstance(asset, str) or not asset or not isinstance(ext, str) or not ext:
        raise CompletenessError(f"matrix row {row.get('target')!r} lacks asset/archive_ext")
    return f"openasr-{version}-{asset}.{ext}"


def required_cli_archives(version: str, matrix: list[dict[str, Any]]) -> set[str]:
    names = {
        f"openasr-{version}-xcframework.zip",
        f"openasr-{version}-xcframework.zip.sha256",
    }
    for row in matrix:
        if not isinstance(row, dict) or row.get("experimental") or row.get("distribution") == "plugin":
            continue
        names.add(_archive_name(version, row))
    return names


def optional_experimental_archives(version: str, matrix: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in matrix:
        if not isinstance(row, dict) or not row.get("experimental"):
            continue
        names.add(_archive_name(version, row))
    return names


def backend_pack_names(matrix: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in matrix:
        if not isinstance(row, dict) or row.get("experimental"):
            continue
        provider = row.get("provider")
        if provider == "cuda":
            names.append(f"backend-pack-cuda-sm_{row['cuda_gpu_target']}.json")
        elif provider == "hip":
            names.append(f"backend-pack-hip-{row['hip_gpu_target']}.json")
        elif provider == "vulkan" and row.get("distribution") == "plugin":
            names.append("backend-pack-vulkan-generic.json")
    return names


def _payload_filenames(entry: dict[str, Any], source: str) -> list[str]:
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise CompletenessError(f"{source} has no files")
    names: list[str] = []
    for file in files:
        name = file.get("filename") if isinstance(file, dict) else None
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise CompletenessError(f"{source} has unsafe filename {name!r}")
        names.append(name)
    return names


def required_from_backend_packs(version: str, entries: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for entry in entries:
        source = str(entry.get("id") or "backend-pack")
        provider, target = qualification_manifest.artifact_cell(entry)
        names.add(f"openasr-{version}-qualification-{provider}-{target}.json")
        names.update(_payload_filenames(entry, source))
    return names


def required_assets(
    version: str,
    matrix: list[dict[str, Any]],
    pack_entries: list[dict[str, Any]],
) -> set[str]:
    names = required_cli_archives(version, matrix)
    names.update(backend_pack_names(matrix))
    names.update(required_from_backend_packs(version, pack_entries))
    names.update(
        {
            "backend-plugin-hints.json",
            "catalog.backends.candidate.json",
            f"openasr-{version}-build-provenance.bundle.json",
            "SHA256SUMS",
        }
    )
    return names


def load_pack_entries(pack_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(pack_dir.glob("backend-pack-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CompletenessError(f"{path.name} must contain a JSON object")
        entries.append(value)
    return entries


def compare_assets(
    *,
    version: str,
    actual: set[str],
    matrix: list[dict[str, Any]],
    pack_entries: list[dict[str, Any]],
) -> tuple[list[str], list[str], str | None]:
    required = required_assets(version, matrix, pack_entries)
    optional = optional_experimental_archives(version, matrix)
    lock_asset = f"openasr-{version}{LOCK_ASSET_SUFFIX}"
    lock = lock_asset if lock_asset in actual else None
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional - ({lock_asset} if lock else set()))
    return missing, extra, lock


def _emit_lines(values: list[str]) -> None:
    for value in values:
        sys.stdout.write(f"{value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    packs = sub.add_parser("pack-names")
    packs.add_argument("--matrix", type=Path, default=MATRIX_PATH)

    compare = sub.add_parser("compare")
    compare.add_argument("--version", required=True)
    compare.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    compare.add_argument("--pack-dir", type=Path, required=True)
    compare.add_argument("--actual-file", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "pack-names":
            _emit_lines(backend_pack_names(load_matrix(args.matrix)))
            return 0

        matrix = load_matrix(args.matrix)
        pack_entries = load_pack_entries(args.pack_dir)
        actual = {
            line
            for line in args.actual_file.read_text(encoding="utf-8").splitlines()
            if line
        }
        missing, extra, lock = compare_assets(
            version=args.version,
            actual=actual,
            matrix=matrix,
            pack_entries=pack_entries,
        )
        required = sorted(required_assets(args.version, matrix, pack_entries))
        optional = sorted(optional_experimental_archives(args.version, matrix))
        sys.stdout.write("== required ==\n")
        _emit_lines(required)
        sys.stdout.write("== optional experimental ==\n")
        _emit_lines(optional)
        sys.stdout.write("== actual ==\n")
        _emit_lines(sorted(actual))
        if lock:
            sys.stderr.write(
                f"release still has qualification mutation lock {lock}\n"
            )
        if missing:
            sys.stderr.write("release is missing expected asset(s):\n")
            sys.stderr.write("\n".join(missing) + "\n")
        if extra:
            sys.stderr.write("release contains unexpected asset(s):\n")
            sys.stderr.write("\n".join(extra) + "\n")
        if lock or missing or extra:
            return 1
        sys.stdout.write(f"release has all {len(required)} expected assets.\n")
        return 0
    except CompletenessError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
