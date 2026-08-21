#!/usr/bin/env python3
"""Fail-closed SHA256SUMS verification for one downloaded release asset."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


class ReleaseAssetVerificationError(RuntimeError):
    pass


def expected_sha256(checksums_text: str, asset_name: str) -> str:
    matches: list[str] = []
    for raw_line in checksums_text.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", raw_line.strip())
        if match and match.group(2).strip() == asset_name:
            matches.append(match.group(1).lower())
    if len(matches) != 1:
        raise ReleaseAssetVerificationError(
            f"SHA256SUMS must contain exactly one entry for {asset_name}; found {len(matches)}"
        )
    return matches[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_asset(asset: Path, checksums: Path) -> str:
    if not asset.is_file():
        raise ReleaseAssetVerificationError(f"release asset is missing: {asset}")
    if not checksums.is_file():
        raise ReleaseAssetVerificationError(f"SHA256SUMS is missing: {checksums}")
    expected = expected_sha256(checksums.read_text(encoding="utf-8"), asset.name)
    actual = file_sha256(asset)
    if actual != expected:
        raise ReleaseAssetVerificationError(
            f"sha256 mismatch for {asset.name}: expected {expected}, got {actual}"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    args = parser.parse_args()
    digest = verify_release_asset(args.asset, args.checksums)
    print(f"verified {args.asset.name} sha256={digest}")


if __name__ == "__main__":
    main()
