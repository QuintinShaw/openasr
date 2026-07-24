#!/usr/bin/env python3
"""Reject workstation paths and private sibling repository names in public source.

The gate scans tracked public-facing text sources. The only intentional
occurrences of forbidden markers are the escaped pattern fragments in this
file itself (so the script does not self-fail).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Broad public tree coverage. Exclude vendored/third-party trees and the gate
# script itself (which keeps forbidden markers only as escaped fixtures).
SCAN_PREFIXES = (
    "crates/",
    "tooling/",
    "docs/",
    ".github/",
    "README.md",
    "SECURITY.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
)
SKIP_PREFIXES = (
    "crates/openasr-core/third_party/",
)
TEXT_SUFFIXES = {
    ".rs",
    ".py",
    ".sh",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".txt",
}
SELF_PATH = Path("tooling/check_public_source_hygiene.py")

# Escaped so this file is not itself a false positive.
FORBIDDEN_PATTERNS = (
    "/" + "Volumes" + "/",
    "/" + "Users" + "/",
    "openasr-" + "legacy",
    "openasr-" + "app",
)


def tracked_source_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=False
    )
    files: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode()
        path = Path(rel)
        if path == SELF_PATH:
            continue
        if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if not (
            any(rel.startswith(prefix) for prefix in SCAN_PREFIXES)
            or rel in SCAN_PREFIXES
        ):
            continue
        if path.suffix and path.suffix not in TEXT_SUFFIXES and path.name not in {
            "README.md",
            "SECURITY.md",
            "AGENTS.md",
            "CONTRIBUTING.md",
        }:
            # Allow extensionless files only when explicitly listed above.
            if path.suffix:
                continue
        files.append(path)
    return files


def main() -> int:
    violations: list[str] = []
    for path in tracked_source_files():
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in line:
                    violations.append(
                        f"{path}:{line_number}: forbidden public-source marker ({pattern})"
                    )
                    break
    if violations:
        print("Public-source hygiene check failed:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
