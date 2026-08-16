#!/usr/bin/env python3
"""Fail closed when a release outlives the frozen Windows GPU migration rail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def semver_triplet(value: str) -> tuple[int, int, int]:
    parts = value.strip().removeprefix("v").split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"expected MAJOR.MINOR.PATCH, got {value!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def legacy_sidecar_allowed(version: str, policy: dict[str, object]) -> bool:
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported Windows GPU migration policy schema")
    last = policy.get("legacy_static_sidecar_last_core_version")
    if not isinstance(last, str):
        raise ValueError("migration policy has no legacy sidecar sunset version")
    return semver_triplet(version) <= semver_triplet(last)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if legacy_sidecar_allowed(args.version, policy):
        return 0
    raise SystemExit(
        "legacy Windows CUDA/HIP whole-engine sidecars have passed their frozen "
        "one-release migration window; remove the legacy matrix/assets instead "
        "of extending the deadline"
    )


if __name__ == "__main__":
    raise SystemExit(main())
