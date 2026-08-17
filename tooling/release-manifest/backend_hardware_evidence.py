#!/usr/bin/env python3
"""Gate live Windows GPU catalog entries on exact real-hardware evidence.

Release CI may build every supported target, but build provenance is not a
correctness claim. This tool selects only entry bytes that have a matching
fresh-process hardware receipt. The production catalog signer and release
finalizer both invoke it, so an untested HIP/CUDA target can remain an inert
draft-release asset without becoming runtime-selectable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must contain a JSON object")
    return value


def _lower_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise EvidenceError(f"{field} must be lowercase {length}-hex")
    return value


def _entry_identity(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    entry = _read(path)
    provider = entry.get("vendor")
    targets = entry.get("targets")
    if provider not in {"cuda", "hip"} or not isinstance(targets, list) or len(targets) != 1:
        raise EvidenceError(f"{path} is not one target-scoped CUDA/HIP entry")
    target = targets[0]
    expected_prefix = "sm_" if provider == "cuda" else "gfx"
    if not isinstance(target, str) or not target.startswith(expected_prefix):
        raise EvidenceError(f"{path} has an invalid {provider} target")
    plugin_files = [file for file in entry.get("files", []) if file.get("role") == "plugin"]
    if len(plugin_files) != 1:
        raise EvidenceError(f"{path} must declare exactly one plugin file")
    return entry, (
        str(provider),
        target,
        str(entry.get("id", "")),
        _lower_hex(entry.get("artifact_fingerprint"), 64, "artifact_fingerprint"),
        _lower_hex(plugin_files[0].get("sha256"), 64, "plugin_sha256"),
        str(entry.get("version", "")),
    )


def _evidence_identity(path: Path) -> tuple[str, ...]:
    evidence = _read(path)
    if evidence.get("schema_version") != 1 or evidence.get("result") != "pass":
        raise EvidenceError(f"{path} is not a passing backend hardware evidence v1 receipt")
    if evidence.get("placement") != "full_device" or evidence.get("cpu_fallback") is not False:
        raise EvidenceError(f"{path} does not prove fail-closed FullDevice execution")
    runs = evidence.get("fresh_process_runs")
    if not isinstance(runs, int) or runs < 5:
        raise EvidenceError(f"{path} must prove at least five fresh-process runs")
    for field in ("binary_sha256", "workload_sha256", "model_pack_sha256", "evidence_sha256"):
        _lower_hex(evidence.get(field), 64, field)
    provider = evidence.get("provider")
    target = evidence.get("device_target")
    if provider not in {"cuda", "hip"}:
        raise EvidenceError(f"{path} has an unsupported provider")
    prefix = "sm_" if provider == "cuda" else "gfx"
    if not isinstance(target, str) or not target.startswith(prefix):
        raise EvidenceError(f"{path} has an invalid device target")
    return (
        str(provider),
        target,
        str(evidence.get("backend_id", "")),
        _lower_hex(evidence.get("artifact_fingerprint"), 64, "artifact_fingerprint"),
        _lower_hex(evidence.get("plugin_sha256"), 64, "plugin_sha256"),
        str(evidence.get("release_version", "")),
    )


def approved_entry_paths(entry_paths: list[Path], evidence_paths: list[Path]) -> list[Path]:
    entries: dict[tuple[str, ...], Path] = {}
    for path in entry_paths:
        _, identity = _entry_identity(path)
        if identity in entries:
            raise EvidenceError(f"duplicate backend entry identity: {identity}")
        entries[identity] = path
    approved: list[Path] = []
    seen: set[tuple[str, ...]] = set()
    for evidence_path in evidence_paths:
        identity = _evidence_identity(evidence_path)
        if identity in seen:
            raise EvidenceError(f"duplicate hardware evidence identity: {identity}")
        seen.add(identity)
        entry_path = entries.get(identity)
        if entry_path is None:
            raise EvidenceError(
                f"{evidence_path} does not match any exact release backend entry"
            )
        approved.append(entry_path)
    return sorted(approved)


def verify_catalog_policy(
    catalog_path: Path,
    version: str,
    approved_paths: list[Path],
) -> None:
    catalog = _read(catalog_path)
    approved_ids = {_entry_identity(path)[0]["id"] for path in approved_paths}
    live_ids = {
        entry.get("id")
        for entry in catalog.get("backends", [])
        if entry.get("vendor") in {"cuda", "hip"} and str(entry.get("version")) == version
    }
    if live_ids != approved_ids:
        raise EvidenceError(
            "live catalog target set does not exactly match hardware-approved release entries: "
            f"approved={sorted(approved_ids)}, live={sorted(live_ids)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", type=Path, required=True)
    parser.add_argument("--evidence", action="append", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    approved = approved_entry_paths(args.entry, args.evidence)
    if args.catalog or args.version:
        if not args.catalog or not args.version:
            raise EvidenceError("--catalog and --version must be supplied together")
        verify_catalog_policy(args.catalog, args.version, approved)
    for path in approved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
