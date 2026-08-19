#!/usr/bin/env python3
"""Gate live Windows GPU catalog entries on real-hardware evidence.

Release CI may build every supported target, but build provenance is not a
correctness claim. Schema v1 receipts approve only the exact target that ran.
Schema v2 receipts may explicitly approve a provider matrix from one
representative target, but only when they bind the complete selected target
set and every target-scoped artifact fingerprint in that matrix. The
production catalog signer and release finalizer both invoke this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend_catalog import artifact_fingerprint


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class EntryIdentity:
    path: Path
    provider: str
    target: str
    backend_id: str
    artifact_fingerprint: str
    plugin_sha256: str
    version: str

    @property
    def tuple(self) -> tuple[str, ...]:
        return (
            self.provider,
            self.target,
            self.backend_id,
            self.artifact_fingerprint,
            self.plugin_sha256,
            self.version,
        )


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


def _entry_identity(path: Path) -> tuple[dict[str, Any], EntryIdentity]:
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
    try:
        fingerprint = artifact_fingerprint(entry)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{path} has no computable artifact fingerprint: {error}") from error
    return entry, EntryIdentity(
        path=path,
        provider=str(provider),
        target=target,
        backend_id=str(entry.get("id", "")),
        artifact_fingerprint=_lower_hex(
            fingerprint, 64, "computed artifact_fingerprint"
        ),
        plugin_sha256=_lower_hex(plugin_files[0].get("sha256"), 64, "plugin_sha256"),
        version=str(entry.get("version", "")),
    )


def _common_evidence_identity(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    evidence = _read(path)
    if evidence.get("schema_version") not in {1, 2} or evidence.get("result") != "pass":
        raise EvidenceError(f"{path} is not a passing backend hardware evidence receipt")
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
    return evidence, (
        str(provider),
        target,
        str(evidence.get("backend_id", "")),
        _lower_hex(evidence.get("artifact_fingerprint"), 64, "artifact_fingerprint"),
        _lower_hex(evidence.get("plugin_sha256"), 64, "plugin_sha256"),
        str(evidence.get("release_version", "")),
    )


def provider_matrix_sha256(entries: list[EntryIdentity]) -> str:
    payload = [
        {
            "artifact_fingerprint": entry.artifact_fingerprint,
            "backend_id": entry.backend_id,
            "plugin_sha256": entry.plugin_sha256,
            "provider": entry.provider,
            "release_version": entry.version,
            "target": entry.target,
        }
        for entry in sorted(entries, key=lambda item: (item.provider, item.version, item.target))
    ]
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def approved_entry_paths(entry_paths: list[Path], evidence_paths: list[Path]) -> list[Path]:
    entries: dict[tuple[str, ...], EntryIdentity] = {}
    all_entries: list[EntryIdentity] = []
    for path in entry_paths:
        _, identity = _entry_identity(path)
        if identity.tuple in entries:
            raise EvidenceError(f"duplicate backend entry identity: {identity.tuple}")
        entries[identity.tuple] = identity
        all_entries.append(identity)

    approved: dict[Path, Path] = {}
    seen_evidence: set[tuple[object, ...]] = set()
    for evidence_path in evidence_paths:
        evidence, identity_tuple = _common_evidence_identity(evidence_path)
        tested_entry = entries.get(identity_tuple)
        if tested_entry is None:
            raise EvidenceError(
                f"{evidence_path} does not match any exact tested release backend entry"
            )
        schema_version = int(evidence["schema_version"])
        evidence_key = (schema_version, *identity_tuple)
        if evidence_key in seen_evidence:
            raise EvidenceError(f"duplicate hardware evidence identity: {evidence_key}")
        seen_evidence.add(evidence_key)

        selected = [tested_entry]
        if schema_version == 2:
            if evidence.get("scope") != "provider_matrix":
                raise EvidenceError(
                    f"{evidence_path} schema v2 must declare scope=provider_matrix"
                )
            targets = evidence.get("approved_targets")
            if (
                not isinstance(targets, list)
                or not targets
                or any(not isinstance(target, str) for target in targets)
                or targets != sorted(set(targets))
            ):
                raise EvidenceError(
                    f"{evidence_path} approved_targets must be a sorted unique non-empty array"
                )
            if tested_entry.target not in targets:
                raise EvidenceError(
                    f"{evidence_path} does not include its tested target in approved_targets"
                )
            provider_entries = [
                entry
                for entry in all_entries
                if entry.provider == tested_entry.provider
                and entry.version == tested_entry.version
            ]
            release_targets = sorted(entry.target for entry in provider_entries)
            if targets != release_targets:
                raise EvidenceError(
                    f"{evidence_path} approved_targets do not equal the complete provider release matrix"
                )
            selected = provider_entries
            expected_matrix = provider_matrix_sha256(selected)
            actual_matrix = _lower_hex(
                evidence.get("provider_matrix_sha256"),
                64,
                "provider_matrix_sha256",
            )
            if actual_matrix != expected_matrix:
                raise EvidenceError(
                    f"{evidence_path} does not bind the selected provider matrix"
                )

        for entry in selected:
            previous = approved.get(entry.path)
            if previous is not None:
                raise EvidenceError(
                    f"{entry.path} is approved by both {previous} and {evidence_path}"
                )
            approved[entry.path] = evidence_path
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
