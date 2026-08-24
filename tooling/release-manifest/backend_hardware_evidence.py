#!/usr/bin/env python3
"""Gate live Windows GPU catalog entries on exact real-hardware evidence.

Release CI may build every supported target, but build provenance is not a
correctness claim. A hardware summary approves exactly one target-scoped
release entry, and its raw audit is a required one-to-one witness. The
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
from backend_target_identity import (
    is_cuda_qualification_target,
    is_hip_qualification_target,
    is_provider_qualification_target,
    is_vulkan_qualification_target,
)
from release_attestation import AttestationError, verify_paths


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

    def matches_evidence(self, evidence: tuple[str, ...]) -> bool:
        provider, target, backend_id, artifact, plugin, version, _driver = evidence
        return (
            provider == self.provider
            and backend_id == self.backend_id
            and artifact == self.artifact_fingerprint
            and plugin == self.plugin_sha256
            and version == self.version
            and (self.provider == "vulkan" or target == self.target)
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


def _driver_version(value: object, field: str = "driver_version") -> str:
    if not (
        isinstance(value, str)
        and 0 < len(value) <= 64
        and all(part and part.isascii() and part.isdigit() for part in value.split("."))
    ):
        raise EvidenceError(f"{field} must be a dotted numeric driver version")
    return value


def _entry_identity(path: Path) -> tuple[dict[str, Any], EntryIdentity]:
    entry = _read(path)
    provider = entry.get("vendor")
    targets = entry.get("targets")
    if provider not in {"cuda", "hip", "vulkan"} or not isinstance(targets, list):
        raise EvidenceError(f"{path} is not a supported Windows GPU entry")
    if provider in {"cuda", "hip"}:
        if len(targets) != 1:
            raise EvidenceError(f"{path} is not one target-scoped CUDA/HIP entry")
        target = targets[0]
        valid_target = (
            is_cuda_qualification_target(target)
            if provider == "cuda"
            else is_hip_qualification_target(target)
        )
        if not valid_target:
            raise EvidenceError(f"{path} has an invalid {provider} target")
    else:
        if targets:
            raise EvidenceError(f"{path} Vulkan entry must be artifact-generic")
        target = ""
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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2:
            raise EvidenceError(f"{path}:{line_number} is not a checksum record")
        digest, filename = parts
        filename = filename.lstrip("*")
        _lower_hex(digest, 64, f"{path}:{line_number} digest")
        if not filename or Path(filename).name != filename or filename in checksums:
            raise EvidenceError(
                f"{path}:{line_number} has an unsafe/duplicate filename"
            )
        checksums[filename] = digest
    return checksums


def _common_evidence_identity(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    evidence = _read(path)
    if evidence.get("schema_version") != 1 or evidence.get("result") != "pass":
        raise EvidenceError(f"{path} is not a passing backend hardware evidence receipt")
    forbidden = {"scope", "approved_targets", "provider_matrix_sha256"}
    present = sorted(field for field in forbidden if field in evidence)
    if present:
        raise EvidenceError(
            f"{path} contains removed provider-matrix fields: {', '.join(present)}"
        )
    if evidence.get("placement") != "full_device" or evidence.get("cpu_fallback") is not False:
        raise EvidenceError(f"{path} does not prove fail-closed FullDevice execution")
    runs = evidence.get("fresh_process_runs")
    if not isinstance(runs, int) or runs < 5:
        raise EvidenceError(f"{path} must prove at least five fresh-process runs")
    for field in ("binary_sha256", "workload_sha256", "model_pack_sha256", "evidence_sha256"):
        _lower_hex(evidence.get(field), 64, field)
    provider = evidence.get("provider")
    target = evidence.get("device_target")
    if provider not in {"cuda", "hip", "vulkan"}:
        raise EvidenceError(f"{path} has an unsupported provider")
    if not is_provider_qualification_target(provider, target):
        raise EvidenceError(f"{path} has an invalid device target")
    return evidence, (
        str(provider),
        target,
        str(evidence.get("backend_id", "")),
        _lower_hex(evidence.get("artifact_fingerprint"), 64, "artifact_fingerprint"),
        _lower_hex(evidence.get("plugin_sha256"), 64, "plugin_sha256"),
        str(evidence.get("release_version", "")),
        _driver_version(evidence.get("driver_version")),
    )


def _raw_activation_matches(
    value: object,
    *,
    identity: tuple[str, ...],
    catalog_sha256: str,
    scope_sha256: str,
    context: str,
) -> None:
    provider, target, backend_id, artifact, _plugin, version, driver = identity
    if not isinstance(value, dict):
        raise EvidenceError(f"{context} is not an activation status object")
    qualification = value.get("qualification")
    if (
        value.get("host_mode") != "neutral_dynamic"
        or value.get("activated") is not None
        or not isinstance(qualification, dict)
    ):
        raise EvidenceError(f"{context} lacks an isolated qualification backend")
    expected = {
        "vendor": provider,
        "device_target": target,
        "backend_id": backend_id,
        "artifact_fingerprint": artifact,
        "version": version,
        "driver_version": driver,
        "catalog_sha256": catalog_sha256,
        "scope_sha256": scope_sha256,
    }
    for field, expected_value in expected.items():
        if qualification.get(field) != expected_value:
            raise EvidenceError(f"{context} {field} does not match exact release entry")


def _validate_hardware_run_receipt(
    receipt: dict[str, Any],
    *,
    identity: tuple[str, ...],
    evidence: dict[str, Any],
    nonce: object,
    context: str,
) -> None:
    provider, _target, _backend_id, _artifact, _plugin, _version, _driver = identity
    if receipt.get("schema") != "openasr.short-audio-receipt.v0":
        raise EvidenceError(f"{context} has the wrong receipt schema")
    if not isinstance(nonce, str) or len(nonce) != 32 or any(
        char not in "0123456789abcdef" for char in nonce
    ):
        raise EvidenceError(f"{context} has an invalid run nonce")
    scope = receipt.get("scope")
    if not isinstance(scope, str) or not scope.endswith(f"/{nonce}"):
        raise EvidenceError(f"{context} receipt is not nonce scoped")
    pack, audio, run = receipt.get("pack"), receipt.get("audio"), receipt.get("run")
    if not isinstance(pack, dict) or pack.get("content_sha256") != evidence.get(
        "model_pack_sha256"
    ):
        raise EvidenceError(f"{context} receipt pack bytes do not match")
    if not isinstance(audio, dict) or audio.get("sha256") != evidence.get("workload_sha256"):
        raise EvidenceError(f"{context} receipt workload bytes do not match")
    if (
        not isinstance(run, dict)
        or run.get("backend") != "native"
        or run.get("device") != provider
        or run.get("os") != "windows"
        or receipt.get("placement") != provider
    ):
        raise EvidenceError(f"{context} receipt run identity does not match")
    env = run.get("env_allowlist")
    if (
        not isinstance(env, dict)
        or env.get("OPENASR_GGML_BACKEND") != provider
        or env.get("OPENASR_OFFLINE") not in {"1", "true"}
    ):
        raise EvidenceError(f"{context} receipt is not provider-confined offline")
    observed = receipt.get("observed_placement")
    if not isinstance(observed, dict):
        raise EvidenceError(f"{context} receipt has no observed placement")
    if (
        type(observed.get("direct_graph_computes")) is not int
        or observed["direct_graph_computes"] <= 0
        or observed.get("scheduler_graph_computes") != 0
        or observed.get("fallback_node_samples_by_backend") not in (None, {})
    ):
        raise EvidenceError(f"{context} receipt is not fail-closed FullDevice")
    compute = observed.get("observed_compute_nodes_by_backend")
    provider_names = {
        "cuda": ("cuda", "nvidia"),
        "hip": ("hip", "rocm"),
        "vulkan": ("vulkan",),
    }[provider]
    if (
        not isinstance(compute, dict)
        or not compute
        or any(type(nodes) is not int or nodes <= 0 for nodes in compute.values())
        or any(not any(tag in str(name).lower() for tag in provider_names) for name in compute)
    ):
        raise EvidenceError(f"{context} receipt compute placement does not match provider")
    transcript = receipt.get("transcript")
    if not isinstance(transcript, dict):
        raise EvidenceError(f"{context} receipt has no transcript")
    _lower_hex(transcript.get("text_sha256"), 64, f"{context} transcript.text_sha256")
    execution = receipt.get("execution")
    if not isinstance(execution, dict):
        raise EvidenceError(f"{context} receipt has no execution projection")
    attempt = execution.get("request_attempt_id")
    _lower_hex(attempt, 32, f"{context} request_attempt_id")
    phases = execution.get("phase_duration_micros")
    if (
        execution.get("request_attempt_conflicted") is not False
        or execution.get("live_lease_reconciliation") != "matched"
        or execution.get("live_state_complete") is not True
        or execution.get("event_history_complete") is not True
        or execution.get("dropped_events") != 0
        or execution.get("timing_complete") is not True
        or execution.get("request_receipt_complete") is not True
        or execution.get("terminal") != "succeeded"
        or not isinstance(phases, dict)
        or set(phases) != {
            "upload-ingest",
            "decode-normalize",
            "admission-wait",
            "compute",
        }
        or any(type(value) is not int or value < 0 for value in phases.values())
    ):
        raise EvidenceError(f"{context} receipt execution evidence is incomplete")


def _validate_raw_provenance(path: Path, raw: dict[str, Any]) -> dict[str, str]:
    for field in (
        "generator_sha256",
        "neutral_archive_sha256",
        "neutral_extracted_tree_sha256",
        "plugin_sha256",
        "vendor_archive_sha256",
        "catalog_candidate_sha256",
        "preview_catalog_sha256",
        "preview_catalog_signature_sha256",
        "checksums_sha256",
    ):
        _lower_hex(raw.get(field), 64, f"{path} {field}")
    _lower_hex(raw.get("source_digest"), 40, f"{path} source_digest")
    for field in ("repository", "signer_workflow", "scope", "preview_catalog_url"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise EvidenceError(f"{path} {field} must be a non-empty string")
    if "/.github/workflows/" not in raw["signer_workflow"]:
        raise EvidenceError(f"{path} signer_workflow is not a repository workflow")

    preflight = raw.get("catalog_signature_preflight")
    if not isinstance(preflight, dict):
        raise EvidenceError(f"{path} lacks catalog signature preflight evidence")
    for field in ("stdout_sha256", "stderr_sha256"):
        _lower_hex(preflight.get(field), 64, f"{path} preflight.{field}")
    if preflight.get("cached_catalog_sha256") != raw.get("preview_catalog_sha256"):
        raise EvidenceError(f"{path} preflight catalog bytes do not match")
    if preflight.get("cached_signature_sha256") != raw.get(
        "preview_catalog_signature_sha256"
    ):
        raise EvidenceError(f"{path} preflight signature bytes do not match")

    subjects = raw.get("attested_release_subjects")
    if not isinstance(subjects, list) or not subjects:
        raise EvidenceError(f"{path} lacks attested release subjects")
    by_name: dict[str, str] = {}
    for index, subject in enumerate(subjects, start=1):
        if not isinstance(subject, dict):
            raise EvidenceError(f"{path} attested subject {index} is not an object")
        filename = subject.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in by_name
        ):
            raise EvidenceError(f"{path} attested subject {index} has an invalid filename")
        by_name[filename] = _lower_hex(
            subject.get("sha256"), 64, f"{path} attested subject {index} sha256"
        )
        _lower_hex(
            subject.get("verification_sha256"),
            64,
            f"{path} attested subject {index} verification_sha256",
        )
    required_hashes = {
        raw["neutral_archive_sha256"],
        raw["plugin_sha256"],
        raw["vendor_archive_sha256"],
        raw["catalog_candidate_sha256"],
    }
    missing_hashes = sorted(required_hashes - set(by_name.values()))
    if missing_hashes:
        raise EvidenceError(
            f"{path} attested release subjects omit required exact bytes: {missing_hashes}"
        )
    return by_name


def _validate_raw_audit(
    path: Path,
    raw: dict[str, Any],
    *,
    evidence: dict[str, Any],
    identity: tuple[str, ...],
) -> None:
    if raw.get("schema") != "openasr.backend-hardware-audit.v1":
        raise EvidenceError(f"{path} has an unsupported raw-audit schema")
    _validate_raw_provenance(path, raw)
    provider, target, backend_id, artifact, plugin, version, driver = identity
    expected_identity = {
        "provider": provider,
        "device_target": target,
        "backend_id": backend_id,
        "artifact_fingerprint": artifact,
        "plugin_sha256": plugin,
        "release_version": version,
        "driver_version": driver,
    }
    for field, expected in expected_identity.items():
        if raw.get(field) != expected:
            raise EvidenceError(f"{path} {field} does not match exact summary/entry")
    for field in ("binary_sha256", "plugin_sha256", "workload_sha256", "model_pack_sha256"):
        if raw.get(field) != evidence.get(field):
            raise EvidenceError(f"{path} {field} does not match evidence summary")
    qualification_scope_sha256 = _lower_hex(
        raw.get("qualification_scope_sha256"),
        64,
        f"{path} qualification_scope_sha256",
    )
    preview_catalog_sha256 = _lower_hex(
        raw.get("preview_catalog_sha256"),
        64,
        f"{path} preview_catalog_sha256",
    )
    runs = raw.get("runs")
    if not isinstance(runs, list) or len(runs) != evidence.get("fresh_process_runs"):
        raise EvidenceError(f"{path} runs do not match fresh_process_runs")
    seen_nonces: set[str] = set()
    seen_process_ids: set[int] = set()
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise EvidenceError(f"{path} run {index} is not an object")
        nonce = run.get("nonce")
        process_id = run.get("process_id")
        if not isinstance(nonce, str) or nonce in seen_nonces:
            raise EvidenceError(f"{path} run {index} has a duplicate/invalid nonce")
        if type(process_id) is not int or process_id <= 0 or process_id in seen_process_ids:
            raise EvidenceError(f"{path} run {index} has a duplicate/invalid process id")
        seen_nonces.add(nonce)
        seen_process_ids.add(process_id)
        receipt = run.get("receipt")
        receipt_sha256 = run.get("receipt_sha256")
        if not isinstance(receipt, dict) or not isinstance(receipt_sha256, str):
            raise EvidenceError(f"{path} run {index} lacks receipt/hash")
        if _canonical_sha256(receipt) != _lower_hex(
            receipt_sha256, 64, f"{path} run {index} receipt_sha256"
        ):
            raise EvidenceError(f"{path} run {index} receipt canonical hash mismatch")
        _validate_hardware_run_receipt(
            receipt,
            identity=identity,
            evidence=evidence,
            nonce=nonce,
            context=f"{path} run {index}",
        )
        _raw_activation_matches(
            run.get("activation_before"),
            identity=identity,
            catalog_sha256=preview_catalog_sha256,
            scope_sha256=qualification_scope_sha256,
            context=f"{path} run {index} activation_before",
        )
        _raw_activation_matches(
            run.get("activation_after"),
            identity=identity,
            catalog_sha256=preview_catalog_sha256,
            scope_sha256=qualification_scope_sha256,
            context=f"{path} run {index} activation_after",
        )


def approved_entry_paths(
    entry_paths: list[Path], evidence_paths: list[Path], raw_audit_paths: list[Path]
) -> list[Path]:
    entries: list[EntryIdentity] = []
    seen_entries: set[tuple[str, ...]] = set()
    for path in entry_paths:
        _, identity = _entry_identity(path)
        if identity.tuple in seen_entries:
            raise EvidenceError(f"duplicate backend entry identity: {identity.tuple}")
        seen_entries.add(identity.tuple)
        entries.append(identity)

    raw_by_sha: dict[str, tuple[Path, dict[str, Any]]] = {}
    for raw_path in raw_audit_paths:
        raw = _read(raw_path)
        digest = _canonical_sha256(raw)
        if digest in raw_by_sha:
            raise EvidenceError(f"duplicate raw audit canonical SHA256: {digest}")
        raw_by_sha[digest] = (raw_path, raw)

    approved: dict[Path, Path] = {}
    seen_evidence: set[tuple[object, ...]] = set()
    for evidence_path in evidence_paths:
        evidence, identity_tuple = _common_evidence_identity(evidence_path)
        matching_entries = [
            entry for entry in entries if entry.matches_evidence(identity_tuple)
        ]
        if len(matching_entries) != 1:
            raise EvidenceError(
                f"{evidence_path} does not match any exact tested release backend entry"
            )
        tested_entry = matching_entries[0]
        evidence_key = identity_tuple[:6]
        if evidence_key in seen_evidence:
            raise EvidenceError(f"duplicate hardware evidence identity: {evidence_key}")
        seen_evidence.add(evidence_key)
        evidence_sha = _lower_hex(evidence.get("evidence_sha256"), 64, "evidence_sha256")
        bound_raw = raw_by_sha.pop(evidence_sha, None)
        if bound_raw is None:
            raise EvidenceError(f"{evidence_path} has no matching --raw-audit")
        raw_path, raw = bound_raw
        _validate_raw_audit(
            raw_path, raw, evidence=evidence, identity=identity_tuple
        )
        previous = approved.get(tested_entry.path)
        if previous is not None:
            raise EvidenceError(
                f"{tested_entry.path} is approved by both {previous} and {evidence_path}"
            )
        approved[tested_entry.path] = evidence_path
    if raw_by_sha:
        unused = ", ".join(str(path) for path, _ in raw_by_sha.values())
        raise EvidenceError(f"unused --raw-audit: {unused}")
    return sorted(approved)


def verify_release_provenance(
    *,
    entry_paths: list[Path],
    raw_audit_paths: list[Path],
    release_subject_paths: list[Path],
    checksums_path: Path,
    repository: str,
    signer_workflow: str,
    source_digest: str,
) -> None:
    """Re-verify raw provenance against downloaded release bytes and Sigstore."""
    _lower_hex(source_digest, 40, "source_digest")
    checksums = _read_checksums(checksums_path)
    checksums_sha256 = _sha256_file(checksums_path)
    subjects: dict[str, Path] = {}
    for path in release_subject_paths:
        if path.name in subjects:
            raise EvidenceError(f"duplicate release subject filename: {path.name}")
        expected = checksums.get(path.name)
        if expected is None or _sha256_file(path) != expected:
            raise EvidenceError(f"{path} does not match SHA256SUMS")
        subjects[path.name] = path
    required_entry_names = {path.name for path in entry_paths}
    if not required_entry_names <= subjects.keys():
        raise EvidenceError("release subjects omit one or more backend entry files")

    attest_names: set[str] = set()
    for raw_path in raw_audit_paths:
        raw = _read(raw_path)
        recorded = _validate_raw_provenance(raw_path, raw)
        if raw.get("repository") != repository:
            raise EvidenceError(f"{raw_path} repository does not match finalizer input")
        if raw.get("signer_workflow") != signer_workflow:
            raise EvidenceError(f"{raw_path} signer workflow does not match finalizer input")
        if raw.get("source_digest") != source_digest:
            raise EvidenceError(f"{raw_path} source digest does not match the release tag")
        if raw.get("checksums_sha256") != checksums_sha256:
            raise EvidenceError(f"{raw_path} SHA256SUMS binding does not match")
        if not required_entry_names <= recorded.keys():
            raise EvidenceError(
                f"{raw_path} attestation set omits one or more backend entries"
            )
        for filename, expected_sha in recorded.items():
            subject = subjects.get(filename)
            if subject is None or _sha256_file(subject) != expected_sha:
                raise EvidenceError(
                    f"{raw_path} attested subject {filename} does not match downloaded release bytes"
                )
            attest_names.add(filename)

    _verify_attested_paths(
        [subjects[filename] for filename in sorted(attest_names)],
        repository=repository,
        signer_workflow=signer_workflow,
        source_digest=source_digest,
        label="release subject",
    )


def preflight_release_subjects(
    *,
    entry_paths: list[Path],
    release_subject_paths: list[Path],
    checksums_path: Path,
    repository: str,
    signer_workflow: str,
    source_digest: str,
) -> None:
    """Authenticate downloaded release bytes before any subject is executed.

    The later raw hardware audit replays the same provenance and records it in
    evidence.  This earlier seam exists solely to keep an unverified archive or
    executable from crossing the qualification runner's execution boundary.
    """

    _lower_hex(source_digest, 40, "source_digest")
    checksums = _read_checksums(checksums_path)
    subjects: dict[str, Path] = {}
    for path in release_subject_paths:
        if path.name in subjects:
            raise EvidenceError(f"duplicate release subject filename: {path.name}")
        expected = checksums.get(path.name)
        if expected is None or _sha256_file(path) != expected:
            raise EvidenceError(f"{path} does not match SHA256SUMS")
        subjects[path.name] = path
    required_entry_names = {path.name for path in entry_paths}
    if not required_entry_names or not required_entry_names <= subjects.keys():
        raise EvidenceError("release subjects omit one or more backend entry files")
    _verify_attested_paths(
        [subjects[name] for name in sorted(subjects)],
        repository=repository,
        signer_workflow=signer_workflow,
        source_digest=source_digest,
        label="pre-execution release subject",
    )


def _verify_attested_paths(
    paths: list[Path],
    *,
    repository: str,
    signer_workflow: str,
    source_digest: str,
    label: str,
) -> None:
    """Verify exact bytes against one repository workflow and source commit."""

    try:
        verify_paths(
            paths,
            repository=repository,
            signer_workflow=signer_workflow,
            source_digest=source_digest,
            label=label,
        )
    except AttestationError as error:
        raise EvidenceError(str(error)) from error


def verify_qualification_provenance(
    *,
    evidence_paths: list[Path],
    raw_audit_paths: list[Path],
    repository: str,
    signer_workflow: str,
    source_digest: str,
) -> None:
    """Authenticate post-release hardware witnesses before parsing their claims."""

    _verify_attested_paths(
        [*evidence_paths, *raw_audit_paths],
        repository=repository,
        signer_workflow=signer_workflow,
        source_digest=source_digest,
        label="qualification evidence",
    )


def verify_catalog_policy(
    catalog_path: Path,
    version: str,
    approved_paths: list[Path],
    evidence_paths: list[Path],
) -> None:
    catalog = _read(catalog_path)
    approved_ids = {_entry_identity(path)[0]["id"] for path in approved_paths}
    evidence_by_backend: dict[str, tuple[str, str, str]] = {}
    for path in evidence_paths:
        evidence, identity = _common_evidence_identity(path)
        backend_id = identity[2]
        if backend_id in evidence_by_backend:
            raise EvidenceError(f"duplicate evidence backend_id: {backend_id}")
        evidence_by_backend[backend_id] = (
            _canonical_sha256(evidence),
            identity[1],
            identity[6],
        )
    for entry in catalog.get("backends", []):
        if (
            not isinstance(entry, dict)
            or entry.get("vendor") not in {"cuda", "hip", "vulkan"}
            or str(entry.get("version")) != version
        ):
            continue
        backend_id = entry.get("id")
        activation = entry.get("activation", {"state": "published-inert"})
        if not isinstance(activation, dict):
            raise EvidenceError(f"catalog backend {backend_id} has invalid activation state")
        state = activation.get("state", "published-inert")
        if state not in {"published-inert", "qualified", "activated", "revoked"}:
            raise EvidenceError(f"catalog backend {backend_id} has unknown activation state")
        if state not in {"qualified", "activated"}:
            continue
        if backend_id not in approved_ids:
            raise EvidenceError(
                f"catalog backend {backend_id} is {state} without exact hardware evidence"
            )
        expected = evidence_by_backend.get(str(backend_id))
        if expected is None or (
            activation.get("hardware_evidence_sha256"),
            activation.get("qualified_device_target"),
            activation.get("qualified_driver_version"),
        ) != expected:
            raise EvidenceError(
                f"catalog backend {backend_id} hardware target/driver evidence binding does not verify"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", type=Path, default=[])
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    parser.add_argument("--raw-audit", action="append", type=Path, default=[])
    parser.add_argument("--release-subject", action="append", type=Path, default=[])
    parser.add_argument("--checksums", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--signer-workflow")
    parser.add_argument("--qualification-signer-workflow")
    parser.add_argument("--source-digest")
    parser.add_argument("--release-preflight-only", action="store_true")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    if (args.evidence or args.raw_audit) and not args.entry:
        raise EvidenceError("hardware evidence validation requires at least one --entry")
    release_provenance_requested = any(
        (args.release_subject, args.checksums, args.signer_workflow)
    )
    if release_provenance_requested:
        if not (
            args.release_subject
            and args.checksums
            and args.repo
            and args.signer_workflow
            and args.source_digest
        ):
            raise EvidenceError(
                "release provenance requires --release-subject, --checksums, --repo, "
                "--signer-workflow, and --source-digest together"
            )
        if args.release_preflight_only:
            if args.evidence or args.raw_audit or args.qualification_signer_workflow:
                raise EvidenceError(
                    "release preflight cannot consume post-execution evidence"
                )
            preflight_release_subjects(
                entry_paths=args.entry,
                release_subject_paths=args.release_subject,
                checksums_path=args.checksums,
                repository=args.repo,
                signer_workflow=args.signer_workflow,
                source_digest=args.source_digest,
            )
        else:
            verify_release_provenance(
                entry_paths=args.entry,
                raw_audit_paths=args.raw_audit,
                release_subject_paths=args.release_subject,
                checksums_path=args.checksums,
                repository=args.repo,
                signer_workflow=args.signer_workflow,
                source_digest=args.source_digest,
            )
    elif args.release_preflight_only:
        raise EvidenceError("release preflight requires exact release provenance inputs")
    if args.qualification_signer_workflow:
        if not (args.repo and args.source_digest):
            raise EvidenceError(
                "qualification provenance requires --repo and --source-digest"
            )
        verify_qualification_provenance(
            evidence_paths=args.evidence,
            raw_audit_paths=args.raw_audit,
            repository=args.repo,
            signer_workflow=args.qualification_signer_workflow,
            source_digest=args.source_digest,
        )
    approved = approved_entry_paths(args.entry, args.evidence, args.raw_audit)
    if args.catalog or args.version:
        if not args.catalog or not args.version:
            raise EvidenceError("--catalog and --version must be supplied together")
        verify_catalog_policy(args.catalog, args.version, approved, args.evidence)
    for path in approved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
