#!/usr/bin/env python3
"""Project and gate pre-publication GPU correctness evidence.

The matrix is derived from the canonical architecture inventory, the public model
catalog, and the staged backend catalog. It describes evidence that is required;
it never turns a build, placement receipt, or CPU result into token correctness.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from backend_catalog import artifact_fingerprint
from backend_target_identity import (
    is_cuda_qualification_target,
    is_hip_qualification_target,
    is_provider_qualification_target,
    is_vulkan_qualification_target,
)
from release_attestation import AttestationError, verify_paths

SCHEMA = "openasr.gpu-correctness-matrix.v1"
RECEIPT_SCHEMA = "openasr.short-audio-receipt.v0"
EVIDENCE_SCHEMA = "openasr.short-audio-receipt.evidence.v1"
GPU_PROVIDERS = ("cuda", "vulkan", "hip")
LaneKey = tuple[str, str, str, str, str, str]
ReceiptKey = tuple[str, str, str, str, str, str, str, str]


class MatrixError(ValueError):
    """Raised when a staging projection or evidence set is incomplete."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MatrixError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_schema(document: dict[str, Any], expected: str, label: str) -> None:
    if document.get("schema") != expected:
        raise MatrixError(f"{label} schema must be {expected!r}")


def _public_models(catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    models = catalog.get("models")
    if not isinstance(models, list):
        raise MatrixError("model catalog must contain a models array")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        if not isinstance(model, dict) or model.get("public") is not True:
            continue
        if model.get("kind", "asr-model") != "asr-model":
            continue
        family = model.get("family")
        model_id = model.get("id")
        if not isinstance(family, str) or not family or not isinstance(model_id, str) or not model_id:
            raise MatrixError("every public asr-model needs family and id")
        grouped.setdefault(family, []).append(model)
    if not grouped:
        raise MatrixError("public model catalog has no asr-model entries")
    return grouped


def _advertised_providers(descriptor: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    execution = descriptor.get("execution", {})
    capabilities = execution.get("execution_capabilities", {}) if isinstance(execution, dict) else {}
    optimization = descriptor.get("optimization", {})
    auto_policy = optimization.get("auto_gpu_policy") if isinstance(optimization, dict) else None
    result: list[tuple[str, str, list[str]]] = []
    providers = capabilities.get("providers", [])
    if not isinstance(providers, list):
        raise MatrixError(f"family {descriptor.get('catalog_family_id')} has invalid providers")
    for item in providers:
        if not isinstance(item, dict) or not isinstance(item.get("provider"), str):
            raise MatrixError(f"family {descriptor.get('catalog_family_id')} has an invalid provider row")
        provider = item["provider"].lower()
        if provider not in GPU_PROVIDERS or (item.get("full_device") is not True and item.get("hybrid") is not True):
            continue
        placement = "full_device" if item.get("full_device") is True else "hybrid"
        modes = ["explicit"]
        if auto_policy == "all-backends" or (auto_policy == "except-metal" and provider != "metal"):
            modes.append("auto")
        result.append((provider, placement, modes))
    return result


def _lane_policies(provider: str) -> tuple[str, str, str]:
    """Return explicit staging policies, not claims of successful hardware runs."""
    capture = "enabled" if provider == "hip" else "disabled"
    scheduler = "disabled"
    # Production reuse evidence is currently Unknown, so qualification must
    # exercise the shipped FreshGraph plan in both cold- and warm-process rows.
    return capture, scheduler, "fresh_graph"


def _tie_policy(family: str) -> str:
    # This is the family oracle contract, not a provider capability. XASR's
    # existing host oracle is last-max; other current token/code paths use the
    # first-max contract. Receipts must still bind and repeat this value.
    return "last_maximum" if family == "xasr-zipformer" else "first_maximum"


def _vulkan_target_contract(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MatrixError("Vulkan qualification targets must be an object")
    result: dict[str, str] = {}
    for backend_id, target in value.items():
        if (
            not isinstance(backend_id, str)
            or not backend_id
            or not is_vulkan_qualification_target(target)
        ):
            raise MatrixError(
                "each Vulkan qualification target must bind backend_id to one canonical vk_caps class"
            )
        result[backend_id] = target
    return dict(sorted(result.items()))


def _backend_candidates(
    backend_catalog: dict[str, Any],
    release_version: str,
    *,
    vulkan_targets: dict[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    backends = backend_catalog.get("backends")
    if not isinstance(backends, list):
        raise MatrixError("backend catalog must contain a backends array")
    candidates: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    vulkan_targets = _vulkan_target_contract(vulkan_targets)
    seen_vulkan_backends: set[str] = set()
    for entry in backends:
        if not isinstance(entry, dict):
            raise MatrixError("backend catalog contains a non-object entry")
        provider = entry.get("vendor")
        if provider not in GPU_PROVIDERS or str(entry.get("version")) != release_version:
            continue
        backend_id = entry.get("id")
        if not isinstance(backend_id, str) or not backend_id:
            raise MatrixError("release backend candidate has no backend_id")
        targets = entry.get("targets")
        if provider in {"cuda", "hip"}:
            if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], str) or not targets[0]:
                raise MatrixError(f"backend {backend_id} is not exact-target scoped")
            device_target = targets[0]
            if not (
                is_cuda_qualification_target(device_target)
                if provider == "cuda"
                else is_hip_qualification_target(device_target)
            ):
                raise MatrixError(
                    f"backend {backend_id} has a non-canonical {provider} target"
                )
            artifact_target = device_target
        else:
            if targets != []:
                raise MatrixError(f"backend {backend_id} Vulkan artifact must be target-generic")
            seen_vulkan_backends.add(backend_id)
            artifact_target = ""
            activation_value = entry.get("activation", {"state": "published-inert"})
            qualified_target = (
                activation_value.get("qualified_device_target", "")
                if isinstance(activation_value, dict)
                else ""
            )
            device_target = vulkan_targets.get(backend_id, qualified_target)
            if device_target and not is_vulkan_qualification_target(device_target):
                raise MatrixError(f"backend {backend_id} has an invalid Vulkan device target")
            if (
                backend_id in vulkan_targets
                and qualified_target
                and qualified_target != device_target
            ):
                raise MatrixError(
                    f"backend {backend_id} signed Vulkan target differs from the matrix contract"
                )
        plugin_files = [
            item
            for item in entry.get("files", [])
            if isinstance(item, dict) and item.get("role") == "plugin"
        ]
        if len(plugin_files) != 1 or not _hex_digest(plugin_files[0].get("sha256")):
            raise MatrixError(f"backend {backend_id} lacks one exact plugin sha256")
        identity = (provider, artifact_target, backend_id)
        if identity in seen:
            raise MatrixError(f"duplicate exact backend candidate {identity}")
        seen.add(identity)
        try:
            fingerprint = artifact_fingerprint(entry)
        except (TypeError, ValueError) as error:
            raise MatrixError(f"backend {backend_id} has no artifact fingerprint: {error}") from error
        if not _hex_digest(fingerprint):
            raise MatrixError(f"backend {backend_id} has an invalid artifact fingerprint")
        activation = entry.get("activation", {"state": "published-inert"})
        if not isinstance(activation, dict) or activation.get("state", "published-inert") not in {
            "published-inert",
            "qualified",
            "activated",
            "revoked",
        }:
            raise MatrixError(f"backend {backend_id} has an invalid activation state")
        candidates.setdefault(provider, []).append(
            {
                "provider": provider,
                "artifact_target": artifact_target,
                "device_target": device_target,
                "backend_id": backend_id,
                "artifact_fingerprint": fingerprint,
                "plugin_sha256": plugin_files[0]["sha256"],
                "activation_state": activation.get("state", "published-inert"),
                "qualification_source_catalog_sha256": activation.get(
                    "qualification_source_catalog_sha256", ""
                ),
                "hardware_evidence_sha256": activation.get(
                    "hardware_evidence_sha256", ""
                ),
                "qualified_device_target": activation.get(
                    "qualified_device_target", ""
                ),
                "qualified_driver_version": activation.get(
                    "qualified_driver_version", ""
                ),
                "correctness_matrix_sha256": activation.get(
                    "correctness_matrix_sha256", ""
                ),
                "correctness_receipts_sha256": activation.get(
                    "correctness_receipts_sha256", ""
                ),
            }
        )
    for values in candidates.values():
        values.sort(key=lambda item: (item["device_target"], item["backend_id"]))
    unknown_vulkan_targets = sorted(set(vulkan_targets) - seen_vulkan_backends)
    if unknown_vulkan_targets:
        raise MatrixError(
            f"Vulkan qualification targets name unknown release backends: {unknown_vulkan_targets}"
        )
    if not candidates:
        raise MatrixError("backend catalog has no exact release GPU candidates")
    return candidates


def _candidate_set_sha256(candidates: dict[str, list[dict[str, str]]]) -> str:
    values = [
        {
            field: item[field]
            for field in (
                "provider",
                "artifact_target",
                "backend_id",
                "artifact_fingerprint",
                "plugin_sha256",
            )
        }
        for provider in sorted(candidates)
        for item in candidates[provider]
    ]
    return _canonical_sha(values)


def correctness_receipt_set_sha256(
    receipt_paths: list[Path], *, provider: str, device_target: str, backend_id: str
) -> str:
    digests: list[str] = []
    for path in receipt_paths:
        _document, evidence = _evidence_for(path)
        if evidence.get("evidence_class") not in {"placement_resource", "token_transcript"}:
            continue
        if (
            evidence.get("provider"),
            evidence.get("device_target"),
            evidence.get("backend_id"),
        ) == (provider, device_target, backend_id):
            digests.append(_sha256(path))
    return _canonical_sha(sorted(digests))


def _validate_activation_catalog(
    matrix: dict[str, Any],
    activation_catalog: dict[str, Any],
    source_digests: dict[str, str],
    receipt_paths: list[Path],
) -> dict[tuple[str, str, str], str]:
    contract = matrix["artifact_contract"]
    vulkan_targets = _vulkan_target_contract(
        contract.get("vulkan_qualification_targets", {})
    )
    candidates = _backend_candidates(
        activation_catalog,
        contract["release_version"],
        vulkan_targets=vulkan_targets,
    )
    if _candidate_set_sha256(candidates) != contract["backend_candidates_sha256"]:
        raise MatrixError("activation catalog backend candidates differ from the matrix bytes")
    activated: dict[tuple[str, str, str], str] = {}
    projected_candidates = {
        (cell.get("provider"), cell.get("device_target"), cell.get("backend_id"))
        for cell in matrix.get("cells", [])
        if isinstance(cell, dict)
    }
    for values in candidates.values():
        for candidate in values:
            state = candidate["activation_state"]
            binding_fields = (
                "qualification_source_catalog_sha256",
                "hardware_evidence_sha256",
                "correctness_matrix_sha256",
                "correctness_receipts_sha256",
            )
            bindings = [candidate[field] for field in binding_fields]
            qualified_target = candidate["qualified_device_target"]
            qualified_driver = candidate["qualified_driver_version"]
            if state == "published-inert" and (
                any(bindings) or qualified_target or qualified_driver
            ):
                raise MatrixError(
                    f"published-inert backend {candidate['backend_id']} carries qualification bindings"
                )
            if state == "qualified" and (
                any(not _hex_digest(value) for value in bindings[:2])
                or any(bindings[2:])
                or qualified_target != candidate["device_target"]
                or not _driver_version(qualified_driver)
            ):
                raise MatrixError(
                    f"qualified backend {candidate['backend_id']} must carry only source and hardware bindings"
                )
            if state == "activated" and any(
                not _hex_digest(value) for value in bindings
            ):
                raise MatrixError(
                    f"backend {candidate['backend_id']} has incomplete qualification bindings"
                )
            if state == "activated" and (
                qualified_target != candidate["device_target"]
                or not _driver_version(qualified_driver)
            ):
                raise MatrixError(
                    f"backend {candidate['backend_id']} has invalid qualified target/driver bindings"
                )
            if state in {"qualified", "activated"} and (
                candidate["qualification_source_catalog_sha256"]
                != source_digests["backend_catalog_sha256"]
            ):
                raise MatrixError(
                    f"backend {candidate['backend_id']} qualification is bound to another source catalog"
                )
            if state != "activated":
                continue
            candidate_identity = (
                candidate["provider"],
                candidate["device_target"],
                candidate["backend_id"],
            )
            if candidate_identity not in projected_candidates:
                raise MatrixError(
                    f"backend {candidate['backend_id']} activation target is not projected by this matrix"
                )
            if (
                candidate["correctness_matrix_sha256"] != matrix["matrix_sha256"]
            ):
                raise MatrixError(
                    f"backend {candidate['backend_id']} activation is bound to another source/matrix"
                )
            expected_receipts = correctness_receipt_set_sha256(
                receipt_paths,
                provider=candidate["provider"],
                device_target=candidate["device_target"],
                backend_id=candidate["backend_id"],
            )
            if candidate["correctness_receipts_sha256"] != expected_receipts:
                raise MatrixError(
                    f"backend {candidate['backend_id']} activation receipt-set hash does not verify"
                )
            activated[candidate_identity] = qualified_driver
    return activated

def project_matrix(
    inventory: dict[str, Any],
    catalog: dict[str, Any],
    backend_catalog: dict[str, Any],
    *,
    source_digests: dict[str, str] | None = None,
    candidate: dict[str, str] | None = None,
    vulkan_targets: dict[str, str] | None = None,
) -> dict[str, Any]:
    _require_schema(inventory, "openasr.model-family-inventory.v1", "architecture inventory")
    grouped = _public_models(catalog)
    if not candidate or not isinstance(candidate.get("release_version"), str):
        raise MatrixError("staging projection requires an immutable candidate contract")
    release_version = candidate["release_version"]
    vulkan_targets = _vulkan_target_contract(vulkan_targets)
    backend_candidates = _backend_candidates(
        backend_catalog, release_version, vulkan_targets=vulkan_targets
    )

    descriptors = inventory.get("families")
    if not isinstance(descriptors, list):
        raise MatrixError("architecture inventory must contain a families array")
    by_family = {
        item.get("catalog_family_id"): item
        for item in descriptors
        if isinstance(item, dict) and isinstance(item.get("catalog_family_id"), str)
    }
    missing = sorted(set(grouped) - set(by_family))
    if missing:
        raise MatrixError(f"public catalog families missing from architecture inventory: {missing}")

    cells: list[dict[str, Any]] = []
    for family in sorted(grouped):
        descriptor = by_family[family]
        topology = descriptor.get("topology")
        if not isinstance(topology, dict):
            raise MatrixError(f"family {family} has no topology projection")
        for provider, placement, modes in _advertised_providers(descriptor):
            provider_candidates = backend_candidates.get(provider, [])
            if not provider_candidates:
                continue
            capture_mode, scheduler_mode, graph_mode = _lane_policies(provider)
            tie_policy = _tie_policy(family)
            for model in sorted(grouped[family], key=lambda item: str(item["id"])):
                quants = [
                    item.get("quant")
                    for item in model.get("quants", [])
                    if isinstance(item, dict) and isinstance(item.get("quant"), str)
                ]
                recommended = model.get("recommended_quant")
                if isinstance(recommended, str) and recommended not in quants:
                    quants.append(recommended)
                if not quants:
                    raise MatrixError(f"public model {model['id']} has no concrete quant")
                for quant in sorted(set(quants)):
                    member = f"{model['id']}:{quant}"
                    for backend_candidate in provider_candidates:
                        if not backend_candidate["device_target"]:
                            # Generic Vulkan bytes remain PublishedInert until
                            # a real host contributes an exact capability class.
                            # Absence of a cell is fail-closed, not a pass.
                            continue
                        cells.append(
                            {
                            "family": family,
                            "model_id": model["id"],
                            "quant": quant,
                            "provider": provider,
                            "device_target": backend_candidate["device_target"],
                            "backend_id": backend_candidate["backend_id"],
                            "artifact_fingerprint": backend_candidate["artifact_fingerprint"],
                            "plugin_sha256": backend_candidate["plugin_sha256"],
                            "activation_modes": modes,
                            "placement": placement,
                            "capture_mode": capture_mode,
                            "scheduler_mode": scheduler_mode,
                            "graph_mode": graph_mode,
                            "topology": {
                                "decode_driver": topology.get("decode_driver"),
                                "decoder_state": topology.get("decoder_state"),
                                "block_stack": topology.get("block_stack"),
                            },
                            "kernel_coverage_bucket": {
                                "id": f"{descriptor.get('quantization', {}).get('tensor_classification', 'unknown')}:{quant}",
                                "members": [member],
                                "equivalence": "no cross-model or cross-quant merge; exact cell required",
                            },
                            "output_plan": {
                                "kind": "full_logits",
                                "requires_complete_output": True,
                                "tie_policy": tie_policy,
                            },
                            "reuse_modes": ["cold", "reuse"],
                            "required_receipt_classes": ["placement_resource", "token_transcript"],
                            "status": backend_candidate["activation_state"],
                            }
                        )
    if not cells:
        raise MatrixError("projection produced no public family/provider cells")
    required_candidate = ("release_subject", "release_version", "core_commit", "binary_sha256")
    for field in required_candidate:
        value = candidate.get(field)
        if field == "core_commit":
            valid = isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value)
        else:
            valid = _hex_digest(value) if field.endswith("sha256") else isinstance(value, str) and bool(value)
        if not valid:
            raise MatrixError(f"candidate contract field {field} is invalid")
    if not source_digests or set(source_digests) != {
        "architecture_inventory_sha256", "model_catalog_sha256", "backend_catalog_sha256"
    } or any(not _hex_digest(value) for value in source_digests.values()):
        raise MatrixError("matrix projection requires all canonical source digests")
    matrix = {
        "schema": SCHEMA,
        "artifact_contract": {
            "schema": "openasr.gpu-correctness-artifact.v1",
            "release_subject": candidate["release_subject"],
            "release_version": candidate["release_version"],
            "core_commit": candidate["core_commit"],
            "binary_sha256": candidate["binary_sha256"],
            "backend_candidates_sha256": _candidate_set_sha256(backend_candidates),
            "vulkan_qualification_targets": vulkan_targets,
            "source_digests": source_digests,
        },
        "source_digests": source_digests,
        # Build/package provenance belongs exclusively to
        # backend_hardware_evidence.py. This matrix owns runtime placement and
        # token/transcript correctness only.
        "required_global_receipt_classes": [],
        "cells": cells,
    }
    matrix["matrix_sha256"] = _canonical_sha(matrix)
    return matrix


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _driver_version(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 64
        and all(part and part.isascii() and part.isdigit() for part in value.split("."))
    )


def _parse_trace_events(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or events[0].get("schema") != "openasr.gpu-correctness-trace.v1" or events[0].get("event") != "header":
        raise MatrixError(f"{path} lacks the strict runtime trace header")
    header = events[0]
    tokens: dict[int, dict[str, Any]] = {}
    topks: dict[int, list[dict[str, Any]]] = {}
    margins: dict[int, float] = {}
    logits_digests: dict[int, dict[str, Any]] = {}
    for event in events[1:]:
        if event.get("schema") != "openasr.gpu-correctness-trace.v1" or not isinstance(event.get("step_index"), int):
            raise MatrixError(f"{path} contains an unversioned or malformed trace event")
        step = event["step_index"]
        if event.get("event") == "token":
            if (
                not isinstance(event.get("token_id"), int)
                or event["token_id"] < 0
                or event.get("is_eot") not in {0, 1, False, True}
                or step in tokens
            ):
                raise MatrixError(f"{path} contains an invalid token event")
            tokens[step] = event
        elif event.get("event") == "top_k":
            items = event.get("items")
            margin = event.get("top1_top2_margin")
            if (
                not isinstance(items, list)
                or len(items) < 2
                or any(
                    not isinstance(item, dict)
                    or type(item.get("token_id")) is not int
                    or item["token_id"] < 0
                    or not isinstance(item.get("value"), (int, float))
                    or not math.isfinite(float(item["value"]))
                    for item in items
                )
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or margin < 0
                or step in topks
            ):
                raise MatrixError(f"{path} contains an invalid top-k event")
            topks[step] = items
            margins[step] = float(margin)
        elif event.get("event") == "logits_digest":
            if (
                type(event.get("element_count")) is not int
                or event["element_count"] <= 0
                or not _hex_digest(event.get("sha256"))
                or event.get("non_finite_count") != 0
                or step in logits_digests
            ):
                raise MatrixError(f"{path} contains an invalid logits digest event")
            logits_digests[step] = event
        else:
            raise MatrixError(f"{path} contains an unknown trace event")
    if not tokens or set(tokens) != set(topks) or set(tokens) != set(logits_digests):
        raise MatrixError(
            f"{path} does not contain matching per-step token, top-k, and logits digest events"
        )
    return header, {
        "tokens": tokens,
        "topks": topks,
        "margins": margins,
        "logits_digests": logits_digests,
    }


def parse_trace_artifact(path: Path) -> dict[str, Any]:
    header, trace = _parse_trace_events(path)
    if (
        header.get("graph_mode") not in {"fresh_graph", "reusable_graph"}
        or not isinstance(header.get("provider"), str)
        or not isinstance(header.get("device_target"), str)
        or not isinstance(header.get("backend_id"), str)
        or not _driver_version(header.get("driver_version"))
        or not _hex_digest(header.get("artifact_fingerprint"))
        or not isinstance(header.get("device"), str)
    ):
        raise MatrixError(f"{path} has invalid runtime trace identity")
    return {
        "graph_mode": header["graph_mode"],
        "provider": header["provider"],
        "device_target": header["device_target"],
        "backend_id": header["backend_id"],
        "driver_version": header["driver_version"],
        "artifact_fingerprint": header["artifact_fingerprint"],
        "device": header["device"],
        "steps": sorted(trace["tokens"]),
        "token_ids": {
            step: event["token_id"] for step, event in trace["tokens"].items()
        },
        "topks": trace["topks"],
        "margins": trace["margins"],
        "logits_sha256": {
            step: event["sha256"]
            for step, event in trace["logits_digests"].items()
        },
    }


def parse_cpu_oracle_trace(path: Path) -> dict[str, Any]:
    header, trace = _parse_trace_events(path)
    if (
        header.get("provider") != "cpu"
        or header.get("graph_mode") not in {"fresh_graph", "reusable_graph"}
        or not isinstance(header.get("device"), str)
    ):
        raise MatrixError(f"{path} is not a CPU family-oracle trace")
    return {
        "graph_mode": header["graph_mode"],
        "device": header["device"],
        "steps": sorted(trace["tokens"]),
        "token_ids": {
            step: event["token_id"] for step, event in trace["tokens"].items()
        },
    }

def _evidence_for(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _read(path)
    if document.get("schema") != RECEIPT_SCHEMA:
        raise MatrixError(f"{path} is not a short-audio receipt")
    evidence = document.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("schema") != EVIDENCE_SCHEMA:
        raise MatrixError(f"{path} has no versioned correctness evidence")
    required = (
        "contract", "matrix_sha256", "candidate_release_subject", "core_commit",
        "catalog_digests", "family", "model_id", "quant", "topology",
        "provider", "device_target", "backend_id", "driver_version", "artifact_fingerprint", "device", "placement", "capture_mode", "scheduler_mode",
        "evidence_class", "artifacts", "result",
    )
    if any(field not in evidence for field in required):
        raise MatrixError(f"{path} has an incomplete strict artifact contract")
    if evidence.get("contract") != "openasr.gpu-correctness-artifact.v1" or evidence.get("result") != "pass":
        raise MatrixError(f"{path} has an invalid or non-passing artifact contract")
    if not _hex_digest(evidence.get("matrix_sha256")):
        raise MatrixError(f"{path} has no matrix digest")
    if not _hex_digest(evidence.get("artifact_fingerprint")):
        raise MatrixError(f"{path} has no exact backend artifact fingerprint")
    if not _driver_version(evidence.get("driver_version")):
        raise MatrixError(f"{path} has no exact driver version")
    core_commit = evidence.get("core_commit")
    if not isinstance(core_commit, str) or len(core_commit) != 40 or any(char not in "0123456789abcdef" for char in core_commit):
        raise MatrixError(f"{path} has invalid core commit")
    if evidence.get("capture_mode") not in {"disabled", "enabled", "unsupported"} or evidence.get("scheduler_mode") not in {"disabled", "enabled"}:
        raise MatrixError(f"{path} has invalid capture/scheduler identity")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict) or any(
        not isinstance(artifacts.get(name), dict) or not _hex_digest(artifacts[name].get("sha256"))
        for name in ("binary", "plugin", "pack", "fixture")
    ):
        raise MatrixError(f"{path} has incomplete artifact identity")
    return document, evidence


def _core_validate_qualification_receipts(receipt_paths: list[Path]) -> None:
    """Call the core-owned strict predicate once for the complete receipt set."""
    if not receipt_paths:
        return
    repository = Path(__file__).resolve().parents[2]
    command = [
        "cargo",
        "run",
        "--quiet",
        "--locked",
        "--manifest-path",
        str(repository / "Cargo.toml"),
        "-p",
        "openasr-cli",
        "--",
        "bench-receipt",
        "validate-qualification",
    ]
    for path in receipt_paths:
        command.extend(("--receipt", str(path)))
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repository,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MatrixError(
            "core qualification predicate rejected the receipt set"
            + (f": {detail}" if detail else "")
        )


def expected_receipt_keys(matrix: dict[str, Any]) -> tuple[
    set[ReceiptKey],
    dict[LaneKey, dict[str, Any]],
]:
    """Return exact-cell receipt keys. Untested lanes stay required; they are not skipped."""
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise MatrixError("correctness matrix has no cells")
    expected: set[ReceiptKey] = set()
    cell_by_lane: dict[LaneKey, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise MatrixError("correctness matrix contains a non-object cell")
        family, provider = cell.get("family"), cell.get("provider")
        model_id, quant = cell.get("model_id"), cell.get("quant")
        device_target, backend_id = cell.get("device_target"), cell.get("backend_id")
        modes = cell.get("reuse_modes")
        if not all(
            isinstance(value, str) and value
            for value in (family, provider, model_id, quant, device_target, backend_id)
        ) or modes != ["cold", "reuse"]:
            raise MatrixError("matrix cell lacks exact family/model/quant/provider/target/backend or cold/reuse requirements")
        if not is_provider_qualification_target(provider, device_target):
            raise MatrixError(
                f"matrix cell {family}/{model_id}:{quant} has a non-canonical provider target"
            )
        if not _hex_digest(cell.get("artifact_fingerprint")) or not _hex_digest(cell.get("plugin_sha256")):
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks exact backend bytes")
        for field in ("topology", "kernel_coverage_bucket", "output_plan"):
            if not isinstance(cell.get(field), dict):
                raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks {field}")
        if cell.get("capture_mode") not in {"disabled", "enabled", "unsupported"} or cell.get("scheduler_mode") not in {"disabled", "enabled"}:
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks capture/scheduler policy")
        if cell.get("graph_mode") not in {"fresh_graph", "reusable_graph"}:
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks graph-mode policy")
        plan = cell["output_plan"]
        if plan.get("kind") not in {"full_logits", "complete_scores", "native_first_max_token"} or plan.get("tie_policy") not in {"first_maximum", "last_maximum"}:
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks a concrete output plan")
        coverage = cell["kernel_coverage_bucket"]
        if not isinstance(coverage.get("members"), list) or coverage.get("members") != [f"{model_id}:{quant}"]:
            raise MatrixError("kernel coverage bucket silently merges cells")
        required_classes = cell.get("required_receipt_classes")
        if not isinstance(required_classes, list) or set(required_classes) != {"placement_resource", "token_transcript"}:
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks separate receipt classes")
        activation_modes = cell.get("activation_modes")
        if not isinstance(activation_modes, list) or any(mode not in {"auto", "explicit"} for mode in activation_modes):
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks Auto/explicit activation modes")
        for mode in modes:
            for evidence_class in required_classes:
                key = (
                    family,
                    model_id,
                    quant,
                    provider,
                    device_target,
                    backend_id,
                    mode,
                    evidence_class,
                )
                if key in expected:
                    raise MatrixError(f"duplicate correctness cell {key}")
                expected.add(key)
                cell_by_lane[(family, model_id, quant, provider, device_target, backend_id)] = cell
    return expected, cell_by_lane


def _bind_matrix_snapshots(
    matrix: dict[str, Any],
    *,
    inventory: dict[str, Any] | None,
    catalog: dict[str, Any] | None,
    backend_catalog: dict[str, Any] | None,
    source_digests: dict[str, str] | None,
) -> None:
    _require_schema(matrix, SCHEMA, "correctness matrix")
    claimed_matrix_sha = matrix.get("matrix_sha256")
    if not _hex_digest(claimed_matrix_sha):
        raise MatrixError("correctness matrix has no lowercase matrix_sha256")
    unsigned_matrix = dict(matrix)
    unsigned_matrix.pop("matrix_sha256", None)
    if _canonical_sha(unsigned_matrix) != claimed_matrix_sha:
        raise MatrixError("correctness matrix hash does not verify")
    contract = matrix.get("artifact_contract")
    if not isinstance(contract, dict) or contract.get("schema") != "openasr.gpu-correctness-artifact.v1":
        raise MatrixError("correctness matrix lacks strict artifact contract")
    if not isinstance(contract.get("release_version"), str) or not contract["release_version"]:
        raise MatrixError("correctness matrix lacks release_version")
    matrix_source_digests = matrix.get("source_digests")
    if contract.get("source_digests") != matrix_source_digests or not isinstance(matrix_source_digests, dict):
        raise MatrixError("matrix source digests are not bound into the artifact contract")
    if source_digests is None or source_digests != matrix_source_digests:
        raise MatrixError("matrix source digests do not match the exact staging snapshots")
    for field in ("binary_sha256", "backend_candidates_sha256"):
        if not _hex_digest(contract.get(field)):
            raise MatrixError(f"matrix artifact contract lacks {field}")
    core_commit = contract.get("core_commit")
    if not isinstance(core_commit, str) or len(core_commit) != 40 or any(char not in "0123456789abcdef" for char in core_commit):
        raise MatrixError("matrix artifact contract has invalid core_commit")
    if any(value is None for value in (inventory, catalog, backend_catalog)):
        raise MatrixError("validation requires the exact inventory, model catalog, and backend catalog snapshots")
    projected = project_matrix(
        inventory,
        catalog,
        backend_catalog,
        source_digests=source_digests,
        candidate={
            "release_subject": contract["release_subject"],
            "release_version": contract["release_version"],
            "core_commit": contract["core_commit"],
            "binary_sha256": contract["binary_sha256"],
        },
        vulkan_targets=_vulkan_target_contract(
            contract.get("vulkan_qualification_targets", {})
        ),
    )
    if projected != matrix:
        raise MatrixError("matrix does not match a fresh canonical projection of its source snapshots")


def closed_receipt_keys(
    matrix: dict[str, Any],
    receipt_paths: list[Path],
    *,
    inventory: dict[str, Any] | None = None,
    catalog: dict[str, Any] | None = None,
    backend_catalog: dict[str, Any] | None = None,
    source_digests: dict[str, str] | None = None,
    trace_paths: list[Path] | None = None,
    activated_candidates: dict[tuple[str, str, str], str] | None = None,
    qualification_validator: Callable[[list[Path]], None] | None = None,
) -> tuple[set[ReceiptKey], set[str]]:
    """Validate each receipt against exact cells without skipping missing lanes."""
    _bind_matrix_snapshots(
        matrix,
        inventory=inventory,
        catalog=catalog,
        backend_catalog=backend_catalog,
        source_digests=source_digests,
    )
    (qualification_validator or _core_validate_qualification_receipts)(receipt_paths)
    trace_paths = trace_paths or []
    trace_hashes = {path.name: _sha256(path) for path in trace_paths}
    trace_semantics = {path.name: parse_trace_artifact(path) for path in trace_paths}
    expected, cell_by_lane = expected_receipt_keys(matrix)
    receipts: set[ReceiptKey] = set()
    classes: set[str] = set()
    for path in receipt_paths:
        document, evidence = _evidence_for(path)
        evidence_class = evidence.get("evidence_class")
        if not isinstance(evidence_class, str):
            raise MatrixError(f"{path} has no evidence class")
        classes.add(evidence_class)
        if evidence_class == "build_packaging":
            continue
        if evidence_class not in {"placement_resource", "token_transcript"}:
            raise MatrixError(f"{path} has unknown evidence class {evidence_class!r}")
        family = evidence.get("family")
        provider = evidence.get("provider")
        model_id = evidence.get("model_id")
        quant = evidence.get("quant")
        device_target = evidence.get("device_target")
        backend_id = evidence.get("backend_id")
        if not all(
            isinstance(value, str) and value
            for value in (family, provider, model_id, quant, device_target, backend_id)
        ):
            raise MatrixError(f"{path} lacks exact family/provider/target/backend/model/quant identity")
        lane_key = (family, model_id, quant, provider, device_target, backend_id)
        lane = cell_by_lane.get(lane_key)
        if lane is None:
            raise MatrixError(f"{path} is not bound to a projected exact target/backend lane")
        if evidence.get("artifact_fingerprint") != lane["artifact_fingerprint"]:
            raise MatrixError(f"{path} backend artifact fingerprint does not match its matrix cell")
        candidate_identity = (provider, device_target, backend_id)
        if activated_candidates is not None and candidate_identity in activated_candidates:
            if evidence.get("driver_version") != activated_candidates[candidate_identity]:
                raise MatrixError(f"{path} driver version does not match activated evidence")
        contract = matrix["artifact_contract"]
        if evidence.get("matrix_sha256") != matrix.get("matrix_sha256") or evidence.get("candidate_release_subject") != contract.get("release_subject") or evidence.get("core_commit") != contract.get("core_commit"):
            raise MatrixError(f"{path} is stale or bound to another release subject")
        expected_catalog_digests = {
            "inventory_sha256": contract["source_digests"]["architecture_inventory_sha256"],
            "model_catalog_sha256": contract["source_digests"]["model_catalog_sha256"],
            "backend_catalog_sha256": contract["source_digests"]["backend_catalog_sha256"],
        }
        if evidence.get("catalog_digests") != expected_catalog_digests:
            raise MatrixError(f"{path} is bound to another inventory or catalog snapshot")
        if evidence.get("topology") != lane["topology"].get("decoder_state"):
            raise MatrixError(f"{path} topology does not match its matrix cell")
        if evidence.get("capture_mode") != lane["capture_mode"] or evidence.get("scheduler_mode") != lane["scheduler_mode"]:
            raise MatrixError(f"{path} capture/scheduler identity does not match its matrix cell")
        artifacts = evidence["artifacts"]
        if (
            artifacts["binary"].get("sha256") != contract["binary_sha256"]
            or artifacts["plugin"].get("sha256") != lane["plugin_sha256"]
        ):
            raise MatrixError(f"{path} binary/plugin identity does not match the candidate")
        pack = document.get("pack")
        if not isinstance(pack, dict) or pack.get("model_id") != f"{model_id}:{quant}" or pack.get("quant") != quant:
            raise MatrixError(f"{path} pack identity does not match its matrix cell")
        if evidence["artifacts"]["pack"].get("sha256") != pack.get("content_sha256"):
            raise MatrixError(f"{path} pack artifact hash does not match the receipt pack")
        audio = document.get("audio")
        if not isinstance(audio, dict) or evidence["artifacts"]["fixture"].get("sha256") != audio.get("sha256"):
            raise MatrixError(f"{path} fixture artifact hash does not match the receipt audio")
        execution = evidence.get("execution")
        if not isinstance(family, str) or not isinstance(provider, str) or not isinstance(execution, dict):
            raise MatrixError(f"{path} lacks family/provider/execution identity")
        mode = execution.get("mode")
        if mode not in {"cold", "reuse"}:
            raise MatrixError(f"{path} has invalid execution mode")
        expected_process_state = (
            ("cold", "empty") if mode == "cold" else ("warm", "populated")
        )
        run = document.get("run")
        if not isinstance(run, dict) or (
            run.get("warmup"), run.get("cache_state")
        ) != expected_process_state:
            raise MatrixError(f"{path} execution mode contradicts process cache state")
        decode_diagnostics = document.get("decode_diagnostics")
        if (
            not isinstance(decode_diagnostics, dict)
            or decode_diagnostics.get("reuse_mode") != lane["graph_mode"]
        ):
            raise MatrixError(f"{path} graph mode does not match its matrix cell")
        key = (
            family,
            model_id,
            quant,
            provider,
            device_target,
            backend_id,
            mode,
            evidence_class,
        )
        if key not in expected:
            raise MatrixError(f"{path} is not bound to a projected matrix cell")
        if evidence_class == "token_transcript":
            output_plan = evidence.get("output_plan")
            oracle = evidence.get("family_oracle")
            if not isinstance(output_plan, dict) or not isinstance(oracle, dict):
                raise MatrixError(f"{path} token evidence lacks output plan or family oracle")
            if output_plan != lane["output_plan"]:
                raise MatrixError(f"{path} output plan does not match its matrix cell")
            if not isinstance(oracle.get("family"), str) or oracle.get("family") != family or oracle.get("tie_policy") != lane["output_plan"]["tie_policy"]:
                raise MatrixError(f"{path} family oracle does not match its matrix cell")
            trace = evidence.get("trace")
            if not isinstance(trace, dict) or not isinstance(trace.get("token_trace"), dict) or not _hex_digest(trace["token_trace"].get("sha256")):
                raise MatrixError(f"{path} token evidence lacks a non-empty trace artifact hash")
            token_trace = trace["token_trace"]
            if token_trace.get("label") not in trace_hashes or trace_hashes[token_trace["label"]] != token_trace.get("sha256"):
                raise MatrixError(f"{path} token trace artifact content hash does not verify")
            semantics = trace_semantics[token_trace["label"]]
            if (
                semantics["graph_mode"] != lane["graph_mode"]
                or semantics["provider"] != provider
                or semantics["device_target"] != device_target
                or semantics["backend_id"] != backend_id
                or semantics["driver_version"] != evidence.get("driver_version")
                or semantics["artifact_fingerprint"] != evidence.get("artifact_fingerprint")
                or semantics["device"] != evidence.get("device")
            ):
                raise MatrixError(f"{path} trace header does not match receipt execution identity")
            if lane["output_plan"]["requires_complete_output"]:
                logits = trace.get("logits")
                if not isinstance(logits, dict) or not _hex_digest(logits.get("sha256")):
                    raise MatrixError(f"{path} complete-output plan lacks logits trace content hash")
                if logits.get("label") not in trace_hashes or trace_hashes[logits["label"]] != logits.get("sha256"):
                    raise MatrixError(f"{path} logits trace artifact content hash does not verify")
            top_k = trace.get("top_k")
            if not isinstance(top_k, list) or not top_k or len(top_k) > 32 or any(not isinstance(item, dict) or not isinstance(item.get("value"), (int, float)) for item in top_k):
                raise MatrixError(f"{path} has invalid top-k summary")
            margin = trace.get("top1_top2_margin")
            if not isinstance(margin, (int, float)) or margin < 0:
                raise MatrixError(f"{path} has invalid top-1/top-2 margin")
        if key in receipts:
            raise MatrixError(f"duplicate evidence for correctness cell {key}")
        receipts.add(key)
    unknown = sorted(receipts - expected)
    if unknown:
        raise MatrixError(f"receipts are not bound to projected matrix cells: {unknown}")
    return receipts, classes


def lane_activation_modes(
    matrix: dict[str, Any],
    closed_keys: set[ReceiptKey],
    *,
    activated_candidates: set[tuple[str, str, str]] | None = None,
) -> dict[LaneKey, tuple[str, ...]]:
    """Return Auto/explicit modes only for lanes with complete receipts.

    Advertised CUDA, physical Vulkan, and HIP cells stay in the map when
    untested. Empty modes mean the lane exists and is not selectable.
    """
    expected, cell_by_lane = expected_receipt_keys(matrix)
    activated_candidates = activated_candidates or set()
    allowlist: dict[LaneKey, tuple[str, ...]] = {}
    for lane, cell in cell_by_lane.items():
        needed = {key for key in expected if key[:6] == lane}
        modes = (
            tuple(cell["activation_modes"])
            if lane[3:] in activated_candidates and needed <= closed_keys
            else ()
        )
        allowlist[lane] = modes
    return allowlist


def require_activation(
    matrix: dict[str, Any],
    closed_keys: set[ReceiptKey],
    *,
    provider: str,
    device_target: str,
    backend_id: str,
    mode: str,
    activated_candidates: set[tuple[str, str, str]] | None = None,
) -> None:
    """Fail closed if Auto or explicit selection is requested for an unproven cell."""
    if mode not in {"auto", "explicit"}:
        raise MatrixError(f"{mode} is not an activation mode")
    allowlist = lane_activation_modes(
        matrix,
        closed_keys,
        activated_candidates=activated_candidates,
    )
    matching = [
        key
        for key in allowlist
        if key[3:] == (provider, device_target, backend_id)
    ]
    if not matching:
        raise MatrixError(
            f"{provider}/{device_target}/{backend_id} is not a projected activation lane"
        )
    blocked = [key for key in matching if mode not in allowlist[key]]
    if blocked:
        raise MatrixError(
            f"{provider}/{device_target}/{backend_id} {mode} is not selectable without closed correctness receipts: {blocked}"
        )


_UNAVAILABLE_HOST_ALIASES = {
    "cuda": ("windows_cuda", "cuda"),
    "vulkan": ("vulkan",),
    "hip": ("hip",),
}


def parse_hardware_unavailable(path: Path) -> dict[str, str]:
    """Record missing CUDA/Vulkan/HIP hosts. Unavailable is not a pass."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MatrixError(f"cannot read hardware-unavailable record {path}: {error}") from error
    lowered = text.lower()
    if "not a pass" not in lowered:
        raise MatrixError(f"{path} does not record that missing hosts are not passes")
    statuses: dict[str, str] = {}
    for provider, aliases in _UNAVAILABLE_HOST_ALIASES.items():
        matched = False
        for line in text.splitlines():
            stripped = line.strip().lower()
            if any(stripped.startswith(f"{alias}:") for alias in aliases):
                if "pass" in stripped.split(":", 1)[-1] or "proven" in stripped.split(":", 1)[-1]:
                    raise MatrixError(
                        f"{path} records {provider} as a pass without receipts"
                    )
                if "unavailable" not in stripped:
                    raise MatrixError(f"{path} does not record {provider} as unavailable")
                statuses[provider] = "unavailable"
                matched = True
                break
        if not matched:
            raise MatrixError(f"{path} does not record {provider} as unavailable")
    return statuses


def require_desktop_plugin_switch(path: Path) -> None:
    """Fail closed unless a desktop plugin-switch log records an actual PASS.

    `skipped=true` and `result=FAIL` are not passes. A missing host is not
    selectable.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MatrixError(f"cannot read desktop plugin-switch log {path}: {error}") from error
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(" ") or line.startswith("\t") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"result", "skipped", "reason", "host_mode"}:
            fields[key] = value.strip()
    if fields.get("skipped", "").lower() == "true":
        raise MatrixError("desktop plugin-switch skip is not a pass")
    if fields.get("result") != "PASS":
        raise MatrixError(
            "desktop plugin-switch is not selectable: "
            f"result={fields.get('result')!r} reason={fields.get('reason')!r}"
        )


def require_untested_hosts_not_activatable(
    matrix: dict[str, Any],
    closed_keys: set[ReceiptKey],
    hardware_unavailable: Path,
) -> None:
    """CUDA/Vulkan/HIP stay projected and unselectable without receipts."""
    statuses = parse_hardware_unavailable(hardware_unavailable)
    for provider, status in statuses.items():
        if status != "unavailable":
            raise MatrixError(f"{provider} missing-host record is not fail-closed")
        lanes = [lane for lane in lane_activation_modes(matrix, closed_keys) if lane[3] == provider]
        for lane in lanes:
            require_activation(
                matrix,
                closed_keys,
                provider=provider,
                device_target=lane[4],
                backend_id=lane[5],
                mode="auto",
            )
            require_activation(
                matrix,
                closed_keys,
                provider=provider,
                device_target=lane[4],
                backend_id=lane[5],
                mode="explicit",
            )


def validate_matrix(
    matrix: dict[str, Any],
    receipt_paths: list[Path],
    *,
    inventory: dict[str, Any] | None = None,
    catalog: dict[str, Any] | None = None,
    backend_catalog: dict[str, Any] | None = None,
    activation_catalog: dict[str, Any] | None = None,
    source_digests: dict[str, str] | None = None,
    trace_paths: list[Path] | None = None,
    qualification_validator: Callable[[list[Path]], None] | None = None,
) -> None:
    expected, _cell_by_lane = expected_receipt_keys(matrix)
    if backend_catalog is None or source_digests is None:
        raise MatrixError("validation requires backend catalog and source digests")
    activated_candidate_drivers = _validate_activation_catalog(
        matrix,
        activation_catalog or backend_catalog,
        source_digests,
        receipt_paths,
    )
    receipts, classes = closed_receipt_keys(
        matrix,
        receipt_paths,
        inventory=inventory,
        catalog=catalog,
        backend_catalog=backend_catalog,
        source_digests=source_digests,
        trace_paths=trace_paths,
        activated_candidates=activated_candidate_drivers,
        qualification_validator=qualification_validator,
    )
    activated_candidates = set(activated_candidate_drivers)
    _expected, cell_by_lane = expected_receipt_keys(matrix)
    activated_lanes = {
        lane for lane in cell_by_lane if lane[3:] in activated_candidates
    }
    required = {key for key in expected if key[:6] in activated_lanes}
    missing = sorted(required - receipts)
    if missing:
        raise MatrixError(f"correctness matrix is incomplete; missing receipts: {missing}")
    required_global = matrix.get("required_global_receipt_classes", [])
    if not isinstance(required_global, list) or any(not isinstance(item, str) for item in required_global):
        raise MatrixError("correctness matrix has invalid global receipt requirements")
    missing_global = sorted(set(required_global) - classes)
    if missing_global:
        raise MatrixError(f"correctness matrix is incomplete; missing global receipts: {missing_global}")
    for cell in matrix["cells"]:
        identity = (cell["provider"], cell["device_target"], cell["backend_id"])
        if identity not in activated_candidates:
            continue
        provider = cell["provider"]
        for mode in cell["activation_modes"]:
            require_activation(
                matrix,
                receipts,
                provider=provider,
                device_target=cell["device_target"],
                backend_id=cell["backend_id"],
                mode=mode,
                activated_candidates=activated_candidates,
            )


def _hardware_qualification(
    *,
    matrix: dict[str, Any],
    source_backend_catalog: dict[str, Any],
    current_activation_catalog: dict[str, Any],
    source_digests: dict[str, str],
    backend_id: str,
    entry_paths: list[Path],
    hardware_evidence_paths: list[Path],
    hardware_raw_audit_paths: list[Path],
) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    """Return one exact candidate and its independently verified hardware bindings."""
    import backend_hardware_evidence as hardware_gate

    contract = matrix.get("artifact_contract")
    if not isinstance(contract, dict):
        raise MatrixError("matrix has no artifact contract")
    release_version = contract.get("release_version")
    if not isinstance(release_version, str) or not release_version:
        raise MatrixError("matrix has no release version")

    vulkan_targets = _vulkan_target_contract(
        contract.get("vulkan_qualification_targets", {})
    )
    source_candidates = _backend_candidates(
        source_backend_catalog, release_version, vulkan_targets=vulkan_targets
    )
    current_candidates = _backend_candidates(
        current_activation_catalog, release_version, vulkan_targets=vulkan_targets
    )
    if _candidate_set_sha256(source_candidates) != _candidate_set_sha256(current_candidates):
        raise MatrixError("current activation catalog changed immutable backend candidate bytes")
    matching_candidates = [
        candidate
        for values in source_candidates.values()
        for candidate in values
        if candidate["backend_id"] == backend_id
    ]
    if len(matching_candidates) != 1:
        raise MatrixError(f"backend_id {backend_id!r} is not one exact release candidate")
    candidate = matching_candidates[0]

    approved_paths = hardware_gate.approved_entry_paths(
        entry_paths, hardware_evidence_paths, hardware_raw_audit_paths
    )
    approved_identity = []
    for path in approved_paths:
        _entry, identity = hardware_gate._entry_identity(path)
        if identity.backend_id == backend_id:
            approved_identity.append(identity)
    if len(approved_identity) != 1:
        raise MatrixError(f"backend {backend_id} lacks one exact hardware-approved entry")
    identity = approved_identity[0]
    expected_hardware_identity = (
        candidate["provider"],
        candidate["device_target"],
        candidate["backend_id"],
        candidate["artifact_fingerprint"],
        candidate["plugin_sha256"],
        release_version,
    )
    matching_evidence: list[dict[str, Any]] = []
    for path in hardware_evidence_paths:
        document, evidence_identity = hardware_gate._common_evidence_identity(path)
        if (
            evidence_identity[:6] == expected_hardware_identity
            and identity.matches_evidence(evidence_identity)
        ):
            matching_evidence.append(document)
    if len(matching_evidence) != 1:
        raise MatrixError(
            f"backend {backend_id} lacks one unambiguous hardware evidence summary"
        )
    hardware_evidence_sha256 = hardware_gate._canonical_sha256(matching_evidence[0])
    hardware_driver_version = matching_evidence[0].get("driver_version")
    if not _driver_version(hardware_driver_version):
        raise MatrixError(f"backend {backend_id} hardware evidence has no exact driver")
    desired_qualified = {
        "state": "qualified",
        "qualification_source_catalog_sha256": source_digests[
            "backend_catalog_sha256"
        ],
        "hardware_evidence_sha256": hardware_evidence_sha256,
        "qualified_device_target": candidate["device_target"],
        "qualified_driver_version": hardware_driver_version,
    }
    return candidate, approved_paths, desired_qualified


def _catalog_backend_entry(
    catalog: dict[str, Any], backend_id: str
) -> dict[str, Any]:
    matches = [
        entry
        for entry in catalog.get("backends", [])
        if isinstance(entry, dict) and entry.get("id") == backend_id
    ]
    if len(matches) != 1:
        raise MatrixError(f"activation catalog does not contain exactly one {backend_id}")
    return matches[0]


def _require_exact_model_snapshot(
    activation_catalog: dict[str, Any],
    model_catalog: dict[str, Any],
    *,
    label: str,
) -> None:
    """Keep a backend-wide activation bound to every advertised model byte.

    Backend activation is catalog-wide, while correctness evidence is projected
    over the public model list.  Accepting a catalog whose model projection
    changed after qualification would silently expose untested families or
    quants through an otherwise valid backend binding.
    """

    expected = model_catalog.get("models")
    actual = activation_catalog.get("models")
    if not isinstance(expected, list) or not expected:
        raise MatrixError("qualification model snapshot has no public models")
    if actual != expected:
        raise MatrixError(
            f"{label} model projection differs from the qualification snapshot"
        )


def _catalog_transition_payload(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return the signed catalog content whose mutation needs policy review.

    `generated_at` is refreshed when the already-verified transition is signed;
    every other top-level field remains part of the immutable replay.
    """

    payload = copy.deepcopy(catalog)
    payload.pop("generated_at", None)
    return payload


def revoke_catalog_backend(
    *, current_activation_catalog: dict[str, Any], backend_id: str
) -> dict[str, Any]:
    """Derive the fail-safe one-way transition to Revoked.

    Existing qualification/correctness bindings are deliberately preserved as
    audit facts.  Runtime selection keys only on the signed state, so a revoked
    entry becomes unusable immediately without erasing why it was once trusted.
    """

    result = copy.deepcopy(current_activation_catalog)
    entry = _catalog_backend_entry(result, backend_id)
    existing = entry.get("activation", {"state": "published-inert"})
    if not isinstance(existing, dict):
        raise MatrixError(f"backend {backend_id} has malformed activation state")
    known_fields = {
        "state",
        "qualification_source_catalog_sha256",
        "hardware_evidence_sha256",
        "qualified_device_target",
        "qualified_driver_version",
        "correctness_matrix_sha256",
        "correctness_receipts_sha256",
    }
    unknown = sorted(set(existing) - known_fields)
    if unknown:
        raise MatrixError(
            f"backend {backend_id} activation has unknown fields: {unknown}"
        )
    state = existing.get("state", "published-inert")
    if state not in {"published-inert", "qualified", "activated", "revoked"}:
        raise MatrixError(f"backend {backend_id} has unsupported activation state {state!r}")
    digest_names = (
        "qualification_source_catalog_sha256",
        "hardware_evidence_sha256",
        "correctness_matrix_sha256",
        "correctness_receipts_sha256",
    )
    digests = [existing.get(name) for name in digest_names]
    target = existing.get("qualified_device_target")
    driver = existing.get("qualified_driver_version")
    provider = entry.get("vendor")
    signed_targets = entry.get("targets", [])
    target_matches_entry = (
        provider == "vulkan" and signed_targets == []
    ) or (
        provider in {"cuda", "hip"}
        and isinstance(target, str)
        and signed_targets == [target]
    )
    hardware_complete = (
        all(_hex_digest(value) for value in digests[:2])
        and is_provider_qualification_target(provider, target)
        and target_matches_entry
        and _driver_version(driver)
    )
    activation_complete = hardware_complete and all(
        _hex_digest(value) for value in digests[2:]
    )
    hardware_only = hardware_complete and not any(digests[2:])
    empty = not any(digests) and target is None and driver is None
    valid_shape = (
        (state == "published-inert" and empty)
        or (state == "qualified" and hardware_only)
        or (state == "activated" and activation_complete)
        or (state == "revoked" and (empty or hardware_only or activation_complete))
    )
    if not valid_shape:
        raise MatrixError(f"backend {backend_id} activation bindings are incomplete")
    existing["state"] = "revoked"
    entry["activation"] = existing
    return result


def verify_catalog_backend_revocation(
    *,
    current_activation_catalog: dict[str, Any],
    candidate_activation_catalog: dict[str, Any],
    backend_id: str,
) -> None:
    """Replay one exact revocation and reject unrelated signed mutations."""

    expected = revoke_catalog_backend(
        current_activation_catalog=current_activation_catalog,
        backend_id=backend_id,
    )
    candidate_entry = _catalog_backend_entry(candidate_activation_catalog, backend_id)
    activation = candidate_entry.get("activation")
    if not isinstance(activation, dict) or activation.get("state") != "revoked":
        raise MatrixError("candidate revocation catalog does not revoke the requested backend")
    if _catalog_transition_payload(expected) != _catalog_transition_payload(
        candidate_activation_catalog
    ):
        raise MatrixError("candidate catalog is not the exact revocation replay")


def qualify_catalog_backend(
    *,
    matrix: dict[str, Any],
    source_backend_catalog: dict[str, Any],
    current_activation_catalog: dict[str, Any],
    source_digests: dict[str, str],
    backend_id: str,
    entry_paths: list[Path],
    hardware_evidence_paths: list[Path],
    hardware_raw_audit_paths: list[Path],
) -> dict[str, Any]:
    """Derive PublishedInert -> Qualified without token-correctness authority."""
    import backend_hardware_evidence as hardware_gate

    current_entry = _catalog_backend_entry(current_activation_catalog, backend_id)
    current_activation = current_entry.get("activation", {"state": "published-inert"})
    if not isinstance(current_activation, dict):
        raise MatrixError(f"backend {backend_id} has malformed activation state")
    current_state = current_activation.get("state", "published-inert")
    if current_state == "revoked":
        raise MatrixError(f"revoked backend {backend_id} cannot be requalified")
    if current_state == "activated":
        raise MatrixError(f"activated backend {backend_id} cannot return to qualified")
    if current_state not in {"published-inert", "qualified"}:
        raise MatrixError(
            f"backend {backend_id} has unsupported activation state {current_state!r}"
        )

    _candidate, approved_paths, desired_qualified = _hardware_qualification(
        matrix=matrix,
        source_backend_catalog=source_backend_catalog,
        current_activation_catalog=current_activation_catalog,
        source_digests=source_digests,
        backend_id=backend_id,
        entry_paths=entry_paths,
        hardware_evidence_paths=hardware_evidence_paths,
        hardware_raw_audit_paths=hardware_raw_audit_paths,
    )
    result = copy.deepcopy(current_activation_catalog)
    entry = _catalog_backend_entry(result, backend_id)
    existing = entry.get("activation", {"state": "published-inert"})
    if not isinstance(existing, dict):
        raise MatrixError(f"backend {backend_id} has malformed activation state")
    state = existing.get("state", "published-inert")
    if state == "qualified" and existing != desired_qualified:
        raise MatrixError(f"backend {backend_id} qualified state has different bindings")
    entry["activation"] = desired_qualified

    _validate_activation_catalog(matrix, result, source_digests, [])
    with tempfile.TemporaryDirectory(prefix="openasr-qualified-catalog-") as temp:
        candidate_path = Path(temp) / "catalog.json"
        candidate_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        hardware_gate.verify_catalog_policy(
            candidate_path,
            matrix["artifact_contract"]["release_version"],
            approved_paths,
            hardware_evidence_paths,
        )
    return result


def activate_catalog_backend(
    *,
    matrix: dict[str, Any],
    inventory: dict[str, Any],
    model_catalog: dict[str, Any],
    source_backend_catalog: dict[str, Any],
    current_activation_catalog: dict[str, Any],
    source_digests: dict[str, str],
    backend_id: str,
    entry_paths: list[Path],
    hardware_evidence_paths: list[Path],
    hardware_raw_audit_paths: list[Path],
    receipt_paths: list[Path],
    trace_paths: list[Path],
    qualification_validator: Callable[[list[Path]], None] | None = None,
) -> dict[str, Any]:
    """Derive Qualified -> Activated after the independent token gate closes.

    This function does not sign or deploy.  It is the sole catalog mutation
    seam: hardware approval remains owned by `backend_hardware_evidence`, token
    correctness remains owned by this module, and an inert candidate is never
    promoted through both states in one invocation.
    """
    import backend_hardware_evidence as hardware_gate

    _require_exact_model_snapshot(
        current_activation_catalog,
        model_catalog,
        label="current activation catalog",
    )
    current_entry = _catalog_backend_entry(current_activation_catalog, backend_id)
    current_activation = current_entry.get("activation", {"state": "published-inert"})
    if not isinstance(current_activation, dict):
        raise MatrixError(f"backend {backend_id} has malformed activation state")
    current_state = current_activation.get("state", "published-inert")
    if current_state == "revoked":
        raise MatrixError(f"revoked backend {backend_id} cannot be reactivated")
    if current_state == "published-inert":
        raise MatrixError(
            f"backend {backend_id} must be independently qualified before activation"
        )
    if current_state not in {"qualified", "activated"}:
        raise MatrixError(
            f"backend {backend_id} has unsupported activation state {current_state!r}"
        )

    candidate, approved_paths, desired_qualified = _hardware_qualification(
        matrix=matrix,
        source_backend_catalog=source_backend_catalog,
        current_activation_catalog=current_activation_catalog,
        source_digests=source_digests,
        backend_id=backend_id,
        entry_paths=entry_paths,
        hardware_evidence_paths=hardware_evidence_paths,
        hardware_raw_audit_paths=hardware_raw_audit_paths,
    )
    release_version = matrix["artifact_contract"]["release_version"]
    receipts_sha256 = correctness_receipt_set_sha256(
        receipt_paths,
        provider=candidate["provider"],
        device_target=candidate["device_target"],
        backend_id=backend_id,
    )
    desired_activation = {
        "state": "activated",
        "qualification_source_catalog_sha256": source_digests[
            "backend_catalog_sha256"
        ],
        "hardware_evidence_sha256": desired_qualified["hardware_evidence_sha256"],
        "qualified_device_target": candidate["device_target"],
        "qualified_driver_version": desired_qualified["qualified_driver_version"],
        "correctness_matrix_sha256": matrix.get("matrix_sha256"),
        "correctness_receipts_sha256": receipts_sha256,
    }

    result = copy.deepcopy(current_activation_catalog)
    entry = _catalog_backend_entry(result, backend_id)
    existing = entry.get("activation", {"state": "published-inert"})
    if not isinstance(existing, dict):
        raise MatrixError(f"backend {backend_id} has malformed activation state")
    state = existing.get("state", "published-inert")
    if state == "qualified" and existing != desired_qualified:
        raise MatrixError(f"backend {backend_id} qualified state has different bindings")
    if state == "activated" and existing != desired_activation:
        raise MatrixError(f"backend {backend_id} activation is immutable for this source")
    entry["activation"] = desired_activation

    with tempfile.TemporaryDirectory(prefix="openasr-activation-catalog-") as temp:
        candidate_path = Path(temp) / "catalog.json"
        candidate_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        hardware_gate.verify_catalog_policy(
            candidate_path,
            release_version,
            approved_paths,
            hardware_evidence_paths,
        )
    validate_matrix(
        matrix,
        receipt_paths,
        inventory=inventory,
        catalog=model_catalog,
        backend_catalog=source_backend_catalog,
        activation_catalog=result,
        source_digests=source_digests,
        trace_paths=trace_paths,
        qualification_validator=qualification_validator,
    )
    return result


def verify_catalog_backend_transition(
    *,
    matrix: dict[str, Any],
    inventory: dict[str, Any],
    model_catalog: dict[str, Any],
    source_backend_catalog: dict[str, Any],
    current_activation_catalog: dict[str, Any],
    candidate_activation_catalog: dict[str, Any],
    source_digests: dict[str, str],
    backend_id: str,
    entry_paths: list[Path],
    hardware_evidence_paths: list[Path],
    hardware_raw_audit_paths: list[Path],
    receipt_paths: list[Path],
    trace_paths: list[Path],
    qualification_validator: Callable[[list[Path]], None] | None = None,
) -> None:
    """Replay both immutable transitions and reject every unrelated mutation."""

    _require_exact_model_snapshot(
        current_activation_catalog,
        model_catalog,
        label="current activation catalog",
    )
    _require_exact_model_snapshot(
        candidate_activation_catalog,
        model_catalog,
        label="candidate activation catalog",
    )

    if current_activation_catalog.get("backends") == candidate_activation_catalog.get(
        "backends"
    ):
        if _catalog_transition_payload(current_activation_catalog) != _catalog_transition_payload(
            candidate_activation_catalog
        ):
            raise MatrixError("idempotent activation replay contains unrelated mutations")
        entry = _catalog_backend_entry(candidate_activation_catalog, backend_id)
        activation = entry.get("activation")
        if not isinstance(activation, dict) or activation.get("state") != "activated":
            raise MatrixError("idempotent activation replay requires an activated backend")
        import backend_hardware_evidence as hardware_gate

        _candidate, approved_paths, _desired = _hardware_qualification(
            matrix=matrix,
            source_backend_catalog=source_backend_catalog,
            current_activation_catalog=candidate_activation_catalog,
            source_digests=source_digests,
            backend_id=backend_id,
            entry_paths=entry_paths,
            hardware_evidence_paths=hardware_evidence_paths,
            hardware_raw_audit_paths=hardware_raw_audit_paths,
        )
        with tempfile.TemporaryDirectory(prefix="openasr-idempotent-activation-") as temp:
            candidate_path = Path(temp) / "catalog.json"
            candidate_path.write_text(
                json.dumps(candidate_activation_catalog, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            hardware_gate.verify_catalog_policy(
                candidate_path,
                matrix["artifact_contract"]["release_version"],
                approved_paths,
                hardware_evidence_paths,
            )
        validate_matrix(
            matrix,
            receipt_paths,
            inventory=inventory,
            catalog=model_catalog,
            backend_catalog=source_backend_catalog,
            activation_catalog=candidate_activation_catalog,
            source_digests=source_digests,
            trace_paths=trace_paths,
            qualification_validator=qualification_validator,
        )
        return

    qualified = qualify_catalog_backend(
        matrix=matrix,
        source_backend_catalog=source_backend_catalog,
        current_activation_catalog=current_activation_catalog,
        source_digests=source_digests,
        backend_id=backend_id,
        entry_paths=entry_paths,
        hardware_evidence_paths=hardware_evidence_paths,
        hardware_raw_audit_paths=hardware_raw_audit_paths,
    )
    activated = activate_catalog_backend(
        matrix=matrix,
        inventory=inventory,
        model_catalog=model_catalog,
        source_backend_catalog=source_backend_catalog,
        current_activation_catalog=qualified,
        source_digests=source_digests,
        backend_id=backend_id,
        entry_paths=entry_paths,
        hardware_evidence_paths=hardware_evidence_paths,
        hardware_raw_audit_paths=hardware_raw_audit_paths,
        receipt_paths=receipt_paths,
        trace_paths=trace_paths,
        qualification_validator=qualification_validator,
    )
    if _catalog_transition_payload(activated) != _catalog_transition_payload(
        candidate_activation_catalog
    ):
        raise MatrixError(
            "candidate activation catalog is not the exact qualify-then-activate replay"
        )


def _require_runtime_placement(
    receipt: dict[str, Any], *, provider: str, scheduler_mode: str
) -> None:
    observed = receipt.get("observed_placement")
    if not isinstance(observed, dict):
        raise MatrixError("GPU receipt lacks observed placement")
    direct = observed.get("direct_graph_computes")
    scheduled = observed.get("scheduler_graph_computes")
    if (
        type(direct) is not int
        or direct <= 0
        or type(scheduled) is not int
        or (scheduler_mode == "disabled" and scheduled != 0)
        or observed.get("fallback_node_samples_by_backend") not in (None, {})
    ):
        raise MatrixError("GPU receipt does not prove fail-closed graph placement")
    compute = observed.get("observed_compute_nodes_by_backend")
    provider_tags = {
        "cuda": ("cuda", "nvidia"),
        "hip": ("hip", "rocm"),
        "vulkan": ("vulkan",),
    }[provider]
    if (
        not isinstance(compute, dict)
        or not compute
        or any(type(count) is not int or count <= 0 for count in compute.values())
        or any(
            not any(tag in str(name).lower() for tag in provider_tags)
            for name in compute
        )
    ):
        raise MatrixError("GPU receipt compute nodes do not match the exact provider")


def _artifact_identity(label: str, path: Path) -> dict[str, Any]:
    if path.name != label or not label or len(label) > 128:
        raise MatrixError(f"unsafe correctness artifact label: {label!r}")
    return {
        "label": label,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def bind_runtime_cell_receipts(
    *,
    matrix: dict[str, Any],
    inventory: dict[str, Any],
    model_catalog: dict[str, Any],
    backend_catalog: dict[str, Any],
    source_digests: dict[str, str],
    backend_id: str,
    process_mode: str,
    gpu_receipt_path: Path,
    gpu_trace_path: Path,
    cpu_receipt_path: Path,
    cpu_trace_path: Path,
    binary_path: Path,
    plugin_path: Path,
    pack_path: Path,
    fixture_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind generic runtime output to one existing exact matrix cell."""

    if process_mode not in {"cold", "reuse"}:
        raise MatrixError("process mode must be cold or reuse")
    _bind_matrix_snapshots(
        matrix,
        inventory=inventory,
        catalog=model_catalog,
        backend_catalog=backend_catalog,
        source_digests=source_digests,
    )
    gpu_receipt = _read(gpu_receipt_path)
    cpu_receipt = _read(cpu_receipt_path)
    for label, document in (("GPU", gpu_receipt), ("CPU", cpu_receipt)):
        if document.get("schema") != RECEIPT_SCHEMA or document.get("evidence") is not None:
            raise MatrixError(f"{label} input must be one unbound short-audio receipt")
    pack = gpu_receipt.get("pack")
    audio = gpu_receipt.get("audio")
    if not isinstance(pack, dict) or not isinstance(audio, dict):
        raise MatrixError("GPU receipt lacks pack/audio identity")
    model_ref = pack.get("model_id")
    quant = pack.get("quant")
    matches = [
        cell
        for cell in matrix.get("cells", [])
        if isinstance(cell, dict)
        and cell.get("backend_id") == backend_id
        and f"{cell.get('model_id')}:{cell.get('quant')}" == model_ref
        and cell.get("quant") == quant
    ]
    if len(matches) != 1:
        raise MatrixError("runtime receipt does not select one exact matrix cell")
    cell = matches[0]
    provider = cell["provider"]
    expected_run_state = (
        ("cold", "empty") if process_mode == "cold" else ("warm", "populated")
    )
    run = gpu_receipt.get("run")
    if (
        not isinstance(run, dict)
        or run.get("backend") != "native"
        or run.get("device") != provider
        or run.get("os") != "windows"
        or (run.get("warmup"), run.get("cache_state")) != expected_run_state
        or gpu_receipt.get("placement") != provider
    ):
        raise MatrixError("GPU receipt run identity does not match the exact cell")
    if gpu_receipt.get("core_commit") != matrix["artifact_contract"]["core_commit"]:
        raise MatrixError("GPU receipt core commit does not match the matrix")
    diagnostics = gpu_receipt.get("decode_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("output_plan") != cell["output_plan"]["kind"]
        or diagnostics.get("reuse_mode") != cell["graph_mode"]
        or type(diagnostics.get("capability_evidence_revision")) is not int
        or diagnostics["capability_evidence_revision"] <= 0
    ):
        raise MatrixError("GPU receipt did not execute the projected output/graph plan")
    _require_runtime_placement(
        gpu_receipt, provider=provider, scheduler_mode=cell["scheduler_mode"]
    )

    gpu_trace = parse_trace_artifact(gpu_trace_path)
    if (
        gpu_trace["provider"] != provider
        or gpu_trace["device_target"] != cell["device_target"]
        or gpu_trace["backend_id"] != backend_id
        or gpu_trace["artifact_fingerprint"] != cell["artifact_fingerprint"]
        or gpu_trace["graph_mode"] != cell["graph_mode"]
    ):
        raise MatrixError("GPU trace identity does not match the exact matrix cell")
    cpu_trace = parse_cpu_oracle_trace(cpu_trace_path)
    if cpu_trace["token_ids"] != gpu_trace["token_ids"]:
        raise MatrixError("GPU token sequence diverges from the CPU family oracle")
    if (
        cpu_receipt.get("pack") != gpu_receipt.get("pack")
        or cpu_receipt.get("audio", {}).get("sha256") != audio.get("sha256")
        or cpu_receipt.get("transcript", {}).get("text_sha256")
        != gpu_receipt.get("transcript", {}).get("text_sha256")
    ):
        raise MatrixError("GPU transcript/inputs diverge from the CPU family oracle")

    for step, items in gpu_trace["topks"].items():
        maximum = max(float(item["value"]) for item in items)
        tied = [
            int(item["token_id"])
            for item in items
            if float(item["value"]) == maximum
        ]
        oracle_token = min(tied) if cell["output_plan"]["tie_policy"] == "first_maximum" else max(tied)
        if gpu_trace["token_ids"][step] != oracle_token:
            raise MatrixError("GPU selected token violates the family tie policy")

    if _sha256(binary_path) != matrix["artifact_contract"]["binary_sha256"]:
        raise MatrixError("runtime binary bytes do not match the matrix")
    if _sha256(plugin_path) != cell["plugin_sha256"]:
        raise MatrixError("runtime plugin bytes do not match the matrix cell")
    if _sha256(pack_path) != pack.get("content_sha256"):
        raise MatrixError("model pack bytes do not match the runtime receipt")
    if _sha256(fixture_path) != audio.get("sha256"):
        raise MatrixError("audio fixture bytes do not match the runtime receipt")

    worst_step = min(gpu_trace["margins"], key=gpu_trace["margins"].get)
    trace_identity = _artifact_identity(gpu_trace_path.name, gpu_trace_path)
    artifacts = {
        "binary": _artifact_identity(binary_path.name, binary_path),
        "plugin": _artifact_identity(plugin_path.name, plugin_path),
        "pack": _artifact_identity(pack_path.name, pack_path),
        "fixture": _artifact_identity(fixture_path.name, fixture_path),
    }
    common_evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "contract": "openasr.gpu-correctness-artifact.v1",
        "matrix_sha256": matrix["matrix_sha256"],
        "candidate_release_subject": matrix["artifact_contract"]["release_subject"],
        "core_commit": matrix["artifact_contract"]["core_commit"],
        "catalog_digests": {
            "inventory_sha256": source_digests["architecture_inventory_sha256"],
            "model_catalog_sha256": source_digests["model_catalog_sha256"],
            "backend_catalog_sha256": source_digests["backend_catalog_sha256"],
        },
        "family": cell["family"],
        "model_id": cell["model_id"],
        "quant": cell["quant"],
        "topology": cell["topology"]["decoder_state"],
        "provider": provider,
        "device_target": cell["device_target"],
        "backend_id": backend_id,
        "driver_version": gpu_trace["driver_version"],
        "artifact_fingerprint": cell["artifact_fingerprint"],
        "device": gpu_trace["device"],
        "placement": cell["placement"],
        "capture_mode": cell["capture_mode"],
        "scheduler_mode": cell["scheduler_mode"],
        "result": "pass",
        "artifacts": artifacts,
        "execution": {
            "mode": process_mode,
            "graph_rebuild_reason": (
                "shipped-fresh-graph-policy"
                if cell["graph_mode"] == "fresh_graph"
                else None
            ),
        },
    }
    placement_receipt = copy.deepcopy(gpu_receipt)
    placement_receipt["evidence"] = {
        **copy.deepcopy(common_evidence),
        "evidence_class": "placement_resource",
    }
    token_receipt = copy.deepcopy(gpu_receipt)
    token_receipt["evidence"] = {
        **copy.deepcopy(common_evidence),
        "evidence_class": "token_transcript",
        "output_plan": cell["output_plan"],
        "family_oracle": {
            "family": cell["family"],
            "tie_policy": cell["output_plan"]["tie_policy"],
        },
        "trace": {
            "token_trace": trace_identity,
            "logits": (
                trace_identity if cell["output_plan"]["requires_complete_output"] else None
            ),
            "top_k": gpu_trace["topks"][worst_step],
            "top1_top2_margin": gpu_trace["margins"][worst_step],
        },
    }
    return placement_receipt, token_receipt


def _write_bound_receipt_pair(
    placement: dict[str, Any],
    token: dict[str, Any],
    *,
    placement_out: Path,
    token_out: Path,
) -> None:
    if placement_out.resolve() == token_out.resolve():
        raise MatrixError("placement and token outputs must be different files")
    for path in (placement_out, token_out):
        if path.exists() or not path.name.startswith("gpu-correctness-receipt-"):
            raise MatrixError(f"refusing unsafe/existing correctness output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    encoded = [
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        for value in (placement, token)
    ]
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        for destination, contents in zip(
            (placement_out, token_out), encoded, strict=True
        ):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                delete=False,
            ) as handle:
                handle.write(contents)
                temp = Path(handle.name)
            temporary.append(temp)
        _core_validate_qualification_receipts(temporary)
        for temp, destination in zip(
            temporary, (placement_out, token_out), strict=True
        ):
            try:
                destination.hardlink_to(temp)
            except FileExistsError as error:
                raise MatrixError(f"refusing to replace correctness output: {destination}") from error
            published.append(destination)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _parse_vulkan_target_args(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        backend_id, separator, target = value.partition("=")
        if not separator or not backend_id or backend_id in result:
            raise MatrixError(
                "--vulkan-target must uniquely bind backend_id to one vk_caps class"
            )
        result[backend_id] = target
    return _vulkan_target_contract(result)


def _add_evidence_attestation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-repo")
    parser.add_argument("--evidence-signer-workflow")
    parser.add_argument("--evidence-source-digest")


def _verify_cli_evidence_attestations(
    args: argparse.Namespace, paths: list[Path]
) -> None:
    values = (
        args.evidence_repo,
        args.evidence_signer_workflow,
        args.evidence_source_digest,
    )
    if not any(values):
        return
    if not all(values):
        raise MatrixError(
            "evidence attestation requires repo, signer workflow, and source digest together"
        )
    try:
        verify_paths(
            paths,
            repository=args.evidence_repo,
            signer_workflow=args.evidence_signer_workflow,
            source_digest=args.evidence_source_digest,
            label="GPU correctness evidence",
        )
    except AttestationError as error:
        raise MatrixError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    project = subparsers.add_parser("project")
    project.add_argument("--inventory", type=Path, required=True)
    project.add_argument("--catalog", type=Path, required=True)
    project.add_argument("--backend-catalog", type=Path, required=True)
    project.add_argument("--out", type=Path, required=True)
    project.add_argument("--release-subject", required=True)
    project.add_argument("--release-version", required=True)
    project.add_argument("--core-commit", required=True)
    project.add_argument("--binary-sha256", required=True)
    project.add_argument(
        "--vulkan-target",
        action="append",
        default=[],
        metavar="BACKEND_ID=VK_CAPS",
        help="bind one generic Vulkan artifact to an exact capability class",
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--inventory", type=Path, required=True)
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--backend-catalog", type=Path, required=True)
    validate.add_argument("--activation-catalog", type=Path)
    validate.add_argument("--receipt", type=Path, action="append", required=True)
    validate.add_argument("--trace", type=Path, action="append")
    _add_evidence_attestation_args(validate)
    qualify = subparsers.add_parser("qualify-catalog")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--inventory", type=Path, required=True)
    qualify.add_argument("--catalog", type=Path, required=True)
    qualify.add_argument("--backend-catalog", type=Path, required=True)
    qualify.add_argument("--current-activation-catalog", type=Path)
    qualify.add_argument("--backend-id", required=True)
    qualify.add_argument("--entry", type=Path, action="append", required=True)
    qualify.add_argument(
        "--hardware-evidence", type=Path, action="append", required=True
    )
    qualify.add_argument(
        "--hardware-raw-audit", type=Path, action="append", required=True
    )
    qualify.add_argument("--out", type=Path, required=True)
    activate = subparsers.add_parser("activate-catalog")
    activate.add_argument("--manifest", type=Path, required=True)
    activate.add_argument("--inventory", type=Path, required=True)
    activate.add_argument("--catalog", type=Path, required=True)
    activate.add_argument("--backend-catalog", type=Path, required=True)
    activate.add_argument("--current-activation-catalog", type=Path)
    activate.add_argument("--backend-id", required=True)
    activate.add_argument("--entry", type=Path, action="append", required=True)
    activate.add_argument(
        "--hardware-evidence", type=Path, action="append", required=True
    )
    activate.add_argument(
        "--hardware-raw-audit", type=Path, action="append", required=True
    )
    activate.add_argument("--receipt", type=Path, action="append", required=True)
    activate.add_argument("--trace", type=Path, action="append")
    activate.add_argument("--out", type=Path, required=True)
    _add_evidence_attestation_args(activate)
    verify_transition = subparsers.add_parser("verify-catalog-transition")
    verify_transition.add_argument("--manifest", type=Path, required=True)
    verify_transition.add_argument("--inventory", type=Path, required=True)
    verify_transition.add_argument("--catalog", type=Path, required=True)
    verify_transition.add_argument("--backend-catalog", type=Path, required=True)
    verify_transition.add_argument("--current-activation-catalog", type=Path, required=True)
    verify_transition.add_argument("--candidate-activation-catalog", type=Path, required=True)
    verify_transition.add_argument("--backend-id", required=True)
    verify_transition.add_argument("--entry", type=Path, action="append", required=True)
    verify_transition.add_argument(
        "--hardware-evidence", type=Path, action="append", required=True
    )
    verify_transition.add_argument(
        "--hardware-raw-audit", type=Path, action="append", required=True
    )
    verify_transition.add_argument("--receipt", type=Path, action="append", required=True)
    verify_transition.add_argument("--trace", type=Path, action="append")
    _add_evidence_attestation_args(verify_transition)
    revoke = subparsers.add_parser("revoke-catalog")
    revoke.add_argument("--current-activation-catalog", type=Path, required=True)
    revoke.add_argument("--backend-id", required=True)
    revoke.add_argument("--out", type=Path, required=True)
    verify_revocation = subparsers.add_parser("verify-revocation-transition")
    verify_revocation.add_argument(
        "--current-activation-catalog", type=Path, required=True
    )
    verify_revocation.add_argument(
        "--candidate-activation-catalog", type=Path, required=True
    )
    verify_revocation.add_argument("--backend-id", required=True)
    bind_cell = subparsers.add_parser("bind-cell")
    bind_cell.add_argument("--manifest", type=Path, required=True)
    bind_cell.add_argument("--inventory", type=Path, required=True)
    bind_cell.add_argument("--catalog", type=Path, required=True)
    bind_cell.add_argument("--backend-catalog", type=Path, required=True)
    bind_cell.add_argument("--backend-id", required=True)
    bind_cell.add_argument("--process-mode", choices=("cold", "reuse"), required=True)
    bind_cell.add_argument("--gpu-receipt", type=Path, required=True)
    bind_cell.add_argument("--gpu-trace", type=Path, required=True)
    bind_cell.add_argument("--cpu-receipt", type=Path, required=True)
    bind_cell.add_argument("--cpu-trace", type=Path, required=True)
    bind_cell.add_argument("--binary", type=Path, required=True)
    bind_cell.add_argument("--plugin", type=Path, required=True)
    bind_cell.add_argument("--pack", type=Path, required=True)
    bind_cell.add_argument("--fixture", type=Path, required=True)
    bind_cell.add_argument("--placement-out", type=Path, required=True)
    bind_cell.add_argument("--token-out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "project":
        matrix = project_matrix(
            _read(args.inventory),
            _read(args.catalog),
            _read(args.backend_catalog),
            source_digests={
                "architecture_inventory_sha256": _sha256(args.inventory),
                "model_catalog_sha256": _sha256(args.catalog),
                "backend_catalog_sha256": _sha256(args.backend_catalog),
            },
            candidate={
                "release_subject": args.release_subject,
                "release_version": args.release_version,
                "core_commit": args.core_commit,
                "binary_sha256": args.binary_sha256,
            },
            vulkan_targets=_parse_vulkan_target_args(args.vulkan_target),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.out)
    elif args.command == "validate":
        _verify_cli_evidence_attestations(
            args,
            [
                args.manifest,
                args.inventory,
                args.catalog,
                args.backend_catalog,
                *args.receipt,
                *(args.trace or []),
            ],
        )
        validate_matrix(
            _read(args.manifest),
            args.receipt,
            inventory=_read(args.inventory),
            catalog=_read(args.catalog),
            backend_catalog=_read(args.backend_catalog),
            activation_catalog=(
                _read(args.activation_catalog) if args.activation_catalog else None
            ),
            source_digests={
                "architecture_inventory_sha256": _sha256(args.inventory),
                "model_catalog_sha256": _sha256(args.catalog),
                "backend_catalog_sha256": _sha256(args.backend_catalog),
            },
            trace_paths=args.trace or [],
        )
        print("gpu correctness matrix passed")
    elif args.command == "qualify-catalog":
        if args.out.exists():
            raise MatrixError(f"refusing to overwrite qualified catalog: {args.out}")
        source_digests = {
            "architecture_inventory_sha256": _sha256(args.inventory),
            "model_catalog_sha256": _sha256(args.catalog),
            "backend_catalog_sha256": _sha256(args.backend_catalog),
        }
        result = qualify_catalog_backend(
            matrix=_read(args.manifest),
            source_backend_catalog=_read(args.backend_catalog),
            current_activation_catalog=_read(
                args.current_activation_catalog or args.backend_catalog
            ),
            source_digests=source_digests,
            backend_id=args.backend_id,
            entry_paths=args.entry,
            hardware_evidence_paths=args.hardware_evidence,
            hardware_raw_audit_paths=args.hardware_raw_audit,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(args.out)
    elif args.command == "activate-catalog":
        _verify_cli_evidence_attestations(
            args,
            [
                args.manifest,
                args.inventory,
                args.catalog,
                args.backend_catalog,
                *args.receipt,
                *(args.trace or []),
            ],
        )
        if args.out.exists():
            raise MatrixError(f"refusing to overwrite activation catalog: {args.out}")
        source_digests = {
            "architecture_inventory_sha256": _sha256(args.inventory),
            "model_catalog_sha256": _sha256(args.catalog),
            "backend_catalog_sha256": _sha256(args.backend_catalog),
        }
        result = activate_catalog_backend(
            matrix=_read(args.manifest),
            inventory=_read(args.inventory),
            model_catalog=_read(args.catalog),
            source_backend_catalog=_read(args.backend_catalog),
            current_activation_catalog=_read(
                args.current_activation_catalog or args.backend_catalog
            ),
            source_digests=source_digests,
            backend_id=args.backend_id,
            entry_paths=args.entry,
            hardware_evidence_paths=args.hardware_evidence,
            hardware_raw_audit_paths=args.hardware_raw_audit,
            receipt_paths=args.receipt,
            trace_paths=args.trace or [],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(args.out)
    elif args.command == "verify-catalog-transition":
        _verify_cli_evidence_attestations(
            args,
            [
                args.manifest,
                args.inventory,
                args.catalog,
                args.backend_catalog,
                *args.receipt,
                *(args.trace or []),
            ],
        )
        source_digests = {
            "architecture_inventory_sha256": _sha256(args.inventory),
            "model_catalog_sha256": _sha256(args.catalog),
            "backend_catalog_sha256": _sha256(args.backend_catalog),
        }
        verify_catalog_backend_transition(
            matrix=_read(args.manifest),
            inventory=_read(args.inventory),
            model_catalog=_read(args.catalog),
            source_backend_catalog=_read(args.backend_catalog),
            current_activation_catalog=_read(args.current_activation_catalog),
            candidate_activation_catalog=_read(args.candidate_activation_catalog),
            source_digests=source_digests,
            backend_id=args.backend_id,
            entry_paths=args.entry,
            hardware_evidence_paths=args.hardware_evidence,
            hardware_raw_audit_paths=args.hardware_raw_audit,
            receipt_paths=args.receipt,
            trace_paths=args.trace or [],
        )
        print("catalog activation transition verified")
    elif args.command == "revoke-catalog":
        if args.out.exists():
            raise MatrixError(f"refusing to overwrite revoked catalog: {args.out}")
        result = revoke_catalog_backend(
            current_activation_catalog=_read(args.current_activation_catalog),
            backend_id=args.backend_id,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(args.out)
    elif args.command == "verify-revocation-transition":
        verify_catalog_backend_revocation(
            current_activation_catalog=_read(args.current_activation_catalog),
            candidate_activation_catalog=_read(args.candidate_activation_catalog),
            backend_id=args.backend_id,
        )
        print("catalog revocation transition verified")
    else:
        source_digests = {
            "architecture_inventory_sha256": _sha256(args.inventory),
            "model_catalog_sha256": _sha256(args.catalog),
            "backend_catalog_sha256": _sha256(args.backend_catalog),
        }
        placement, token = bind_runtime_cell_receipts(
            matrix=_read(args.manifest),
            inventory=_read(args.inventory),
            model_catalog=_read(args.catalog),
            backend_catalog=_read(args.backend_catalog),
            source_digests=source_digests,
            backend_id=args.backend_id,
            process_mode=args.process_mode,
            gpu_receipt_path=args.gpu_receipt,
            gpu_trace_path=args.gpu_trace,
            cpu_receipt_path=args.cpu_receipt,
            cpu_trace_path=args.cpu_trace,
            binary_path=args.binary,
            plugin_path=args.plugin,
            pack_path=args.pack,
            fixture_path=args.fixture,
        )
        _write_bound_receipt_pair(
            placement,
            token,
            placement_out=args.placement_out,
            token_out=args.token_out,
        )
        print(args.placement_out)
        print(args.token_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as error:
        raise SystemExit(f"gpu correctness gate failed: {error}")
