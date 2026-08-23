#!/usr/bin/env python3
"""Project and gate pre-publication GPU correctness evidence.

The matrix is derived from the canonical architecture inventory, the public model
catalog, and the staged backend catalog. It describes evidence that is required;
it never turns a build, placement receipt, or CPU result into token correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "openasr.gpu-correctness-matrix.v1"
RECEIPT_SCHEMA = "openasr.short-audio-receipt.v0"
EVIDENCE_SCHEMA = "openasr.short-audio-receipt.evidence.v1"
PROVIDERS = ("cpu", "metal", "cuda", "vulkan", "hip")


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
    if capabilities.get("cpu") is True:
        result.append(("cpu", "full_device", ["explicit", "auto"]))
    providers = capabilities.get("providers", [])
    if not isinstance(providers, list):
        raise MatrixError(f"family {descriptor.get('catalog_family_id')} has invalid providers")
    for item in providers:
        if not isinstance(item, dict) or not isinstance(item.get("provider"), str):
            raise MatrixError(f"family {descriptor.get('catalog_family_id')} has an invalid provider row")
        provider = item["provider"].lower()
        if provider not in PROVIDERS or (item.get("full_device") is not True and item.get("hybrid") is not True):
            continue
        placement = "full_device" if item.get("full_device") is True else "hybrid"
        modes = ["explicit"]
        if auto_policy == "all-backends" or (auto_policy == "except-metal" and provider != "metal"):
            modes.append("auto")
        result.append((provider, placement, modes))
    return result


def _lane_policies(provider: str) -> tuple[str, str]:
    """Return explicit staging policies, not claims of successful hardware runs."""
    capture = "enabled" if provider == "hip" else "disabled"
    scheduler = "disabled"
    return capture, scheduler


def _tie_policy(family: str) -> str:
    # This is the family oracle contract, not a provider capability. XASR's
    # existing host oracle is last-max; other current token/code paths use the
    # first-max contract. Receipts must still bind and repeat this value.
    return "last_maximum" if family == "xasr-zipformer" else "first_maximum"

def project_matrix(
    inventory: dict[str, Any],
    catalog: dict[str, Any],
    backend_catalog: dict[str, Any],
    *,
    source_digests: dict[str, str] | None = None,
    candidate: dict[str, str] | None = None,
) -> dict[str, Any]:
    _require_schema(inventory, "openasr.model-family-inventory.v1", "architecture inventory")
    grouped = _public_models(catalog)
    backends = backend_catalog.get("backends")
    if not isinstance(backends, list):
        raise MatrixError("backend catalog must contain a backends array")
    backend_ids: dict[str, list[str]] = {}
    for backend in backends:
        if not isinstance(backend, dict):
            raise MatrixError("backend catalog contains a non-object entry")
        vendor = backend.get("vendor")
        identifier = backend.get("id")
        if isinstance(vendor, str) and isinstance(identifier, str) and identifier:
            backend_ids.setdefault(vendor.lower(), []).append(identifier)

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
            capture_mode, scheduler_mode = _lane_policies(provider)
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
                    cells.append(
                        {
                            "family": family,
                            "model_id": model["id"],
                            "quant": quant,
                            "provider": provider,
                            "activation_modes": modes,
                            "placement": placement,
                            "capture_mode": capture_mode,
                            "scheduler_mode": scheduler_mode,
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
                            "backend_catalog_ids": sorted(backend_ids.get(provider, [])),
                            "required_receipt_classes": ["placement_resource", "token_transcript"],
                            "status": "pending",
                        }
                    )
    if not cells:
        raise MatrixError("projection produced no public family/provider cells")
    if not candidate:
        raise MatrixError("staging projection requires an immutable candidate contract")
    required_candidate = ("release_subject", "core_commit", "binary_sha256", "plugin_sha256")
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
            "core_commit": candidate["core_commit"],
            "binary_sha256": candidate["binary_sha256"],
            "plugin_sha256": candidate["plugin_sha256"],
            "source_digests": source_digests,
        },
        "source_digests": source_digests,
        "required_global_receipt_classes": ["build_packaging"],
        "cells": cells,
    }
    matrix["matrix_sha256"] = _canonical_sha(matrix)
    return matrix


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def parse_trace_artifact(path: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or events[0].get("schema") != "openasr.gpu-correctness-trace.v1" or events[0].get("event") != "header":
        raise MatrixError(f"{path} lacks the strict runtime trace header")
    header = events[0]
    if header.get("mode") not in {"cold", "reuse"} or not isinstance(header.get("provider"), str) or not isinstance(header.get("device"), str):
        raise MatrixError(f"{path} has invalid runtime trace identity")
    tokens: dict[int, dict[str, Any]] = {}
    topks: dict[int, list[dict[str, Any]]] = {}
    for event in events[1:]:
        if event.get("schema") != "openasr.gpu-correctness-trace.v1" or not isinstance(event.get("step_index"), int):
            raise MatrixError(f"{path} contains an unversioned or malformed trace event")
        step = event["step_index"]
        if event.get("event") == "token":
            if not isinstance(event.get("token_id"), int) or event["token_id"] < 0:
                raise MatrixError(f"{path} contains an invalid token event")
            tokens[step] = event
        elif event.get("event") == "top_k":
            items = event.get("items")
            if not isinstance(items, list) or not items or any(not isinstance(item, dict) or not isinstance(item.get("value"), (int, float)) for item in items):
                raise MatrixError(f"{path} contains an invalid top-k event")
            topks[step] = items
        else:
            raise MatrixError(f"{path} contains an unknown trace event")
    if not tokens or set(tokens) != set(topks):
        raise MatrixError(f"{path} does not contain matching per-step token and top-k events")
    return {"mode": header["mode"], "provider": header["provider"], "device": header["device"], "steps": sorted(tokens)}

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
        "provider", "device", "placement", "capture_mode", "scheduler_mode",
        "evidence_class", "artifacts", "result",
    )
    if any(field not in evidence for field in required):
        raise MatrixError(f"{path} has an incomplete strict artifact contract")
    if evidence.get("contract") != "openasr.gpu-correctness-artifact.v1" or evidence.get("result") != "pass":
        raise MatrixError(f"{path} has an invalid or non-passing artifact contract")
    if not _hex_digest(evidence.get("matrix_sha256")):
        raise MatrixError(f"{path} has no matrix digest")
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


def expected_receipt_keys(matrix: dict[str, Any]) -> tuple[
    set[tuple[str, str, str, str, str, str]],
    dict[tuple[str, str, str, str], dict[str, Any]],
]:
    """Return exact-cell receipt keys. Untested lanes stay required; they are not skipped."""
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise MatrixError("correctness matrix has no cells")
    expected: set[tuple[str, str, str, str, str, str]] = set()
    cell_by_lane: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise MatrixError("correctness matrix contains a non-object cell")
        family, provider = cell.get("family"), cell.get("provider")
        model_id, quant = cell.get("model_id"), cell.get("quant")
        modes = cell.get("reuse_modes")
        if not all(isinstance(value, str) and value for value in (family, provider, model_id, quant)) or modes != ["cold", "reuse"]:
            raise MatrixError("matrix cell lacks family/model/quant/provider or cold/reuse requirements")
        for field in ("topology", "kernel_coverage_bucket", "output_plan"):
            if not isinstance(cell.get(field), dict):
                raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks {field}")
        if cell.get("capture_mode") not in {"disabled", "enabled", "unsupported"} or cell.get("scheduler_mode") not in {"disabled", "enabled"}:
            raise MatrixError(f"matrix cell {family}/{model_id}:{quant} lacks capture/scheduler policy")
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
                key = (family, model_id, quant, provider, mode, evidence_class)
                if key in expected:
                    raise MatrixError(f"duplicate correctness cell {key}")
                expected.add(key)
                cell_by_lane[(family, model_id, quant, provider)] = cell
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
    matrix_source_digests = matrix.get("source_digests")
    if contract.get("source_digests") != matrix_source_digests or not isinstance(matrix_source_digests, dict):
        raise MatrixError("matrix source digests are not bound into the artifact contract")
    if source_digests is None or source_digests != matrix_source_digests:
        raise MatrixError("matrix source digests do not match the exact staging snapshots")
    for field in ("binary_sha256", "plugin_sha256"):
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
            "core_commit": contract["core_commit"],
            "binary_sha256": contract["binary_sha256"],
            "plugin_sha256": contract["plugin_sha256"],
        },
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
) -> tuple[set[tuple[str, str, str, str, str, str]], set[str]]:
    """Validate each receipt against exact cells without skipping missing lanes."""
    _bind_matrix_snapshots(
        matrix,
        inventory=inventory,
        catalog=catalog,
        backend_catalog=backend_catalog,
        source_digests=source_digests,
    )
    if not trace_paths:
        raise MatrixError("validation requires immutable token/logits trace artifacts")
    trace_hashes = {path.name: _sha256(path) for path in trace_paths}
    trace_semantics = {path.name: parse_trace_artifact(path) for path in trace_paths}
    expected, cell_by_lane = expected_receipt_keys(matrix)
    receipts: set[tuple[str, str, str, str, str, str]] = set()
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
        if not all(isinstance(value, str) and value for value in (family, provider, model_id, quant)):
            raise MatrixError(f"{path} lacks family/provider/model/quant identity")
        lane = cell_by_lane.get((family, model_id, quant, provider))
        if lane is None:
            raise MatrixError(f"{path} is not bound to a projected family/model/quant/provider lane")
        contract = matrix["artifact_contract"]
        if evidence.get("matrix_sha256") != matrix.get("matrix_sha256") or evidence.get("candidate_release_subject") != contract.get("release_subject") or evidence.get("core_commit") != contract.get("core_commit"):
            raise MatrixError(f"{path} is stale or bound to another release subject")
        if evidence.get("catalog_digests") != contract.get("source_digests"):
            raise MatrixError(f"{path} is bound to another inventory or catalog snapshot")
        if evidence.get("topology") != lane["topology"].get("decoder_state"):
            raise MatrixError(f"{path} topology does not match its matrix cell")
        if evidence.get("capture_mode") != lane["capture_mode"] or evidence.get("scheduler_mode") != lane["scheduler_mode"]:
            raise MatrixError(f"{path} capture/scheduler identity does not match its matrix cell")
        artifacts = evidence["artifacts"]
        if artifacts["binary"].get("sha256") != contract["binary_sha256"] or artifacts["plugin"].get("sha256") != contract["plugin_sha256"]:
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
        key = (family, model_id, quant, provider, mode, evidence_class)
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
            if semantics["mode"] != mode or semantics["provider"] != provider or semantics["device"] != evidence.get("device"):
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
    closed_keys: set[tuple[str, str, str, str, str, str]],
) -> dict[tuple[str, str, str, str], tuple[str, ...]]:
    """Return Auto/explicit modes only for lanes with complete receipts.

    Advertised CUDA, physical Vulkan, and HIP cells stay in the map when
    untested. Empty modes mean the lane exists and is not selectable.
    """
    expected, cell_by_lane = expected_receipt_keys(matrix)
    allowlist: dict[tuple[str, str, str, str], tuple[str, ...]] = {}
    for lane, cell in cell_by_lane.items():
        needed = {key for key in expected if key[:4] == lane}
        modes = tuple(cell["activation_modes"]) if needed <= closed_keys else ()
        allowlist[lane] = modes
    return allowlist


def require_activation(
    matrix: dict[str, Any],
    closed_keys: set[tuple[str, str, str, str, str, str]],
    *,
    provider: str,
    mode: str,
) -> None:
    """Fail closed if Auto or explicit selection is requested for an unproven cell."""
    if mode not in {"auto", "explicit"}:
        raise MatrixError(f"{mode} is not an activation mode")
    allowlist = lane_activation_modes(matrix, closed_keys)
    matching = [key for key in allowlist if key[3] == provider]
    if not matching:
        raise MatrixError(f"{provider} is not a projected activation lane")
    blocked = [key for key in matching if mode not in allowlist[key]]
    if blocked:
        raise MatrixError(
            f"{provider} {mode} is not selectable without closed correctness receipts: {blocked}"
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
    closed_keys: set[tuple[str, str, str, str, str, str]],
    hardware_unavailable: Path,
) -> None:
    """CUDA/Vulkan/HIP stay projected and unselectable without receipts."""
    statuses = parse_hardware_unavailable(hardware_unavailable)
    for provider, status in statuses.items():
        if status != "unavailable":
            raise MatrixError(f"{provider} missing-host record is not fail-closed")
        require_activation(matrix, closed_keys, provider=provider, mode="auto")
        require_activation(matrix, closed_keys, provider=provider, mode="explicit")


def validate_matrix(
    matrix: dict[str, Any],
    receipt_paths: list[Path],
    *,
    inventory: dict[str, Any] | None = None,
    catalog: dict[str, Any] | None = None,
    backend_catalog: dict[str, Any] | None = None,
    source_digests: dict[str, str] | None = None,
    trace_paths: list[Path] | None = None,
) -> None:
    expected, _cell_by_lane = expected_receipt_keys(matrix)
    receipts, classes = closed_receipt_keys(
        matrix,
        receipt_paths,
        inventory=inventory,
        catalog=catalog,
        backend_catalog=backend_catalog,
        source_digests=source_digests,
        trace_paths=trace_paths,
    )
    missing = sorted(expected - receipts)
    if missing:
        raise MatrixError(f"correctness matrix is incomplete; missing receipts: {missing}")
    required_global = matrix.get("required_global_receipt_classes", [])
    if not isinstance(required_global, list) or any(not isinstance(item, str) for item in required_global):
        raise MatrixError("correctness matrix has invalid global receipt requirements")
    missing_global = sorted(set(required_global) - classes)
    if missing_global:
        raise MatrixError(f"correctness matrix is incomplete; missing global receipts: {missing_global}")
    for cell in matrix["cells"]:
        provider = cell["provider"]
        for mode in cell["activation_modes"]:
            require_activation(matrix, receipts, provider=provider, mode=mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    project = subparsers.add_parser("project")
    project.add_argument("--inventory", type=Path, required=True)
    project.add_argument("--catalog", type=Path, required=True)
    project.add_argument("--backend-catalog", type=Path, required=True)
    project.add_argument("--out", type=Path, required=True)
    project.add_argument("--release-subject", required=True)
    project.add_argument("--core-commit", required=True)
    project.add_argument("--binary-sha256", required=True)
    project.add_argument("--plugin-sha256", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--inventory", type=Path, required=True)
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--backend-catalog", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, action="append", required=True)
    validate.add_argument("--trace", type=Path, action="append", required=True)
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
                "core_commit": args.core_commit,
                "binary_sha256": args.binary_sha256,
                "plugin_sha256": args.plugin_sha256,
            },
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.out)
    else:
        validate_matrix(
            _read(args.manifest),
            args.receipt,
            inventory=_read(args.inventory),
            catalog=_read(args.catalog),
            backend_catalog=_read(args.backend_catalog),
            source_digests={
                "architecture_inventory_sha256": _sha256(args.inventory),
                "model_catalog_sha256": _sha256(args.catalog),
                "backend_catalog_sha256": _sha256(args.backend_catalog),
            },
            trace_paths=args.trace,
        )
        print("gpu correctness matrix passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as error:
        raise SystemExit(f"gpu correctness gate failed: {error}")
