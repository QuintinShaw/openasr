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


def _kernel_classes(descriptor: dict[str, Any]) -> list[str]:
    topology = descriptor.get("topology", {})
    quantization = descriptor.get("quantization", {})
    values = [
        topology.get("decode_driver"),
        topology.get("decoder_state"),
        topology.get("block_stack"),
        quantization.get("tensor_classification"),
    ]
    return sorted({str(value) for value in values if isinstance(value, str) and value})


def project_matrix(inventory: dict[str, Any], catalog: dict[str, Any], backend_catalog: dict[str, Any], *, source_digests: dict[str, str] | None = None) -> dict[str, Any]:
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
        quant_names: set[str] = set()
        for model in grouped[family]:
            for quant in model.get("quants", []):
                if isinstance(quant, dict) and isinstance(quant.get("quant"), str):
                    quant_names.add(quant["quant"])
            recommended = model.get("recommended_quant")
            if isinstance(recommended, str):
                quant_names.add(recommended)
        if not quant_names:
            quant_names.add("unknown")
        for provider, placement, modes in _advertised_providers(descriptor):
            cells.append(
                {
                    "family": family,
                    "model_ids": sorted(str(model["id"]) for model in grouped[family]),
                    "provider": provider,
                    "activation_modes": modes,
                    "placement": placement,
                    "topology": {
                        "decode_driver": topology.get("decode_driver"),
                        "decoder_state": topology.get("decoder_state"),
                        "block_stack": topology.get("block_stack"),
                    },
                    "representative_kernel_classes": _kernel_classes(descriptor),
                    "weight_types": sorted(quant_names),
                    "reuse_modes": ["cold", "reuse"],
                    "backend_catalog_ids": sorted(backend_ids.get(provider, [])),
                    "required_receipt_classes": ["placement_resource", "token_transcript"],
                    "status": "pending",
                }
            )
    if not cells:
        raise MatrixError("projection produced no public family/provider cells")
    matrix = {
        "schema": SCHEMA,
        "source_digests": source_digests or {},
        "required_global_receipt_classes": ["build_packaging"],
        "cells": cells,
    }
    matrix["matrix_sha256"] = _canonical_sha(matrix)
    return matrix


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _evidence_for(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _read(path)
    if document.get("schema") != RECEIPT_SCHEMA:
        raise MatrixError(f"{path} is not a short-audio receipt")
    evidence = document.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("schema") != EVIDENCE_SCHEMA:
        raise MatrixError(f"{path} has no versioned correctness evidence")
    if evidence.get("result") != "pass":
        raise MatrixError(f"{path} does not contain passing evidence")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict) or any(
        not isinstance(artifacts.get(name), dict) or not _hex_digest(artifacts[name].get("sha256"))
        for name in ("binary", "pack", "fixture")
    ):
        raise MatrixError(f"{path} has incomplete artifact identity")
    return document, evidence


def validate_matrix(matrix: dict[str, Any], receipt_paths: list[Path]) -> None:
    _require_schema(matrix, SCHEMA, "correctness matrix")
    claimed_matrix_sha = matrix.get("matrix_sha256")
    if not _hex_digest(claimed_matrix_sha):
        raise MatrixError("correctness matrix has no lowercase matrix_sha256")
    unsigned_matrix = dict(matrix)
    unsigned_matrix.pop("matrix_sha256", None)
    if _canonical_sha(unsigned_matrix) != claimed_matrix_sha:
        raise MatrixError("correctness matrix hash does not verify")
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise MatrixError("correctness matrix has no cells")
    expected: set[tuple[str, str, str, str]] = set()
    model_ids_by_lane: dict[tuple[str, str], set[str]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise MatrixError("correctness matrix contains a non-object cell")
        family, provider = cell.get("family"), cell.get("provider")
        modes = cell.get("reuse_modes")
        if not isinstance(family, str) or not isinstance(provider, str) or modes != ["cold", "reuse"]:
            raise MatrixError("matrix cell lacks family, provider, or cold/reuse requirements")
        model_ids = cell.get("model_ids")
        if not isinstance(model_ids, list) or not model_ids or any(not isinstance(item, str) or not item for item in model_ids):
            raise MatrixError(f"matrix cell {family}/{provider} lacks model identities")
        model_ids_by_lane[(family, provider)] = set(model_ids)
        required_classes = cell.get("required_receipt_classes")
        if not isinstance(required_classes, list) or set(required_classes) != {"placement_resource", "token_transcript"}:
            raise MatrixError(f"matrix cell {family}/{provider} lacks separate receipt classes")
        for mode in modes:
            for evidence_class in required_classes:
                key = (family, provider, mode, evidence_class)
                if key in expected:
                    raise MatrixError(f"duplicate correctness cell {key}")
                expected.add(key)
    receipts: set[tuple[str, str, str, str]] = set()
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
        if (not isinstance(document.get("pack"), dict) or
                document["pack"].get("model_id") not in model_ids_by_lane[(family, provider)]):
            raise MatrixError(f"{path} is bound to a model outside its matrix lane")
        execution = evidence.get("execution")
        if not isinstance(family, str) or not isinstance(provider, str) or not isinstance(execution, dict):
            raise MatrixError(f"{path} lacks family/provider/execution identity")
        mode = execution.get("mode")
        if mode not in {"cold", "reuse"}:
            raise MatrixError(f"{path} has invalid execution mode")
        key = (family, provider, mode, evidence_class)
        if key not in expected:
            raise MatrixError(f"{path} is not bound to a projected matrix cell")
        if evidence_class == "token_transcript":
            output_plan = evidence.get("output_plan")
            oracle = evidence.get("family_oracle")
            trace = evidence.get("trace")
            if not isinstance(output_plan, dict) or not isinstance(oracle, dict) or not isinstance(trace, dict):
                raise MatrixError(f"{path} token evidence lacks output plan, oracle, or trace")
            if not _hex_digest(trace.get("token_trace_sha256")):
                raise MatrixError(f"{path} token evidence lacks a token trace hash")
        if key in receipts:
            raise MatrixError(f"duplicate evidence for correctness cell {key}")
        receipts.add(key)
    missing = sorted(expected - receipts)
    if missing:
        raise MatrixError(f"correctness matrix is incomplete; missing receipts: {missing}")
    required_global = matrix.get("required_global_receipt_classes", [])
    if not isinstance(required_global, list) or any(not isinstance(item, str) for item in required_global):
        raise MatrixError("correctness matrix has invalid global receipt requirements")
    missing_global = sorted(set(required_global) - classes)
    if missing_global:
        raise MatrixError(f"correctness matrix is incomplete; missing global receipts: {missing_global}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    project = subparsers.add_parser("project")
    project.add_argument("--inventory", type=Path, required=True)
    project.add_argument("--catalog", type=Path, required=True)
    project.add_argument("--backend-catalog", type=Path, required=True)
    project.add_argument("--out", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, action="append", required=True)
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
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.out)
    else:
        validate_matrix(_read(args.manifest), args.receipt)
        print("gpu correctness matrix passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as error:
        raise SystemExit(f"gpu correctness gate failed: {error}")
