#!/usr/bin/env python3
"""Create and verify the single external release-correctness binding.

The matrix and receipts are declarations. This binding is anchored to actual
release bytes, SHA256SUMS, canonical source snapshots, deployed catalog bytes,
and a verified reusable deploy run. The finalizer consumes this file fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "openasr.release-correctness-binding.v1"


class BindingError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise BindingError(f"{field} must be lowercase 64-hex")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BindingError(f"{path} must be a JSON object")
    return value


def sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        checksum, filename = parts
        entries[filename.lstrip("* ")] = hex64(checksum.lower(), f"SHA256SUMS {filename}")
    if not entries:
        raise BindingError("SHA256SUMS is empty")
    return entries


def validate_deploy_run(deploy: dict[str, Any], orchestrator_run_id: str, release_tag: str, tag_commit: str) -> None:
    if deploy.get("workflow_name") != "Deploy catalog" or deploy.get("conclusion") != "success":
        raise BindingError("deploy run is not the successful Deploy catalog workflow")
    if deploy.get("event") != "workflow_call":
        raise BindingError("catalog deploy was not called by the release orchestrator")
    if deploy.get("caller_run_id") != orchestrator_run_id:
        raise BindingError("catalog deploy is not bound to this orchestrator run")
    if deploy.get("release_tag") != release_tag:
        raise BindingError("catalog deploy tag does not match release tag")
    if deploy.get("head_sha") != tag_commit:
        raise BindingError("catalog deploy head is not the release tag commit")


def build_binding(args: argparse.Namespace) -> dict[str, Any]:
    matrix = read_json(args.matrix)
    if matrix.get("schema") != "openasr.gpu-correctness-matrix.v1":
        raise BindingError("matrix schema is not the canonical correctness matrix")
    matrix_sha = hex64(matrix.get("matrix_sha256"), "matrix_sha256")
    source_digests = matrix.get("source_digests")
    if not isinstance(source_digests, dict) or set(source_digests) != {
        "architecture_inventory_sha256", "model_catalog_sha256", "backend_catalog_sha256"
    }:
        raise BindingError("matrix has incomplete source digests")
    source_paths = [args.inventory, args.model_catalog, args.backend_catalog]
    source_names = list(source_digests)
    actual_sources = {name: sha256(path) for name, path in zip(source_names, source_paths)}
    if actual_sources != source_digests:
        raise BindingError("matrix source digests do not match source snapshot bytes")
    catalog_sha = sha256(args.public_catalog)
    if source_digests["model_catalog_sha256"] != catalog_sha:
        raise BindingError("matrix model catalog digest does not equal the deployed catalog bytes")
    signature_sha = sha256(args.public_signature)
    sums = sha256sums(args.sha256sums)
    assets = []
    for path in args.asset:
        actual = sha256(path)
        expected = sums.get(path.name)
        if expected != actual:
            raise BindingError(f"release asset {path.name} is absent or differs from SHA256SUMS")
        assets.append({"name": path.name, "sha256": actual, "size_bytes": path.stat().st_size})
    deploy = read_json(args.deploy_run)
    validate_deploy_run(deploy, args.orchestrator_run_id, args.tag, args.tag_commit)
    binding = {
        "schema": SCHEMA,
        "release": {
            "tag": args.tag,
            "tag_commit": args.tag_commit,
            "orchestrator_run_id": args.orchestrator_run_id,
        },
        "candidate_assets": assets,
        "sha256sums_sha256": sha256(args.sha256sums),
        "plugin_sha256": hex64(args.plugin_sha256, "plugin_sha256"),
        "matrix_sha256": matrix_sha,
        "source_digests": source_digests,
        "deployed_catalog_sha256": catalog_sha,
        "deployed_catalog_signature_sha256": signature_sha,
        "deploy_run": deploy,
    }
    binding["binding_sha256"] = digest(binding)
    return binding


def verify_binding(binding: dict[str, Any], args: argparse.Namespace) -> None:
    if binding.get("schema") != SCHEMA:
        raise BindingError("binding schema mismatch")
    claimed = binding.get("binding_sha256")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    if digest(unsigned) != claimed:
        raise BindingError("binding digest does not verify")
    rebuilt = build_binding(args)
    if rebuilt != binding:
        raise BindingError("binding does not match independently re-hashed release/source/deploy inputs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tag-commit", required=True)
    parser.add_argument("--orchestrator-run-id", required=True)
    parser.add_argument("--plugin-sha256", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--model-catalog", type=Path, required=True)
    parser.add_argument("--backend-catalog", type=Path, required=True)
    parser.add_argument("--public-catalog", type=Path, required=True)
    parser.add_argument("--public-signature", type=Path, required=True)
    parser.add_argument("--sha256sums", type=Path, required=True)
    parser.add_argument("--deploy-run", type=Path, required=True)
    parser.add_argument("--asset", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.command == "build":
        args.binding.write_text(json.dumps(build_binding(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        verify_binding(read_json(args.binding), args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindingError as error:
        raise SystemExit(f"release correctness binding failed: {error}")
