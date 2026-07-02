#!/usr/bin/env python3
"""Audit publish-model candidates before writing registry or catalog state.

This script is intentionally read-only. It answers the question "what is the
next safe publishing action for each configured model?" from three inputs:

- tooling/publish-model/models-core.toml: expected model/quants;
- model-registry/catalog.json: committed public/staging catalog state;
- tmp/publish/<model>: local packs, result sidecars, metrics, and HF revision.

It does not hash large packs. Hash evidence must already exist in
<model>.<quant>.result.json, either from convert.sh or materialize_result_sidecars.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from _catalog import CATALOG_CORE, load as load_publish_catalog
from _file_loaders import load_json, load_toml
from _pathlib_helpers import repo_root

DEFAULT_PUBLISH_CATALOG = CATALOG_CORE
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
HF_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = repo_root(SCRIPT_DIR)


@dataclass
class QuantReadiness:
    quant: str
    pack_exists: bool
    result_exists: bool
    metric_exists: bool
    size_bytes: int | None
    sha256: str | None
    issues: list[str]


@dataclass
class ModelReadiness:
    model: str
    status: str
    catalog_state: str
    expected_quants: list[str]
    artifact_root: str
    hf_repo: str | None
    hf_revision: str | None
    quant_readiness: list[QuantReadiness]
    blockers: list[str]
    next_action: str


def catalog_models(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    models = data.get("models", [])
    if not isinstance(models, list):
        raise SystemExit(f"catalog models must be a list: {path}")
    return {model.get("id"): model for model in models if isinstance(model, dict)}


def read_text_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text().strip() or None


def load_sidecar(path: Path, model: str, quant: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"missing result sidecar for {quant}"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return None, [f"invalid result sidecar for {quant}: {error}"]
    if not isinstance(data, dict):
        return None, [f"result sidecar for {quant} is not an object"]

    issues: list[str] = []
    if data.get("model") != model:
        issues.append(f"result sidecar model mismatch for {quant}")
    if data.get("quant") != quant:
        issues.append(f"result sidecar quant mismatch for {quant}")
    size = data.get("size_bytes")
    if not isinstance(size, int) or size <= 0:
        issues.append(f"invalid size_bytes in result sidecar for {quant}")
    sha = data.get("sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        issues.append(f"invalid sha256 in result sidecar for {quant}")
    pack = data.get("pack")
    if not isinstance(pack, str) or not pack.endswith(".oasr"):
        issues.append(f"invalid pack path in result sidecar for {quant}")
    elif not Path(pack).exists():
        issues.append(f"result sidecar pack path missing for {quant}")
    return data, issues


def metrics_for_quant(path: Path, quant: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"missing metrics.json for {quant}"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return None, [f"invalid metrics.json: {error}"]
    if not isinstance(data, dict):
        return None, ["metrics.json is not an object"]
    quants = data.get("quants")
    if not isinstance(quants, dict):
        return None, ["metrics.json missing quants object"]
    metric = quants.get(quant)
    if not isinstance(metric, dict):
        return None, [f"metrics.json missing quant {quant}"]
    measured = any(metric.get(key) is not None for key in ("rtf_cpu", "rtf_metal", "peak_rss_bytes", "jfk_wer_vs_fp16"))
    if not measured:
        return metric, [f"metrics.json quant {quant} has no measured values"]
    return metric, []


def quant_readiness(model_root: Path, model: str, quant: str) -> QuantReadiness:
    pack = model_root / "packs" / f"{model}-{quant}.oasr"
    sidecar_path = model_root / "packs" / f"{model}.{quant}.result.json"
    sidecar, sidecar_issues = load_sidecar(sidecar_path, model, quant)
    metric, metric_issues = metrics_for_quant(model_root / "metrics.json", quant)

    issues = list(sidecar_issues) + list(metric_issues)
    if not pack.exists():
        issues.append(f"missing pack for {quant}")
    if sidecar is not None and metric is not None:
        sidecar_size = sidecar.get("size_bytes")
        metric_size = metric.get("size_bytes")
        if metric_size is not None and metric_size != sidecar_size:
            issues.append(f"metrics size mismatch for {quant}")

    return QuantReadiness(
        quant=quant,
        pack_exists=pack.exists(),
        result_exists=sidecar_path.exists(),
        metric_exists=metric is not None,
        size_bytes=sidecar.get("size_bytes") if sidecar else None,
        sha256=sidecar.get("sha256") if sidecar else None,
        issues=issues,
    )


def catalog_issues(model: dict[str, Any] | None, expected_quants: list[str]) -> list[str]:
    if model is None:
        return []
    issues: list[str] = []
    if not isinstance(model.get("hf_repo"), str) or "/" not in model.get("hf_repo", ""):
        issues.append("catalog missing hf_repo")
    revision = model.get("hf_revision")
    if not isinstance(revision, str) or not HF_REVISION_RE.fullmatch(revision):
        issues.append("catalog hf_revision is not a 40-hex commit")
    if not isinstance(model.get("public"), bool):
        issues.append("catalog public field is not boolean")
    quants = model.get("quants")
    if not isinstance(quants, list):
        return issues + ["catalog quants field is not a list"]
    found = [quant.get("quant") for quant in quants if isinstance(quant, dict)]
    missing = [quant for quant in expected_quants if quant not in found]
    if missing:
        issues.append(f"catalog missing quants: {', '.join(missing)}")
    for quant in quants:
        if not isinstance(quant, dict):
            issues.append("catalog quant entry is not an object")
            continue
        name = quant.get("quant", "<unknown>")
        if not isinstance(quant.get("sha256"), str) or not SHA256_RE.fullmatch(quant.get("sha256", "")):
            issues.append(f"catalog {name} missing sha256")
        if not isinstance(quant.get("size_bytes"), int) or quant.get("size_bytes", 0) <= 0:
            issues.append(f"catalog {name} missing size_bytes")
        perf = quant.get("perf")
        if not isinstance(perf, dict) or not any(perf.get(key) is not None for key in ("rtf_cpu", "rtf_metal", "peak_rss_bytes", "jfk_wer_vs_fp16")):
            issues.append(f"catalog {name} missing perf")
    return issues


def status_for(
    *,
    catalog_model: dict[str, Any] | None,
    catalog_blockers: list[str],
    quant_rows: list[QuantReadiness],
    hf_repo: str | None,
    hf_revision: str | None,
) -> tuple[str, list[str], str]:
    blockers: list[str] = []
    artifact_ready = all(not row.issues for row in quant_rows)
    revision_ready = hf_revision is not None and HF_REVISION_RE.fullmatch(hf_revision) is not None
    repo_ready = hf_repo is not None and "/" in hf_repo

    if catalog_model is not None and not catalog_blockers:
        if catalog_model.get("public") is True:
            return (
                "public_cataloged",
                [],
                "No onboarding action; rerun public gate and E2E only if catalog metadata changes.",
            )
        return (
            "staging_cataloged",
            [],
            "Keep public:false until the HF repo is public and anonymous public_gate.py passes.",
        )

    if artifact_ready and revision_ready and repo_ready:
        return (
            "ready_for_manifest",
            catalog_blockers,
            "Run _registry.py and _manifest.py; the manifest defaults to public:false.",
        )

    for row in quant_rows:
        blockers.extend(row.issues)
    if not repo_ready:
        blockers.append("missing hf_repo.txt or catalog hf_repo")
    if not revision_ready:
        blockers.append("missing immutable hf_revision.txt or catalog hf_revision")

    if artifact_ready and not revision_ready:
        return (
            "needs_private_hf_upload",
            sorted(set(blockers)),
            "Run publish.sh privately to record hf_repo.txt/hf_revision.txt, then manifest.",
        )
    return (
        "needs_artifacts",
        sorted(set(blockers)),
        "Run download/convert, materialize sidecars if needed, then bench sequentially.",
    )


def audit_model(
    *,
    model: str,
    config: dict[str, Any],
    artifact_root: Path,
    catalog_model: dict[str, Any] | None,
) -> ModelReadiness:
    expected_quants = list(config.get("quants", []))
    model_root = artifact_root / model
    quant_rows = [quant_readiness(model_root, model, quant) for quant in expected_quants]
    local_hf_repo = read_text_file(model_root / "hf_repo.txt")
    local_hf_revision = read_text_file(model_root / "hf_revision.txt")
    hf_repo = local_hf_repo or (catalog_model or {}).get("hf_repo")
    hf_revision = local_hf_revision or (catalog_model or {}).get("hf_revision")
    cat_blockers = catalog_issues(catalog_model, expected_quants)
    status, blockers, next_action = status_for(
        catalog_model=catalog_model,
        catalog_blockers=cat_blockers,
        quant_rows=quant_rows,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
    )
    if catalog_model is None:
        catalog_state = "absent"
    elif catalog_model.get("public") is True:
        catalog_state = "public"
    else:
        catalog_state = "staging"
    return ModelReadiness(
        model=model,
        status=status,
        catalog_state=catalog_state,
        expected_quants=expected_quants,
        artifact_root=str(model_root),
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        quant_readiness=quant_rows,
        blockers=blockers,
        next_action=next_action,
    )


def audit_models(
    *,
    publish_catalog: Path,
    machine_catalog: Path,
    artifact_root: Path,
    selected_models: list[str] | None = None,
) -> list[ModelReadiness]:
    publish_models = load_publish_catalog() if publish_catalog == DEFAULT_PUBLISH_CATALOG else load_toml(publish_catalog)
    machine_models = catalog_models(machine_catalog)
    names = selected_models if selected_models else sorted(publish_models)
    unknown = sorted(set(names) - set(publish_models))
    if unknown:
        raise SystemExit(f"unknown publish model(s): {', '.join(unknown)}")
    return [
        audit_model(
            model=name,
            config=publish_models[name],
            artifact_root=artifact_root,
            catalog_model=machine_models.get(name),
        )
        for name in names
    ]


def status_rank(status: str) -> int:
    return {
        "ready_for_manifest": 0,
        "needs_private_hf_upload": 1,
        "staging_cataloged": 2,
        "needs_artifacts": 3,
        "public_cataloged": 4,
    }.get(status, 99)


def render_text(rows: list[ModelReadiness]) -> str:
    ordered = sorted(rows, key=lambda row: (status_rank(row.status), row.model))
    lines = ["model\tstatus\tcatalog\tquants\tblockers\tnext_action"]
    for row in ordered:
        blockers = "; ".join(row.blockers) if row.blockers else "-"
        lines.append(
            "\t".join(
                [
                    row.model,
                    row.status,
                    row.catalog_state,
                    ",".join(row.expected_quants),
                    blockers,
                    row.next_action,
                ]
            )
        )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--publish-catalog", type=Path, default=DEFAULT_PUBLISH_CATALOG)
    parser.add_argument("--machine-catalog", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    machine_catalog = args.machine_catalog or (repo_root / "model-registry" / "catalog.json")
    artifact_root = args.artifact_root or (repo_root / "tmp" / "publish")
    rows = audit_models(
        publish_catalog=args.publish_catalog,
        machine_catalog=machine_catalog,
        artifact_root=artifact_root,
        selected_models=args.models,
    )
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    else:
        print(render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
