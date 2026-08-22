#!/usr/bin/env python3
"""Fail-closed release-audit-form gate for model families.

Policy (docs/model-audits/README.md): before a family's first catalog entry
may flip `public:true`, a completed release audit form must exist at
docs/model-audits/<family>.md, copied from docs/model-audits/TEMPLATE.md. The
form records, per performance/completeness dimension, whether the family ships
in its best known state -- and a detailed justification plus unlock condition
for anything consciously skipped.

This module validates both mechanical completeness and the structured backend
cell semantics projected from the architecture inventory and public model
catalog. A public family/provider lane cannot be approved by prose, a CPU
result, placement-only evidence, or an old migration exemption. _manifest.py
calls it on every `--public` write.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from _pathlib_helpers import repo_root

AUDIT_DIR_RELATIVE = Path("docs") / "model-audits"
TEMPLATE_RELATIVE = AUDIT_DIR_RELATIVE / "TEMPLATE.md"

# The migration ledger remains exported for source-tree and documentation checks.
def _load_pre_audit_families() -> frozenset[str]:
    path = repo_root(Path(__file__).resolve().parent) / AUDIT_DIR_RELATIVE / "pre_audit_families.txt"
    families: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            families.add(line)
    return frozenset(families)


PRE_AUDIT_FAMILIES = _load_pre_audit_families()
# The sentinel TEMPLATE.md places at every fill site. A published form must
# have replaced every occurrence; one leftover marker means a half-filled form.
FILL_SENTINEL = "<!-- TODO:fill -->"

# The ten audit dimensions. Heading lines are matched verbatim; renaming or
# deleting one in a family form (or in TEMPLATE.md) fails the gate.
REQUIRED_SECTIONS = (
    "## 1. Graph & scheduling",
    "## 2. Precision & quantization",
    "## 3. Memory & data movement",
    "## 4. Decode algorithms",
    "## 5. Frontend & IO",
    "## 6. Platform-specific",
    "## 7. Backend coverage matrix",
    "## 8. Correctness & quality",
    "## 9. Resource limits & fail-closed",
    "## 10. Engineering completeness",
)

# The migration ledger is loaded once above and is not a release exemption.
MODEL_INVENTORY_RELATIVE = Path("tooling") / "model-family-inventory.v1.json"
MODEL_CATALOG_RELATIVE = Path("model-registry") / "catalog.json"
BACKEND_CELL_HEADER = "| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |"
BACKEND_NAMES = ("cpu", "metal", "cuda", "vulkan", "hip")


class AuditFormError(RuntimeError):
    """Raised when a family's release audit form blocks a public release."""



def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditFormError(f"{path} must contain a JSON object")
    return value


def _status(value: str) -> str:
    match = re.match(r"\s*([A-Za-z]+)", value)
    return match.group(1).lower() if match else ""


def _advertised_lanes(family: str, inventory_path: Path, catalog_path: Path) -> dict[str, tuple[str, ...]]:
    """Project only lanes the live inventory exposes for public models.

    The audit form is documentation, but its backend cells are a release input.
    This projection deliberately does not infer support from a shared ggml path.
    """
    inventory = _load_json(inventory_path)
    catalog = _load_json(catalog_path)
    families = inventory.get("families")
    models = catalog.get("models")
    if not isinstance(families, list) or not isinstance(models, list):
        raise AuditFormError("architecture inventory and model catalog must contain arrays")
    descriptor = next(
        (item for item in families if isinstance(item, dict) and item.get("catalog_family_id") == family),
        None,
    )
    public = any(
        isinstance(item, dict)
        and item.get("family") == family
        and item.get("kind", "asr-model") == "asr-model"
        and item.get("public") is True
        for item in models
    )
    if descriptor is None or not public:
        return {}
    execution = descriptor.get("execution", {})
    capabilities = execution.get("execution_capabilities", {}) if isinstance(execution, dict) else {}
    optimization = descriptor.get("optimization", {})
    auto_policy = optimization.get("auto_gpu_policy") if isinstance(optimization, dict) else None
    lanes: dict[str, tuple[str, ...]] = {}
    if capabilities.get("cpu") is True:
        lanes["cpu"] = ("explicit", "auto")
    providers = capabilities.get("providers", [])
    if isinstance(providers, list):
        for item in providers:
            if not isinstance(item, dict) or not isinstance(item.get("provider"), str):
                continue
            provider = item["provider"].lower()
            if item.get("full_device") is not True and item.get("hybrid") is not True:
                continue
            modes = ["explicit"]
            if auto_policy == "all-backends" or (
                auto_policy == "except-metal" and provider != "metal"
            ):
                modes.append("auto")
            lanes[provider] = tuple(modes)
    return lanes


def _parse_backend_cells(text: str) -> dict[str, tuple[str, str, str, str]]:
    """Parse the fixed backend table without treating prose as evidence."""
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == BACKEND_CELL_HEADER)
    except StopIteration:
        return {}
    cells: dict[str, tuple[str, str, str, str]] = {}
    for line in lines[start + 2 :]:
        if not line.lstrip().startswith("|"):
            if cells:
                break
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 5:
            continue
        provider = parts[0].lower()
        if provider in BACKEND_NAMES:
            cells[provider] = (parts[1], parts[2], parts[3], parts[4])
    return cells


def _validate_backend_semantics(
    family: str,
    text: str,
    *,
    inventory_path: Path,
    catalog_path: Path,
) -> None:
    lanes = _advertised_lanes(family, inventory_path, catalog_path)
    if not lanes:
        return
    cells = _parse_backend_cells(text)
    if not cells:
        raise AuditFormError(
            f"release audit form for public family '{family}' has no structured backend coverage table"
        )
    stale_terms = ("untested", "deferred", "stale", "historical", "pending")
    for provider, modes in sorted(lanes.items()):
        row = cells.get(provider)
        if row is None:
            raise AuditFormError(
                f"family '{family}' advertises {provider} ({'/'.join(modes)}) but its backend cell is missing"
            )
        supported, golden, utilization, justification = row
        values = (supported, golden, utilization)
        if any(not value.strip() for value in values):
            raise AuditFormError(f"family '{family}' has an incomplete {provider} backend cell")
        if any(any(term in value.lower() for term in stale_terms) for value in values):
            raise AuditFormError(
                f"family '{family}' {provider} backend cell contains unproven or stale status"
            )
        if _status(supported) not in {"yes", "supported"}:
            raise AuditFormError(
                f"family '{family}' advertises {provider} ({'/'.join(modes)}) but Supported? is {supported!r}"
            )
        if _status(golden) not in {"yes", "supported"}:
            raise AuditFormError(
                f"family '{family}' advertises {provider} but Golden-verified? is {golden!r}"
            )
        utilization_status = _status(utilization)
        if provider == "cpu" and utilization_status in {"n/a", "na"}:
            pass
        elif utilization_status not in {"yes", "supported"}:
            raise AuditFormError(
                f"family '{family}' advertises {provider} but Utilization measured? is {utilization!r}"
            )
        if not justification.strip() and not any(
            marker in value for value in values for marker in ("(", "`", "#", "http")
        ):
            raise AuditFormError(f"family '{family}' {provider} backend cell has no evidence text")


def default_audit_dir() -> Path:
    return repo_root(Path(__file__).resolve().parent) / AUDIT_DIR_RELATIVE


def validate_family_audit_form(
    family: str,
    *,
    audit_dir: Path | None = None,
    inventory_path: Path | None = None,
    catalog_path: Path | None = None,
) -> None:
    """Refuse a public release unless its form and advertised lanes are complete.

    Missing forms always fail. ``PRE_AUDIT_FAMILIES`` remains a migration ledger
    for reporting and source-tree audits, but is not an evidence-free release
    exemption. For a family present in the live projections, every Auto or
    explicitly selectable backend cell must be a current, positive answer.
    """
    if not isinstance(family, str) or not family.strip():
        raise AuditFormError("model family is missing; cannot locate its release audit form")
    directory = audit_dir if audit_dir is not None else default_audit_dir()
    path = directory / f"{family}.md"
    if not path.exists():
        raise AuditFormError(
            f"family '{family}' has no release audit form at {path}; copy "
            f"{TEMPLATE_RELATIVE} to {AUDIT_DIR_RELATIVE / (family + '.md')} and complete it "
            "before releasing (see docs/model-audits/README.md)"
        )
    text = path.read_text()
    leftover = text.count(FILL_SENTINEL)
    if leftover:
        raise AuditFormError(
            f"release audit form {path} still contains {leftover} '{FILL_SENTINEL}' "
            "marker(s); complete every fill site before releasing"
        )
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    if missing:
        raise AuditFormError(
            f"release audit form {path} is missing required section(s): "
            f"{', '.join(missing)}; restore the ten headings from {TEMPLATE_RELATIVE}"
        )
    root = repo_root(Path(__file__).resolve().parent)
    _validate_backend_semantics(
        family,
        text,
        inventory_path=inventory_path or root / MODEL_INVENTORY_RELATIVE,
        catalog_path=catalog_path or root / MODEL_CATALOG_RELATIVE,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: audit_form.py <family>", file=sys.stderr)
        return 2
    try:
        validate_family_audit_form(argv[0])
    except AuditFormError as error:
        print(f"release-audit gate failed: {error}", file=sys.stderr)
        return 1
    print(f"release-audit gate passed: {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
