#!/usr/bin/env python3
"""Fail-closed release-audit-form gate for model families.

Policy (docs/model-audits/README.md): before a family's first catalog entry
may flip `public:true`, a completed release audit form must exist at
docs/model-audits/<family>.md, copied from docs/model-audits/TEMPLATE.md. The
form records, per performance/completeness dimension, whether the family ships
in its best known state -- and a detailed justification plus unlock condition
for anything consciously skipped.

This module validates the form's mechanical completeness only (present, no
leftover fill markers, all ten sections intact); the content quality is on the
auditor. _manifest.py calls it on every `--public` write.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _pathlib_helpers import repo_root

AUDIT_DIR_RELATIVE = Path("docs") / "model-audits"
TEMPLATE_RELATIVE = AUDIT_DIR_RELATIVE / "TEMPLATE.md"

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

# Families already public before the audit-form policy (2026-07). They are
# exempt from the missing-form check until backfilled on the rolling audit
# matrix; a form that DOES exist for them is still validated. Remove a family
# from this set when its form lands -- the set only shrinks, never grows.
PRE_AUDIT_FAMILIES = frozenset(
    {
        "cohere",
        "dolphin",
        "firered-aed",
        "firered-punc",
        "hymt2",
        "mimo-asr",
        "moonshine",
        "parakeet-tdt",
        "pyannote-segmentation",
        "qwen",
        "qwen3-forced-aligner",
        "sensevoice",
        "wespeaker",
        "whisper",
        "xasr-zipformer",
    }
)


class AuditFormError(RuntimeError):
    """Raised when a family's release audit form blocks a public release."""


def default_audit_dir() -> Path:
    return repo_root(Path(__file__).resolve().parent) / AUDIT_DIR_RELATIVE


def validate_family_audit_form(family: str, *, audit_dir: Path | None = None) -> None:
    """Refuse a public release unless docs/model-audits/<family>.md is complete.

    Fail-closed checks, in order: the form file exists (families in
    PRE_AUDIT_FAMILIES are exempt from this one check until backfilled), no
    FILL_SENTINEL marker remains, and every REQUIRED_SECTIONS heading is
    present. Raises AuditFormError with the offending path.
    """
    if not isinstance(family, str) or not family.strip():
        raise AuditFormError("model family is missing; cannot locate its release audit form")
    directory = audit_dir if audit_dir is not None else default_audit_dir()
    path = directory / f"{family}.md"
    if not path.exists():
        if family in PRE_AUDIT_FAMILIES:
            return
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
