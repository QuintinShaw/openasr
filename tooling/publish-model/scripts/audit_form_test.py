#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import _manifest  # noqa: E402
import audit_form  # noqa: E402
from _pathlib_helpers import repo_root  # noqa: E402

TEMPLATE_PATH = repo_root(SCRIPT_DIR) / "docs" / "model-audits" / "TEMPLATE.md"


def completed_form_text() -> str:
    """A form derived the way contributors derive one: copy TEMPLATE.md and
    replace every fill marker."""
    return TEMPLATE_PATH.read_text().replace(audit_form.FILL_SENTINEL, "Supported")


class AuditFormTemplateTest(unittest.TestCase):
    def test_template_contains_every_required_section_and_fill_markers(self) -> None:
        text = TEMPLATE_PATH.read_text()

        for section in audit_form.REQUIRED_SECTIONS:
            self.assertIn(section, text)
        self.assertIn(audit_form.FILL_SENTINEL, text)

    def test_template_mentions_the_fill_marker_only_at_fill_sites(self) -> None:
        """The gate counts raw FILL_SENTINEL occurrences, so the template's
        prose must never spell the marker out verbatim (e.g. inside backticks
        in the how-to paragraph): a contributor who keeps the instructions and
        fills every real site would otherwise fail the gate forever. Fill
        sites are table rows (contain '|') or the title line ('# ')."""
        for line in TEMPLATE_PATH.read_text().splitlines():
            if audit_form.FILL_SENTINEL in line:
                self.assertTrue(
                    "|" in line or line.startswith("# "),
                    f"template prose spells out the fill marker verbatim: {line!r}",
                )


class ValidateFamilyAuditFormTest(unittest.TestCase):
    def test_missing_form_fails_closed_for_a_new_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(audit_form.AuditFormError, "no release audit form"):
                audit_form.validate_family_audit_form("new-family", audit_dir=Path(temp))

    def test_missing_form_is_tolerated_for_a_pre_audit_family(self) -> None:
        self.assertIn("whisper", audit_form.PRE_AUDIT_FAMILIES)
        with tempfile.TemporaryDirectory() as temp:
            audit_form.validate_family_audit_form("whisper", audit_dir=Path(temp))

    def test_half_filled_form_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audit_dir = Path(temp)
            text = completed_form_text() + f"\n| Late item | {audit_form.FILL_SENTINEL} | |\n"
            (audit_dir / "new-family.md").write_text(text)

            with self.assertRaisesRegex(audit_form.AuditFormError, "1 '<!-- TODO:fill -->' marker"):
                audit_form.validate_family_audit_form("new-family", audit_dir=audit_dir)

    def test_half_filled_form_fails_even_for_a_pre_audit_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audit_dir = Path(temp)
            (audit_dir / "whisper.md").write_text(TEMPLATE_PATH.read_text())

            with self.assertRaisesRegex(audit_form.AuditFormError, "marker"):
                audit_form.validate_family_audit_form("whisper", audit_dir=audit_dir)

    def test_form_missing_a_section_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audit_dir = Path(temp)
            text = completed_form_text().replace("## 7. Backend coverage matrix", "## 7. Backends")
            (audit_dir / "new-family.md").write_text(text)

            with self.assertRaisesRegex(
                audit_form.AuditFormError, r"missing required section\(s\): ## 7\. Backend coverage matrix"
            ):
                audit_form.validate_family_audit_form("new-family", audit_dir=audit_dir)

    def test_completed_form_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audit_dir = Path(temp)
            (audit_dir / "new-family.md").write_text(completed_form_text())

            audit_form.validate_family_audit_form("new-family", audit_dir=audit_dir)

    def test_missing_family_value_fails_closed(self) -> None:
        with self.assertRaisesRegex(audit_form.AuditFormError, "family is missing"):
            audit_form.validate_family_audit_form("", audit_dir=Path("/nonexistent"))


class ManifestAuditGateTest(unittest.TestCase):
    def test_public_generation_requires_a_completed_audit_form(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(SystemExit) as error:
                _manifest.ensure_release_audit_form(
                    "new-model",
                    {"registry_id": "new-model", "family": "new-family"},
                    True,
                    audit_dir=Path(temp),
                )

        self.assertIn("release-audit gate failed", str(error.exception))

    def test_public_generation_accepts_a_completed_audit_form(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audit_dir = Path(temp)
            (audit_dir / "new-family.md").write_text(completed_form_text())

            _manifest.ensure_release_audit_form(
                "new-model",
                {"registry_id": "new-model", "family": "new-family"},
                True,
                audit_dir=audit_dir,
            )

    def test_private_generation_does_not_require_an_audit_form(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _manifest.ensure_release_audit_form(
                "new-model",
                {"registry_id": "new-model", "family": "new-family"},
                False,
                audit_dir=Path(temp),
            )


if __name__ == "__main__":
    unittest.main()
