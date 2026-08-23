#!/usr/bin/env python3
from __future__ import annotations

import json
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
    return TEMPLATE_PATH.read_text().replace(audit_form.FILL_SENTINEL, "Supported").replace(
        "| Supported | Supported | Supported | |",
        "| Supported | Supported | Supported | evidence |",
    )


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

    def test_missing_form_fails_closed_for_a_pre_audit_family(self) -> None:
        self.assertIn("whisper", audit_form.PRE_AUDIT_FAMILIES)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(audit_form.AuditFormError, "no release audit form"):
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
    def advertised_gpu_sources(self, root: Path, family: str = "gpu-family") -> tuple[Path, Path]:
        inventory = root / "inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "schema": "openasr.model-family-inventory.v1",
                    "families": [
                        {
                            "catalog_family_id": family,
                            "execution": {
                                "execution_capabilities": {
                                    "cpu": True,
                                    "providers": [
                                        {"provider": "cuda", "full_device": True, "hybrid": False},
                                        {"provider": "vulkan", "full_device": True, "hybrid": False},
                                        {"provider": "hip", "full_device": True, "hybrid": False},
                                    ],
                                }
                            },
                            "optimization": {"auto_gpu_policy": "all-backends"},
                        }
                    ],
                }
            )
        )
        catalog = root / "catalog.json"
        catalog.write_text(
            json.dumps(
                {
                    "models": [
                        {"id": "gpu", "family": family, "kind": "asr-model", "public": True}
                    ]
                }
            )
        )
        return inventory, catalog

    def test_advertised_unproven_backend_cell_fails_closed(self) -> None:
        cases = (
            ("Deferred", "unproven or stale status"),
            ("Untested", "unproven or stale status"),
            ("Supported (stale)", "unproven or stale status"),
        )
        for status, pattern in cases:
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    audit_dir = root / "audits"
                    audit_dir.mkdir()
                    text = completed_form_text().replace(
                        "| CUDA | Supported | Supported | Supported | evidence |",
                        f"| CUDA | {status} | No | No | not tested; unlock: run the matrix |",
                    )
                    (audit_dir / "gpu-family.md").write_text(text)
                    inventory, catalog = self.advertised_gpu_sources(root)
                    with self.assertRaisesRegex(audit_form.AuditFormError, pattern):
                        audit_form.validate_family_audit_form(
                            "gpu-family",
                            audit_dir=audit_dir,
                            inventory_path=inventory,
                            catalog_path=catalog,
                        )

    def test_missing_advertised_backend_cell_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit_dir = root / "audits"
            audit_dir.mkdir()
            text = "\n".join(
                line
                for line in completed_form_text().splitlines()
                if not line.startswith("| CUDA |")
            )
            (audit_dir / "gpu-family.md").write_text(text + "\n")
            inventory, catalog = self.advertised_gpu_sources(root)
            with self.assertRaisesRegex(audit_form.AuditFormError, "cuda"):
                audit_form.validate_family_audit_form(
                    "gpu-family",
                    audit_dir=audit_dir,
                    inventory_path=inventory,
                    catalog_path=catalog,
                )

    def test_public_generation_blocks_untested_advertised_gpu_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit_dir = root / "audits"
            audit_dir.mkdir()
            text = completed_form_text().replace(
                "| CUDA | Supported | Supported | Supported | evidence |",
                "| CUDA | Untested | No | No | not tested; unlock: run the matrix |",
            ).replace(
                "| Vulkan | Supported | Supported | Supported | evidence |",
                "| Vulkan | Untested | No | No | software lavapipe is not physical evidence |",
            ).replace(
                "| HIP | Supported | Supported | Supported | evidence |",
                "| HIP | Deferred | No | No | no HIP host; unlock: run the matrix |",
            )
            (audit_dir / "gpu-family.md").write_text(text)
            inventory, catalog = self.advertised_gpu_sources(root)
            with self.assertRaises(SystemExit) as error:
                _manifest.ensure_release_audit_form(
                    "gpu",
                    {"registry_id": "gpu", "family": "gpu-family"},
                    True,
                    audit_dir=audit_dir,
                    inventory_path=inventory,
                    catalog_path=catalog,
                )
            message = str(error.exception)
            self.assertIn("release-audit gate failed", message)
            self.assertRegex(message, "unproven or stale status|cuda backend cell|vulkan|hip")

    def test_explicit_nonactivation_is_not_an_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit_dir = root / "audits"
            audit_dir.mkdir()
            text = completed_form_text().replace(
                "| CUDA | Supported | Supported | Supported | evidence |",
                "| CUDA | No | No | No | not shipped; unlock: run the provider matrix |",
            )
            (audit_dir / "cpu-family.md").write_text(text)
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema": "openasr.model-family-inventory.v1",
                        "families": [
                            {
                                "catalog_family_id": "cpu-family",
                                "execution": {"execution_capabilities": {"cpu": True, "providers": []}},
                                "optimization": {"auto_gpu_policy": "cpu-only"},
                            }
                        ],
                    }
                )
            )
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps({"models": [{"family": "cpu-family", "kind": "asr-model", "public": True}]})
            )
            audit_form.validate_family_audit_form(
                "cpu-family", audit_dir=audit_dir, inventory_path=inventory, catalog_path=catalog
            )

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
