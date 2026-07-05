#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_catalog_drift as drift  # noqa: E402


class ExtractSectionTest(unittest.TestCase):
    def test_extracts_between_start_and_end_marker(self) -> None:
        text = "before\n## Model support\nbody text\n## Benchmarks\nafter"

        section = drift.extract_section(text, "## Model support", ("\n## ",))

        self.assertEqual(section.strip(), "body text")

    def test_extracts_to_end_of_file_when_no_marker_matches(self) -> None:
        text = "## Model support\nbody text with no trailing heading"

        section = drift.extract_section(text, "## Model support", ("\n## ",))

        self.assertIn("body text with no trailing heading", section)

    def test_missing_start_marker_raises(self) -> None:
        with self.assertRaises(KeyError):
            drift.extract_section("no marker here", "## Model support", ("\n## ",))


class FamilyCountStringsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_root = drift.REPO_ROOT

    def tearDown(self) -> None:
        drift.REPO_ROOT = self.old_root

    def _run_with(self, readme_text: str) -> list[str]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drift.REPO_ROOT = root
            (root / "README.md").write_text(readme_text)
            errors: list[str] = []
            drift.check_family_count_strings(errors)
            return errors

    def test_matching_count_passes(self) -> None:
        expected = len(drift.LANG_BY_FAMILY)
        words = {v: k for k, v in drift.WORD_TO_NUMBER.items()}
        word = words[expected]
        errors = self._run_with(f"OpenASR runs {word} native families offline.")

        self.assertEqual(errors, [])

    def test_stale_count_fails(self) -> None:
        errors = self._run_with("OpenASR runs across three model families on CPU.")

        self.assertEqual(len(errors), 1)
        self.assertIn("family count phrase", errors[0])
        self.assertIn("says 3", errors[0])

    def test_digit_count_is_also_checked(self) -> None:
        errors = self._run_with("OpenASR runs 3 families offline.")

        self.assertEqual(len(errors), 1)
        self.assertIn("says 3", errors[0])


class PublicFamilyDocsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_root = drift.REPO_ROOT

    def tearDown(self) -> None:
        drift.REPO_ROOT = self.old_root

    def _catalog(self, family: str) -> dict:
        return {
            "models": [
                {"id": f"{family}-model", "family": family, "public": True, "kind": "asr-model"},
            ]
        }

    def test_documented_family_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drift.REPO_ROOT = root
            (root / "README.md").write_text("## Model support\n\nWhisper is supported.\n\n## Benchmarks\n")
            (root / "ACKNOWLEDGMENTS.md").write_text(
                "**Speech recognition**\n\n- Whisper -- link\n\n**Speaker diarization**\n"
            )
            errors: list[str] = []

            drift.check_public_family_docs(self._catalog("whisper"), errors)

            self.assertEqual(errors, [])

    def test_undocumented_public_family_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drift.REPO_ROOT = root
            (root / "README.md").write_text("## Model support\n\nWhisper is supported.\n\n## Benchmarks\n")
            (root / "ACKNOWLEDGMENTS.md").write_text(
                "**Speech recognition**\n\n- Whisper -- link\n\n**Speaker diarization**\n"
            )
            errors: list[str] = []

            drift.check_public_family_docs(self._catalog("dolphin"), errors)

            self.assertEqual(len(errors), 2)
            self.assertIn("README.md", errors[0])
            self.assertIn("ACKNOWLEDGMENTS.md", errors[1])

    def test_private_family_is_not_required_in_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drift.REPO_ROOT = root
            (root / "README.md").write_text("## Model support\n\nWhisper is supported.\n\n## Benchmarks\n")
            (root / "ACKNOWLEDGMENTS.md").write_text(
                "**Speech recognition**\n\n- Whisper -- link\n\n**Speaker diarization**\n"
            )
            catalog = {
                "models": [
                    {"id": "dolphin-model", "family": "dolphin", "public": False, "kind": "asr-model"},
                ]
            }
            errors: list[str] = []

            drift.check_public_family_docs(catalog, errors)

            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
