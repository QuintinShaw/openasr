#!/usr/bin/env python3
"""No-network tests for the public-HF E2E helper.

The real public-HF smoke still downloads a public pack and runs native
transcription. These tests only cover local argument/evidence guard behavior so
CI can keep the helper safe without doing network I/O.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SH = REPO_ROOT / "tooling" / "public-hf-e2e" / "run.sh"


def run_helper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUN_SH), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class PublicHfE2ETests(unittest.TestCase):
    def test_dry_run_summary_is_redacted_and_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_json = Path(temp) / "nested" / "summary.json"
            summary_md = Path(temp) / "nested" / "summary.md"

            result = run_helper(
                "--dry-run",
                "--audio",
                str(REPO_ROOT / "fixtures" / "jfk.wav"),
                "--summary-json",
                str(summary_json),
                "--summary-md",
                str(summary_md),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            rendered = json.dumps(summary, sort_keys=True)
            markdown = summary_md.read_text(encoding="utf-8")

            self.assertTrue(summary["dry_run"])
            self.assertFalse(summary["executed"])
            self.assertEqual(summary["audio_file"], "jfk.wav")
            self.assertEqual(summary["tool"], "public-hf-e2e")
            self.assertTrue(summary["catalog_is_canonical_public_hf"])
            self.assertNotIn(str(REPO_ROOT), rendered)
            self.assertNotIn(str(REPO_ROOT), markdown)
            self.assertIn("Public-HF E2E evidence", markdown)

    def test_strict_evidence_rejects_dry_run_before_writing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_json = Path(temp) / "summary.json"

            result = run_helper(
                "--dry-run",
                "--strict-evidence",
                "--summary-json",
                str(summary_json),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--strict-evidence cannot be used with --dry-run", result.stderr)
            self.assertFalse(summary_json.exists())

    def test_strict_evidence_requires_canonical_public_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_json = Path(temp) / "summary.json"

            result = run_helper(
                "--strict-evidence",
                "--catalog-url",
                "model-registry/catalog.json",
                "--summary-json",
                str(summary_json),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires the canonical public catalog URL", result.stderr)
            self.assertFalse(summary_json.exists())

    def test_strict_evidence_rejects_markdown_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_md = Path(temp) / "summary.md"

            result = run_helper(
                "--strict-evidence",
                "--summary-md",
                str(summary_md),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--summary-json is required with --strict-evidence", result.stderr)
            self.assertFalse(summary_md.exists())

    def test_summary_path_directory_fails_before_dry_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_dir = Path(temp) / "summary.json"
            summary_dir.mkdir()

            result = run_helper("--dry-run", "--summary-json", str(summary_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--summary-json path is a directory", result.stderr)

    def test_help_lists_evidence_options(self) -> None:
        result = run_helper("--help")

        self.assertEqual(result.returncode, 0)
        self.assertIn("--summary-json", result.stdout)
        self.assertIn("--summary-md", result.stdout)
        self.assertIn("--strict-evidence", result.stdout)


if __name__ == "__main__":
    unittest.main()
