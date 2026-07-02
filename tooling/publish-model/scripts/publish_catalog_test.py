#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from _catalog import CATALOG_URL


SCRIPT = Path(__file__).with_name("publish_catalog.sh")


def catalog_with(models: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-05-31T00:00:00Z",
        "catalog_url": CATALOG_URL,
        "models": models,
    }


class PublishCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.catalog = self.root / "catalog.json"
        self.public_dir = self.root / "public"
        self.cargo_log = self.root / "cargo.args"

        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        cargo = bin_dir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            "{ printf 'COMMAND\\n'; printf '%s\\n' \"$@\"; } >> \"$OPENASR_FAKE_CARGO_LOG\"\n"
            "if [[ \"$1\" == \"run\" ]]; then\n"
            "  out=''\n"
            "  prev=''\n"
            "  for arg in \"$@\"; do\n"
            "    if [[ \"$prev\" == \"--out\" ]]; then out=\"$arg\"; break; fi\n"
            "    prev=\"$arg\"\n"
            "  done\n"
            "  [[ -n \"$out\" ]] || exit 2\n"
            "  mkdir -p \"$(dirname \"$out\")\"\n"
            f"  printf '{{\"schema_version\":1,\"catalog_url\":\"{CATALOG_URL}\",\"catalog_sha256\":\"%064d\",\"catalog_epoch\":2026060101,\"signature\":{{\"algorithm\":\"ed25519\",\"key_id\":\"openasr-catalog-v1\",\"value\":\"%0128d\"}}}}\\n' 0 0 > \"$out\"\n"
            "fi\n"
            "exit 0\n"
        )
        cargo.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = f"{bin_dir}{os.pathsep}{self.env['PATH']}"
        self.env["OPENASR_FAKE_CARGO_LOG"] = str(self.cargo_log)
        self.env["OPENASR_CATALOG_SRC"] = str(self.catalog)
        self.env["OPENASR_PUBLIC_DIR"] = str(self.public_dir)
        self.env["OPENASR_CATALOG_EPOCH"] = "2026060101"
        self.env["OPENASR_CATALOG_SIGNING_KEY_SEED_HEX"] = "a" * 64
        self.env.pop("HF_TOKEN", None)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_publish_catalog(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), "--dry-run", *args],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_writes_public_projection_and_reports_count(self) -> None:
        self.catalog.write_text(
            json.dumps(
                catalog_with(
                    [
                        {"id": "public-model", "public": True},
                        {"id": "private-model", "public": False},
                    ]
                )
            )
        )

        result = self.run_publish_catalog()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("signed public catalog projection written", result.stderr)
        self.assertIn("1 public model(s)", result.stderr)
        self.assertIn("committed artifacts not touched", result.stderr)
        projection = json.loads((self.public_dir / "catalog.json").read_text())
        self.assertEqual([model["id"] for model in projection["models"]], ["public-model"])
        manifest = json.loads((self.public_dir / "catalog.signature.json").read_text())
        self.assertEqual(manifest["catalog_epoch"], 2026060101)
        self.assertEqual(manifest["signature"]["algorithm"], "ed25519")
        commands = self.cargo_log.read_text().split("COMMAND\n")
        self.assertEqual(commands[0], "")
        self.assertEqual(
            commands[1].splitlines(),
            [
                "test",
                "-p",
                "openasr-core",
                "bundled_catalog_json_parses_and_matches_registry_cards",
            ],
        )
        self.assertEqual(
            commands[2].splitlines()[:7],
            [
                "run",
                "--quiet",
                "-p",
                "openasr-cli",
                "--",
                "__openasr-sign-catalog-manifest",
                str(self.public_dir / "catalog.json"),
            ],
        )

    def test_dry_run_can_write_redacted_summary_artifacts(self) -> None:
        self.catalog.write_text(
            json.dumps(
                catalog_with(
                    [
                        {"id": "public-model", "public": True},
                        {"id": "private-model", "public": False},
                    ]
                )
            )
        )
        summary_json = self.root / "summary" / "catalog.json"
        summary_md = self.root / "summary" / "catalog.md"

        result = self.run_publish_catalog(
            "--summary-json",
            str(summary_json),
            "--summary-md",
            str(summary_md),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(summary_json.read_text())
        rendered_summary = json.dumps(summary, sort_keys=True)
        markdown = summary_md.read_text()
        self.assertEqual(summary["probe"], "catalog_publish")
        self.assertEqual(summary["target"], "catalog.openasr.org")
        self.assertTrue(summary["dry_run"])
        self.assertFalse(summary["strict_evidence"])
        self.assertFalse(summary["signed"])
        self.assertEqual(summary["public_model_count"], 1)
        self.assertEqual(summary["public_model_ids"], ["public-model"])
        self.assertEqual(summary["catalog_epoch"], 2026060101)
        self.assertEqual(summary["catalog_file"], "catalog.json")
        self.assertEqual(summary["manifest_file"], "catalog.signature.json")
        self.assertEqual(len(summary["catalog_sha256"]), 64)
        self.assertEqual(len(summary["manifest_sha256"]), 64)
        self.assertIn("Signed public catalog publish evidence", markdown)
        self.assertIn("- dry run: `True`", markdown)
        self.assertIn("- signed: `False`", markdown)
        self.assertIn("- target: `catalog.openasr.org`", markdown)
        self.assertIn("- public models: `public-model`", markdown)
        self.assertNotIn(self.env["OPENASR_CATALOG_SIGNING_KEY_SEED_HEX"], rendered_summary)
        self.assertNotIn(str(self.public_dir), rendered_summary)
        self.assertNotIn(str(self.public_dir), markdown)

    def test_strict_evidence_rejects_dry_run_before_writing_projection(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "public-model", "public": True}]))
        )

        result = self.run_publish_catalog(
            "--strict-evidence",
            "--summary-json",
            str(self.root / "summary.json"),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--strict-evidence cannot be used with --dry-run", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())
        self.assertFalse(self.cargo_log.exists())

    def test_strict_evidence_rejects_markdown_only_summary_before_writing_projection(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "public-model", "public": True}]))
        )

        result = subprocess.run(
            [
                str(SCRIPT),
                "--strict-evidence",
                "--summary-md",
                str(self.root / "summary.md"),
            ],
            env={**self.env, "HF_TOKEN": "redacted"},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--summary-json is required with --strict-evidence", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())
        self.assertFalse((self.root / "summary.md").exists())
        self.assertFalse(self.cargo_log.exists())

    def test_strict_evidence_requires_summary_next_to_public_artifacts(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "public-model", "public": True}]))
        )

        result = subprocess.run(
            [
                str(SCRIPT),
                "--strict-evidence",
                "--summary-json",
                str(self.root / "summary" / "publish-summary.json"),
            ],
            env={**self.env, "HF_TOKEN": "redacted"},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("next to catalog.json", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())
        self.assertFalse((self.public_dir / "catalog.signature.json").exists())
        self.assertFalse(self.cargo_log.exists())

    def test_summary_output_rejects_directory_before_writing_projection(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "public-model", "public": True}]))
        )
        summary_dir = self.root / "summary-dir"
        summary_dir.mkdir()

        result = self.run_publish_catalog("--summary-json", str(summary_dir))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("summary output points to a directory", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())
        self.assertFalse(self.cargo_log.exists())

    def test_dry_run_fails_when_public_projection_would_be_empty(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "private-model", "public": False}]))
        )

        result = self.run_publish_catalog()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no public:true models", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())

    def test_dry_run_requires_signing_key_before_writing_projection(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "public-model", "public": True}]))
        )
        self.env.pop("OPENASR_CATALOG_SIGNING_KEY_SEED_HEX", None)

        result = self.run_publish_catalog()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OPENASR_CATALOG_SIGNING_KEY_SEED_HEX", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())
        self.assertFalse((self.public_dir / "catalog.signature.json").exists())
        self.assertFalse(self.cargo_log.exists())

    def test_dry_run_rejects_malformed_signing_key_before_writing_projection(self) -> None:
        self.catalog.write_text(
            json.dumps(catalog_with([{"id": "public-model", "public": True}]))
        )
        self.env["OPENASR_CATALOG_SIGNING_KEY_SEED_HEX"] = "not-hex"

        result = self.run_publish_catalog()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("64-hex", result.stderr)
        self.assertFalse((self.public_dir / "catalog.json").exists())
        self.assertFalse((self.public_dir / "catalog.signature.json").exists())
        self.assertFalse(self.cargo_log.exists())


if __name__ == "__main__":
    unittest.main()
