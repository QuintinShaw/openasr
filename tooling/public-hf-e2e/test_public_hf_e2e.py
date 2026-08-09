#!/usr/bin/env python3
"""No-network tests for the public-HF E2E helper.

The real public-HF smoke still downloads a public pack and runs native
transcription. These tests only cover local argument/evidence guard behavior so
CI can keep the helper safe without doing network I/O.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SH = REPO_ROOT / "tooling" / "public-hf-e2e" / "run.sh"
PACK_PATH_HELPER = REPO_ROOT / "tooling" / "public-hf-e2e" / "installed_pack_path.py"

PACK_PATH_SPEC = importlib.util.spec_from_file_location(
    "installed_pack_path", PACK_PATH_HELPER
)
assert PACK_PATH_SPEC is not None and PACK_PATH_SPEC.loader is not None
installed_pack_path = importlib.util.module_from_spec(PACK_PATH_SPEC)
PACK_PATH_SPEC.loader.exec_module(installed_pack_path)


def run_helper(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    return subprocess.run(
        [str(RUN_SH), *args],
        cwd=REPO_ROOT,
        env=command_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class PublicHfE2ETests(unittest.TestCase):
    def test_installed_pack_path_accepts_legacy_oasr_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            pack = home / "models" / "legacy" / "pack.oasr"
            pack.parent.mkdir(parents=True)
            pack.write_bytes(b"legacy pack")
            digest = hashlib.sha256(pack.read_bytes()).hexdigest()

            installed_pack_path.validate_installed_pack_path(
                str(home), str(pack), digest
            )

    def test_installed_pack_path_accepts_content_addressed_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            payload = b"content addressed pack"
            digest = hashlib.sha256(payload).hexdigest()
            pack = home / "models" / "objects" / "sha256" / digest / "content"
            pack.parent.mkdir(parents=True)
            pack.write_bytes(payload)

            installed_pack_path.validate_installed_pack_path(
                str(home), str(pack), digest
            )

    def test_installed_pack_path_rejects_content_object_with_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            payload = b"content addressed pack"
            digest = hashlib.sha256(payload).hexdigest()
            wrong_digest = hashlib.sha256(b"other pack").hexdigest()
            pack = home / "models" / "objects" / "sha256" / wrong_digest / "content"
            pack.parent.mkdir(parents=True)
            pack.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "canonical content-addressed"):
                installed_pack_path.validate_installed_pack_path(
                    str(home), str(pack), digest
                )

    def test_installed_pack_path_rejects_content_file_outside_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            payload = b"content addressed pack"
            digest = hashlib.sha256(payload).hexdigest()
            pack = home / "models" / "objects" / "sha256" / "content"
            pack.parent.mkdir(parents=True)
            pack.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "canonical content-addressed"):
                installed_pack_path.validate_installed_pack_path(
                    str(home), str(pack), digest
                )

    def test_installed_pack_path_rejects_file_outside_home(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            home.mkdir()
            pack = root / "outside.oasr"
            pack.write_bytes(b"outside")
            digest = hashlib.sha256(pack.read_bytes()).hexdigest()

            with self.assertRaisesRegex(ValueError, "outside OPENASR_HOME"):
                installed_pack_path.validate_installed_pack_path(
                    str(home), str(pack), digest
                )

    def test_installed_pack_path_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            home.mkdir()
            payload = b"outside content addressed pack"
            digest = hashlib.sha256(payload).hexdigest()
            outside = root / "outside-content"
            outside.write_bytes(payload)
            pack = home / "models" / "objects" / "sha256" / digest / "content"
            pack.parent.mkdir(parents=True)
            pack.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "outside OPENASR_HOME"):
                installed_pack_path.validate_installed_pack_path(
                    str(home), str(pack), digest
                )

    def test_runner_ignores_inherited_models_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workdir = root / "work"
            inherited_models = root / "host-models"
            capture = root / "models-dir.txt"
            fake_openasr = root / "openasr"
            fake_openasr.write_text(
                """#!/usr/bin/env python3
import hashlib
import os
import sys
from pathlib import Path

models_dir = Path(os.environ["OPENASR_MODELS_DIR"])
Path(os.environ["FAKE_OPENASR_ENV_CAPTURE"]).write_text(
    str(models_dir), encoding="utf-8"
)
if sys.argv[1] == "pull":
    payload = b"fake public pack"
    digest = hashlib.sha256(payload).hexdigest()
    pack = models_dir / "objects" / "sha256" / digest / "content"
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_bytes(payload)
    print(f"installed\\tfake\\tq8\\t{pack}")
elif sys.argv[1] == "transcribe":
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.write_text("And so my fellow Americans ask not", encoding="utf-8")
else:
    raise SystemExit(f"unexpected command: {sys.argv[1]}")
""",
                encoding="utf-8",
            )
            fake_openasr.chmod(0o755)

            result = run_helper(
                "--bin",
                str(fake_openasr),
                "--workdir",
                str(workdir),
                "--catalog-url",
                "model-registry/catalog.json",
                env={
                    "OPENASR_MODELS_DIR": str(inherited_models),
                    "FAKE_OPENASR_ENV_CAPTURE": str(capture),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                Path(capture.read_text(encoding="utf-8")),
                workdir / "openasr-home" / "models",
            )
            self.assertFalse(inherited_models.exists())

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
