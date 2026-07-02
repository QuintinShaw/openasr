#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("publish.sh")


class PublishFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.log = self.root / "publish.log"
        self.work_root = self.root / "work"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.env = os.environ.copy()
        self.env["OPENASR_PUBLISH_WORK_ROOT"] = str(self.work_root)
        self.env["OPENASR_FAKE_PUBLISH_LOG"] = str(self.log)
        self.env["OPENASR_PUBLISH_MATERIALIZE_CMD"] = str(self.fake_command("materialize"))
        self.env["OPENASR_PUBLISH_TARGET_CMD"] = str(self.fake_command("target"))
        self.env["OPENASR_PUBLISH_REGISTRY_CMD"] = str(self.fake_command("registry"))
        self.env["OPENASR_PUBLISH_MANIFEST_CMD"] = str(self.fake_command("manifest"))
        self.env["OPENASR_PUBLISH_CATALOG_CMD"] = str(self.fake_command("catalog"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def fake_command(self, step: str) -> Path:
        path = self.bin_dir / step
        path.write_text(
            "#!/usr/bin/env bash\n"
            f"printf 'COMMAND:{step}\\n' >> \"$OPENASR_FAKE_PUBLISH_LOG\"\n"
            "for arg in \"$@\"; do printf 'ARG:%s\\n' \"$arg\" >> \"$OPENASR_FAKE_PUBLISH_LOG\"; done\n"
        )
        path.chmod(0o755)
        return path

    def run_publish(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), *args],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def commands(self) -> list[str]:
        return [
            line.removeprefix("COMMAND:")
            for line in self.log.read_text().splitlines()
            if line.startswith("COMMAND:")
        ]

    def test_public_flow_writes_checkpoints_and_skips_completed_steps(self) -> None:
        result = self.run_publish("--public")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.commands(),
            ["materialize", "target", "registry", "manifest", "catalog"],
        )
        checkpoint_dir = self.work_root / "checkpoints"
        for step in [
            "materialize_results",
            "publish_hf",
            "registry",
            "manifest",
            "public_catalog",
        ]:
            data = json.loads((checkpoint_dir / f"{step}.done.json").read_text())
            self.assertEqual(data["step"], step)
            self.assertRegex(data["input_sha256"], r"^[0-9a-f]{64}$")

        self.log.write_text("")
        rerun = self.run_publish("--public")

        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        self.assertEqual(self.log.read_text(), "")
        self.assertIn("skip publish_hf", rerun.stderr)

    def test_force_reruns_completed_steps(self) -> None:
        first = self.run_publish("--public")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.log.write_text("")

        forced = self.run_publish("--public", "--force")

        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertEqual(
            self.commands(),
            ["materialize", "target", "registry", "manifest", "catalog"],
        )

    def test_dry_run_stops_before_registry_manifest_and_catalog(self) -> None:
        result = self.run_publish("--dry-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), ["materialize", "target"])
        rendered = self.log.read_text()
        self.assertIn("ARG:--target", rendered)
        self.assertIn("ARG:hf", rendered)
        self.assertIn("ARG:--dry-run", rendered)
        self.assertNotIn("COMMAND:registry", rendered)
        self.assertNotIn("COMMAND:manifest", rendered)
        self.assertNotIn("COMMAND:catalog", rendered)


if __name__ == "__main__":
    unittest.main()
