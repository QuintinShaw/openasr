#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from _test_support import native_path, posix_path, posix_script_command


SCRIPT = Path(__file__).with_name("convert.sh")
REAL_PYTHON = shutil.which("python3") or "/usr/bin/python3"


class ConvertTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.bin_dir = self.base / "bin"
        self.bin_dir.mkdir()
        self.log = self.base / "calls.log"
        self._write_fake_python()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_fake_python(self) -> None:
        # Keep the test at the shell boundary: the real catalog remains the
        # source of the recipe shape, while this shim supplies only the fields
        # and converter side effects needed to exercise convert.sh.
        path = self.bin_dir / "python3"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "script=\"${1:-}\"\n"
            "if [[ \"$script\" == */_catalog.py ]]; then\n"
            "  command=\"${2:-}\"\n"
            "  case \"$command\" in\n"
            "    field)\n"
            "      field=\"${4:-}\"\n"
            "      case \"$field\" in\n"
            "        registry_id) printf '%s\\n' 'mimo-v2.5-asr' ;;\n"
            "        external_converter) printf '%s\\n' 'python3 tooling/mimo-asr/convert_mimo_asr.py --main-dir {src} --tokenizer {src}/tokenizer-src/model.safetensors --out {out} --package-id {registry_id} --quant {quant}' ;;\n"
            "        requant_source_quant) printf '%s\\n' 'fp16' ;;\n"
            "        *) exit 1 ;;\n"
            "      esac\n"
            "      ;;\n"
            "    field-lines) exit 0 ;;\n"
            "    token)\n"
            "      case \"${3:-}\" in\n"
            "        fp16) printf '%s\\n' 'fp16' ;;\n"
            "        q8_0) printf '%s\\n' 'q8-0' ;;\n"
            "        q4_k) printf '%s\\n' 'q4-k' ;;\n"
            "        *) exit 1 ;;\n"
            "      esac\n"
            "      ;;\n"
            "    *) exit 1 ;;\n"
            "  esac\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"$script\" == */_require_files.py ]]; then exit 0; fi\n"
            "if [[ \"$script\" == *convert_mimo_asr.py ]]; then\n"
            "  printf 'EXTERNAL' >> \"$OPENASR_FAKE_CONVERT_LOG\"\n"
            "  for arg in \"$@\"; do printf '\\t%s' \"$(printf '%q' \"$arg\")\" >> \"$OPENASR_FAKE_CONVERT_LOG\"; done\n"
            "  printf '\\n' >> \"$OPENASR_FAKE_CONVERT_LOG\"\n"
            "  out=''\n"
            "  while (($#)); do\n"
            "    if [[ \"$1\" == '--out' && $# -ge 2 ]]; then out=\"$2\"; shift 2; else shift; fi\n"
            "  done\n"
            "  [[ \"${OPENASR_FAKE_CONVERTER_MODE:-}\" != 'fail' ]] || exit 42\n"
            "  [[ \"${OPENASR_FAKE_CONVERTER_MODE:-}\" != 'missing' ]] || exit 0\n"
            "  [[ -n \"$out\" ]] || exit 43\n"
            "  if [[ \"$out\" == */.requant-source.*/source.oasr ]]; then\n"
            "    [[ \"${TMPDIR:-}\" == \"$(dirname \"$out\")\" ]] || exit 46\n"
            "    printf 'scratch\\n' > \"$TMPDIR/gguf-writer.tmp\"\n"
            "    rm -f \"$TMPDIR/gguf-writer.tmp\"\n"
            "  fi\n"
            "  mkdir -p \"$(dirname \"$out\")\"\n"
            "  printf 'source pack\\n' > \"$out\"\n"
            "  exit 0\n"
            "fi\n"
            f'exec "{REAL_PYTHON}" "$@"\n'
        )
        path.chmod(0o755)

    def _write_fake_binary(self, root: Path) -> None:
        path = root / "target" / "release" / "openasr"
        path.parent.mkdir(parents=True)
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf 'BIN' >> \"$OPENASR_FAKE_CONVERT_LOG\"\n"
            "for arg in \"$@\"; do printf '\\t%s' \"$(printf '%q' \"$arg\")\" >> \"$OPENASR_FAKE_CONVERT_LOG\"; done\n"
            "printf '\\n' >> \"$OPENASR_FAKE_CONVERT_LOG\"\n"
            "if [[ \"${1:-}\" == 'model-pack' && \"${2:-}\" == 'requant' ]]; then\n"
            "  source=\"$3\"\n"
            "  output=\"$4\"\n"
            "  [[ -f \"$source\" ]] || exit 44\n"
            "  [[ \"${OPENASR_FAKE_REQUANT_MODE:-}\" != 'fail' ]] || exit 45\n"
            "  mkdir -p \"$(dirname \"$output\")\"\n"
            "  printf 'requantized pack\\n' > \"$output\"\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"${1:-}\" == 'verify' ]]; then [[ -f \"$2\" ]]; exit; fi\n"
            "exit 0\n"
        )
        path.chmod(0o755)

    def _repo(self, name: str) -> Path:
        root = self.base / name
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "convert-test@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Convert Test"], cwd=root, check=True)
        subprocess.run(["git", "config", "commit.gpgSign", "false"], cwd=root, check=True)
        (root / "README").write_text("fixture\n")
        subprocess.run(["git", "add", "README"], cwd=root, check=True)
        subprocess.run(["git", "commit", "--no-gpg-sign", "--no-verify", "-qm", "fixture"], cwd=root, check=True)
        (root / "tmp" / "publish" / "mimo-v2.5-asr" / "src").mkdir(parents=True)
        self._write_fake_binary(root)
        return root

    def _run(self, root: Path, quant: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "OPENASR_REPO_ROOT": posix_path(root),
                "OPENASR_FAKE_CONVERT_LOG": posix_path(self.log),
                "PATH": f"{self.bin_dir}{os.pathsep}{env['PATH']}",
                **extra_env,
            }
        )
        return subprocess.run(
            posix_script_command(SCRIPT, "mimo-v2.5-asr", quant),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        calls: list[list[str]] = []
        for line in self.log.read_text().splitlines():
            _, *encoded = line.split("\t")
            calls.append([line.split("\t", 1)[0], *[shlex.split(value)[0] for value in encoded]])
        return calls

    def test_q4_k_external_converter_stages_source_then_requants_and_cleans(self) -> None:
        root = self._repo("repo with spaces")

        result = self._run(root, "q4_k")

        self.assertEqual(result.returncode, 0, result.stderr)
        output = root / "tmp" / "publish" / "mimo-v2.5-asr" / "packs" / "mimo-v2.5-asr-q4_k.oasr"
        self.assertTrue(output.is_file())
        calls = self._calls()
        external = next(call for call in calls if call[0] == "EXTERNAL")
        self.assertIn("--quant", external)
        self.assertEqual(external[external.index("--quant") + 1], "fp16")
        staging_arg = external[external.index("--out") + 1]
        staging = native_path(staging_arg)
        self.assertEqual(staging.parent.parent, root / "tmp" / "publish" / "mimo-v2.5-asr")
        self.assertEqual(staging.name, "source.oasr")
        self.assertFalse(staging.exists(), "successful requant must remove the derived staging pack")
        self.assertFalse(staging.parent.exists(), "successful requant must remove the staging directory")
        requant = next(call for call in calls if call[:2] == ["BIN", "model-pack"])
        self.assertEqual(requant[2], "requant")
        self.assertEqual(native_path(requant[3]), staging)
        self.assertEqual(native_path(requant[4]), output)
        self.assertEqual(requant[6], "q4-k")
        self.assertTrue(
            any(
                call[:2] == ["BIN", "verify"] and native_path(call[2]) == output
                for call in calls
            )
        )
        self.assertTrue((output.parent / "mimo-v2.5-asr.q4_k.result.json").is_file())

    def test_q4_k_fails_closed_when_converter_does_not_write_staging(self) -> None:
        root = self._repo("repo with missing staging")

        result = self._run(root, "q4_k", OPENASR_FAKE_CONVERTER_MODE="missing")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("produced no staging pack", result.stderr)
        packs = root / "tmp" / "publish" / "mimo-v2.5-asr" / "packs"
        self.assertFalse((packs / "mimo-v2.5-asr-q4_k.oasr").exists())
        work_files = list((root / "tmp" / "publish" / "mimo-v2.5-asr").glob(".requant-source.*"))
        self.assertEqual(work_files, [])

    def test_non_q4_external_converter_keeps_direct_output_path(self) -> None:
        root = self._repo("repo")

        result = self._run(root, "q8_0")

        self.assertEqual(result.returncode, 0, result.stderr)
        output = root / "tmp" / "publish" / "mimo-v2.5-asr" / "packs" / "mimo-v2.5-asr-q8_0.oasr"
        self.assertTrue(output.is_file())
        calls = self._calls()
        external = next(call for call in calls if call[0] == "EXTERNAL")
        self.assertEqual(native_path(external[external.index("--out") + 1]), output)
        self.assertEqual(external[external.index("--quant") + 1], "q8-0")
        self.assertFalse(any(call[:3] == ["BIN", "model-pack", "requant"] for call in calls))


if __name__ == "__main__":
    unittest.main()
