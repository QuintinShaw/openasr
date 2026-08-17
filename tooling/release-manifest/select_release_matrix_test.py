from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from select_release_matrix import MATRIX_PATH, load_matrix, select_matrix


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release-binaries.yml"
SCRIPT = Path(__file__).resolve().parent / "select_release_matrix.py"


class SelectReleaseMatrixTests(unittest.TestCase):
    def test_empty_only_target_returns_the_full_matrix(self) -> None:
        rows = load_matrix()
        selected = select_matrix(rows, "")
        self.assertEqual(selected, rows)
        self.assertGreater(len(selected), 1)

    def test_exact_target_returns_one_row(self) -> None:
        selected = select_matrix(load_matrix(), "x86_64-apple-darwin")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["target"], "x86_64-apple-darwin")
        self.assertEqual(selected[0]["os"], "macos-15-intel")

    def test_unknown_target_fails_closed(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            select_matrix(load_matrix(), "not-a-real-target")
        message = str(raised.exception)
        self.assertIn("not-a-real-target", message)
        self.assertIn("x86_64-apple-darwin", message)

    def test_cli_prints_only_compact_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--only-target", ""],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stderr, "")
        parsed = json.loads(completed.stdout)
        self.assertIsInstance(parsed, list)
        self.assertEqual(completed.stdout, json.dumps(parsed, separators=(",", ":")))
        self.assertNotIn("\n", completed.stdout)
        self.assertEqual(len(parsed), len(load_matrix()))

    def test_cli_unknown_target_exits_one_without_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--only-target", "not-a-real-target"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("not-a-real-target", completed.stderr)
        self.assertIn("x86_64-apple-darwin", completed.stderr)

    def test_workflow_wires_selected_json_into_build_include(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertTrue(MATRIX_PATH.is_file())
        self.assertIsInstance(json.loads(MATRIX_PATH.read_text(encoding="utf-8")), list)
        self.assertIn("select_release_matrix.py", workflow)
        self.assertIn("fromJSON(needs.select-matrix.outputs.include)", workflow)
        self.assertIn("\n  select-matrix:\n", workflow)


if __name__ == "__main__":
    unittest.main()
