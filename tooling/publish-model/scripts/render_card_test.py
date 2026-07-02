#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("render_card.py")


class RenderCardTest(unittest.TestCase):
    def test_renderer_uses_repo_owned_template_and_prose(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "moonshine-tiny"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Moonshine Tiny · OpenASR", result.stdout)
        self.assertIn("Tiny 27M-parameter English ASR", result.stdout)
        self.assertIn("Native in OpenASR", result.stdout)
        self.assertNotIn("{{", result.stdout)


if __name__ == "__main__":
    unittest.main()
