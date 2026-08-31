from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

import gh_release


class GhReleaseDownloadTests(unittest.TestCase):
    def test_retries_transient_failures_then_succeeds(self) -> None:
        dest = Path("/tmp")
        with mock.patch("gh_release.subprocess.run") as run, mock.patch(
            "gh_release.time.sleep"
        ) as sleep:
            run.side_effect = [
                subprocess.CalledProcessError(1, ["gh"]),
                subprocess.CalledProcessError(1, ["gh"]),
                None,
            ]
            gh_release.download_asset("v0.1.37", "backend-pack-vulkan-generic.json", dest)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_refuses_unsafe_asset_names(self) -> None:
        with self.assertRaises(ValueError):
            gh_release.download_asset("v0.1.37", "../escape.dll", Path("/tmp"))
