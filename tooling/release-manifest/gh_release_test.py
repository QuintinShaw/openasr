from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gh_release


def _asset(name: str, api_url: str = "https://api.github.com/repos/o/r/releases/assets/1") -> dict[str, str]:
    return {"name": name, "apiUrl": api_url, "id": "RA_not_numeric"}


class GhReleaseDownloadTests(unittest.TestCase):
    def test_retries_transient_failures_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp)
            dest = dest_dir / "backend-pack-vulkan-generic.json"

            def run(command, check=True, timeout=None, capture_output=False, text=False):
                del check, timeout, capture_output, text
                if command[:1] == ["curl"]:
                    if run.calls == 0:
                        run.calls += 1
                        raise subprocess.CalledProcessError(22, command)
                    if run.calls == 1:
                        run.calls += 1
                        raise subprocess.TimeoutExpired(command, 1800)
                    dest.write_text("ok\n", encoding="utf-8")
                    run.calls += 1
                    return subprocess.CompletedProcess(command, 0)
                raise AssertionError(command)

            run.calls = 0
            view = json.dumps({"assets": [_asset("backend-pack-vulkan-generic.json")]})
            with mock.patch.dict(os.environ, {"GH_TOKEN": "test-token"}, clear=False), mock.patch(
                "gh_release.subprocess.check_output", return_value=view
            ), mock.patch("gh_release.subprocess.run", side_effect=run) as run_mock, mock.patch(
                "gh_release.time.sleep"
            ) as sleep:
                gh_release.download_asset(
                    "v0.1.37", "backend-pack-vulkan-generic.json", dest_dir
                )
            self.assertEqual(dest.read_text(encoding="utf-8"), "ok\n")
            self.assertEqual(run_mock.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            curl = run_mock.call_args.args[0]
            self.assertIn("--http1.1", curl)
            self.assertIn("--speed-time", curl)
            self.assertIn("--speed-limit", curl)
            self.assertIn("Bearer test-token", " ".join(curl))
            self.assertIn("application/octet-stream", " ".join(curl))

    def test_uses_api_url_not_node_id_or_browser_url(self) -> None:
        view = json.dumps(
            {
                "assets": [
                    {
                        "name": "SHA256SUMS",
                        "id": "RA_kwDOTLO0884gEJX_",
                        "apiUrl": "https://api.github.com/repos/QuintinShaw/openasr/releases/assets/537957887",
                        "url": "https://github.com/QuintinShaw/openasr/releases/download/untagged-draft/SHA256SUMS",
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "SHA256SUMS"

            def run(command, check=True, timeout=None):
                del check, timeout
                self.assertEqual(command[-1], "https://api.github.com/repos/QuintinShaw/openasr/releases/assets/537957887")
                dest.write_text("sums\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.dict(os.environ, {"GH_TOKEN": "test-token"}, clear=False), mock.patch(
                "gh_release.subprocess.check_output", return_value=view
            ), mock.patch("gh_release.subprocess.run", side_effect=run):
                gh_release.download_asset("v0.1.37", "SHA256SUMS", Path(tmp))
            self.assertEqual(dest.read_text(encoding="utf-8"), "sums\n")

    def test_refuses_unsafe_asset_names(self) -> None:
        with self.assertRaises(ValueError):
            gh_release.download_asset("v0.1.37", "../escape.dll", Path("/tmp"))

    def test_download_url_retries_and_requires_https(self) -> None:
        with self.assertRaises(ValueError):
            gh_release.download_url("http://dl.openasr.org/core/v0.1.37/x.zip", Path("/tmp/x.zip"))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "payload.bin"

            def run(command, check=True, timeout=None):
                del check, timeout
                dest.write_bytes(b"payload")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("gh_release.subprocess.run", side_effect=run) as run_mock:
                gh_release.download_url("https://dl.openasr.org/core/v0.1.37/payload.bin", dest)
            curl = run_mock.call_args.args[0]
            self.assertIn("--http1.1", curl)
            self.assertNotIn("Authorization", " ".join(curl))
            self.assertEqual(dest.read_bytes(), b"payload")

    def test_download_assets_lists_the_release_once(self) -> None:
        view = json.dumps(
            {
                "assets": [
                    _asset("SHA256SUMS", "https://api.github.com/repos/o/r/releases/assets/1"),
                    _asset("hints.json", "https://api.github.com/repos/o/r/releases/assets/2"),
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp)

            def run(command, check=True, timeout=None):
                del check, timeout
                Path(command[command.index("-o") + 1]).write_text("ok\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.dict(os.environ, {"GH_TOKEN": "test-token"}, clear=False), mock.patch(
                "gh_release.subprocess.check_output", return_value=view
            ) as view_mock, mock.patch("gh_release.subprocess.run", side_effect=run):
                gh_release.download_assets(
                    "v0.1.37", ["SHA256SUMS", "hints.json", "SHA256SUMS"], dest_dir
                )
            self.assertEqual(view_mock.call_count, 1)
            self.assertEqual((dest_dir / "SHA256SUMS").read_text(encoding="utf-8"), "ok\n")
            self.assertEqual((dest_dir / "hints.json").read_text(encoding="utf-8"), "ok\n")

    def test_cli_download_many(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp)
            with mock.patch.object(gh_release, "download_assets") as download_assets:
                self.assertEqual(
                    gh_release.main(
                        [
                            "download",
                            "v0.1.37",
                            str(dest_dir),
                            "SHA256SUMS",
                            "backend-plugin-hints.json",
                            "--repo",
                            "QuintinShaw/openasr",
                        ]
                    ),
                    0,
                )
            download_assets.assert_called_once_with(
                "v0.1.37",
                ["SHA256SUMS", "backend-plugin-hints.json"],
                dest_dir,
                repository="QuintinShaw/openasr",
            )
