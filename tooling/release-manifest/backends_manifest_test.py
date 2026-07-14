from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from backends_manifest import (
    BackendsManifestError,
    asset_filename,
    build_manifest,
    manifest_url_for,
    write_manifest,
)

VERSION = "0.1.10"
COMMIT = "a" * 40


def _write_fake_archive(dist_dir: Path, filename: str, content: bytes) -> tuple[str, int]:
    path = dist_dir / filename
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


class BuildManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dist_dir = Path(self._tmp.name)
        self.expected = {}
        for backend, suffix in (("vulkan", "vulkan"), ("cuda", "cuda"), ("hip", "rocm")):
            filename = asset_filename(VERSION, suffix)
            sha256, size = _write_fake_archive(self.dist_dir, filename, f"fake-{backend}-bytes".encode())
            self.expected[backend] = (filename, sha256, size)

    def test_generates_all_three_backends_with_correct_hashes(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["core_version"], VERSION)
        self.assertEqual(manifest["source_commit"], COMMIT)

        backends = manifest["platforms"]["windows-x86_64"]["backends"]
        self.assertEqual(set(backends), {"vulkan", "cuda", "hip"})
        for backend, (filename, sha256, size) in self.expected.items():
            entry = backends[backend]
            self.assertEqual(entry["asset"], filename)
            self.assertEqual(entry["sha256"], sha256)
            self.assertEqual(entry["size_bytes"], size)

    def test_hip_backend_key_uses_rocm_asset_name(self) -> None:
        # Cargo feature name is `hip`; the shipped asset is named `-rocm` for
        # cross-platform consistency with the Linux AMD leg (see
        # release-binaries.yml's matrix comment) -- pin that the manifest
        # keeps the key/asset-name split straight.
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        hip_entry = manifest["platforms"]["windows-x86_64"]["backends"]["hip"]
        self.assertEqual(hip_entry["asset"], f"openasr-{VERSION}-windows-x86_64-rocm.zip")

    def test_urls_prefer_dl_openasr_org_then_github_fallback(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        vulkan_entry = manifest["platforms"]["windows-x86_64"]["backends"]["vulkan"]
        urls = vulkan_entry["urls"]
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith("https://dl.openasr.org/core/v0.1.10/"))
        self.assertTrue(
            urls[1].startswith("https://github.com/QuintinShaw/openasr/releases/download/v0.1.10/")
        )

    def test_pe_import_markers_are_backend_specific(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        backends = manifest["platforms"]["windows-x86_64"]["backends"]
        self.assertEqual(backends["vulkan"]["pe_import_markers"], ["vulkan-1.dll"])
        self.assertEqual(backends["cuda"]["pe_import_markers"], ["cublas64_"])
        self.assertEqual(backends["hip"]["pe_import_markers"], ["amdhip64_"])

    def test_missing_archive_fails_loudly(self) -> None:
        (self.dist_dir / asset_filename(VERSION, "cuda")).unlink()
        with self.assertRaisesRegex(BackendsManifestError, "cuda.*not found"):
            build_manifest(VERSION, COMMIT, self.dist_dir)

    def test_empty_archive_fails_loudly(self) -> None:
        (self.dist_dir / asset_filename(VERSION, "vulkan")).write_bytes(b"")
        with self.assertRaisesRegex(BackendsManifestError, "vulkan.*empty"):
            build_manifest(VERSION, COMMIT, self.dist_dir)

    def test_rejects_malformed_version(self) -> None:
        with self.assertRaises(BackendsManifestError):
            build_manifest("not-a-version", COMMIT, self.dist_dir)

    def test_rejects_malformed_commit(self) -> None:
        with self.assertRaises(BackendsManifestError):
            build_manifest(VERSION, "not-a-sha", self.dist_dir)

    def test_restrict_to_specific_backends(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir, backends=["vulkan"])
        backends = manifest["platforms"]["windows-x86_64"]["backends"]
        self.assertEqual(set(backends), {"vulkan"})

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(BackendsManifestError):
            build_manifest(VERSION, COMMIT, self.dist_dir, backends=["metal"])

    def test_write_manifest_round_trips_through_json(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        out = self.dist_dir / "out" / "backends-manifest.json"
        write_manifest(manifest, out)
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(reloaded, manifest)

    def test_manifest_url_for(self) -> None:
        self.assertEqual(
            manifest_url_for(VERSION),
            "https://dl.openasr.org/core/v0.1.10/backends-manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
