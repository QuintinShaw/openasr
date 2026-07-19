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

VERSION = "0.1.20"
COMMIT = "a" * 40


def _write_fake_archive(dist_dir: Path, filename: str, content: bytes) -> tuple[str, int]:
    path = dist_dir / filename
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


def _write_fake_vendor_archive(dist_dir: Path, vendor_key: str, content: bytes) -> tuple[str, str, int]:
    """Mirrors release-binaries.yml's staging convention: the vendor archive's
    filename embeds the first 12 hex chars of its OWN sha256 (computed once the
    deterministic zip is built), so the fixture must compute the hash first and
    then name the file after it -- not the other way around."""
    sha256 = hashlib.sha256(content).hexdigest()
    filename = f"openasr-vendor-{vendor_key}-{sha256[:12]}.zip"
    path = dist_dir / filename
    path.write_bytes(content)
    return filename, sha256, len(content)


class BuildManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dist_dir = Path(self._tmp.name)
        self.expected = {}
        for backend in ("vulkan", "cuda", "hip"):
            filename = asset_filename(VERSION, backend)
            sha256, size = _write_fake_archive(self.dist_dir, filename, f"fake-{backend}-bytes".encode())
            self.expected[backend] = (filename, sha256, size)
        self.vendor_expected = {}
        for vendor_key in ("cuda-runtime", "rocm-runtime"):
            filename, sha256, size = _write_fake_vendor_archive(
                self.dist_dir, vendor_key, f"fake-{vendor_key}-bytes".encode()
            )
            self.vendor_expected[vendor_key] = (filename, sha256, size)

    def test_generates_all_three_backends_with_correct_hashes(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["core_version"], VERSION)
        self.assertEqual(manifest["source_commit"], COMMIT)

        backends = manifest["platforms"]["windows-x86_64"]["backends"]
        self.assertEqual(set(backends), {"vulkan", "cuda", "hip"})
        for backend, (filename, sha256, size) in self.expected.items():
            entry = backends[backend]
            self.assertEqual(entry["asset"], filename)
            self.assertEqual(entry["sha256"], sha256)
            self.assertEqual(entry["size_bytes"], size)

    def test_sidecar_asset_names_split_cuda_and_hip_from_vulkan(self) -> None:
        # vulkan stays self-contained (v1-shaped asset name); cuda/hip ship a
        # small "-sidecar" archive that references a separate vendor layer --
        # see the module docstring and docs/backend-kernels.md.
        self.assertEqual(asset_filename(VERSION, "vulkan"), f"openasr-{VERSION}-windows-x86_64-vulkan.zip")
        self.assertEqual(
            asset_filename(VERSION, "cuda"), f"openasr-{VERSION}-windows-x86_64-cuda-sidecar.zip"
        )
        self.assertEqual(
            asset_filename(VERSION, "hip"), f"openasr-{VERSION}-windows-x86_64-rocm-sidecar.zip"
        )

    def test_hip_backend_key_uses_rocm_asset_name(self) -> None:
        # Cargo feature name is `hip`; the shipped asset is named `-rocm` for
        # cross-platform consistency with the Linux AMD leg (see
        # release-binaries.yml's matrix comment) -- pin that the manifest
        # keeps the key/asset-name split straight.
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        hip_entry = manifest["platforms"]["windows-x86_64"]["backends"]["hip"]
        self.assertEqual(hip_entry["asset"], f"openasr-{VERSION}-windows-x86_64-rocm-sidecar.zip")

    def test_urls_prefer_dl_openasr_org_then_github_fallback(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        vulkan_entry = manifest["platforms"]["windows-x86_64"]["backends"]["vulkan"]
        urls = vulkan_entry["urls"]
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith(f"https://dl.openasr.org/core/v{VERSION}/"))
        self.assertTrue(
            urls[1].startswith(f"https://github.com/QuintinShaw/openasr/releases/download/v{VERSION}/")
        )

    def test_pe_import_markers_are_backend_specific(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        backends = manifest["platforms"]["windows-x86_64"]["backends"]
        self.assertEqual(backends["vulkan"]["pe_import_markers"], ["vulkan-1.dll"])
        self.assertEqual(backends["cuda"]["pe_import_markers"], ["cublas64_"])
        self.assertEqual(backends["hip"]["pe_import_markers"], ["amdhip64_"])

    def test_vulkan_is_self_contained_cuda_and_hip_reference_a_vendor_layer(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        backends = manifest["platforms"]["windows-x86_64"]["backends"]
        self.assertNotIn("vendor_layer", backends["vulkan"])
        self.assertEqual(backends["cuda"]["vendor_layer"], "cuda-runtime")
        self.assertEqual(backends["hip"]["vendor_layer"], "rocm-runtime")

    def test_vendor_layers_are_assembled_with_correct_hashes_and_content_addressed_urls(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir)
        vendor_layers = manifest["vendor_layers"]
        self.assertEqual(set(vendor_layers), {"cuda-runtime", "rocm-runtime"})
        for vendor_key, (filename, sha256, size) in self.vendor_expected.items():
            entry = vendor_layers[vendor_key]
            self.assertEqual(entry["asset"], filename)
            self.assertEqual(entry["sha256"], sha256)
            self.assertEqual(entry["size_bytes"], size)
            self.assertEqual(entry["urls"][0], f"https://dl.openasr.org/core/vendor/{sha256}/{filename}")
            self.assertEqual(
                entry["urls"][1],
                f"https://github.com/QuintinShaw/openasr/releases/download/v{VERSION}/{filename}",
            )
            self.assertTrue(entry["toolchain"])

    def test_vendor_layers_omitted_entirely_when_only_vulkan_is_selected(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir, backends=["vulkan"])
        self.assertNotIn("vendor_layers", manifest)

    def test_vendor_layers_only_includes_layers_needed_by_selected_backends(self) -> None:
        manifest = build_manifest(VERSION, COMMIT, self.dist_dir, backends=["cuda"])
        self.assertEqual(set(manifest["vendor_layers"]), {"cuda-runtime"})

    def test_missing_vendor_archive_fails_loudly(self) -> None:
        (self.dist_dir / self.vendor_expected["cuda-runtime"][0]).unlink()
        with self.assertRaisesRegex(BackendsManifestError, "cuda-runtime.*not found"):
            build_manifest(VERSION, COMMIT, self.dist_dir)

    def test_mislabeled_vendor_archive_short_hash_fails_loudly(self) -> None:
        # Filename claims a short hash that does not actually prefix the
        # archive's real sha256 -- a staging bug the generator must catch,
        # since content-addressed dedup on the desktop side trusts the
        # manifest's sha256 field, not the filename.
        real_filename, _, _ = self.vendor_expected["cuda-runtime"]
        (self.dist_dir / real_filename).unlink()
        (self.dist_dir / "openasr-vendor-cuda-runtime-000000000000.zip").write_bytes(b"fake-cuda-runtime-bytes")
        with self.assertRaisesRegex(BackendsManifestError, "does not prefix-match"):
            build_manifest(VERSION, COMMIT, self.dist_dir)

    def test_ambiguous_vendor_archive_glob_fails_loudly(self) -> None:
        (self.dist_dir / "openasr-vendor-cuda-runtime-aaaaaaaaaaaa.zip").write_bytes(b"a second candidate")
        with self.assertRaisesRegex(BackendsManifestError, "multiple candidate archives"):
            build_manifest(VERSION, COMMIT, self.dist_dir)

    def test_empty_vendor_archive_fails_loudly(self) -> None:
        filename, _, _ = self.vendor_expected["cuda-runtime"]
        (self.dist_dir / filename).write_bytes(b"")
        with self.assertRaisesRegex(BackendsManifestError, "cuda-runtime.*empty"):
            build_manifest(VERSION, COMMIT, self.dist_dir)

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
            f"https://dl.openasr.org/core/v{VERSION}/backends-manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
