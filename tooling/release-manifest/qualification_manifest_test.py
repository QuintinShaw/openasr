from __future__ import annotations

import argparse
import base64
import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import qualification_manifest


HOST_ABI = {
    "schema_version": 3,
    "fingerprint": "1" * 64,
    "target": "x86_64-pc-windows-msvc",
    "crt": "msvc-md",
    "toolchain": "msvc-v143",
    "compile_flags_sha256": "2" * 64,
    "ggml_backend_api_version": 1,
    "ggml_revision": "3" * 40,
    "ggml_headers_sha256": "4" * 64,
    "openasr_ffi_sha256": "5" * 64,
    "openasr_extension_sha256": "6" * 64,
}


def sha256_size(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def write_zip(path: Path, entries: list[tuple[str, bytes, int | None]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            if mode is not None:
                info.external_attr = mode << 16
            archive.writestr(info, payload)


class QualificationManifestCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.version = "1.2.3"
        self.tag = f"v{self.version}"
        self.base_url = f"https://dl.openasr.org/core/{self.tag}"
        self.mirror_url = (
            f"https://github.com/QuintinShaw/openasr/releases/download/{self.tag}"
        )
        self.prefix = f"openasr-{self.version}-windows-x86_64-neutral"
        self.neutral = self.root / f"{self.prefix}.zip"
        write_zip(
            self.neutral,
            [
                (
                    f"{self.prefix}/openasr.exe",
                    b"MZ-exact-neutral-host",
                    stat.S_IFREG | 0o644,
                ),
                (
                    f"{self.prefix}/openasr-backend-host-abi-v1.json",
                    (json.dumps(HOST_ABI, sort_keys=True) + "\n").encode(),
                    stat.S_IFREG | 0o644,
                ),
                (
                    f"{self.prefix}/ggml-base.dll",
                    b"MZ-base",
                    stat.S_IFREG | 0o644,
                ),
            ],
        )
        self.plugin = (
            self.root
            / f"openasr-{self.version}-windows-x86_64-cuda-sm_89-plugin.dll"
        )
        self.plugin.write_bytes(b"MZ-cuda-plugin")
        self.vendor = self.root / "vendor-raw.zip"
        write_zip(
            self.vendor,
            [
                ("cudart64_12.dll", b"runtime", stat.S_IFREG | 0o644),
                ("nested/cublas64_12.dll", b"blas", stat.S_IFREG | 0o644),
            ],
        )
        plugin_sha, plugin_size = sha256_size(self.plugin)
        vendor_sha, vendor_size = sha256_size(self.vendor)
        content_addressed_vendor = (
            self.root / f"openasr-vendor-cuda-runtime-{vendor_sha[:12]}.zip"
        )
        self.vendor.rename(content_addressed_vendor)
        self.vendor = content_addressed_vendor
        vendor_tree = qualification_manifest.inspect_zip(self.vendor)
        self.entry = self.root / "backend-pack-cuda-sm_89.json"
        self.entry_data = {
            "id": "cuda-windows-x86_64-111111111111-sm_89",
            "vendor": "cuda",
            "version": self.version,
            "display_name": "CUDA",
            "description": "fixture",
            "targets": ["sm_89"],
            "min_driver_api": "12.0.0",
            "min_cli_version": self.version,
            "host_abi": HOST_ABI,
            "files": [
                {
                    "filename": self.plugin.name,
                    "url": f"{self.base_url}/{self.plugin.name}",
                    "mirrors": [
                        {
                            "source": "github",
                            "url": f"{self.mirror_url}/{self.plugin.name}",
                        }
                    ],
                    "sha256": plugin_sha,
                    "size_bytes": plugin_size,
                    "role": "plugin",
                },
                {
                    "filename": self.vendor.name,
                    "url": f"{self.base_url}/{self.vendor.name}",
                    "mirrors": [
                        {
                            "source": "github",
                            "url": f"{self.mirror_url}/{self.vendor.name}",
                        }
                    ],
                    "sha256": vendor_sha,
                    "size_bytes": vendor_size,
                    "role": "archive",
                    "extract_subdir": "vendor",
                    "extracted_tree_sha256": vendor_tree.digest("vendor"),
                },
            ],
        }
        self._write_entry()
        self.attestation = self.root / f"openasr-{self.version}-build-provenance.bundle.json"
        self._write_attestation([self.neutral, self.plugin, self.vendor])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_entry(self) -> None:
        self.entry.write_text(json.dumps(self.entry_data), encoding="utf-8")

    def _write_attestation(self, subjects: list[Path]) -> None:
        statement = {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [
                {"name": path.name, "digest": {"sha256": sha256_size(path)[0]}}
                for path in subjects
            ],
            "predicateType": qualification_manifest.ATTESTATION_PREDICATE_TYPE,
            "predicate": {},
        }
        bundle = {
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "dsseEnvelope": {
                "payload": base64.b64encode(
                    json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
                ).decode(),
                "payloadType": "application/vnd.in-toto+json",
                "signatures": [{"keyid": "", "sig": "AA=="}],
            },
            "verificationMaterial": {},
        }
        self.attestation.write_text(json.dumps(bundle), encoding="utf-8")

    def _args(
        self,
        entry: Path | None = None,
        *,
        provider: str = "cuda",
        target: str = "sm_89",
    ) -> argparse.Namespace:
        output = qualification_manifest.manifest_asset_name(
            self.version, provider, target
        )
        return argparse.Namespace(
            neutral_archive=self.neutral,
            backend_entry=entry or self.entry,
            asset_directory=self.root,
            attestation_bundle=self.attestation,
            release_subject=self.tag,
            source_digest="7" * 40,
            base_url=self.base_url,
            mirror_base_url=self.mirror_url,
            out=self.root / output,
        )

    def _vulkan_pack(self) -> tuple[Path, Path, Path]:
        plugin = (
            self.root
            / f"openasr-{self.version}-windows-x86_64-vulkan-generic-plugin.dll"
        )
        plugin.write_bytes(b"MZ-vulkan-plugin")
        vendor_raw = self.root / "vulkan-vendor-raw.zip"
        write_zip(
            vendor_raw,
            [("vulkan-1.dll", b"loader", stat.S_IFREG | 0o644)],
        )
        vendor_sha, vendor_size = sha256_size(vendor_raw)
        vendor = (
            self.root / f"openasr-vendor-vulkan-loader-{vendor_sha[:12]}.zip"
        )
        vendor_raw.rename(vendor)
        plugin_sha, plugin_size = sha256_size(plugin)
        vendor_tree = qualification_manifest.inspect_zip(vendor)
        entry = self.root / "backend-pack-vulkan-generic.json"
        entry.write_text(
            json.dumps(
                {
                    "id": "vulkan-windows-x86_64-generic",
                    "vendor": "vulkan",
                    "version": self.version,
                    "display_name": "Vulkan",
                    "description": "fixture",
                    "targets": [],
                    "min_driver_api": "1.2.0",
                    "min_cli_version": self.version,
                    "host_abi": HOST_ABI,
                    "files": [
                        {
                            "filename": plugin.name,
                            "url": f"{self.base_url}/{plugin.name}",
                            "mirrors": [
                                {
                                    "source": "github",
                                    "url": f"{self.mirror_url}/{plugin.name}",
                                }
                            ],
                            "sha256": plugin_sha,
                            "size_bytes": plugin_size,
                            "role": "plugin",
                        },
                        {
                            "filename": vendor.name,
                            "url": f"{self.base_url}/{vendor.name}",
                            "mirrors": [
                                {
                                    "source": "github",
                                    "url": f"{self.mirror_url}/{vendor.name}",
                                }
                            ],
                            "sha256": vendor_sha,
                            "size_bytes": vendor_size,
                            "role": "archive",
                            "extract_subdir": "vendor",
                            "extracted_tree_sha256": vendor_tree.digest("vendor"),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self._write_attestation([self.neutral, plugin, vendor])
        return entry, plugin, vendor

    def test_compile_binds_host_plugin_vendor_and_attestation_without_policy(self) -> None:
        manifest = qualification_manifest.compile_manifest(self._args())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["release_subject"], self.tag)
        self.assertEqual(manifest["host_abi"], HOST_ABI)
        self.assertEqual(
            manifest["provider_target"], {"provider": "cuda", "target": "sm_89"}
        )
        self.assertEqual(manifest["artifacts"]["plugin"]["sha256"], sha256_size(self.plugin)[0])
        vendor = manifest["artifacts"]["vendor"][0]
        vendor_tree = qualification_manifest.inspect_zip(self.vendor)
        self.assertEqual(vendor["unpacked_size_bytes"], vendor_tree.unpacked_size_bytes)
        self.assertEqual(vendor["unpacked_tree_sha256"], vendor_tree.digest())
        self.assertNotEqual(
            vendor["unpacked_tree_sha256"],
            self.entry_data["files"][1]["extracted_tree_sha256"],
        )
        self.assertEqual(
            manifest["attestation"]["bundle"]["file_name"], self.attestation.name
        )
        rendered = json.dumps(manifest, sort_keys=True)
        for forbidden in ("activation_mode", "activatable", "model_id", "pack_sha256"):
            self.assertNotIn(forbidden, rendered)

    def test_compile_generic_vulkan_binds_plugin_and_loader_not_a_device_target(
        self,
    ) -> None:
        entry, plugin, vendor = self._vulkan_pack()
        manifest = qualification_manifest.compile_manifest(
            self._args(entry, provider="vulkan", target="generic")
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            manifest["provider_target"],
            {"provider": "vulkan", "target": "generic"},
        )
        self.assertEqual(
            manifest["artifacts"]["plugin"]["sha256"], sha256_size(plugin)[0]
        )
        self.assertEqual(
            manifest["artifacts"]["vendor"][0]["file_name"], vendor.name
        )
        self.assertTrue(vendor.name.startswith("openasr-vendor-vulkan-loader-"))

    def test_generic_vulkan_entry_cannot_encode_a_physical_device(self) -> None:
        entry, _, _ = self._vulkan_pack()
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["targets"] = [
            "vk_caps_00001002_0000744c_00112233445566778899aabbccddeeff"
        ]
        entry.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "must not encode a physical device target",
        ):
            qualification_manifest.compile_manifest(
                self._args(entry, provider="vulkan", target="generic")
            )

    def test_expected_artifact_cells_use_generic_vulkan_and_compiled_gpu_targets(
        self,
    ) -> None:
        matrix = [
            {"provider": "cuda", "cuda_gpu_target": "89"},
            {"provider": "cuda", "cuda_gpu_target": "120", "experimental": True},
            {"provider": "hip", "hip_gpu_target": "gfx1200"},
            {"provider": "vulkan"},
        ]
        self.assertEqual(
            qualification_manifest.expected_artifact_cells(matrix),
            {("cuda", "sm_89"), ("hip", "gfx1200"), ("vulkan", "generic")},
        )
        self.assertEqual(
            qualification_manifest.expected_artifact_cells(
                matrix, promoted_cuda_targets={"120"}
            ),
            {
                ("cuda", "sm_89"),
                ("cuda", "sm_120"),
                ("hip", "gfx1200"),
                ("vulkan", "generic"),
            },
        )

    def test_backend_host_abi_must_equal_neutral_archive(self) -> None:
        self.entry_data["host_abi"] = dict(HOST_ABI, fingerprint="8" * 64)
        self._write_entry()
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "host ABI differs",
        ):
            qualification_manifest.compile_manifest(self._args())

    def test_attestation_must_bind_every_referenced_release_subject(self) -> None:
        self._write_attestation([self.neutral, self.vendor])
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "does not bind exact release subject",
        ):
            qualification_manifest.compile_manifest(self._args())

    def test_candidate_vendor_tree_uses_its_install_prefix(self) -> None:
        self.entry_data["files"][1]["extracted_tree_sha256"] = (
            qualification_manifest.inspect_zip(self.vendor).digest()
        )
        self._write_entry()
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "tree differs",
        ):
            qualification_manifest.compile_manifest(self._args())

    def test_plugin_filename_must_bind_exact_provider_target(self) -> None:
        wrong = self.root / self.plugin.name.replace("sm_89", "sm_90")
        self.plugin.rename(wrong)
        plugin_sha, plugin_size = sha256_size(wrong)
        self.entry_data["files"][0].update(
            {
                "filename": wrong.name,
                "url": f"{self.base_url}/{wrong.name}",
                "mirrors": [
                    {"source": "github", "url": f"{self.mirror_url}/{wrong.name}"}
                ],
                "sha256": plugin_sha,
                "size_bytes": plugin_size,
            }
        )
        self._write_entry()
        self._write_attestation([self.neutral, wrong, self.vendor])
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "must bind exact provider/target",
        ):
            qualification_manifest.compile_manifest(self._args())

    def test_vendor_filename_must_bind_provider_and_content_digest(self) -> None:
        wrong_suffix = "0" * 12
        if self.entry_data["files"][1]["sha256"].startswith(wrong_suffix):
            wrong_suffix = "f" * 12
        wrong = self.root / f"openasr-vendor-cuda-runtime-{wrong_suffix}.zip"
        self.vendor.rename(wrong)
        vendor_sha, vendor_size = sha256_size(wrong)
        self.entry_data["files"][1].update(
            {
                "filename": wrong.name,
                "url": f"{self.base_url}/{wrong.name}",
                "mirrors": [
                    {"source": "github", "url": f"{self.mirror_url}/{wrong.name}"}
                ],
                "sha256": vendor_sha,
                "size_bytes": vendor_size,
            }
        )
        self._write_entry()
        self._write_attestation([self.neutral, self.plugin, wrong])
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "not named by its sha256 prefix",
        ):
            qualification_manifest.compile_manifest(self._args())

    def test_zip_rejects_case_collisions_and_non_regular_entries(self) -> None:
        collision = self.root / "collision.zip"
        write_zip(
            collision,
            [
                ("A.dll", b"a", stat.S_IFREG | 0o644),
                ("a.dll", b"b", stat.S_IFREG | 0o644),
            ],
        )
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "case-colliding",
        ):
            qualification_manifest.inspect_zip(collision)
        linked = self.root / "linked.zip"
        write_zip(linked, [("escape.dll", b"target", stat.S_IFLNK | 0o777)])
        with self.assertRaisesRegex(
            qualification_manifest.QualificationManifestError,
            "non-regular",
        ):
            qualification_manifest.inspect_zip(linked)

    def test_zip_rejects_windows_characters_unicode_and_resource_bombs(self) -> None:
        for name in ("bad?.dll", "bad<name.dll", "unicode-" + chr(0xE9) + ".dll"):
            archive = self.root / f"{hashlib.sha256(name.encode()).hexdigest()}.zip"
            write_zip(archive, [(name, b"payload", stat.S_IFREG | 0o644)])
            with self.assertRaisesRegex(
                qualification_manifest.QualificationManifestError,
                "portable Windows relative path",
            ):
                qualification_manifest.inspect_zip(archive)
        with mock.patch.object(qualification_manifest, "MAX_ZIP_UNPACKED_BYTES", 4):
            with self.assertRaisesRegex(
                qualification_manifest.QualificationManifestError,
                "unpacked bytes",
            ):
                qualification_manifest.inspect_zip(self.vendor)
        compressed = self.root / "compressed.zip"
        write_zip(compressed, [("zeros.bin", b"0" * 4096, stat.S_IFREG | 0o644)])
        with mock.patch.object(qualification_manifest, "MAX_ZIP_COMPRESSION_RATIO", 1):
            with self.assertRaisesRegex(
                qualification_manifest.QualificationManifestError,
                "compression ratio",
            ):
                qualification_manifest.inspect_zip(compressed)

    def test_cli_writes_deterministic_utf8_lf(self) -> None:
        args = self._args()
        result = qualification_manifest.main(
            [
                "--neutral-archive",
                str(args.neutral_archive),
                "--backend-entry",
                str(args.backend_entry),
                "--asset-directory",
                str(args.asset_directory),
                "--attestation-bundle",
                str(args.attestation_bundle),
                "--release-subject",
                args.release_subject,
                "--source-digest",
                args.source_digest,
                "--base-url",
                args.base_url,
                "--mirror-base-url",
                args.mirror_base_url,
                "--out",
                str(args.out),
            ]
        )
        self.assertEqual(result, 0)
        payload = args.out.read_bytes()
        self.assertTrue(payload.endswith(b"\n"))
        self.assertNotIn(b"\r", payload)
        self.assertEqual(json.loads(payload), qualification_manifest.compile_manifest(args))


if __name__ == "__main__":
    unittest.main()
