import argparse
import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import backend_catalog


def minimal_pe(marker: bytes, certificate: bytes = b"") -> bytes:
    optional = 0x98
    security = optional + 112 + 4 * 8
    certificate_offset = 0x200
    data = bytearray(certificate_offset)
    data[:2] = b"MZ"
    data[0x3C:0x40] = struct.pack("<I", 0x80)
    data[0x80:0x84] = b"PE\0\0"
    data[0x94:0x96] = struct.pack("<H", 0xF0)
    data[optional : optional + 2] = struct.pack("<H", 0x20B)
    data[optional + 108 : optional + 112] = struct.pack("<I", 16)
    data[0x1F0 : 0x1F0 + min(len(marker), 16)] = marker[:16]
    if certificate:
        data[security : security + 8] = struct.pack(
            "<II", certificate_offset, len(certificate)
        )
        data.extend(certificate)
    return bytes(data)


class BackendCatalogTest(unittest.TestCase):
    def test_vendor_tree_digest_sorts_posix_paths_not_filesystem_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "B.bin").write_bytes(b"upper")
            (root / "a.bin").write_bytes(b"lower")
            digest = backend_catalog.materialized_tree_sha256(root, "vendor")
            rows = [("vendor/B.bin", 5, hashlib.sha256(b"upper").hexdigest()),
                    ("vendor/a.bin", 5, hashlib.sha256(b"lower").hexdigest())]
            rows.sort(key=lambda item: item[0])
            expected = hashlib.sha256(b"openasr-backend-tree-v1\0")
            for relative, size, sha256 in rows:
                encoded = relative.encode("utf-8")
                expected.update(struct.pack("<Q", len(encoded)))
                expected.update(encoded)
                expected.update(struct.pack("<Q", size))
                expected.update(sha256.encode("ascii"))
            self.assertEqual(digest, expected.hexdigest())
            self.assertEqual([item[0] for item in rows], ["vendor/B.bin", "vendor/a.bin"])

    def test_compile_binds_actual_plugin_vendor_tree_and_host_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "openasr-1.2.3-windows-x86_64-cuda-sm_86-plugin.dll"
            plugin.write_bytes(b"MZ-plugin")
            vendor = root / "vendor"
            vendor.mkdir()
            (vendor / "cudart64_13.dll").write_bytes(b"runtime")
            archive = root / "vendor.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.write(vendor / "cudart64_13.dll", "cudart64_13.dll")
            build = root / "build.json"
            cmake_contract = {
                "schema_version": 1,
                "cmake_version": "cmake version 4.0.0",
                "entries": {
                    "CMAKE_BUILD_TYPE": "Release",
                    "BUILD_SHARED_LIBS": "ON",
                    "GGML_BACKEND_DL": "ON",
                    "OPENASR_VERIFIED_BACKEND_LOADING_ONLY": "ON",
                    "GGML_NATIVE": "OFF",
                },
                "compilers": {
                    role: {
                        "filename": filename,
                        "sha256": character * 64,
                        "size_bytes": 1,
                    }
                    for role, filename, character in (
                        ("c", "cl.exe", "1"),
                        ("cxx", "cl.exe", "1"),
                        ("cuda", "nvcc.exe", "2"),
                    )
                },
            }
            cmake_contract_sha256 = hashlib.sha256(
                json.dumps(
                    cmake_contract, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            build.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "topology": "neutral-backend-dl",
                        "host_abi": {"schema_version": 1, "fingerprint": "a" * 64},
                        "providers": {"cuda": True},
                        "backend_targets": {"cuda": ["sm_86"]},
                        "build_flags": {
                            "backend_dl": True,
                            "shared": True,
                            "verified_backend_loading_only": True,
                        },
                        "cmake_contract": cmake_contract,
                        "cmake_contract_sha256": cmake_contract_sha256,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                build_manifest=build,
                provider="cuda",
                plugin=plugin,
                vendor_archive=archive,
                vendor_tree=vendor,
                vendor_extract_subdir="vendor",
                version="1.2.3",
                minimum_cli_version="1.2.3",
                minimum_driver_api="13.0.0",
                base_url="https://dl.example/v1.2.3",
                mirror_base_url=None,
                backend_id=None,
                display_name=None,
                require_single_target=True,
            )
            entry = backend_catalog.compile_entry(args)
            self.assertEqual(entry["id"], "cuda-windows-x86_64-aaaaaaaaaaaa-sm_86")
            self.assertEqual(entry["targets"], ["sm_86"])
            self.assertEqual(entry["min_driver_api"], "13.0.0")
            self.assertEqual(entry["files"][0]["sha256"], backend_catalog.sha256_size(plugin)[0])
            self.assertEqual(entry["files"][1]["extract_subdir"], "vendor")
            self.assertEqual(len(entry["files"][1]["extracted_tree_sha256"]), 64)

            build_data = json.loads(build.read_text(encoding="utf-8"))
            build_data["build_flags"]["verified_backend_loading_only"] = False
            build.write_text(json.dumps(build_data), encoding="utf-8")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.compile_entry(args)

    def test_merge_retains_old_abi_and_rejects_ambiguous_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = {
                "id": "cuda-old",
                "vendor": "cuda",
                "host_abi": {"fingerprint": "1" * 64},
                "targets": ["sm_86"],
            }
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"backends": [old]}), encoding="utf-8")
            new = dict(old, id="cuda-new", host_abi={"fingerprint": "2" * 64})
            entry = root / "entry.json"
            entry.write_text(json.dumps(new), encoding="utf-8")
            out = root / "out.json"
            backend_catalog.merge_catalog(catalog, [entry], out)
            self.assertEqual(
                [item["id"] for item in json.loads(out.read_text())["backends"]],
                ["cuda-new", "cuda-old"],
            )
            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps(dict(old, id="cuda-duplicate")), encoding="utf-8")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.merge_catalog(catalog, [duplicate], out)

    def test_update_hints_bind_target_scoped_provider_candidates_to_one_host_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for provider, target in (("cuda", "sm_86"), ("cuda", "sm_89"), ("hip", "gfx1100")):
                entry = {
                    "id": f"{provider}-pack-{target}",
                    "vendor": provider,
                    "version": "1.2.3",
                    "host_abi": {
                        "fingerprint": "a" * 64,
                        "ggml_revision": "f" * 40,
                    },
                    "targets": [target],
                    "min_driver_api": "1.0",
                    "files": [
                        {
                            "filename": f"ggml-{provider}.dll",
                            "sha256": "b" * 64,
                            "size_bytes": 10,
                            "role": "plugin",
                        },
                        {
                            "filename": f"{provider}-vendor.zip",
                            "sha256": "c" * 64,
                            "size_bytes": 20,
                            "role": "archive",
                            "extract_subdir": "vendor",
                            "extracted_tree_sha256": "d" * 64,
                        },
                    ],
                }
                path = root / f"{provider}-{target}.json"
                path.write_text(json.dumps(entry), encoding="utf-8")
                paths.append(path)
            out = root / "hints.json"
            backend_catalog.compile_update_hints(paths, out)
            hints = json.loads(out.read_text(encoding="utf-8"))["windows-x86_64"]
            self.assertEqual(hints["core_version"], "1.2.3")
            self.assertEqual(hints["host_abi_fingerprint"], "a" * 64)
            self.assertEqual(hints["ggml_revision"], "f" * 40)
            self.assertEqual(len(hints["catalog_entries_sha256"]), 64)
            self.assertEqual(
                list(hints["providers"]["cuda"]["targets"]),
                ["sm_86", "sm_89"],
            )
            self.assertEqual(
                hints["providers"]["cuda"]["targets"]["sm_86"]["size_bytes"], 30
            )
            self.assertEqual(
                len(
                    hints["providers"]["hip"]["targets"]["gfx1100"]
                    ["artifact_fingerprint"]
                ),
                64,
            )
            self.assertEqual(
                hints["providers"]["cuda"]["vendor"]["filename"], "cuda-vendor.zip"
            )

            bad = json.loads(paths[-1].read_text(encoding="utf-8"))
            bad["host_abi"]["fingerprint"] = "e" * 64
            paths[-1].write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.compile_update_hints(paths, out)

    def test_update_hints_allow_content_addressed_cuda_vendor_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for provider, target, vendor_name in (
                ("cuda", "sm_86", "cuda-12-6.zip"),
                ("cuda", "sm_120", "cuda-12-8.zip"),
                ("hip", "gfx1100", "hip-vendor.zip"),
            ):
                entry = {
                    "id": f"{provider}-pack-{target}",
                    "vendor": provider,
                    "version": "1.2.3",
                    "host_abi": {
                        "fingerprint": "a" * 64,
                        "ggml_revision": "f" * 40,
                    },
                    "targets": [target],
                    "min_driver_api": "1.0",
                    "files": [
                        {
                            "filename": f"ggml-{provider}-{target}.dll",
                            "sha256": "b" * 64,
                            "size_bytes": 10,
                            "role": "plugin",
                        },
                        {
                            "filename": vendor_name,
                            "sha256": "c" * 64,
                            "size_bytes": 20,
                            "role": "archive",
                            "extract_subdir": "vendor",
                            "extracted_tree_sha256": "d" * 64,
                        },
                    ],
                }
                path = root / f"{provider}-{target}.json"
                path.write_text(json.dumps(entry), encoding="utf-8")
                paths.append(path)
            out = root / "hints.json"
            backend_catalog.compile_update_hints(paths, out)
            hints = json.loads(out.read_text(encoding="utf-8"))["windows-x86_64"]
            self.assertNotIn("vendor", hints["providers"]["cuda"])
            self.assertEqual(
                hints["providers"]["cuda"]["targets"]["sm_86"]["vendor"]["filename"],
                "cuda-12-6.zip",
            )
            self.assertEqual(
                hints["providers"]["cuda"]["targets"]["sm_120"]["vendor"]["filename"],
                "cuda-12-8.zip",
            )
            self.assertEqual(
                hints["providers"]["hip"]["vendor"]["filename"], "hip-vendor.zip"
            )

    def test_release_asset_and_published_catalog_gates_bind_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            expected_entries = []
            for provider, marker in (("cuda", b"cuda"), ("hip", b"hip")):
                plugin = root / f"ggml-{provider}.dll"
                vendor = root / f"{provider}-vendor.zip"
                plugin.write_bytes(marker + b"-plugin")
                vendor.write_bytes(marker + b"-vendor")
                files = []
                for asset, role in ((plugin, "plugin"), (vendor, "archive")):
                    digest, size = backend_catalog.sha256_size(asset)
                    record = {
                        "filename": asset.name,
                        "url": f"https://dl.openasr.org/core/v1.2.3/{asset.name}",
                        "mirrors": [
                            {
                                "source": "github",
                                "url": "https://github.com/QuintinShaw/openasr/"
                                f"releases/download/v1.2.3/{asset.name}",
                            }
                        ],
                        "sha256": digest,
                        "size_bytes": size,
                        "role": role,
                    }
                    if role == "archive":
                        record.update(
                            extract_subdir="vendor", extracted_tree_sha256="d" * 64
                        )
                    files.append(record)
                entry = {
                    "id": f"{provider}-release",
                    "vendor": provider,
                    "version": "1.2.3",
                    "host_abi": {"fingerprint": "a" * 64},
                    "targets": ["sm_86" if provider == "cuda" else "gfx1100"],
                    "files": files,
                }
                path = root / f"entry-{provider}.json"
                path.write_text(json.dumps(entry), encoding="utf-8")
                entries.append(path)
                expected_entries.append(entry)

            report = backend_catalog.verify_release_assets(entries, root, "1.2.3")
            self.assertEqual(report["verified_files"], 4)
            self.assertEqual(set(report["providers"]), {"cuda", "hip"})

            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps({"schema_version": 1, "models": [], "backends": expected_entries}),
                encoding="utf-8",
            )
            catalog_report = backend_catalog.verify_catalog_entries(catalog, entries)
            self.assertEqual(
                catalog_report["verified_backend_ids"], ["cuda-release", "hip-release"]
            )

            plugin = root / "ggml-cuda.dll"
            plugin.write_bytes(b"tampered")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.verify_release_assets(entries, root, "1.2.3")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.verify_release_assets(entries, root, "9.9.9")

            bad_catalog = json.loads(catalog.read_text(encoding="utf-8"))
            bad_catalog["backends"][0]["targets"] = ["sm_89"]
            catalog.write_text(json.dumps(bad_catalog), encoding="utf-8")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.verify_catalog_entries(catalog, entries)

    def test_release_asset_gate_rejects_path_and_version_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for provider in ("cuda", "hip"):
                entry = {
                    "id": provider,
                    "vendor": provider,
                    "version": "9.9.9",
                    "host_abi": {"fingerprint": "a" * 64},
                    "targets": [provider],
                    "files": [
                        {
                            "filename": "../escape.dll",
                            "url": "https://example.invalid/escape.dll",
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                            "role": "plugin",
                        }
                    ],
                }
                path = root / f"{provider}.json"
                path.write_text(json.dumps(entry), encoding="utf-8")
                entries.append(path)
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.verify_release_assets(entries, root, "1.2.3")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.verify_release_assets(entries, root, "9.9.9")

    def test_artifact_fingerprint_binds_target_driver_and_payload(self):
        entry = {
            "id": "cuda-pack",
            "vendor": "cuda",
            "version": "1.2.3",
            "host_abi": {"fingerprint": "a" * 64},
            "targets": ["sm_86"],
            "min_driver_api": "13.0",
            "files": [
                {
                    "filename": "ggml-cuda.dll",
                    "sha256": "b" * 64,
                    "size_bytes": 10,
                    "role": "plugin",
                }
            ],
        }
        baseline = backend_catalog.artifact_fingerprint(entry)
        for field, value in (
            ("targets", ["sm_89"]),
            ("min_driver_api", "13.1"),
        ):
            changed = dict(entry, **{field: value})
            self.assertNotEqual(
                backend_catalog.artifact_fingerprint(changed), baseline
            )
        changed_file = json.loads(json.dumps(entry))
        changed_file["files"][0]["sha256"] = "c" * 64
        self.assertNotEqual(
            backend_catalog.artifact_fingerprint(changed_file), baseline
        )

    def test_bundle_manifest_hashes_final_bytes_and_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fingerprint = "a" * 64
            host_abi = root / "openasr-backend-host-abi-v1.json"
            host_abi.write_text(
                json.dumps({"schema_version": 1, "fingerprint": fingerprint}),
                encoding="utf-8",
            )
            payloads = {
                "ggml.dll": minimal_pe(b"host"),
                "ggml-base.dll": minimal_pe(b"base"),
                "ggml-cpu-avx2.dll": minimal_pe(b"cpu"),
                "ggml-vulkan.dll": minimal_pe(b"vulkan"),
                "vulkan-1.dll": minimal_pe(b"loader"),
            }
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)

            out = root / "openasr-backend-bundle-v1.json"
            backend_catalog.compile_bundled_manifest(root, host_abi, out, True)
            result = json.loads(out.read_text(encoding="utf-8"))

            self.assertEqual(result["host_abi_fingerprint"], fingerprint)
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(len(result["bundle_contract_sha256"]), 64)
            self.assertEqual(
                {entry["provider"] for entry in result["files"]},
                {"host", "cpu", "vulkan", "dependency"},
            )
            for entry in result["files"]:
                payload = payloads[entry["filename"]]
                self.assertEqual(entry["size_bytes"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
                image_sha256, image_size = backend_catalog.pe_image_identity(
                    root / entry["filename"]
                )
                self.assertEqual(entry["image_sha256"], image_sha256)
                self.assertEqual(entry["image_size_bytes"], image_size)

    def test_pe_image_identity_is_stable_across_authenticode_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unsigned = root / "unsigned.dll"
            signed = root / "signed.dll"
            unsigned.write_bytes(minimal_pe(b"same"))
            signed.write_bytes(minimal_pe(b"same", b"certificate"))
            self.assertEqual(
                backend_catalog.pe_image_identity(unsigned),
                backend_catalog.pe_image_identity(signed),
            )

    def test_bundle_manifest_requires_release_vulkan_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host_abi = root / "abi.json"
            host_abi.write_text(json.dumps({"fingerprint": "b" * 64}), encoding="utf-8")
            for name in ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-vulkan.dll"):
                (root / name).write_bytes(minimal_pe(name.encode("ascii")))
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.compile_bundled_manifest(
                    root, host_abi, root / "bundle.json", True
                )


if __name__ == "__main__":
    unittest.main()
