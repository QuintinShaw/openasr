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
    def test_compile_binds_actual_plugin_vendor_tree_and_host_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "ggml-cuda.dll"
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
                        "backend_targets": {"cuda": ["sm_86", "sm_89"]},
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
            )
            entry = backend_catalog.compile_entry(args)
            self.assertEqual(entry["id"], "cuda-windows-x86_64-aaaaaaaaaaaa-fat")
            self.assertEqual(entry["targets"], ["sm_86", "sm_89"])
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

    def test_update_hints_bind_both_providers_to_one_host_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for provider in ("cuda", "hip"):
                entry = {
                    "id": f"{provider}-pack",
                    "vendor": provider,
                    "version": "1.2.3",
                    "host_abi": {"fingerprint": "a" * 64},
                    "targets": ["sm_86" if provider == "cuda" else "gfx1100"],
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
                path = root / f"{provider}.json"
                path.write_text(json.dumps(entry), encoding="utf-8")
                paths.append(path)
            out = root / "hints.json"
            backend_catalog.compile_update_hints(paths, out)
            hints = json.loads(out.read_text(encoding="utf-8"))["windows-x86_64"]
            self.assertEqual(hints["host_abi_fingerprint"], "a" * 64)
            self.assertEqual(hints["providers"]["cuda"]["size_bytes"], 30)
            self.assertEqual(len(hints["providers"]["hip"]["artifact_fingerprint"]), 64)

            bad = json.loads(paths[1].read_text(encoding="utf-8"))
            bad["host_abi"]["fingerprint"] = "e" * 64
            paths[1].write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(backend_catalog.BackendCatalogError):
                backend_catalog.compile_update_hints(paths, out)

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
