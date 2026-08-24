from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import backend_hardware_evidence as gate
import generate_backend_hardware_evidence as generate


class GenerateBackendHardwareEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _bytes(self, name: str, value: bytes) -> Path:
        path = self.root / name
        path.write_bytes(value)
        return path

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fixture(self) -> tuple[argparse.Namespace, Path]:
        plugin = self._bytes("cuda-sm_86.dll", b"plugin")
        vendor = self._bytes("cuda-runtime.zip", b"vendor")
        model_pack = self._bytes("model.oasr", b"model")
        audio = self._bytes("audio.wav", b"audio")
        entry_path = self.root / "backend-pack-cuda-sm_86.json"
        entry = {
            "id": "cuda-windows-x86_64-test-sm_86",
            "vendor": "cuda",
            "version": "1.2.3",
            "targets": ["sm_86"],
            "host_abi": {"fingerprint": "9" * 64},
            "files": [
                {
                    "role": "plugin",
                    "filename": plugin.name,
                    "sha256": self._sha(plugin),
                    "size_bytes": plugin.stat().st_size,
                },
                {
                    "role": "archive",
                    "filename": vendor.name,
                    "sha256": self._sha(vendor),
                    "size_bytes": vendor.stat().st_size,
                },
            ],
        }
        entry_path.write_text(json.dumps(entry), encoding="utf-8")
        _, identity = gate._entry_identity(entry_path)
        activation = {
            "host_mode": "neutral_dynamic",
            "host_abi": {"fingerprint": "9" * 64},
            "activated": None,
            "qualification": {
                "backend_id": identity.backend_id,
                "vendor": identity.provider,
                "version": identity.version,
                "artifact_fingerprint": identity.artifact_fingerprint,
                "host_abi_fingerprint": "9" * 64,
                "device_target": identity.target,
                "driver_version": "12.7.0",
            },
        }
        extracted = self.root / "neutral-extracted"
        extracted.mkdir()
        fake_binary = extracted / "openasr.exe"
        fake_binary.write_text(
            """#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import shutil
import sys

args = sys.argv[1:]
if args == ["doctor"]:
    catalog = pathlib.Path(os.environ["OPENASR_CATALOG_FILE"])
    home = pathlib.Path(os.environ["OPENASR_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(catalog, home / "catalog.json")
    shutil.copyfile(catalog.with_name("catalog.signature.json"), home / "catalog.signature.json")
    print("Model registry: ok")
    raise SystemExit(0)
if args[:3] == ["__openasr-backend-plugin", "prepare-qualification", %s]:
    catalog = pathlib.Path(os.environ["OPENASR_CATALOG_FILE"])
    home = pathlib.Path(os.environ["OPENASR_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(catalog, home / "catalog.json")
    shutil.copyfile(catalog.with_name("catalog.signature.json"), home / "catalog.signature.json")
    print(json.dumps({"schema_version": 1, "event": "qualification_prepared"}))
    raise SystemExit(0)
if args[:2] == ["__openasr-backend-plugin", "clear-qualification"]:
    print(json.dumps({"schema_version": 1, "event": "qualification_cleared"}))
    raise SystemExit(0)
if args == ["__openasr-backend-plugin", "status"]:
    status = json.loads(%s)
    scope = os.environ["OPENASR_BACKEND_QUALIFICATION_SCOPE"]
    status["qualification"]["scope_sha256"] = hashlib.sha256(scope.encode("ascii")).hexdigest()
    status["qualification"]["catalog_sha256"] = hashlib.sha256(
        (pathlib.Path(os.environ["OPENASR_HOME"]) / "catalog.json").read_bytes()
    ).hexdigest()
    print(json.dumps(status))
    raise SystemExit(0)
if args[:2] != ["bench-receipt", "short-audio"]:
    raise SystemExit(2)
def value(flag):
    return args[args.index(flag) + 1]
def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
receipt = {
    "schema": "openasr.short-audio-receipt.v0",
    "core_commit": value("--core-commit"),
    "scope": value("--scope"),
    "pack": {"content_sha256": sha(value("--model-pack"))},
    "audio": {"sha256": sha(value("--audio"))},
    "run": {
        "backend": "native",
        "device": "cuda",
        "os": "windows",
        "env_allowlist": {
            "OPENASR_GGML_BACKEND": os.environ["OPENASR_GGML_BACKEND"],
            "OPENASR_OFFLINE": os.environ["OPENASR_OFFLINE"],
        },
    },
    "placement": "cuda",
    "observed_placement": {
        "direct_graph_computes": 1,
        "scheduler_graph_computes": 0,
        "observed_compute_nodes_by_backend": {"CUDA0": 10},
        "fallback_node_samples_by_backend": {},
    },
    "transcript": {"text_sha256": "2" * 64},
    "execution": {
        "request_attempt_id": hashlib.sha256(value("--scope").encode()).hexdigest()[:32],
        "request_attempt_conflicted": False,
        "live_lease_reconciliation": "matched",
        "live_state_complete": True,
        "event_history_complete": True,
        "dropped_events": 0,
        "phase_duration_micros": {
            "upload-ingest": 1,
            "decode-normalize": 1,
            "admission-wait": 1,
            "compute": 1,
        },
        "timing_complete": True,
        "terminal": "succeeded",
        "request_receipt_complete": True,
    },
}
pathlib.Path(value("--out")).write_text(json.dumps(receipt), encoding="utf-8")
if os.environ.get("OPENASR_TEST_MUTATE_CATALOG") == "1":
    (pathlib.Path(os.environ["OPENASR_HOME"]) / "catalog.json").write_text("tampered")
"""
            % (json.dumps(identity.backend_id), json.dumps(json.dumps(activation))),
            encoding="utf-8",
        )
        fake_binary.chmod(0o755)
        companion = extracted / "ggml.dll"
        companion.write_bytes(b"companion")
        neutral = self.root / "neutral.zip"
        with zipfile.ZipFile(neutral, "w") as bundle:
            bundle.write(
                fake_binary,
                "openasr-1.2.3-windows-x86_64-neutral/openasr.exe",
            )
            bundle.write(
                companion,
                "openasr-1.2.3-windows-x86_64-neutral/ggml.dll",
            )
        candidate = self.root / "catalog.backends.candidate.json"
        candidate_value = {
            "schema_version": 1,
            "models": [
                {
                    "id": "test",
                    "quants": [
                        {
                            "pull": "test:q8",
                            "sha256": self._sha(model_pack),
                            "size_bytes": model_pack.stat().st_size,
                        }
                    ],
                }
            ],
            "backends": [entry],
        }
        candidate.write_text(json.dumps(candidate_value), encoding="utf-8")
        preview = self.root / "catalog.json"
        preview_value = json.loads(json.dumps(candidate_value))
        for file in preview_value["backends"][0]["files"]:
            file["url"] = (
                generate._core_payload_file_url(plugin)
                if file["role"] == "plugin"
                else generate._core_payload_file_url(vendor)
            )
            file["mirrors"] = []
        preview.write_text(json.dumps(preview_value), encoding="utf-8")
        signature = self.root / "catalog.signature.json"
        signature.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "catalog_url": preview.resolve().as_uri(),
                    "catalog_sha256": self._sha(preview),
                    "catalog_epoch": 1,
                    "signature": {
                        "algorithm": "ed25519",
                        "key_id": "openasr-catalog-local-dev-v1",
                        "value": "7" * 128,
                    },
                }
            ),
            encoding="utf-8",
        )
        checksums = self.root / "SHA256SUMS"
        subjects = [neutral, plugin, vendor, candidate, entry_path]
        checksums.write_text(
            "".join(f"{self._sha(path)}  {path.name}\n" for path in subjects),
            encoding="utf-8",
        )
        home = self.root / "home"
        home.mkdir()
        shutil.copyfile(preview, home / "catalog.json")
        shutil.copyfile(signature, home / "catalog.signature.json")
        args = argparse.Namespace(
            entry=[entry_path],
            provider="cuda",
            device_target="sm_86",
            binary=fake_binary,
            neutral_archive=neutral,
            plugin=plugin,
            vendor_archive=vendor,
            catalog_candidate=candidate,
            catalog=preview,
            catalog_signature=signature,
            checksums=checksums,
            repo="example/openasr",
            signer_workflow="example/openasr/.github/workflows/release-binaries.yml",
            model="test:q8",
            model_pack=model_pack,
            audio=audio,
            home=home,
            catalog_url=preview.resolve().as_uri(),
            core_commit="1" * 40,
            fresh_process_runs=5,
            output=self.root / "backend-hardware-evidence-v1.2.3-cuda.json",
            raw_output=self.root / "backend-hardware-audit-v1.2.3-cuda.json",
        )
        return args, entry_path

    def test_runner_spawns_five_attested_nonce_bound_processes(self) -> None:
        args, entry_path = self._fixture()

        def attested(path: Path, **_: str) -> dict[str, str]:
            return {
                "filename": path.name,
                "sha256": self._sha(path),
                "verification_sha256": "a" * 64,
            }

        with mock.patch.object(generate, "_verify_attestation", side_effect=attested):
            evidence, raw_audit = generate.generate(args)
        generate._write_validated_outputs(
            evidence=evidence,
            raw_audit=raw_audit,
            entry_paths=[entry_path],
            output=args.output,
            raw_output=args.raw_output,
        )
        self.assertEqual(
            gate.approved_entry_paths([entry_path], [args.output], [args.raw_output]),
            [entry_path],
        )
        self.assertEqual(evidence["schema_version"], 1)
        self.assertNotIn("approved_targets", evidence)
        self.assertNotIn("provider_matrix_sha256", evidence)
        self.assertEqual(evidence["evidence_sha256"], generate._canonical_sha256(raw_audit))
        self.assertEqual(len(raw_audit["runs"]), 5)
        self.assertEqual(len({run["nonce"] for run in raw_audit["runs"]}), 5)
        self.assertTrue(
            all(
                run["receipt"]["scope"].endswith(run["nonce"])
                and run["activation_before"] == run["activation_after"]
                for run in raw_audit["runs"]
            )
        )
        existing_summary = args.output.read_bytes()
        rollback_raw = self.root / "backend-hardware-audit-rollback.json"
        with self.assertRaises(gate.EvidenceError):
            generate._write_validated_outputs(
                evidence=evidence,
                raw_audit=raw_audit,
                entry_paths=[entry_path],
                output=args.output,
                raw_output=rollback_raw,
            )
        self.assertEqual(args.output.read_bytes(), existing_summary)
        self.assertFalse(rollback_raw.exists())
        cleanup_summary = self.root / "backend-hardware-evidence-cleanup.json"
        cleanup_raw = self.root / "backend-hardware-audit-cleanup.json"
        with mock.patch.object(generate, "_unlink_best_effort"):
            generate._write_validated_outputs(
                evidence=evidence,
                raw_audit=raw_audit,
                entry_paths=[entry_path],
                output=cleanup_summary,
                raw_output=cleanup_raw,
            )
        self.assertTrue(cleanup_summary.is_file())
        self.assertTrue(cleanup_raw.is_file())

    def test_attestation_command_pins_repo_workflow_and_source_digest(self) -> None:
        subject = self._bytes("subject.bin", b"subject")
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"[{}]", stderr=b""
        )
        with mock.patch.object(generate.subprocess, "run", return_value=completed) as run:
            generate._verify_attestation(
                subject,
                repo="example/openasr",
                signer_workflow="example/openasr/.github/workflows/release-binaries.yml",
                source_digest="1" * 40,
            )
        command = run.call_args.args[0]
        self.assertIn("--repo", command)
        self.assertIn("example/openasr", command)
        self.assertIn("--signer-workflow", command)
        self.assertIn("example/openasr/.github/workflows/release-binaries.yml", command)
        self.assertIn("--source-digest", command)
        self.assertIn("1" * 40, command)

    def test_neutral_extraction_rejects_changed_companion_dll(self) -> None:
        args, _ = self._fixture()
        companion = args.binary.parent / "ggml.dll"
        companion.write_bytes(b"tampered")
        with self.assertRaises(gate.EvidenceError):
            generate._verify_neutral_extraction(args.neutral_archive, args.binary)

    def test_preview_catalog_rejects_unattested_model_pack(self) -> None:
        args, _ = self._fixture()
        args.model_pack.write_bytes(b"different-model")
        with self.assertRaises(gate.EvidenceError):
            generate.generate(args)

    def test_preview_catalog_rejects_changes_outside_local_payload_urls(self) -> None:
        args, _ = self._fixture()
        preview = json.loads(args.catalog.read_text(encoding="utf-8"))
        preview["models"][0]["id"] = "tampered"
        args.catalog.write_text(json.dumps(preview), encoding="utf-8")
        signature = json.loads(args.catalog_signature.read_text(encoding="utf-8"))
        signature["catalog_sha256"] = self._sha(args.catalog)
        args.catalog_signature.write_text(json.dumps(signature), encoding="utf-8")
        with self.assertRaises(gate.EvidenceError):
            generate.generate(args)

    def test_child_environment_removes_higher_priority_catalog_overrides(self) -> None:
        args, _ = self._fixture()
        inherited = {
            "OPENASR_CATALOG_FILE": "C:/wrong/catalog.json",
            "OPENASR_CATALOG_IDENTITY": "https://wrong.invalid/catalog.json",
            "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX": "secret",
        }
        with mock.patch.dict(os.environ, inherited, clear=False):
            env = generate._evidence_environment(args)
        self.assertEqual(env["OPENASR_CATALOG_FILE"], str(args.catalog.resolve()))
        self.assertEqual(env["OPENASR_CATALOG_IDENTITY"], args.catalog_url)
        self.assertNotIn("OPENASR_CATALOG_SIGNING_KEY_SEED_HEX", env)
        self.assertEqual(env["OPENASR_CATALOG_URL"], args.catalog_url)

    def test_runner_rejects_catalog_cache_changed_by_child(self) -> None:
        args, _ = self._fixture()

        def attested(path: Path, **_: str) -> dict[str, str]:
            return {
                "filename": path.name,
                "sha256": self._sha(path),
                "verification_sha256": "a" * 64,
            }

        with (
            mock.patch.object(generate, "_verify_attestation", side_effect=attested),
            mock.patch.dict(os.environ, {"OPENASR_TEST_MUTATE_CATALOG": "1"}),
            self.assertRaises(gate.EvidenceError),
        ):
            generate.generate(args)

    def test_runner_rechecks_neutral_tree_after_child_processes(self) -> None:
        args, _ = self._fixture()
        companion = args.binary.parent / "ggml.dll"
        counter = 0

        def attested(path: Path, **_: str) -> dict[str, str]:
            return {
                "filename": path.name,
                "sha256": self._sha(path),
                "verification_sha256": "a" * 64,
            }

        def mutating_run(**_: object) -> tuple[dict[str, object], str]:
            nonlocal counter
            counter += 1
            companion.write_bytes(b"changed-during-run")
            return (
                {"process_id": counter, "nonce": f"{counter:032x}"},
                "2" * 64,
            )

        with (
            mock.patch.object(generate, "_verify_attestation", side_effect=attested),
            mock.patch.object(generate, "_run_receipt", side_effect=mutating_run),
            self.assertRaises(gate.EvidenceError),
        ):
            generate.generate(args)

    def test_receipt_rejects_cross_provider_compute(self) -> None:
        args, entry_path = self._fixture()
        _, tested = gate._entry_identity(entry_path)
        nonce = "1" * 32
        receipt = {
            "schema": "openasr.short-audio-receipt.v0",
            "core_commit": args.core_commit,
            "scope": f"scope/{nonce}",
            "pack": {"content_sha256": self._sha(args.model_pack)},
            "audio": {"sha256": self._sha(args.audio)},
            "run": {
                "backend": "native",
                "device": "cuda",
                "os": "windows",
                "env_allowlist": {
                    "OPENASR_GGML_BACKEND": "cuda",
                    "OPENASR_OFFLINE": "1",
                },
            },
            "placement": "cuda",
            "observed_placement": {
                "direct_graph_computes": 1,
                "scheduler_graph_computes": 0,
                "observed_compute_nodes_by_backend": {"CPU": 1},
            },
            "transcript": {"text_sha256": "2" * 64},
        }
        with self.assertRaises(gate.EvidenceError):
            generate._validate_receipt(
                receipt,
                tested=tested,
                nonce=nonce,
                scope="scope",
                core_commit=args.core_commit,
                workload_sha=self._sha(args.audio),
                model_pack_sha=self._sha(args.model_pack),
            )

    def test_generic_vulkan_entry_is_bound_to_the_requested_capability_class(self) -> None:
        target = "vk_caps_00001002_0000744c_0123456789abcdef0123456789abcdef"
        entry = self.root / "backend-pack-vulkan-generic.json"
        entry.write_text(
            json.dumps(
                {
                    "id": "vulkan-windows-generic",
                    "vendor": "vulkan",
                    "version": "1.2.3",
                    "targets": [],
                    "host_abi": {"fingerprint": "9" * 64},
                    "files": [
                        {
                            "role": "plugin",
                            "filename": "ggml-vulkan.dll",
                            "sha256": "8" * 64,
                            "size_bytes": 123,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        selected, tested = generate._release_entries([entry], "vulkan", target)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].target, "")
        self.assertEqual(tested.target, target)

        with self.assertRaisesRegex(gate.EvidenceError, "canonical vk_caps"):
            generate._release_entries([entry], "vulkan", "vk_caps_invalid")

    def test_qualification_command_always_carries_the_exact_target(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"{}", b"")
        target = "vk_caps_00001002_0000744c_0123456789abcdef0123456789abcdef"
        with mock.patch.object(generate.subprocess, "run", return_value=completed) as run:
            generate._prepare_qualification(
                Path("openasr.exe"),
                {"OPENASR_HOME": "C:/isolated"},
                backend_id="vulkan-windows-generic",
                device_target=target,
                scope="backend-hardware-evidence/selector/1234567890abcdef",
            )
        command = run.call_args.args[0]
        self.assertEqual(command[3:5], ["vulkan-windows-generic", "--device-target"])
        self.assertEqual(command[5], target)

    def test_output_paths_must_be_distinct_and_role_named(self) -> None:
        shared = self.root / "backend-hardware-evidence-v1.2.3-cuda.json"
        with self.assertRaises(gate.EvidenceError):
            generate._validate_output_paths(shared, shared)
        with self.assertRaises(gate.EvidenceError):
            generate._validate_output_paths(
                self.root / "summary.json",
                self.root / "backend-hardware-audit-v1.2.3-cuda.json",
            )
        with self.assertRaises(gate.EvidenceError):
            generate._validate_output_paths(
                self.root / "backend-hardware-evidence-v1.2.3-cuda.json",
                self.root / "raw.json",
            )

    def test_writer_rejects_one_path_for_both_outputs(self) -> None:
        args, entry_path = self._fixture()
        shared = self.root / "backend-hardware-evidence-v1.2.3-cuda.json"
        with self.assertRaises(gate.EvidenceError):
            generate._write_validated_outputs(
                evidence={},
                raw_audit={},
                entry_paths=[entry_path],
                output=shared,
                raw_output=shared,
            )
        self.assertFalse(shared.exists())


if __name__ == "__main__":
    unittest.main()
