from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import backend_hardware_evidence as gate


class BackendHardwareEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def entry(
        self,
        target: str = "sm_86",
        plugin_sha: str = "b" * 64,
        provider: str = "cuda",
    ) -> Path:
        targets = [] if provider == "vulkan" else [target]
        min_driver = (
            "12.0.0" if provider == "cuda" else "7.2.0" if provider == "hip" else None
        )
        value = {
            "id": f"{provider}-{target}",
            "vendor": provider,
            "version": "1.2.3",
            "targets": targets,
            "host_abi": {"fingerprint": "9" * 64},
            "files": [
                {
                    "role": "plugin",
                    "filename": f"{provider}-{target}.dll",
                    "sha256": plugin_sha,
                    "size_bytes": 123,
                }
            ],
        }
        if min_driver is not None:
            value["min_driver_api"] = min_driver
        return self.write(
            f"entry-{provider}-{target}.json",
            value,
        )

    def evidence(
        self,
        entry: Path,
        *,
        plugin_sha: str | None = None,
        runs: int = 5,
        schema_version: int = 1,
        device_target: str | None = None,
    ) -> Path:
        _, identity = gate._entry_identity(entry)
        value: dict[str, object] = {
            "schema_version": schema_version,
            "result": "pass",
            "provider": identity.provider,
            "device_target": device_target or identity.target,
            "backend_id": identity.backend_id,
            "release_version": identity.version,
            "driver_version": "12.7.0",
            "artifact_fingerprint": identity.artifact_fingerprint,
            "plugin_sha256": plugin_sha or identity.plugin_sha256,
            "binary_sha256": "c" * 64,
            "workload_sha256": "d" * 64,
            "model_pack_sha256": "e" * 64,
            "evidence_sha256": "f" * 64,
            "fresh_process_runs": runs,
            "placement": "full_device",
            "cpu_fallback": False,
        }
        return self.write(f"evidence-v{schema_version}.json", value)

    def pair(self, entry: Path, **kwargs: object) -> tuple[Path, Path]:
        evidence = self.evidence(entry, **kwargs)
        value = json.loads(evidence.read_text(encoding="utf-8"))
        _, identity = gate._entry_identity(entry)
        device_target = value["device_target"]
        activation = {
            "host_mode": "neutral_dynamic",
            "activated": None,
            "qualification": {
                "vendor": identity.provider,
                "device_target": device_target,
                "backend_id": identity.backend_id,
                "artifact_fingerprint": identity.artifact_fingerprint,
                "version": identity.version,
                "driver_version": value["driver_version"],
                "catalog_sha256": "a" * 64,
                "scope_sha256": "b" * 64,
            }
        }
        raw = {
            "schema": "openasr.backend-hardware-audit.v1",
            "provider": identity.provider,
            "device_target": device_target,
            "backend_id": identity.backend_id,
            "artifact_fingerprint": identity.artifact_fingerprint,
            "plugin_sha256": identity.plugin_sha256,
            "release_version": identity.version,
            "driver_version": value["driver_version"],
            "generator_sha256": "0" * 64,
            "repository": "QuintinShaw/openasr",
            "signer_workflow": "QuintinShaw/openasr/.github/workflows/release-binaries.yml",
            "source_digest": "7" * 40,
            "scope": f"backend-hardware-evidence-v{identity.version}-{identity.provider}",
            "binary_sha256": value["binary_sha256"],
            "neutral_archive_sha256": "1" * 64,
            "neutral_extracted_tree_sha256": "2" * 64,
            "workload_sha256": value["workload_sha256"],
            "model_pack_sha256": value["model_pack_sha256"],
            "vendor_archive_sha256": "3" * 64,
            "catalog_candidate_sha256": "4" * 64,
            "preview_catalog_url": "file:///qualification/catalog.json",
            "preview_catalog_sha256": "a" * 64,
            "preview_catalog_signature_sha256": "5" * 64,
            "catalog_signature_preflight": {
                "stdout_sha256": "8" * 64,
                "stderr_sha256": "9" * 64,
                "cached_catalog_sha256": "a" * 64,
                "cached_signature_sha256": "5" * 64,
            },
            "checksums_sha256": "6" * 64,
            "attested_release_subjects": [
                {
                    "filename": "openasr-neutral.zip",
                    "sha256": "1" * 64,
                    "verification_sha256": "a" * 64,
                },
                {
                    "filename": f"{identity.provider}-plugin.dll",
                    "sha256": identity.plugin_sha256,
                    "verification_sha256": "b" * 64,
                },
                {
                    "filename": f"{identity.provider}-vendor.zip",
                    "sha256": "3" * 64,
                    "verification_sha256": "c" * 64,
                },
                {
                    "filename": "catalog.backends.candidate.json",
                    "sha256": "4" * 64,
                    "verification_sha256": "d" * 64,
                },
                {
                    "filename": identity.path.name,
                    "sha256": gate._sha256_file(identity.path),
                    "verification_sha256": "e" * 64,
                },
            ],
            "qualification_scope_sha256": "b" * 64,
            "runs": [],
        }
        for index in range(value["fresh_process_runs"]):
            nonce = f"{index + 1:032x}"
            receipt = {
                "schema": "openasr.short-audio-receipt.v0",
                "scope": f"backend-hardware-evidence/{nonce}",
                "pack": {"content_sha256": value["model_pack_sha256"]},
                "audio": {"sha256": value["workload_sha256"]},
                "run": {
                    "backend": "native",
                    "device": identity.provider,
                    "os": "windows",
                    "env_allowlist": {
                        "OPENASR_GGML_BACKEND": identity.provider,
                        "OPENASR_OFFLINE": "1",
                    },
                },
                "placement": identity.provider,
                "observed_placement": {
                    "direct_graph_computes": 1,
                    "scheduler_graph_computes": 0,
                    "observed_compute_nodes_by_backend": {
                        {
                            "cuda": "CUDA0",
                            "hip": "ROCm0",
                            "vulkan": "Vulkan0",
                        }[identity.provider]: 10
                    },
                    "fallback_node_samples_by_backend": {},
                },
                "transcript": {"text_sha256": "2" * 64},
                "execution": {
                    "request_attempt_id": f"{index + 10:032x}",
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
            raw["runs"].append(
                {
                    "nonce": nonce,
                    "process_id": index + 1,
                    "receipt": receipt,
                    "receipt_sha256": gate._canonical_sha256(receipt),
                    "activation_before": activation,
                    "activation_after": activation,
                }
            )
        raw_path = self.write("raw-audit.json", raw)
        value["evidence_sha256"] = gate._canonical_sha256(raw)
        evidence.write_text(json.dumps(value), encoding="utf-8")
        return evidence, raw_path

    def test_exact_receipt_approves_only_its_entry(self) -> None:
        tested = self.entry()
        other = self.entry(target="sm_89", plugin_sha="8" * 64)
        evidence, raw = self.pair(tested)
        self.assertEqual(
            gate.approved_entry_paths([tested, other], [evidence], [raw]),
            [tested],
        )

    def test_vulkan_evidence_binds_one_capability_class_to_generic_bytes(self) -> None:
        target = "vk_caps_00001002_0000744c_0123456789abcdef0123456789abcdef"
        entry = self.entry(target="generic", provider="vulkan")
        evidence, raw = self.pair(entry, device_target=target)
        self.assertEqual(gate.approved_entry_paths([entry], [evidence], [raw]), [entry])

        invalid = json.loads(evidence.read_text(encoding="utf-8"))
        invalid["device_target"] = "vk_caps_invalid"
        invalid_path = self.write("invalid-vulkan-evidence.json", invalid)
        with self.assertRaisesRegex(gate.EvidenceError, "invalid device target"):
            gate.approved_entry_paths([entry], [invalid_path], [])

    def test_provider_targets_use_canonical_non_interchangeable_grammars(self) -> None:
        for provider, invalid in (
            ("cuda", "gfx1100"),
            ("cuda", "sm_gfx1100"),
            ("hip", "sm_86"),
            ("hip", "gfx_sm86"),
        ):
            with self.subTest(provider=provider, invalid=invalid):
                with self.assertRaisesRegex(gate.EvidenceError, "invalid .* target"):
                    gate._entry_identity(self.entry(target=invalid, provider=provider))

    def test_raw_pack_artifact_fingerprint_is_computed(self) -> None:
        entry = self.entry()
        raw = json.loads(entry.read_text(encoding="utf-8"))
        self.assertNotIn("artifact_fingerprint", raw)
        _, identity = gate._entry_identity(entry)
        self.assertEqual(len(identity.artifact_fingerprint), 64)

    def test_schema_v2_and_cross_target_approval_are_rejected(self) -> None:
        tested = self.entry()
        other = self.entry(target="sm_89", plugin_sha="8" * 64)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths(
                [tested, other], [self.evidence(tested, schema_version=2)], []
            )
        evidence, raw = self.pair(tested)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([other], [evidence], [raw])

    def test_different_bytes_or_insufficient_runs_are_rejected(self) -> None:
        entry = self.entry()
        wrong_bytes, wrong_bytes_raw = self.pair(entry, plugin_sha="9" * 64)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([entry], [wrong_bytes], [wrong_bytes_raw])
        few_runs, few_runs_raw = self.pair(entry, runs=4)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([entry], [few_runs], [few_runs_raw])

    def test_raw_audit_must_be_exact_bound_and_consumed_once(self) -> None:
        entry = self.entry()
        evidence, raw = self.pair(entry)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([entry], [evidence], [])
        tampered = json.loads(raw.read_text(encoding="utf-8"))
        tampered["backend_id"] = "wrong-backend"
        tampered_path = self.write("raw-tampered.json", tampered)
        bound_evidence = json.loads(evidence.read_text(encoding="utf-8"))
        bound_evidence["evidence_sha256"] = gate._canonical_sha256(tampered)
        bound_evidence_path = self.write("evidence-bound-tampered.json", bound_evidence)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([entry], [bound_evidence_path], [tampered_path])
        receipt_tampered = json.loads(raw.read_text(encoding="utf-8"))
        receipt_tampered["runs"][0]["receipt"] = {"attempt": "tampered"}
        receipt_tampered_path = self.write("raw-receipt-tampered.json", receipt_tampered)
        receipt_bound_evidence = json.loads(evidence.read_text(encoding="utf-8"))
        receipt_bound_evidence["evidence_sha256"] = gate._canonical_sha256(receipt_tampered)
        receipt_bound_evidence_path = self.write(
            "evidence-receipt-tampered.json", receipt_bound_evidence
        )
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths(
                [entry], [receipt_bound_evidence_path], [receipt_tampered_path]
            )
        extra = json.loads(raw.read_text(encoding="utf-8"))
        extra["scope"] = "unused-but-distinct"
        extra_path = self.write("raw-extra.json", extra)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([entry], [evidence], [raw, extra_path])

    def test_raw_provenance_requires_signature_preflight_and_attested_subjects(self) -> None:
        entry = self.entry()
        evidence, raw = self.pair(entry)
        document = json.loads(raw.read_text(encoding="utf-8"))
        document.pop("catalog_signature_preflight")
        raw.write_text(json.dumps(document), encoding="utf-8")
        summary = json.loads(evidence.read_text(encoding="utf-8"))
        summary["evidence_sha256"] = gate._canonical_sha256(document)
        evidence.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(gate.EvidenceError, "signature preflight"):
            gate.approved_entry_paths([entry], [evidence], [raw])

    def test_release_provenance_reverifies_downloaded_bytes_and_attestations(self) -> None:
        plugin = self.root / "cuda-plugin.dll"
        plugin.write_bytes(b"real plugin bytes")
        entry = self.entry(plugin_sha=gate._sha256_file(plugin))
        evidence, raw_path = self.pair(entry)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))

        subjects = {
            "openasr-neutral.zip": self.root / "openasr-neutral.zip",
            "cuda-plugin.dll": plugin,
            "cuda-vendor.zip": self.root / "cuda-vendor.zip",
            "catalog.backends.candidate.json": self.root
            / "catalog.backends.candidate.json",
            entry.name: entry,
        }
        for name, path in subjects.items():
            if not path.exists():
                path.write_bytes(f"subject:{name}".encode())
        subject_hashes = {name: gate._sha256_file(path) for name, path in subjects.items()}
        raw["neutral_archive_sha256"] = subject_hashes["openasr-neutral.zip"]
        raw["plugin_sha256"] = subject_hashes["cuda-plugin.dll"]
        raw["vendor_archive_sha256"] = subject_hashes["cuda-vendor.zip"]
        raw["catalog_candidate_sha256"] = subject_hashes[
            "catalog.backends.candidate.json"
        ]
        raw["attested_release_subjects"] = [
            {
                "filename": name,
                "sha256": digest,
                "verification_sha256": f"{index + 1:064x}",
            }
            for index, (name, digest) in enumerate(sorted(subject_hashes.items()))
        ]
        checksums = self.root / "SHA256SUMS"
        checksums.write_text(
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(subject_hashes.items())
            ),
            encoding="utf-8",
        )
        raw["checksums_sha256"] = gate._sha256_file(checksums)
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        summary = json.loads(evidence.read_text(encoding="utf-8"))
        summary["plugin_sha256"] = subject_hashes["cuda-plugin.dll"]
        summary["evidence_sha256"] = gate._canonical_sha256(raw)
        evidence.write_text(json.dumps(summary), encoding="utf-8")

        completed = subprocess.CompletedProcess([], 0, b"[{}]", b"")
        with mock.patch("release_attestation.subprocess.run", return_value=completed) as run:
            gate.verify_release_provenance(
                entry_paths=[entry],
                raw_audit_paths=[raw_path],
                release_subject_paths=list(subjects.values()),
                checksums_path=checksums,
                repository="QuintinShaw/openasr",
                signer_workflow="QuintinShaw/openasr/.github/workflows/release-binaries.yml",
                source_digest="7" * 40,
            )
        self.assertEqual(run.call_count, len(subjects))

    def test_release_preflight_authenticates_every_subject_before_execution(self) -> None:
        entry = self.entry()
        neutral = self.root / "openasr-neutral.zip"
        neutral.write_bytes(b"neutral")
        subjects = [entry, neutral]
        checksums = self.root / "SHA256SUMS"
        checksums.write_text(
            "".join(f"{gate._sha256_file(path)}  {path.name}\n" for path in subjects),
            encoding="utf-8",
        )
        completed = subprocess.CompletedProcess([], 0, b"[{}]", b"")
        with mock.patch("release_attestation.subprocess.run", return_value=completed) as run:
            gate.preflight_release_subjects(
                entry_paths=[entry],
                release_subject_paths=subjects,
                checksums_path=checksums,
                repository="QuintinShaw/openasr",
                signer_workflow=(
                    "QuintinShaw/openasr/.github/workflows/release-binaries.yml"
                ),
                source_digest="7" * 40,
            )
        self.assertEqual(run.call_count, len(subjects))

        neutral.write_bytes(b"replaced after checksums")
        with self.assertRaisesRegex(gate.EvidenceError, "does not match SHA256SUMS"):
            gate.preflight_release_subjects(
                entry_paths=[entry],
                release_subject_paths=subjects,
                checksums_path=checksums,
                repository="QuintinShaw/openasr",
                signer_workflow=(
                    "QuintinShaw/openasr/.github/workflows/release-binaries.yml"
                ),
                source_digest="7" * 40,
            )

    def test_qualification_witnesses_require_their_own_attestations(self) -> None:
        entry = self.entry()
        evidence, raw = self.pair(entry)
        completed = subprocess.CompletedProcess([], 0, b"[{}]", b"")
        with mock.patch("release_attestation.subprocess.run", return_value=completed) as run:
            gate.verify_qualification_provenance(
                evidence_paths=[evidence],
                raw_audit_paths=[raw],
                repository="QuintinShaw/openasr",
                signer_workflow=(
                    "QuintinShaw/openasr/.github/workflows/"
                    "qualify-windows-backend.yml"
                ),
                source_digest="7" * 40,
            )
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIn("--source-digest", command)
            self.assertIn("7" * 40, command)
            self.assertIn("qualify-windows-backend.yml", " ".join(command))

        failed = subprocess.CompletedProcess([], 1, b"", b"untrusted")
        with mock.patch("release_attestation.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(gate.EvidenceError, "attestation failed"):
                gate.verify_qualification_provenance(
                    evidence_paths=[evidence],
                    raw_audit_paths=[raw],
                    repository="QuintinShaw/openasr",
                    signer_workflow=(
                        "QuintinShaw/openasr/.github/workflows/"
                        "qualify-windows-backend.yml"
                    ),
                    source_digest="7" * 40,
                )

    def test_inert_catalog_may_publish_but_activated_catalog_requires_exact_evidence(self) -> None:
        entry = self.entry()
        entry_value = json.loads(entry.read_text(encoding="utf-8"))
        catalog = self.write(
            "catalog.json",
            {"backends": [entry_value]},
        )
        gate.verify_catalog_policy(catalog, "1.2.3", [], [])

        entry_value["activation"] = {
            "state": "activated",
            "hardware_evidence_sha256": "0" * 64,
        }
        activated = self.write("catalog-activated.json", {"backends": [entry_value]})
        with self.assertRaisesRegex(gate.EvidenceError, "without exact hardware"):
            gate.verify_catalog_policy(activated, "1.2.3", [], [])

        evidence, raw = self.pair(entry)
        approved = gate.approved_entry_paths([entry], [evidence], [raw])
        entry_value["activation"]["hardware_evidence_sha256"] = gate._canonical_sha256(
            json.loads(evidence.read_text(encoding="utf-8"))
        )
        entry_value["activation"]["qualified_device_target"] = "sm_86"
        entry_value["activation"]["qualified_driver_version"] = "12.7.0"
        activated.write_text(json.dumps({"backends": [entry_value]}), encoding="utf-8")
        gate.verify_catalog_policy(activated, "1.2.3", approved, [evidence])


if __name__ == "__main__":
    unittest.main()
