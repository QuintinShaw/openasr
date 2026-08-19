from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
        return self.write(
            f"entry-{provider}-{target}.json",
            {
                "id": f"{provider}-{target}",
                "vendor": provider,
                "version": "1.2.3",
                "targets": [target],
                "min_driver_api": "12.0.0" if provider == "cuda" else "7.2.0",
                "host_abi": {"fingerprint": "9" * 64},
                "files": [
                    {
                        "role": "plugin",
                        "filename": f"{provider}-{target}.dll",
                        "sha256": plugin_sha,
                        "size_bytes": 123,
                    }
                ],
            },
        )

    def evidence(
        self,
        entry: Path,
        *,
        plugin_sha: str | None = None,
        runs: int = 5,
        schema_version: int = 1,
        matrix_entries: list[Path] | None = None,
    ) -> Path:
        _, identity = gate._entry_identity(entry)
        value: dict[str, object] = {
            "schema_version": schema_version,
            "result": "pass",
            "provider": identity.provider,
            "device_target": identity.target,
            "backend_id": identity.backend_id,
            "release_version": identity.version,
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
        if schema_version == 2:
            identities = [gate._entry_identity(path)[1] for path in matrix_entries or [entry]]
            value.update(
                {
                    "scope": "provider_matrix",
                    "approved_targets": sorted(item.target for item in identities),
                    "provider_matrix_sha256": gate.provider_matrix_sha256(identities),
                }
            )
        return self.write(f"evidence-v{schema_version}.json", value)

    def test_exact_receipt_approves_only_its_entry(self) -> None:
        tested = self.entry()
        other = self.entry(target="sm_89", plugin_sha="8" * 64)
        self.assertEqual(
            gate.approved_entry_paths([tested, other], [self.evidence(tested)]),
            [tested],
        )

    def test_raw_pack_artifact_fingerprint_is_computed(self) -> None:
        entry = self.entry()
        raw = json.loads(entry.read_text(encoding="utf-8"))
        self.assertNotIn("artifact_fingerprint", raw)
        _, identity = gate._entry_identity(entry)
        self.assertEqual(len(identity.artifact_fingerprint), 64)

    def test_provider_matrix_receipt_approves_bound_targets(self) -> None:
        tested = self.entry()
        other = self.entry(target="sm_89", plugin_sha="8" * 64)
        hip = self.entry(target="gfx1200", plugin_sha="7" * 64, provider="hip")
        evidence = self.evidence(
            tested,
            schema_version=2,
            matrix_entries=[tested, other],
        )
        self.assertEqual(
            gate.approved_entry_paths([tested, other, hip], [evidence]),
            sorted([tested, other]),
        )

    def test_provider_matrix_rejects_changed_or_missing_target(self) -> None:
        tested = self.entry()
        other = self.entry(target="sm_89", plugin_sha="8" * 64)
        evidence = self.evidence(
            tested,
            schema_version=2,
            matrix_entries=[tested, other],
        )
        changed = self.entry(target="sm_89", plugin_sha="7" * 64)
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([tested, changed], [evidence])
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([tested], [evidence])
        partial = self.evidence(
            tested,
            schema_version=2,
            matrix_entries=[tested],
        )
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([tested, other], [partial])

    def test_different_bytes_or_insufficient_runs_are_rejected(self) -> None:
        entry = self.entry()
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths(
                [entry], [self.evidence(entry, plugin_sha="9" * 64)]
            )
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([entry], [self.evidence(entry, runs=4)])

    def test_live_catalog_must_equal_approved_subset(self) -> None:
        entry = self.entry()
        catalog = self.write("catalog.json", {"backends": []})
        with self.assertRaises(gate.EvidenceError):
            gate.verify_catalog_policy(catalog, "1.2.3", [entry])
        catalog = self.write(
            "catalog.json",
            {
                "backends": [
                    {"id": "cuda-sm_86", "vendor": "cuda", "version": "1.2.3"}
                ]
            },
        )
        gate.verify_catalog_policy(catalog, "1.2.3", [entry])


if __name__ == "__main__":
    unittest.main()
