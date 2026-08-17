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

    def entry(self, target: str = "sm_86", plugin_sha: str = "b" * 64) -> Path:
        return self.write(
            "entry.json",
            {
                "id": "cuda-sm86",
                "vendor": "cuda",
                "version": "1.2.3",
                "targets": [target],
                "artifact_fingerprint": "a" * 64,
                "files": [{"role": "plugin", "sha256": plugin_sha}],
            },
        )

    def evidence(self, plugin_sha: str = "b" * 64, runs: int = 5) -> Path:
        return self.write(
            "evidence.json",
            {
                "schema_version": 1,
                "result": "pass",
                "provider": "cuda",
                "device_target": "sm_86",
                "backend_id": "cuda-sm86",
                "release_version": "1.2.3",
                "artifact_fingerprint": "a" * 64,
                "plugin_sha256": plugin_sha,
                "binary_sha256": "c" * 64,
                "workload_sha256": "d" * 64,
                "model_pack_sha256": "e" * 64,
                "evidence_sha256": "f" * 64,
                "fresh_process_runs": runs,
                "placement": "full_device",
                "cpu_fallback": False,
            },
        )

    def test_exact_receipt_approves_only_its_entry(self) -> None:
        entry = self.entry()
        self.assertEqual(gate.approved_entry_paths([entry], [self.evidence()]), [entry])

    def test_different_bytes_or_insufficient_runs_are_rejected(self) -> None:
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([self.entry()], [self.evidence(plugin_sha="9" * 64)])
        with self.assertRaises(gate.EvidenceError):
            gate.approved_entry_paths([self.entry()], [self.evidence(runs=4)])

    def test_live_catalog_must_equal_approved_subset(self) -> None:
        entry = self.entry()
        catalog = self.write("catalog.json", {"backends": []})
        with self.assertRaises(gate.EvidenceError):
            gate.verify_catalog_policy(catalog, "1.2.3", [entry])
        catalog = self.write(
            "catalog.json",
            {"backends": [{"id": "cuda-sm86", "vendor": "cuda", "version": "1.2.3"}]},
        )
        gate.verify_catalog_policy(catalog, "1.2.3", [entry])


if __name__ == "__main__":
    unittest.main()
