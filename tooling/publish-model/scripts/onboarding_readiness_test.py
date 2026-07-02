#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _catalog import CATALOG_URL
from onboarding_readiness import audit_models


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def sidecar(model: str, quant: str, pack: Path, size: int = 8) -> dict:
    return {
        "model": model,
        "quant": quant,
        "cli_token": quant.replace("_", "-"),
        "pack": str(pack),
        "size_bytes": size,
        "sha256": "a" * 64,
    }


def metric(size: int = 8) -> dict:
    return {
        "size_bytes": size,
        "rtf_cpu": 0.1,
        "rtf_metal": None,
        "peak_rss_bytes": 1024,
        "jfk_wer_vs_fp16": 0.0,
    }


class OnboardingReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.publish_catalog = self.root / "models.toml"
        self.publish_catalog.write_text(
            '["new-model"]\n'
            'quants = ["fp16", "q8_0"]\n\n'
            '["staged-model"]\n'
            'quants = ["fp16"]\n\n'
            '["public-model"]\n'
            'quants = ["fp16"]\n'
        )
        self.machine_catalog = self.root / "catalog.json"
        self.artifacts = self.root / "tmp" / "publish"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_complete_artifacts(self, model: str, quants: list[str]) -> None:
        model_root = self.artifacts / model
        packs = model_root / "packs"
        packs.mkdir(parents=True)
        metrics = {"quants": {}}
        for quant in quants:
            pack = packs / f"{model}-{quant}.oasr"
            pack.write_bytes(b"oasrpack")
            write_json(packs / f"{model}.{quant}.result.json", sidecar(model, quant, pack))
            metrics["quants"][quant] = metric()
        write_json(model_root / "metrics.json", metrics)
        (model_root / "hf_repo.txt").write_text(f"OpenASR/{model}\n")
        (model_root / "hf_revision.txt").write_text("b" * 40 + "\n")

    def write_catalog(self, models: list[dict]) -> None:
        write_json(
            self.machine_catalog,
            {
                "schema_version": 1,
                "generated_at": "2026-05-31T00:00:00Z",
                "catalog_url": CATALOG_URL,
                "models": models,
            },
        )

    def catalog_model(self, model: str, public: bool) -> dict:
        return {
            "id": model,
            "hf_repo": f"OpenASR/{model}",
            "hf_revision": "c" * 40,
            "public": public,
            "quants": [
                {
                    "quant": "fp16",
                    "sha256": "d" * 64,
                    "size_bytes": 8,
                    "perf": {"rtf_cpu": 0.1},
                }
            ],
        }

    def audit(self, models: list[str]) -> dict[str, str]:
        rows = audit_models(
            publish_catalog=self.publish_catalog,
            machine_catalog=self.machine_catalog,
            artifact_root=self.artifacts,
            selected_models=models,
        )
        return {row.model: row.status for row in rows}

    def test_complete_local_evidence_without_catalog_is_ready_for_manifest(self) -> None:
        self.write_catalog([])
        self.write_complete_artifacts("new-model", ["fp16", "q8_0"])

        self.assertEqual(self.audit(["new-model"]), {"new-model": "ready_for_manifest"})

    def test_committed_public_and_staging_catalog_entries_win_without_tmp(self) -> None:
        self.write_catalog(
            [
                self.catalog_model("staged-model", public=False),
                self.catalog_model("public-model", public=True),
            ]
        )

        self.assertEqual(
            self.audit(["staged-model", "public-model"]),
            {
                "staged-model": "staging_cataloged",
                "public-model": "public_cataloged",
            },
        )

    def test_missing_artifacts_stays_blocked(self) -> None:
        self.write_catalog([])

        rows = audit_models(
            publish_catalog=self.publish_catalog,
            machine_catalog=self.machine_catalog,
            artifact_root=self.artifacts,
            selected_models=["new-model"],
        )

        self.assertEqual(rows[0].status, "needs_artifacts")
        self.assertIn("missing pack for fp16", rows[0].blockers)


if __name__ == "__main__":
    unittest.main()
