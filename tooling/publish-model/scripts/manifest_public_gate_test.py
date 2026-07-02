#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import _manifest  # noqa: E402


class ManifestPublicGateTest(unittest.TestCase):
    def test_manifest_reads_repo_owned_prose_cards(self) -> None:
        prose = _manifest.read_prose("moonshine-tiny")

        self.assertIn("Tiny 27M-parameter", prose["tagline"])

    def test_build_catalog_model_reads_sidecars_and_repo_owned_cards(self) -> None:
        old_root = _manifest.REPO_ROOT
        model = "moonshine-tiny"
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _manifest.REPO_ROOT = root
                work = root / "tmp" / "publish" / model
                packs = work / "packs"
                packs.mkdir(parents=True)
                (work / "hf_revision.txt").write_text("b" * 40)
                metrics = {"quants": {}}
                for index, quant in enumerate(("fp16", "q8_0"), start=1):
                    filename = f"{model}-{quant}.oasr"
                    size = 1000 + index
                    (packs / f"{model}.{quant}.result.json").write_text(
                        json.dumps(
                            {
                                "pack": str(packs / filename),
                                "sha256": f"{index}" * 64,
                                "size_bytes": size,
                            }
                        )
                    )
                    metrics["quants"][quant] = {
                        "size_bytes": size,
                        "rtf_cpu": 0.1 * index,
                        "rtf_metal": None,
                        "peak_rss_bytes": 2_000_000 * index,
                        "jfk_wer_vs_fp16": 0.0,
                    }
                (work / "metrics.json").write_text(json.dumps(metrics))

                entry = _manifest.load_publish_catalog()[model]
                args = argparse.Namespace(
                    hf_repo=None,
                    hf_revision=None,
                    public=False,
                    min_cli_version="0.1.0",
                )

                catalog_model = _manifest.build_catalog_model(model, entry, args)

                self.assertEqual(catalog_model["id"], "moonshine-tiny")
                self.assertEqual(catalog_model["kind"], "asr-model")
                self.assertNotIn("capability", catalog_model)
                self.assertEqual(catalog_model["hf_revision"], "b" * 40)
                self.assertEqual(catalog_model["prose"]["tagline"], _manifest.read_prose(model)["tagline"])
                self.assertEqual(catalog_model["pull_recommended"], "moonshine-tiny:q8")
                self.assertTrue(all(q["url"].startswith("https://huggingface.co/OpenASR/moonshine-tiny/resolve/") for q in catalog_model["quants"]))
        finally:
            _manifest.REPO_ROOT = old_root

    def test_build_catalog_model_emits_capability_pack_metadata(self) -> None:
        old_root = _manifest.REPO_ROOT
        model = "wespeaker-voxceleb-resnet34-lm"
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _manifest.REPO_ROOT = root
                work = root / "tmp" / "publish" / model
                packs = work / "packs"
                packs.mkdir(parents=True)
                (work / "hf_revision.txt").write_text("c" * 40)
                filename = f"{model}-f32.oasr"
                (packs / f"{model}.f32.result.json").write_text(
                    json.dumps(
                        {
                            "pack": str(packs / filename),
                            "sha256": "a" * 64,
                            "size_bytes": 1234,
                        }
                    )
                )
                (work / "metrics.json").write_text(
                    json.dumps(
                        {
                            "quants": {
                                "f32": {
                                    "size_bytes": 1234,
                                    "rtf_cpu": 0.01,
                                    "peak_rss_bytes": 128,
                                }
                            }
                        }
                    )
                )

                entry = _manifest.load_publish_catalog()[model]
                args = argparse.Namespace(
                    hf_repo=None,
                    hf_revision=None,
                    public=True,
                    min_cli_version="0.1.0",
                )

                catalog_model = _manifest.build_catalog_model(model, entry, args)

                self.assertEqual(catalog_model["id"], "wespeaker-voxceleb-resnet34-lm")
                self.assertEqual(catalog_model["kind"], "capability-pack")
                self.assertEqual(
                    catalog_model["capability"],
                    {
                        "feature": "speaker-diarization",
                        "role": "speaker-embedder",
                    },
                )
                self.assertTrue(catalog_model["public"])
                self.assertEqual(
                    catalog_model["pull_recommended"],
                    "wespeaker-voxceleb-resnet34-lm:f32",
                )
        finally:
            _manifest.REPO_ROOT = old_root

    def test_build_catalog_model_emits_translation_model_metadata(self) -> None:
        old_root = _manifest.REPO_ROOT
        model = "hymt2-1.8b"
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _manifest.REPO_ROOT = root
                work = root / "tmp" / "publish" / model
                packs = work / "packs"
                packs.mkdir(parents=True)
                (work / "hf_revision.txt").write_text("d" * 40)
                filename = f"{model}-q4_k_m.oasr"
                (packs / f"{model}.q4_k_m.result.json").write_text(
                    json.dumps(
                        {
                            "pack": str(packs / filename),
                            "sha256": "f" * 64,
                            "size_bytes": 123456,
                        }
                    )
                )
                (work / "metrics.json").write_text(
                    json.dumps(
                        {
                            "quants": {
                                "q4_k_m": {
                                    "size_bytes": 123456,
                                    "peak_rss_bytes": 1_400_000_000,
                                }
                            }
                        }
                    )
                )

                entry = _manifest.load_publish_catalog()[model]
                args = argparse.Namespace(
                    hf_repo=None,
                    hf_revision=None,
                    public=False,
                    min_cli_version="0.1.0",
                )

                catalog_model = _manifest.build_catalog_model(model, entry, args)

                self.assertEqual(catalog_model["id"], "hymt2-1.8b")
                self.assertEqual(catalog_model["kind"], "translation-model")
                self.assertEqual(catalog_model["source_langs"], ["zh"])
                self.assertEqual(catalog_model["target_langs"], ["en"])
                self.assertTrue(catalog_model["experimental"])
                self.assertFalse(catalog_model["public"])
                self.assertEqual(catalog_model["recommended_quant"], "q4_k_m")
                self.assertEqual(catalog_model["pull_recommended"], "hymt2-1.8b:q4km")
                self.assertEqual(catalog_model["quants"][0]["suffix"], "q4km")
                self.assertEqual(
                    catalog_model["upstream_gguf_revision"],
                    "1cd5208700acedef4ef93019b6cfc148b8522d45",
                )
                self.assertEqual(
                    catalog_model["license_files"],
                    ["LICENSE.txt", "NOTICE.openasr.txt"],
                )
        finally:
            _manifest.REPO_ROOT = old_root

    def test_public_generation_requires_release_public_metadata(self) -> None:
        with self.assertRaises(SystemExit) as error:
            _manifest.ensure_release_public_allowed(
                "staging-model",
                {"registry_id": "staging-model"},
                True,
            )

        self.assertIn("release_public=true", str(error.exception))

    def test_release_public_metadata_allows_public_generation(self) -> None:
        _manifest.ensure_release_public_allowed(
            "release-model",
            {"registry_id": "release-model", "release_public": True},
            True,
        )

    def test_private_generation_does_not_require_release_public_metadata(self) -> None:
        _manifest.ensure_release_public_allowed(
            "staging-model",
            {"registry_id": "staging-model"},
            False,
        )


if __name__ == "__main__":
    unittest.main()
