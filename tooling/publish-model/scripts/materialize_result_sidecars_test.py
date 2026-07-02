#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from materialize_result_sidecars import (
    ResultSidecarError,
    build_sidecar,
    materialize_model,
)


MODEL = "qwen3-asr-0.6b"


class MaterializeResultSidecarsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.catalog = self.root / "models.toml"
        self.catalog.write_text(
            f'["{MODEL}"]\n'
            'quants = ["fp16", "q8_0"]\n'
        )
        self.packs = self.root / "tmp" / "publish" / MODEL / "packs"
        self.packs.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_pack(
        self,
        quant: str,
        contents: bytes = b"general.architecture\0qwen3-asr\0oasr-pack",
    ) -> Path:
        if b"general.architecture" not in contents:
            contents = b"general.architecture\0qwen3-asr\0" + contents
        path = self.packs / f"{MODEL}-{quant}.oasr"
        path.write_bytes(contents)
        return path

    def read_result(self, quant: str) -> dict:
        return json.loads((self.packs / f"{MODEL}.{quant}.result.json").read_text())

    def test_writes_standard_sidecars_from_existing_packs(self) -> None:
        fp16 = self.write_pack("fp16", b"fp16-data")
        q8 = self.write_pack("q8_0", b"q8-data")

        results = materialize_model(repo_root=self.root, catalog_path=self.catalog, model=MODEL)

        self.assertEqual(
            results,
            [
                self.packs / f"{MODEL}.fp16.result.json",
                self.packs / f"{MODEL}.q8_0.result.json",
            ],
        )
        self.assertEqual(
            self.read_result("fp16"),
            {
                "model": MODEL,
                "quant": "fp16",
                "cli_token": "fp16",
                "pack": str(fp16),
                "size_bytes": fp16.stat().st_size,
                "sha256": hashlib.sha256(fp16.read_bytes()).hexdigest(),
            },
        )
        self.assertEqual(self.read_result("q8_0")["pack"], str(q8))

    def test_existing_matching_sidecar_is_accepted(self) -> None:
        pack = self.write_pack("fp16", b"stable")
        expected = {
            "model": MODEL,
            "quant": "fp16",
            "cli_token": "fp16",
            "pack": str(pack),
            "size_bytes": pack.stat().st_size,
            "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
        }
        (self.packs / f"{MODEL}.fp16.result.json").write_text(json.dumps(expected, separators=(",", ":")) + "\n")

        materialize_model(
            repo_root=self.root,
            catalog_path=self.catalog,
            model=MODEL,
            quants=["fp16"],
        )

        self.assertEqual(self.read_result("fp16"), expected)

    def test_rewrites_relocated_sidecar_pack_path_when_bytes_match(self) -> None:
        pack = self.write_pack("fp16", b"stable")
        expected = {
            "model": MODEL,
            "quant": "fp16",
            "cli_token": "fp16",
            "pack": str(pack),
            "size_bytes": pack.stat().st_size,
            "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
        }
        stale = {
            **expected,
            "pack": "/old/checkout/tmp/publish/qwen3-asr-0.6b/packs/qwen3-asr-0.6b-fp16.oasr",
        }
        (self.packs / f"{MODEL}.fp16.result.json").write_text(json.dumps(stale, separators=(",", ":")) + "\n")

        materialize_model(
            repo_root=self.root,
            catalog_path=self.catalog,
            model=MODEL,
            quants=["fp16"],
        )

        self.assertEqual(self.read_result("fp16"), expected)

    def test_rejects_existing_mismatched_sidecar(self) -> None:
        self.write_pack("fp16", b"actual")
        (self.packs / f"{MODEL}.fp16.result.json").write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "quant": "fp16",
                    "cli_token": "fp16",
                    "pack": "wrong",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                }
            )
        )

        with self.assertRaisesRegex(ResultSidecarError, "mismatch"):
            materialize_model(
                repo_root=self.root,
                catalog_path=self.catalog,
                model=MODEL,
                quants=["fp16"],
            )

    def test_rejects_missing_pack(self) -> None:
        with self.assertRaisesRegex(ResultSidecarError, "pack missing"):
            materialize_model(
                repo_root=self.root,
                catalog_path=self.catalog,
                model=MODEL,
                quants=["fp16"],
            )

    def test_rejects_non_oasr_pack_path(self) -> None:
        pack = self.packs / f"{MODEL}-fp16.gguf"
        pack.write_bytes(b"not-oasr")

        with self.assertRaisesRegex(ResultSidecarError, ".oasr"):
            build_sidecar(MODEL, "fp16", pack)

    def test_rejects_legacy_qwen_architecture_metadata(self) -> None:
        pack = self.write_pack("fp16", b"general.architecture\0qwen3asr\0oasr-pack")

        with self.assertRaisesRegex(ResultSidecarError, "legacy qwen"):
            build_sidecar(MODEL, "fp16", pack)

    def test_hymt2_sidecar_requires_translation_pack_metadata(self) -> None:
        model = "hymt2-1.8b"
        packs = self.root / "tmp" / "publish" / model / "packs"
        packs.mkdir(parents=True)
        pack = packs / f"{model}-q4_k_m.oasr"
        pack.write_bytes(
            b"general.architecture\0hunyuan-dense\0"
            b"openasr.model.kind\0translation-model\0"
            b"openasr.translation.source_langs\0zh\0"
            b"openasr.translation.target_langs\0en\0"
            b"openasr.upstream.base_revision\0"
            b"9a341cd1b679d3efd23b46e847b01745a71ed792\0"
            b"openasr.upstream.gguf_revision\0"
            b"1cd5208700acedef4ef93019b6cfc148b8522d45\0"
            b"openasr.license.files\0LICENSE.txt\0NOTICE.openasr.txt\0"
        )

        sidecar = build_sidecar(model, "q4_k_m", pack)

        self.assertEqual(sidecar["model"], model)
        self.assertEqual(sidecar["quant"], "q4_k_m")
        self.assertEqual(sidecar["cli_token"], "q4-k-m")

    def test_hymt2_sidecar_rejects_missing_notice_marker(self) -> None:
        model = "hymt2-1.8b"
        packs = self.root / "tmp" / "publish" / model / "packs"
        packs.mkdir(parents=True)
        pack = packs / f"{model}-q4_k_m.oasr"
        pack.write_bytes(
            b"general.architecture\0hunyuan-dense\0"
            b"openasr.model.kind\0translation-model\0"
            b"openasr.translation.source_langs\0zh\0"
            b"openasr.translation.target_langs\0en\0"
            b"openasr.upstream.base_revision\0"
            b"9a341cd1b679d3efd23b46e847b01745a71ed792\0"
            b"openasr.upstream.gguf_revision\0"
            b"1cd5208700acedef4ef93019b6cfc148b8522d45\0"
            b"openasr.license.files\0LICENSE.txt\0"
        )

        with self.assertRaisesRegex(ResultSidecarError, "NOTICE.openasr.txt"):
            build_sidecar(model, "q4_k_m", pack)

    def test_rejects_quant_outside_catalog(self) -> None:
        with self.assertRaisesRegex(ResultSidecarError, "not in publish catalog"):
            materialize_model(
                repo_root=self.root,
                catalog_path=self.catalog,
                model=MODEL,
                quants=["q4_k"],
            )


if __name__ == "__main__":
    unittest.main()
