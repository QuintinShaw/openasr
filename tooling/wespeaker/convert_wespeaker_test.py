#!/usr/bin/env python3
"""Conversion-correctness tests for the WeSpeaker ResNet -> .oasr converter.

Covers remap/type selection plus a synthetic state-dict -> GGUF round-trip
(no real weights): reads the pack back with the ``gguf`` reader and checks
tensor set, dims (ggml ne order), metadata, and f32 payload fidelity.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import convert_wespeaker as C


class RemapTest(unittest.TestCase):
    def test_keeps_official_names(self):
        self.assertEqual(C.remap_tensor("conv1.weight"), "conv1.weight")
        self.assertEqual(C.remap_tensor("layer1.0.conv1.weight"), "layer1.0.conv1.weight")
        self.assertEqual(C.remap_tensor("seg_1.weight"), "seg_1.weight")
        self.assertEqual(C.remap_tensor("bn1.running_mean"), "bn1.running_mean")

    def test_strips_pyannote_resnet_prefix(self):
        self.assertEqual(C.remap_tensor("resnet.conv1.weight"), "conv1.weight")
        self.assertEqual(
            C.remap_tensor("resnet.layer2.0.shortcut.0.weight"),
            "layer2.0.shortcut.0.weight",
        )
        self.assertEqual(C.remap_tensor("resnet.seg_1.bias"), "seg_1.bias")

    def test_drops_projection_and_counters(self):
        self.assertIsNone(C.remap_tensor("projection.weight"))
        self.assertIsNone(C.remap_tensor("resnet.projection.weight"))
        self.assertIsNone(C.remap_tensor("bn1.num_batches_tracked"))
        self.assertIsNone(C.remap_tensor("layer1.0.bn1.num_batches_tracked"))
        self.assertIsNone(C.remap_tensor("resnet.layer1.0.bn2.num_batches_tracked"))

    def test_drops_identity_second_emb_layer(self):
        self.assertIsNone(C.remap_tensor("seg_2.weight"))
        self.assertIsNone(C.remap_tensor("resnet.seg_2.bias"))
        self.assertIsNone(C.remap_tensor("seg_bn_1.weight"))
        self.assertIsNone(C.remap_tensor("resnet.seg_bn_1.running_mean"))


class TensorTypeTest(unittest.TestCase):
    def test_bn_and_bias_force_f32(self):
        self.assertEqual(C.choose_tensor_type("bn1.weight", (32,), "f16"), "f32")
        self.assertEqual(C.choose_tensor_type("seg_1.bias", (256,), "f16"), "f32")
        self.assertEqual(
            C.choose_tensor_type("layer1.0.bn1.running_var", (32,), "f16"), "f32"
        )
        self.assertEqual(
            C.choose_tensor_type("layer2.0.shortcut.1.weight", (64,), "f16"), "f32"
        )

    def test_rank2_plus_conv_takes_quant(self):
        self.assertEqual(
            C.choose_tensor_type("conv1.weight", (32, 1, 3, 3), "f16"), "f16"
        )
        self.assertEqual(
            C.choose_tensor_type("seg_1.weight", (256, 5120), "f16"), "f16"
        )
        self.assertEqual(
            C.choose_tensor_type("layer2.0.shortcut.0.weight", (64, 32, 1, 1), "f16"),
            "f16",
        )

    def test_f32_quant_overrides_everything(self):
        self.assertEqual(
            C.choose_tensor_type("conv1.weight", (32, 1, 3, 3), "f32"), "f32"
        )
        self.assertEqual(C.choose_tensor_type("seg_1.weight", (256, 5120), "f32"), "f32")


class TopologyInferTest(unittest.TestCase):
    def test_resnet34_from_official_keys(self):
        state = {
            "conv1.weight": np.zeros((32, 1, 3, 3), np.float32),
            **{
                f"layer{stage}.{block}.conv1.weight": np.zeros((1,), np.float32)
                for stage, n in enumerate([3, 4, 6, 3], start=1)
                for block in range(n)
            },
        }
        self.assertEqual(C.infer_num_blocks(state), [3, 4, 6, 3])
        self.assertEqual(C.infer_block_kind(state), "basic")
        self.assertEqual(C.infer_depth(state), 34)

    def test_bottleneck_depths_from_official_keys(self):
        for depth, num_blocks in (
            (152, [3, 8, 36, 3]),
            (221, [6, 16, 48, 3]),
            (293, [10, 20, 64, 3]),
        ):
            with self.subTest(depth=depth):
                state = {
                    "conv1.weight": np.zeros((32, 1, 3, 3), np.float32),
                    "layer1.0.conv3.weight": np.zeros((1,), np.float32),
                    **{
                        f"layer{stage}.{block}.conv1.weight": np.zeros((1,), np.float32)
                        for stage, n in enumerate(num_blocks, start=1)
                        for block in range(n)
                    },
                }
                self.assertEqual(C.infer_num_blocks(state), num_blocks)
                self.assertEqual(C.infer_block_kind(state), "bottleneck")
                self.assertEqual(C.infer_depth(state), depth)


class QuantAliasTest(unittest.TestCase):
    def test_catalog_fp16_maps_to_converter_f16(self):
        self.assertEqual(C.normalize_quant("fp16"), "f16")
        self.assertEqual(C.normalize_quant("f16"), "f16")
        self.assertEqual(C.normalize_quant("f32"), "f32")


class BuildProvenanceTest(unittest.TestCase):
    def test_missing_env_is_optional(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop(C.BUILD_COMMIT_ENV, None)
            self.assertIsNone(C.build_provenance_from_env())

    def test_rejects_non_sha(self):
        with mock.patch.dict("os.environ", {C.BUILD_COMMIT_ENV: "not-a-commit"}):
            with self.assertRaises(C.ConversionError):
                C.build_provenance_from_env()

    def test_accepts_40_hex(self):
        commit = "a" * 40
        with mock.patch.dict("os.environ", {C.BUILD_COMMIT_ENV: commit}):
            self.assertEqual(C.build_provenance_from_env(), commit)


class RoundTripTest(unittest.TestCase):
    def _synthetic_state(self, *, pyannote_prefix: bool = False):
        rng = np.random.default_rng(0)

        def name(official: str) -> str:
            return f"resnet.{official}" if pyannote_prefix else official

        return {
            name("conv1.weight"): rng.standard_normal((32, 1, 3, 3)).astype(np.float32),
            name("bn1.weight"): rng.standard_normal((32,)).astype(np.float32),
            name("bn1.bias"): rng.standard_normal((32,)).astype(np.float32),
            name("bn1.running_mean"): rng.standard_normal((32,)).astype(np.float32),
            name("bn1.running_var"): rng.standard_normal((32,)).astype(np.float32),
            name("bn1.num_batches_tracked"): np.array(12, dtype=np.int64),
            name("seg_1.weight"): rng.standard_normal((256, 5120)).astype(np.float32),
            name("seg_1.bias"): rng.standard_normal((256,)).astype(np.float32),
            name("projection.weight"): rng.standard_normal((5994, 256)).astype(np.float32),
            name("layer1.0.conv1.weight"): rng.standard_normal((32, 32, 3, 3)).astype(
                np.float32
            ),
            name("layer1.1.conv1.weight"): rng.standard_normal((32, 32, 3, 3)).astype(
                np.float32
            ),
            name("layer1.2.conv1.weight"): rng.standard_normal((32, 32, 3, 3)).astype(
                np.float32
            ),
            name("layer2.0.conv1.weight"): rng.standard_normal((64, 32, 3, 3)).astype(
                np.float32
            ),
            name("layer2.1.conv1.weight"): rng.standard_normal((64, 64, 3, 3)).astype(
                np.float32
            ),
            name("layer2.2.conv1.weight"): rng.standard_normal((64, 64, 3, 3)).astype(
                np.float32
            ),
            name("layer2.3.conv1.weight"): rng.standard_normal((64, 64, 3, 3)).astype(
                np.float32
            ),
            name("layer3.0.conv1.weight"): rng.standard_normal((128, 64, 3, 3)).astype(
                np.float32
            ),
            name("layer3.1.conv1.weight"): rng.standard_normal((128, 128, 3, 3)).astype(
                np.float32
            ),
            name("layer3.2.conv1.weight"): rng.standard_normal((128, 128, 3, 3)).astype(
                np.float32
            ),
            name("layer3.3.conv1.weight"): rng.standard_normal((128, 128, 3, 3)).astype(
                np.float32
            ),
            name("layer3.4.conv1.weight"): rng.standard_normal((128, 128, 3, 3)).astype(
                np.float32
            ),
            name("layer3.5.conv1.weight"): rng.standard_normal((128, 128, 3, 3)).astype(
                np.float32
            ),
            name("layer4.0.conv1.weight"): rng.standard_normal((256, 128, 3, 3)).astype(
                np.float32
            ),
            name("layer4.1.conv1.weight"): rng.standard_normal((256, 256, 3, 3)).astype(
                np.float32
            ),
            name("layer4.2.conv1.weight"): rng.standard_normal((256, 256, 3, 3)).astype(
                np.float32
            ),
        }

    def test_roundtrip_f32_official_keys(self):
        import gguf

        state = self._synthetic_state()
        plan = C.build_tensor_plan(state, "f32")
        names = {p[0] for p in plan}
        self.assertIn("conv1.weight", names)
        self.assertIn("seg_1.weight", names)
        self.assertNotIn("projection.weight", names)
        self.assertNotIn("bn1.num_batches_tracked", names)
        self.assertNotIn("resnet.conv1.weight", names)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "wespeaker-test.oasr"
            C.write_pack(
                out,
                plan,
                model_id="wespeaker-voxceleb-resnet34-lm",
                quant="f32",
                depth=34,
                block_kind="basic",
                num_blocks=[3, 4, 6, 3],
            )
            reader = gguf.GGUFReader(str(out))
            kv = {field.name: field for field in reader.fields.values()}
            self.assertIn("openasr.model.architecture", kv)
            self.assertIn("wespeaker.embed_dim", kv)
            self.assertIn("wespeaker.num_blocks", kv)

            rt = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(set(rt.keys()), names)

            conv = rt["conv1.weight"]
            self.assertEqual(list(conv.shape), [3, 3, 1, 32])
            lin = rt["seg_1.weight"]
            self.assertEqual(list(lin.shape), [5120, 256])

            got = np.array(conv.data, dtype=np.float32).reshape(-1)
            want = state["conv1.weight"].astype(np.float32).reshape(-1)
            np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def test_roundtrip_strips_pyannote_prefix(self):
        import gguf

        state = self._synthetic_state(pyannote_prefix=True)
        plan = C.build_tensor_plan(state, "f32")
        names = {p[0] for p in plan}
        self.assertIn("conv1.weight", names)
        self.assertNotIn("resnet.conv1.weight", names)
        self.assertNotIn("resnet.projection.weight", names)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "wespeaker-pyannote.oasr"
            C.write_pack(
                out,
                plan,
                model_id="wespeaker-voxceleb-resnet34-lm",
                quant="f32",
                depth=34,
                block_kind="basic",
                num_blocks=[3, 4, 6, 3],
            )
            reader = gguf.GGUFReader(str(out))
            rt = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(set(rt.keys()), names)
            kv = {field.name: field for field in reader.fields.values()}
            self.assertIn("general.architecture", kv)
            self.assertIn("wespeaker.num_blocks", kv)
            self.assertIn("openasr.license.name", kv)

    def test_roundtrip_f16_types(self):
        import gguf

        state = self._synthetic_state()
        plan = C.build_tensor_plan(state, "f16")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "wespeaker-f16.oasr"
            C.write_pack(
                out,
                plan,
                model_id="wespeaker-voxceleb-resnet34-lm",
                quant="f16",
                depth=34,
                block_kind="basic",
                num_blocks=[3, 4, 6, 3],
            )
            reader = gguf.GGUFReader(str(out))
            rt = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(rt["conv1.weight"].tensor_type, gguf.GGMLQuantizationType.F16)
            self.assertEqual(rt["seg_1.weight"].tensor_type, gguf.GGMLQuantizationType.F16)
            self.assertEqual(rt["seg_1.bias"].tensor_type, gguf.GGMLQuantizationType.F32)
            self.assertEqual(rt["bn1.weight"].tensor_type, gguf.GGMLQuantizationType.F32)


if __name__ == "__main__":
    unittest.main()
