#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

import convert_diarizen as C


class TensorTypeTest(unittest.TestCase):
    def test_sensitive_and_affine_tensors_stay_f32(self):
        cases = [
            ("weight_sum.weight", (1, 13)),
            ("x.layer_norm.weight", (768,)),
            ("x.rel_attn_embed.weight", (320, 12)),
            ("x.gru_rel_pos_linear.weight", (8, 64)),
            ("x.bias", (768,)),
        ]
        for name, shape in cases:
            with self.subTest(name=name):
                self.assertEqual(C.choose_tensor_type(name, shape, "q8_0"), "f32")

    def test_dense_projections_take_requested_quant(self):
        self.assertEqual(C.choose_tensor_type("proj.weight", (256, 768), "f16"), "f16")
        self.assertEqual(C.choose_tensor_type("proj.weight", (256, 768), "q8_0"), "q8_0")
        self.assertEqual(
            C.choose_tensor_type("odd.weight", (256, 137), "q8_0"), "f16"
        )
        self.assertEqual(
            C.choose_tensor_type("conv.weight", (256, 256, 1), "q8_0"), "f16"
        )

    def test_catalog_fp16_spelling_is_accepted(self):
        self.assertEqual(C.normalize_quant("fp16"), "f16")
        self.assertEqual(C.normalize_quant("q8_0"), "q8_0")


class RuntimeTensorNameTest(unittest.TestCase):
    def test_compacts_every_long_upstream_namespace(self):
        cases = {
            "wavlm_model.feature_extractor.conv_layers.0.layer_norm.weight":
                "dz.fe.conv_layers.0.layer_norm.weight",
            "wavlm_model.encoder.feature_projection.projection.weight":
                "dz.fp.projection.weight",
            "wavlm_model.encoder.transformer.layers.0.feed_forward.intermediate_dense.weight":
                "dz.tr.layers.0.feed_forward.intermediate_dense.weight",
            "conformer.conformer_layer.0.conv.depthwise_conv.weight":
                "dz.cf.0.conv.depthwise_conv.weight",
            "classifier.weight": "classifier.weight",
        }
        for upstream, expected in cases.items():
            with self.subTest(upstream=upstream):
                actual = C.runtime_tensor_name(upstream)
                self.assertEqual(actual, expected)
                self.assertLessEqual(
                    len(actual.encode("utf-8")), C.GGUF_MAX_TENSOR_NAME_BYTES
                )

    def test_rejects_an_unmapped_overlong_name(self):
        with self.assertRaisesRegex(C.ConversionError, "exceeds 63 bytes"):
            C.runtime_tensor_name("x" * 64)


class PositionalConvFoldTest(unittest.TestCase):
    def test_fold_matches_weight_norm_dim_two(self):
        rng = np.random.default_rng(0)
        g = rng.standard_normal((1, 1, 128)).astype(np.float32)
        v = rng.standard_normal((768, 48, 128)).astype(np.float32)
        state = {C.POS_CONV_G: g, C.POS_CONV_V: v}
        result = C.materialize_runtime_state(state)
        self.assertNotIn(C.POS_CONV_G, result)
        self.assertNotIn(C.POS_CONV_V, result)
        norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=(0, 1), keepdims=True))
        expected = (g.astype(np.float64) * v.astype(np.float64) / norm).astype(np.float32)
        np.testing.assert_allclose(result[C.POS_CONV_WEIGHT], expected, rtol=0, atol=0)


class PackRoundTripTest(unittest.TestCase):
    def test_metadata_and_tensor_types_roundtrip(self):
        import gguf

        rng = np.random.default_rng(1)
        plan = [
            C.TensorPlan("proj.weight", rng.standard_normal((32, 64)).astype(np.float32), "q8_0"),
            C.TensorPlan("proj.bias", rng.standard_normal((32,)).astype(np.float32), "f32"),
            C.TensorPlan("classifier.weight", rng.standard_normal((11, 32)).astype(np.float32), "f16"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "diarizen-test.oasr"
            C.write_pack(out, plan, quant="q8_0")
            reader = gguf.GGUFReader(str(out))
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(
                tensors["proj.weight"].tensor_type, gguf.GGMLQuantizationType.Q8_0
            )
            self.assertEqual(
                tensors["proj.bias"].tensor_type, gguf.GGMLQuantizationType.F32
            )
            self.assertEqual(
                tensors["classifier.weight"].tensor_type,
                gguf.GGMLQuantizationType.F16,
            )
            fields = set(reader.fields)
            self.assertIn("diarizen.wavlm_config_json", fields)
            self.assertIn("diarizen.median_filter_frames", fields)
            self.assertIn("diarizen.tensor_schema", fields)


if __name__ == "__main__":
    unittest.main()
