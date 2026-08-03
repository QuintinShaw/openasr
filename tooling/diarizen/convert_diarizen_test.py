#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import convert_diarizen as C


class TensorTypeTest(unittest.TestCase):
    def test_sensitive_and_affine_tensors_stay_f32(self):
        cases = [
            ("weight_sum.weight", (1, 25)),
            ("x.layer_norm.weight", (1024,)),
            ("x.rel_attn_embed.weight", (320, 16)),
            ("x.gru_rel_pos_linear.weight", (8, 64)),
            ("x.bias", (1024,)),
        ]
        for name, shape in cases:
            with self.subTest(name=name):
                self.assertEqual(C.choose_tensor_type(name, shape, "f16"), "f32")

    def test_dense_projections_take_requested_quant(self):
        self.assertEqual(C.choose_tensor_type("proj.weight", (256, 1024), "f16"), "f16")
        self.assertEqual(C.choose_tensor_type("odd.weight", (256, 137), "f16"), "f16")
        self.assertEqual(C.choose_tensor_type("conv.weight", (256, 256, 1), "f16"), "f16")

    def test_catalog_fp16_spelling_is_accepted(self):
        self.assertEqual(C.normalize_quant("fp16"), "f16")
        with self.assertRaisesRegex(C.ConversionError, "FP16 packs only"):
            C.normalize_quant("q8_0")


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
        v = rng.standard_normal((1024, 64, 128)).astype(np.float32)
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
            C.TensorPlan("proj.weight", rng.standard_normal((32, 64)).astype(np.float32), "f16"),
            C.TensorPlan("proj.bias", rng.standard_normal((32,)).astype(np.float32), "f32"),
            C.TensorPlan("classifier.weight", rng.standard_normal((16, 32)).astype(np.float32), "f16"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "diarizen-test.oasr"
            C.write_pack(out, plan, quant="f16", model_id="diarizen-large-s80-v2")
            reader = gguf.GGUFReader(str(out))
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(
                tensors["proj.weight"].tensor_type, gguf.GGMLQuantizationType.F16
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
            self.assertEqual(_kv_str(reader, "openasr.model.id"), "diarizen-large-s80-v2")
            self.assertEqual(_kv_str(reader, "openasr.quantization"), "fp16")
            self.assertEqual(
                _kv_str(reader, "diarizen.upstream_model_id"),
                "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
            )
            self.assertEqual(
                _kv_str(reader, "openasr.source.name"), C.UPSTREAM_MODEL_ID
            )
            self.assertEqual(
                _kv_str(reader, "openasr.source.revision"), C.PINNED_REVISION
            )
            self.assertEqual(_kv_str(reader, "openasr.license.name"), C.LICENSE_NAME)
            self.assertEqual(
                _kv_str(reader, "openasr.license.source"), C.LICENSE_SOURCE
            )
            wavlm = json.loads(_kv_str(reader, "diarizen.wavlm_config_json"))
            self.assertEqual(wavlm["extractor_norm"], "layer_norm")
            self.assertTrue(wavlm["normalize_waveform"])
            self.assertTrue(wavlm["encoder_layer_norm_first"])
            self.assertFalse(wavlm["transformer_layer_norm_first"])


def _kv_str(reader, key):
    field = reader.get_field(key)
    part = field.parts[field.data[-1]]
    return bytes(part).decode("utf-8") if not isinstance(part, str) else part


if __name__ == "__main__":
    unittest.main()
