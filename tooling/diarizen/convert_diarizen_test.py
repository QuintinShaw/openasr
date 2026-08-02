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


if __name__ == "__main__":
    unittest.main()
