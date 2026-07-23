#!/usr/bin/env python3
"""Conversion-correctness tests for the ReDimNet2 -> .oasr converter.

Covers the pure remap/type logic plus a full tiny synthetic state-dict -> GGUF
round-trip (no real weights needed): reads the pack back with the ``gguf``
reader and checks tensor set, dims (ggml ne order), metadata, and f32 payload
fidelity.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import convert_redimnet2 as C


class RemapTest(unittest.TestCase):
    def test_keeps_neural_namespaces(self):
        self.assertEqual(C.remap_tensor("backbone.stem.0.weight"), "backbone.stem.0.weight")
        self.assertEqual(C.remap_tensor("pool.linear1.weight"), "pool.linear1.weight")
        self.assertEqual(C.remap_tensor("bn.weight"), "bn.weight")
        self.assertEqual(C.remap_tensor("linear.weight"), "linear.weight")
        self.assertEqual(
            C.remap_tensor("backbone.stage0.8.tcm.4.attention.q_proj.weight"),
            "backbone.stage0.8.tcm.4.attention.q_proj.weight",
        )

    def test_drops_spec_frontend(self):
        self.assertIsNone(C.remap_tensor("spec.torchfbank.2.real_kernel_pt"))
        self.assertIsNone(C.remap_tensor("spec.torchfbank.2.melbanks_pt"))
        self.assertIsNone(C.remap_tensor("spec.torchfbank.1.flipped_filter"))

    def test_drops_num_batches_tracked(self):
        self.assertIsNone(
            C.remap_tensor("backbone.stage0.3.conv_block.bn1.num_batches_tracked")
        )


class TensorTypeTest(unittest.TestCase):
    def test_norms_and_bias_force_f32(self):
        self.assertEqual(C.choose_tensor_type("bn.weight", (4032,), "f16"), "f32")
        self.assertEqual(
            C.choose_tensor_type("backbone.stem.0.bias", (64,), "f16"), "f32"
        )
        self.assertEqual(
            C.choose_tensor_type("backbone.stage0.3.conv_block.bn1.running_var", (192,), "q8_0"),
            "f32",
        )

    def test_weigth1d_and_pool_force_f32(self):
        self.assertEqual(
            C.choose_tensor_type("backbone.stage0.0.w", (1, 1, 4608, 1), "f16"), "f32"
        )
        self.assertEqual(
            C.choose_tensor_type("backbone.fin_wght1d.w", (1, 7, 4608, 1), "q8_0"), "f32"
        )
        self.assertEqual(
            C.choose_tensor_type("pool.linear1.weight", (128, 6048, 1), "f16"), "f32"
        )

    def test_projection_weight_takes_quant(self):
        # rank-2 attention projection -> honors quant choice.
        self.assertEqual(
            C.choose_tensor_type("backbone.stage0.8.tcm.4.attention.q_proj.weight", (72, 72), "f16"),
            "f16",
        )
        # q8_0 needs innermost % 32 == 0; 72 is not -> f16 fallback.
        self.assertEqual(
            C.choose_tensor_type("backbone.stage0.8.tcm.4.attention.q_proj.weight", (72, 72), "q8_0"),
            "f16",
        )
        # innermost dim % 32 == 0 -> q8_0 honored (quantization is along the
        # last torch axis = ggml ne0). linear.weight is (192, 4032).
        self.assertEqual(
            C.choose_tensor_type("backbone.stage5.head.weight", (224, 512), "q8_0"),
            "q8_0",
        )

    def test_f32_quant_overrides_everything(self):
        self.assertEqual(
            C.choose_tensor_type("backbone.stage5.head.weight", (224, 512), "f32"),
            "f32",
        )


class RoundTripTest(unittest.TestCase):
    def _synthetic_state(self):
        rng = np.random.default_rng(0)
        return {
            "backbone.stem.0.weight": rng.standard_normal((64, 1, 3, 3)).astype(np.float32),
            "backbone.stem.0.bias": rng.standard_normal((64,)).astype(np.float32),
            "backbone.stem_gnorm.weight": rng.standard_normal((4608,)).astype(np.float32),
            "backbone.stage0.0.w": rng.standard_normal((1, 1, 4608, 1)).astype(np.float32),
            "linear.weight": rng.standard_normal((192, 4032)).astype(np.float32),
            "linear.bias": rng.standard_normal((192,)).astype(np.float32),
            # dropped:
            "spec.torchfbank.2.melbanks_pt": rng.standard_normal((72, 256, 1)).astype(np.float32),
            "backbone.stage0.3.conv_block.bn1.num_batches_tracked": np.array(0, dtype=np.int64),
        }

    def test_roundtrip_f32(self):
        import gguf

        state = self._synthetic_state()
        plan = C.build_tensor_plan(state, "f32")
        names = {p[0] for p in plan}
        self.assertIn("backbone.stem.0.weight", names)
        self.assertIn("linear.weight", names)
        self.assertNotIn("spec.torchfbank.2.melbanks_pt", names)
        self.assertNotIn("backbone.stage0.3.conv_block.bn1.num_batches_tracked", names)
        self.assertEqual(len(plan), 6)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "redimnet2-test.oasr"
            C.write_pack(out, plan)
            self.assertTrue(out.exists())

            reader = gguf.GGUFReader(str(out))
            # metadata present.
            kv = {f.name: f for f in reader.fields.values()}
            self.assertIn("redimnet2.embed_dim", kv)
            self.assertIn("redimnet2.model_config_json", kv)

            rt = {t.name: t for t in reader.tensors}
            self.assertEqual(set(rt.keys()), names)

            # gguf stores dims in ggml ne order (torch shape reversed).
            stem_w = rt["backbone.stem.0.weight"]
            self.assertEqual(list(stem_w.shape), [3, 3, 1, 64])
            lin_w = rt["linear.weight"]
            self.assertEqual(list(lin_w.shape), [4032, 192])

            # f32 payload fidelity: reader returns data in ne (reversed) order;
            # reversing our source array's axes matches ggml memory order.
            got = np.array(stem_w.data, dtype=np.float32)
            want = state["backbone.stem.0.weight"].astype(np.float32).reshape(-1)
            np.testing.assert_allclose(got.reshape(-1), want, rtol=0, atol=0)

    def test_roundtrip_f16_types(self):
        import gguf

        state = self._synthetic_state()
        plan = C.build_tensor_plan(state, "f16")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "redimnet2-test-f16.oasr"
            C.write_pack(out, plan)
            reader = gguf.GGUFReader(str(out))
            rt = {t.name: t for t in reader.tensors}
            # rank-2 projection -> f16; norms/bias/weigth1d -> f32.
            self.assertEqual(rt["linear.weight"].tensor_type, gguf.GGMLQuantizationType.F16)
            self.assertEqual(rt["linear.bias"].tensor_type, gguf.GGMLQuantizationType.F32)
            self.assertEqual(
                rt["backbone.stem_gnorm.weight"].tensor_type, gguf.GGMLQuantizationType.F32
            )
            self.assertEqual(rt["backbone.stage0.0.w"].tensor_type, gguf.GGMLQuantizationType.F32)


if __name__ == "__main__":
    unittest.main()
