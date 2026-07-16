#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

import numpy as np
import tempfile

from firered_llm_merge_lora import (
    DEFAULT_SCALING,
    LORA_TARGET_MODULES,
    MergeCounter,
    SafetensorsReader,
    SafetensorsShardSet,
    build_f32_header,
    discover_lora_layer_count,
    lora_tensor_names_for_layer,
    merge_lora_delta,
    plan_merge_targets,
    write_f32_safetensors_streaming,
)


def write_raw_f32_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    blob = bytearray()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        start = len(blob)
        blob += arr.tobytes(order="C")
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [start, len(blob)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(bytes(blob))


class MergeLoraDeltaHandComputedTests(unittest.TestCase):
    """Three modules (mimicking a q_proj-shaped, an o_proj-shaped, and a
    gate_proj-shaped matrix at toy dimensions) with fully hand-computed
    expected merge results -- the "3 modules, hand calc vs script" assertion
    the task requires."""

    def test_module_1_square_2x2_rank_1(self) -> None:
        # base = [[1, 2], [3, 4]], lora_A = [[1, 0]] (r=1, in=2),
        # lora_B = [[2], [1]] (out=2, r=1), scaling = 0.5.
        # delta = 0.5 * (lora_B @ lora_A) = 0.5 * [[2, 0], [1, 0]] = [[1, 0], [0.5, 0]]
        # expected = [[2, 2], [3.5, 4]]
        base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        lora_a = np.array([[1.0, 0.0]], dtype=np.float32)
        lora_b = np.array([[2.0], [1.0]], dtype=np.float32)
        result = merge_lora_delta(base, lora_a, lora_b, scaling=0.5)
        expected = np.array([[2.0, 2.0], [3.5, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)

    def test_module_2_rectangular_out3_in4_rank2(self) -> None:
        # base is [3, 4] (out=3, in=4, like a small o_proj), lora_A [2, 4] (r=2),
        # lora_B [3, 2] (out=3, r=2), scaling = 16/64 = 0.25 (the real
        # fireredasr_llm.py LoraConfig scaling).
        base = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            dtype=np.float32,
        )
        lora_a = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        lora_b = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
        # lora_B @ lora_A:
        #   row0 = 1*[1,0,1,0] + 2*[0,1,0,1] = [1, 2, 1, 2]
        #   row1 = 0*[1,0,1,0] + 1*[0,1,0,1] = [0, 1, 0, 1]
        #   row2 = 3*[1,0,1,0] + 0*[0,1,0,1] = [3, 0, 3, 0]
        delta_unscaled = np.array([[1.0, 2.0, 1.0, 2.0], [0.0, 1.0, 0.0, 1.0], [3.0, 0.0, 3.0, 0.0]])
        expected = base + DEFAULT_SCALING * delta_unscaled
        result = merge_lora_delta(base, lora_a, lora_b, scaling=DEFAULT_SCALING)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)

    def test_module_3_zero_lora_is_a_pure_passthrough(self) -> None:
        # A gate_proj-shaped [4, 3] base with an all-zero lora_B: the merge
        # must be an exact no-op regardless of lora_A's contents.
        base = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
        lora_a = np.array([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], dtype=np.float32)
        lora_b = np.zeros((4, 2), dtype=np.float32)
        result = merge_lora_delta(base, lora_a, lora_b, scaling=0.25)
        np.testing.assert_array_equal(result, base)

    def test_rejects_mismatched_shapes(self) -> None:
        base = np.zeros((3, 4), dtype=np.float32)
        lora_a = np.zeros((2, 5), dtype=np.float32)  # in-dim mismatch (5 != 4)
        lora_b = np.zeros((3, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            merge_lora_delta(base, lora_a, lora_b, scaling=0.25)


class DiscoverLoraLayerCountTests(unittest.TestCase):
    def test_counts_contiguous_layers(self) -> None:
        names = set()
        for layer in range(5):
            name_a, name_b = lora_tensor_names_for_layer(layer, LORA_TARGET_MODULES[0])
            names.add(name_a)
            names.add(name_b)
        self.assertEqual(discover_lora_layer_count(names), 5)

    def test_stops_at_first_gap(self) -> None:
        names = set()
        for layer in [0, 1, 2]:
            name_a, _ = lora_tensor_names_for_layer(layer, LORA_TARGET_MODULES[0])
            names.add(name_a)
        # skip layer 3, add layer 4 -- must still report 3.
        name_a, _ = lora_tensor_names_for_layer(4, LORA_TARGET_MODULES[0])
        names.add(name_a)
        self.assertEqual(discover_lora_layer_count(names), 3)

    def test_empty_set_is_zero_layers(self) -> None:
        self.assertEqual(discover_lora_layer_count(set()), 0)


class BuildF32HeaderTests(unittest.TestCase):
    def test_computes_sequential_non_overlapping_offsets(self) -> None:
        header, total = build_f32_header([("a", [2, 3]), ("b", [4])])
        self.assertEqual(header["a"], {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]})
        self.assertEqual(header["b"], {"dtype": "F32", "shape": [4], "data_offsets": [24, 40]})
        self.assertEqual(total, 40)


class EndToEndSyntheticMergeTests(unittest.TestCase):
    """A tiny 2-layer synthetic checkpoint (not the real 28-layer/7.6B model)
    exercising the full plan -> stream -> write path, including the
    passthrough of non-target tensors (bias/norm/embedding)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.n_layers = 2
        self.hidden = 4
        self.rank = 2
        rng = np.random.default_rng(1234)

        # --- synthetic Qwen2-like base shards (2 shards, standard naming) ---
        base_tensors: dict[str, np.ndarray] = {
            "model.embed_tokens.weight": rng.normal(size=(6, self.hidden)).astype(np.float32),
            "model.norm.weight": np.ones(self.hidden, dtype=np.float32),
            "lm_head.weight": rng.normal(size=(6, self.hidden)).astype(np.float32),
        }
        for layer in range(self.n_layers):
            for module in LORA_TARGET_MODULES:
                base_tensors[f"model.layers.{layer}.{module}.weight"] = rng.normal(
                    size=(self.hidden, self.hidden)
                ).astype(np.float32)
            base_tensors[f"model.layers.{layer}.self_attn.q_proj.bias"] = rng.normal(
                size=(self.hidden,)
            ).astype(np.float32)
            base_tensors[f"model.layers.{layer}.input_layernorm.weight"] = np.ones(
                self.hidden, dtype=np.float32
            )

        shard_a = {k: v for k, v in base_tensors.items() if "layers.0" in k or "layers" not in k}
        shard_b = {k: v for k, v in base_tensors.items() if k not in shard_a}
        write_raw_f32_safetensors(self.root / "model-00001-of-00002.safetensors", shard_a)
        write_raw_f32_safetensors(self.root / "model-00002-of-00002.safetensors", shard_b)
        weight_map = {name: "model-00001-of-00002.safetensors" for name in shard_a}
        weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard_b})
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
        self.base_tensors = base_tensors

        # --- synthetic LoRA source (pt_to_safetensors.py-shaped) ---
        lora_tensors: dict[str, np.ndarray] = {}
        self.lora_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for layer in range(self.n_layers):
            for module in LORA_TARGET_MODULES:
                name_a, name_b = lora_tensor_names_for_layer(layer, module)
                lora_a = rng.normal(size=(self.rank, self.hidden)).astype(np.float32)
                lora_b = rng.normal(size=(self.hidden, self.rank)).astype(np.float32)
                lora_tensors[name_a] = lora_a
                lora_tensors[name_b] = lora_b
                self.lora_pairs[f"model.layers.{layer}.{module}.weight"] = (lora_a, lora_b)
        # A handful of encoder/adapter tensors that must be ignored by the
        # LoRA-name scan (the real pt_to_safetensors.py output contains them
        # alongside llm.* in the same file).
        lora_tensors["encoder.layer_stack.0.mhsa.w_qs.weight"] = rng.normal(
            size=(self.hidden, self.hidden)
        ).astype(np.float32)
        write_raw_f32_safetensors(self.root / "lora_source.safetensors", lora_tensors)

    def test_plan_merge_targets_discovers_expected_matrix_count(self) -> None:
        lora_reader = SafetensorsReader(self.root / "lora_source.safetensors")
        try:
            merge_targets, n_layers = plan_merge_targets(set(lora_reader.names()))
        finally:
            lora_reader.close()
        self.assertEqual(n_layers, self.n_layers)
        self.assertEqual(len(merge_targets), self.n_layers * len(LORA_TARGET_MODULES))

    def test_full_streaming_merge_matches_hand_computed_values_and_passthrough(self) -> None:
        lora_reader = SafetensorsReader(self.root / "lora_source.safetensors")
        qwen2_shards = SafetensorsShardSet(self.root)
        out_path = self.root / "merged.safetensors"
        try:
            merge_targets, n_layers = plan_merge_targets(set(lora_reader.names()))
            ordered_names = qwen2_shards.ordered_names()
            names_and_shapes = [(name, qwen2_shards.shape(name)) for name in ordered_names]
            counter = MergeCounter(lora_reader, qwen2_shards, merge_targets, scaling=DEFAULT_SCALING)
            write_f32_safetensors_streaming(out_path, names_and_shapes, counter)
        finally:
            lora_reader.close()
            qwen2_shards.close()

        self.assertEqual(counter.merged_count, self.n_layers * len(LORA_TARGET_MODULES))
        # embed_tokens, norm, lm_head, + per-layer bias, + per-layer input_layernorm
        self.assertEqual(counter.passthrough_count, 3 + self.n_layers * 2)

        result_reader = SafetensorsReader(out_path)
        try:
            self.assertEqual(set(result_reader.names()), set(self.base_tensors.keys()))
            for name, (lora_a, lora_b) in self.lora_pairs.items():
                expected = self.base_tensors[name] + DEFAULT_SCALING * (lora_b @ lora_a)
                np.testing.assert_allclose(
                    result_reader.get_f32(name), expected, rtol=1e-5, atol=1e-6
                )
            # passthrough tensors must be byte-identical to the base.
            np.testing.assert_array_equal(
                result_reader.get_f32("model.norm.weight"), self.base_tensors["model.norm.weight"]
            )
            np.testing.assert_array_equal(
                result_reader.get_f32("model.embed_tokens.weight"),
                self.base_tensors["model.embed_tokens.weight"],
            )
            np.testing.assert_array_equal(
                result_reader.get_f32("model.layers.0.self_attn.q_proj.bias"),
                self.base_tensors["model.layers.0.self_attn.q_proj.bias"],
            )
        finally:
            result_reader.close()

    def test_incomplete_lora_source_raises_before_touching_base_weights(self) -> None:
        # Drop one lora_B tensor for layer 0 -> plan_merge_targets must fail
        # closed rather than silently merging a partial adapter.
        partial_root = self.root / "partial"
        partial_root.mkdir()
        lora_reader = SafetensorsReader(self.root / "lora_source.safetensors")
        try:
            names = set(lora_reader.names())
        finally:
            lora_reader.close()
        _name_a, name_b_to_drop = lora_tensor_names_for_layer(0, LORA_TARGET_MODULES[0])
        names.discard(name_b_to_drop)
        with self.assertRaises(SystemExit):
            plan_merge_targets(names)


if __name__ == "__main__":
    unittest.main()
