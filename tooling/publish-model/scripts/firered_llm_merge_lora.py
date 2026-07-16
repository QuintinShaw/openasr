#!/usr/bin/env python3
"""Merge the FireRedASR2-LLM PEFT LoRA adapter (28 layers x 7 target modules,
r=64, alpha=16) into the official `Qwen/Qwen2-7B-Instruct` base weights,
producing a single flat F32 `.safetensors` with standard (un-prefixed)
`Qwen2ForCausalLM.state_dict()` tensor names, ready for the Rust
`firered_llm::package_import` GGUF converter.

Background (stage-1 reconnaissance, `scratchpad/fr2/T1-findings.md`, "重大修正"):
`model.pth.tar`'s `llm.*` tensors are NOT a standalone finetuned Qwen2 -- they
are 392 PEFT LoRA low-rank tensors (`lora_A`/`lora_B`, 28 layers x 7 target
modules x 2), all under
`llm.base_model.model.model.layers.{i}.{module}.lora_{A,B}.default.weight`.
The actual base weights (`embed_tokens`, `lm_head`, `norm`, every
`{q,k,v,o}_proj`/`{gate,up,down}_proj` weight, every `q/k/v_proj` bias, every
layernorm) come entirely from the official `Qwen2-7B-Instruct/*.safetensors`
checkpoint and are untouched by this checkpoint. Standard PEFT merge formula
(`fireredasr_llm.py`'s `LoraConfig(r=64, lora_alpha=16, ...)`, default scaling,
`use_rslora` not set)::

    scaling = lora_alpha / r = 16 / 64 = 0.25
    W_merged[out, in] = W_base[out, in] + scaling * (lora_B[out, r] @ lora_A[r, in])

Only the 28 * 7 = 196 target-module weight matrices
(`self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`) are
merged; every other tensor (biases, layernorms, embeddings, `lm_head`, `norm`)
passes through byte-identical (upcast to f32 like everything else this
importer touches -- LoRA's `target_modules` never included any of them, so
there is nothing to add).

This script is a standalone reader/writer for both input formats (no
`safetensors`/`peft` package dependency -- only `struct`+`json`+`mmap`+
`numpy`, mirroring `pt_to_safetensors.py`'s "dumb weight normalizer"
philosophy and the wire-format `SafetensorsFile` parser in
`crates/openasr-core/src/models/local_source_import.rs`) so it can stream
tensor-by-tensor without ever holding the full ~7.6B-parameter model in RAM.

Example::

    python3 tooling/publish-model/scripts/firered_llm_merge_lora.py \\
        --lora-safetensors tmp-weights/fr2/derived/model.safetensors \\
        --qwen2-dir tmp-weights/fr2/fr2_qwen2_download/Qwen2-7B-Instruct \\
        --out tmp-weights/fr2/derived/qwen2-merged.safetensors
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_RANK = 64
DEFAULT_SCALING = DEFAULT_LORA_ALPHA / DEFAULT_LORA_RANK  # 0.25

# The 7 PEFT target modules (fireredasr_llm.py's LoraConfig.target_modules),
# each producing one `model.layers.{i}.{TARGET}.weight` base tensor name.
LORA_TARGET_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

LORA_NAME_PREFIX = "llm.base_model.model.model.layers."


# --- minimal safetensors reader (no `safetensors` package dependency) ------


class SafetensorsReader:
    """mmap-backed, lazy, single-file safetensors reader. Mirrors the Rust
    `SafetensorsFile` parser's wire-format understanding (8-byte LE header
    length, JSON header, raw tensor bytes) closely enough to decode the same
    files, without requiring the `safetensors` python package (which errors
    on BF16 under the numpy framework)."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header_bytes = handle.read(header_len)
        self.header: dict = json.loads(header_bytes)
        self.header.pop("__metadata__", None)
        self.data_offset = 8 + header_len
        self._file = path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def names(self) -> list[str]:
        return list(self.header.keys())

    def shape(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.header[name]["dtype"]

    def raw_bytes(self, name: str) -> bytes:
        start, end = self.header[name]["data_offsets"]
        return self._mmap[self.data_offset + start : self.data_offset + end]

    def get_f32(self, name: str) -> np.ndarray:
        return decode_to_f32(self.raw_bytes(name), self.dtype(name), self.shape(name))

    def close(self) -> None:
        self._mmap.close()
        self._file.close()


def decode_to_f32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if dtype == "BF16":
        as_u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
        as_u32 = as_u16.astype(np.uint32) << 16
        return as_u32.view(np.float32)
    raise ValueError(f"unsupported safetensors dtype {dtype!r}")


class SafetensorsShardSet:
    """Opens every `*.safetensors` shard in a directory (using its
    `model.safetensors.index.json` `weight_map`, or -- if there is no index --
    every `.safetensors` file directly) and resolves any tensor name to its
    shard on demand."""

    def __init__(self, directory: Path):
        self.directory = directory
        index_path = directory / "model.safetensors.index.json"
        self._shards: dict[str, SafetensorsReader] = {}
        self._weight_map: dict[str, str] = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            self._weight_map = dict(index["weight_map"])
            for filename in sorted(set(self._weight_map.values())):
                self._shards[filename] = SafetensorsReader(directory / filename)
        else:
            for path in sorted(directory.glob("*.safetensors")):
                reader = SafetensorsReader(path)
                self._shards[path.name] = reader
                for name in reader.names():
                    self._weight_map[name] = path.name

    def ordered_names(self) -> list[str]:
        # Deterministic emission order: shard file, then in-shard header order.
        names: list[str] = []
        for filename in sorted(self._shards):
            names.extend(self._shards[filename].names())
        return names

    def shape(self, name: str) -> list[int]:
        return self._shards[self._weight_map[name]].shape(name)

    def get_f32(self, name: str) -> np.ndarray:
        return self._shards[self._weight_map[name]].get_f32(name)

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()


# --- minimal safetensors writer (streaming, F32 output) --------------------


def build_f32_header(names_and_shapes: list[tuple[str, list[int]]]) -> tuple[dict, int]:
    """Compute the safetensors JSON header (and total data-section byte
    length) for an F32 output from `(name, shape)` pairs alone -- shapes are
    already known from the source shard headers, so this never touches
    tensor data and can run entirely up front."""
    header: dict[str, object] = {}
    offset = 0
    for name, shape in names_and_shapes:
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    return header, offset


def write_f32_safetensors_streaming(
    out_path: Path,
    names_and_shapes: list[tuple[str, list[int]]],
    tensor_source,
) -> None:
    """Write an F32 `.safetensors` file at `out_path` by writing the header
    (computed purely from `names_and_shapes`, see `build_f32_header`) and then
    calling `tensor_source(name)` once per tensor, in order, writing its bytes
    immediately and discarding the array before requesting the next one. This
    is the memory-bounded path: at most one merged/passthrough tensor (at most
    the largest MLP matrix, ~271 MB in f32) is ever resident at a time, never
    the full ~7.6B-parameter model."""
    header, _total_bytes = build_f32_header(names_and_shapes)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for name, _shape in names_and_shapes:
            array = tensor_source(name)
            handle.write(np.ascontiguousarray(array, dtype=np.float32).tobytes(order="C"))


# --- LoRA merge core ---------------------------------------------------


def merge_lora_delta(
    base: np.ndarray, lora_a: np.ndarray, lora_b: np.ndarray, scaling: float
) -> np.ndarray:
    """`W_merged = W_base + scaling * (lora_B @ lora_A)`, matching PEFT's
    default (non-rslora) merge formula. `base`/`W_merged` are `[out, in]`,
    `lora_a` is `[r, in]`, `lora_b` is `[out, r]`. All arithmetic happens in
    float32 regardless of the input dtypes (the LoRA tensors and the
    upcast-from-bf16 base are both already f32 by the time this is called)."""
    if base.dtype != np.float32 or lora_a.dtype != np.float32 or lora_b.dtype != np.float32:
        raise ValueError("merge_lora_delta requires float32 inputs")
    if base.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("merge_lora_delta requires rank-2 (matrix) inputs")
    out_dim, in_dim = base.shape
    rank_b, in_dim_a = lora_a.shape[0], lora_a.shape[1]
    out_dim_b, rank_a = lora_b.shape
    if in_dim_a != in_dim:
        raise ValueError(f"lora_A in-dim {in_dim_a} != base in-dim {in_dim}")
    if out_dim_b != out_dim:
        raise ValueError(f"lora_B out-dim {out_dim_b} != base out-dim {out_dim}")
    if rank_a != rank_b:
        raise ValueError(f"lora_A rank {rank_b} != lora_B rank {rank_a}")
    delta = scaling * (lora_b @ lora_a)
    return (base + delta).astype(np.float32)


def lora_tensor_names_for_layer(layer_idx: int, module: str) -> tuple[str, str]:
    prefix = f"{LORA_NAME_PREFIX}{layer_idx}.{module}"
    return f"{prefix}.lora_A.default.weight", f"{prefix}.lora_B.default.weight"


def discover_lora_layer_count(lora_names: set[str]) -> int:
    layer_idx = 0
    while True:
        name_a, _ = lora_tensor_names_for_layer(layer_idx, LORA_TARGET_MODULES[0])
        if name_a not in lora_names:
            return layer_idx
        layer_idx += 1


def plan_merge_targets(
    lora_names: set[str],
) -> tuple[dict[str, tuple[str, str]], int]:
    """Returns `(merge_targets, n_layers)`, where `merge_targets` maps each
    base tensor name (`model.layers.{i}.{module}.weight`) that must be merged
    to its `(lora_A_name, lora_B_name)` pair. Pure name-table computation --
    no tensor data touched."""
    n_layers = discover_lora_layer_count(lora_names)
    if n_layers == 0:
        raise SystemExit("found no llm.*.lora_A.default.weight tensors in the LoRA source")

    merge_targets: dict[str, tuple[str, str]] = {}
    for layer_idx in range(n_layers):
        for module in LORA_TARGET_MODULES:
            name_a, name_b = lora_tensor_names_for_layer(layer_idx, module)
            if name_a not in lora_names or name_b not in lora_names:
                raise SystemExit(
                    f"layer {layer_idx} module {module!r} is missing a lora_A/lora_B tensor"
                )
            base_name = f"model.layers.{layer_idx}.{module}.weight"
            merge_targets[base_name] = (name_a, name_b)
    return merge_targets, n_layers


class MergeCounter:
    """Tracks merged-vs-passthrough tensor counts across a streaming write,
    and resolves each tensor's f32 array on demand for
    `write_f32_safetensors_streaming`'s `tensor_source` callback."""

    def __init__(
        self,
        lora_reader: SafetensorsReader,
        qwen2_shards: SafetensorsShardSet,
        merge_targets: dict[str, tuple[str, str]],
        scaling: float,
    ):
        self.lora_reader = lora_reader
        self.qwen2_shards = qwen2_shards
        self.merge_targets = merge_targets
        self.scaling = scaling
        self.merged_count = 0
        self.passthrough_count = 0

    def __call__(self, name: str) -> np.ndarray:
        if name in self.merge_targets:
            name_a, name_b = self.merge_targets[name]
            base = self.qwen2_shards.get_f32(name)
            lora_a = self.lora_reader.get_f32(name_a)
            lora_b = self.lora_reader.get_f32(name_b)
            self.merged_count += 1
            return merge_lora_delta(base, lora_a, lora_b, self.scaling)
        self.passthrough_count += 1
        return self.qwen2_shards.get_f32(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge the FireRedASR2-LLM LoRA adapter into Qwen2-7B-Instruct base weights."
    )
    parser.add_argument(
        "--lora-safetensors",
        required=True,
        type=Path,
        help="pt_to_safetensors.py output over model.pth.tar (contains the llm.*.lora_{A,B} tensors)",
    )
    parser.add_argument(
        "--qwen2-dir",
        required=True,
        type=Path,
        help="directory with the official Qwen2-7B-Instruct model-*.safetensors + index.json",
    )
    parser.add_argument("--out", required=True, type=Path, help="output merged .safetensors path")
    parser.add_argument(
        "--scaling",
        type=float,
        default=DEFAULT_SCALING,
        help=f"LoRA merge scaling (lora_alpha / r); default {DEFAULT_SCALING} (16/64)",
    )
    args = parser.parse_args(argv)

    if not args.lora_safetensors.is_file():
        raise SystemExit(f"--lora-safetensors not found: {args.lora_safetensors}")
    if not args.qwen2_dir.is_dir():
        raise SystemExit(f"--qwen2-dir not found: {args.qwen2_dir}")

    print(f"opening LoRA source {args.lora_safetensors} ...", file=sys.stderr)
    lora_reader = SafetensorsReader(args.lora_safetensors)
    print(f"opening Qwen2 base shards under {args.qwen2_dir} ...", file=sys.stderr)
    qwen2_shards = SafetensorsShardSet(args.qwen2_dir)

    try:
        merge_targets, n_layers = plan_merge_targets(set(lora_reader.names()))
        ordered_names = qwen2_shards.ordered_names()
        names_and_shapes = [(name, qwen2_shards.shape(name)) for name in ordered_names]
        total_params = sum(int(np.prod(shape)) for _name, shape in names_and_shapes)

        counter = MergeCounter(lora_reader, qwen2_shards, merge_targets, args.scaling)
        print(
            f"streaming {len(ordered_names)} tensors ({n_layers} layers, "
            f"{total_params / 1e9:.2f}B params, scaling={args.scaling}) to {args.out} ...",
            file=sys.stderr,
        )
        write_f32_safetensors_streaming(args.out, names_and_shapes, counter)

        expected_merged = n_layers * len(LORA_TARGET_MODULES)
        if counter.merged_count != expected_merged:
            args.out.unlink(missing_ok=True)
            raise SystemExit(
                f"merged {counter.merged_count} tensors but expected {expected_merged} "
                f"({n_layers} layers x {len(LORA_TARGET_MODULES)} target modules) "
                "-- deleted the incomplete output rather than leaving a partially-merged pack"
            )
    finally:
        lora_reader.close()
        qwen2_shards.close()

    print(
        f"wrote {args.out} ({len(ordered_names)} tensors, {total_params / 1e9:.2f}B params, F32; "
        f"{counter.merged_count} merged, {counter.passthrough_count} passthrough)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
