#!/usr/bin/env python3
"""Normalize ONNX initializers into an F32 safetensors weight pack.

This is a companion to `pt_to_safetensors.py` for upstreams whose ONNX
deployment artifacts are the numerically authoritative source. The motivating
case is `GilgameshWind/X-ASR-zh-en`: the public sherpa ONNX weights do not match
`streaming_exp/pretrained.pt`, so the `.pt` can guide architecture mapping but
must not be used as the parity weight source.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def write_safetensors(out_path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    blob = bytearray()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        data = arr.tobytes(order="C")
        start = len(blob)
        blob.extend(data)
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [start, len(blob)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)


def load_initializers(path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(path)
    return {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        for initializer in model.graph.initializer
    }


def onnx_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(path)
    return {prop.key: prop.value for prop in model.metadata_props}


def parse_int_list(metadata: dict[str, str], key: str) -> list[int]:
    value = metadata.get(key)
    if value is None:
        raise SystemExit(f"ONNX metadata key {key!r} is missing")
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(
            f"ONNX metadata key {key!r} is not a comma-separated int list: {value!r}"
        ) from error


def parse_int(metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        raise SystemExit(f"ONNX metadata key {key!r} is missing")
    try:
        return int(value)
    except ValueError as error:
        raise SystemExit(f"ONNX metadata key {key!r} is not an int: {value!r}") from error


def encoder_feature_dim(path: Path) -> int:
    model = onnx.load(path)
    for graph_input in model.graph.input:
        if graph_input.name != "x":
            continue
        dims = graph_input.type.tensor_type.shape.dim
        if len(dims) != 3 or not dims[2].dim_value:
            raise SystemExit("encoder input 'x' does not expose static feature dim")
        return int(dims[2].dim_value)
    raise SystemExit("encoder ONNX input 'x' not found")


def xasr_config_from_onnx_metadata(
    *,
    encoder: Path,
    decoder: Path,
    joiner: Path,
) -> dict[str, object]:
    encoder_meta = onnx_metadata(encoder)
    decoder_meta = onnx_metadata(decoder)
    joiner_meta = onnx_metadata(joiner)
    return {
        "num_encoder_layers": parse_int_list(encoder_meta, "num_encoder_layers"),
        "encoder_dims": parse_int_list(encoder_meta, "encoder_dims"),
        "query_head_dims": parse_int_list(encoder_meta, "query_head_dims"),
        "value_head_dims": parse_int_list(encoder_meta, "value_head_dims"),
        "num_heads": parse_int_list(encoder_meta, "num_heads"),
        "cnn_module_kernels": parse_int_list(encoder_meta, "cnn_module_kernels"),
        "left_context_len": parse_int_list(encoder_meta, "left_context_len"),
        "downsampling_factors": [1, 2, 4, 8, 4, 2],
        "feature_dim": encoder_feature_dim(encoder),
        "decode_chunk_len": parse_int(encoder_meta, "decode_chunk_len"),
        "joiner_dim": parse_int(joiner_meta, "joiner_dim"),
        "decoder_context_size": parse_int(decoder_meta, "context_size"),
        "vocab_size": parse_int(decoder_meta, "vocab_size"),
        "blank_id": 0,
    }


def write_json_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def xasr_layer_prefixes() -> list[str]:
    return [prefix for prefix, _dim, _kernel in xasr_layer_specs()]


def xasr_layer_specs() -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    num_layers = [2, 2, 4, 5, 4, 2]
    encoder_dims = [192, 256, 512, 768, 512, 256]
    cnn_module_kernels = [31, 31, 15, 15, 15, 31]
    for stack, count in enumerate(num_layers):
        for layer in range(count):
            if stack == 0:
                prefix = f"encoder.encoders.{stack}.layers.{layer}"
            else:
                prefix = f"encoder.encoders.{stack}.encoder.layers.{layer}"
            specs.append((prefix, encoder_dims[stack], cnn_module_kernels[stack]))
    return specs


def xasr_encoder_matmul_remaps(encoder: Path) -> dict[str, str]:
    model = onnx.load(encoder)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for node_input in node.input:
            consumers.setdefault(node_input, []).append(node)

    remaps: dict[str, str] = {}
    linear_pos_nodes: list[tuple[str, str]] = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2 or node.input[1] not in initializer_names:
            continue
        weight_name = node.input[1]
        mapped_name = None
        for consumer in consumers.get(node.output[0], []):
            if consumer.op_type != "Add":
                continue
            bias_inputs = [
                item
                for item in consumer.input
                if item in initializer_names and item.endswith(".bias")
            ]
            if bias_inputs:
                mapped_name = bias_inputs[0].removesuffix(".bias") + ".weight"
                break
        if mapped_name is None and node.name.startswith("/linear_pos"):
            linear_pos_nodes.append((weight_name, node.name))
            continue
        if mapped_name is None:
            raise SystemExit(
                f"could not map encoder MatMul initializer {weight_name!r} from node {node.name!r}"
            )
        if mapped_name == "encoder_proj.weight":
            mapped_name = "joiner.encoder_proj.weight"
        remaps[weight_name] = mapped_name

    prefixes = xasr_layer_prefixes()
    if len(linear_pos_nodes) != len(prefixes):
        raise SystemExit(
            f"expected {len(prefixes)} linear_pos MatMuls, got {len(linear_pos_nodes)}"
        )
    for (weight_name, _node_name), prefix in zip(linear_pos_nodes, prefixes):
        remaps[weight_name] = f"{prefix}.self_attn_weights.linear_pos.weight"

    matmul_count = sum(
        1
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) >= 2 and node.input[1] in initializer_names
    )
    if len(remaps) != matmul_count:
        raise SystemExit(f"mapped {len(remaps)} MatMuls, expected {matmul_count}")
    return remaps


def xasr_encoder_chunkwise_scale_tensors(
    encoder: Path,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Remap ONNX-exported ChunkCausalDepthwiseConv1d edge scales.

    icefall stores one semantic tensor per convolution module:
    `[left_edge, right_edge]` with shape `[2, channels, kernel]`. The X-ASR
    ONNX export bakes these as anonymous 2-D initializers consumed by Slice
    nodes: the left edge is sliced as `[:chunk_size]`, and the right edge as
    `[-chunk_size:]`. Keep the safetensors/GGUF contract semantic and skip the
    anonymous source initializers.
    """

    model = onnx.load(encoder)
    initializers = {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        for initializer in model.graph.initializer
    }
    slice_initializers: list[tuple[str, np.ndarray, str]] = []
    for node in model.graph.node:
        if node.op_type != "Slice" or not node.input:
            continue
        name = node.input[0]
        if not name.startswith("onnx::Slice_") or name not in initializers:
            continue
        value = initializers[name]
        if value.ndim != 2:
            continue
        # The positional embedding table is also a Slice initializer in some
        # exports; the chunkwise conv scales are exactly the per-stack
        # channel/kernel matrices below.
        if (value.shape[0], value.shape[1]) not in {
            (192, 31),
            (256, 31),
            (512, 15),
            (768, 15),
        }:
            continue
        slice_initializers.append((name, value, node.name))

    layer_specs = xasr_layer_specs()
    expected_count = len(layer_specs) * 2 * 2
    if len(slice_initializers) != expected_count:
        raise SystemExit(
            f"expected {expected_count} X-ASR chunkwise scale Slice initializers, "
            f"got {len(slice_initializers)}"
        )

    tensors: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    cursor = 0
    for prefix, channels, kernel in layer_specs:
        expected_shape = (channels, kernel)
        for module in ["conv_module1", "conv_module2"]:
            left_name, left, left_node = slice_initializers[cursor]
            right_name, right, right_node = slice_initializers[cursor + 1]
            cursor += 2
            if left.shape != expected_shape or right.shape != expected_shape:
                raise SystemExit(
                    f"chunkwise scale shape mismatch for {prefix}.{module}: "
                    f"{left_name} from {left_node} has {list(left.shape)}, "
                    f"{right_name} from {right_node} has {list(right.shape)}, "
                    f"expected {list(expected_shape)}"
                )
            target = f"{prefix}.{module}.depthwise_conv.chunkwise_conv_scale"
            tensors[target] = np.stack([left, right], axis=0)
            consumed.add(left_name)
            consumed.add(right_name)
    return tensors, consumed


def xasr_encoder_downsample_bias_tensors(
    encoder: Path,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Remap ONNX-exported downsample softmax weights back to semantic logits."""

    model = onnx.load(encoder)
    initializers = {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        for initializer in model.graph.initializer
    }
    node_to_target = {
        "/downsample/Mul": "encoder.encoders.1.downsample.bias",
        "/downsample_1/Mul": "encoder.encoders.2.downsample.bias",
        "/downsample_2/Mul": "encoder.encoders.3.downsample.bias",
        "/downsample_3/Mul": "encoder.encoders.4.downsample.bias",
        "/downsample_4/Mul": "encoder.encoders.5.downsample.bias",
        "/downsample_output/Mul": "encoder.downsample_output.bias",
    }

    tensors: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    for node in model.graph.node:
        target = node_to_target.get(node.name)
        if target is None:
            continue
        source_names = [node_input for node_input in node.input if node_input in initializers]
        if len(source_names) != 1:
            raise SystemExit(
                f"expected one initializer input for {node.name!r}, got {source_names!r}"
            )
        source_name = source_names[0]
        weights = np.asarray(initializers[source_name], dtype=np.float32).reshape(-1)
        if np.any(weights <= 0.0):
            raise SystemExit(f"downsample weights for {target} contain non-positive values")
        if not np.isclose(float(weights.sum()), 1.0, rtol=1.0e-4, atol=1.0e-5):
            raise SystemExit(
                f"downsample weights for {target} sum to {float(weights.sum())}, expected 1.0"
            )
        tensors[target] = np.log(weights).astype(np.float32)
        consumed.add(source_name)

    missing = sorted(set(node_to_target.values()) - set(tensors))
    if missing:
        raise SystemExit(f"missing X-ASR downsample remaps for {missing}")
    return tensors, consumed


def add_tensor(
    tensors: dict[str, np.ndarray],
    name: str,
    value: np.ndarray,
    *,
    overwrite: bool = False,
) -> None:
    if name in tensors and not overwrite:
        raise SystemExit(f"duplicate tensor name after ONNX remap: {name}")
    tensors[name] = np.ascontiguousarray(value, dtype=np.float32)


def add_xasr_onnx_initializers(
    tensors: dict[str, np.ndarray],
    *,
    encoder: Path,
    decoder: Path,
    joiner: Path,
) -> None:
    encoder_tensors = load_initializers(encoder)
    decoder_tensors = load_initializers(decoder)
    joiner_tensors = load_initializers(joiner)
    encoder_matmul_remaps = xasr_encoder_matmul_remaps(encoder)
    encoder_chunkwise_scales, consumed_encoder_initializers = (
        xasr_encoder_chunkwise_scale_tensors(encoder)
    )
    encoder_downsample_biases, consumed_downsample_initializers = (
        xasr_encoder_downsample_bias_tensors(encoder)
    )
    consumed_encoder_initializers.update(consumed_downsample_initializers)

    for name, value in encoder_tensors.items():
        if name in consumed_encoder_initializers:
            continue
        if name in encoder_matmul_remaps:
            # ONNX MatMul stores `[in, out]`; the Rust package importer expects
            # safetensors `.weight` values in PyTorch `[out, in]`, then reverses
            # dims for GGML at pack time.
            add_tensor(tensors, encoder_matmul_remaps[name], value.T)
        elif name == "encoder_proj.bias":
            add_tensor(tensors, "joiner.encoder_proj.bias", value)
        else:
            add_tensor(tensors, name, value)
    for name, value in encoder_chunkwise_scales.items():
        add_tensor(tensors, name, value)
    for name, value in encoder_downsample_biases.items():
        add_tensor(tensors, name, value)

    for name, value in decoder_tensors.items():
        if name == "decoder_proj.weight":
            add_tensor(tensors, "joiner.decoder_proj.weight", value)
        elif name == "decoder_proj.bias":
            add_tensor(tensors, "joiner.decoder_proj.bias", value)
        else:
            add_tensor(tensors, name, value)

    for name, value in joiner_tensors.items():
        if name == "output_linear.weight":
            add_tensor(tensors, "joiner.output_linear.weight", value)
        elif name == "output_linear.bias":
            add_tensor(tensors, "joiner.output_linear.bias", value)
        else:
            add_tensor(tensors, name, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize ONNX initializers into an F32 safetensors pack."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--joiner", required=True, type=Path)
    parser.add_argument(
        "--config-out",
        type=Path,
        help="write X-ASR config.json derived from ONNX metadata (default: alongside --out when --xasr-remap is set)",
    )
    parser.add_argument(
        "--xasr-remap",
        action="store_true",
        help="apply X-ASR decoder/joiner/final-encoder-proj semantic remaps",
    )
    args = parser.parse_args(argv)

    for path in [args.encoder, args.decoder, args.joiner]:
        if not path.is_file():
            raise SystemExit(f"ONNX file not found: {path}")

    tensors: dict[str, np.ndarray] = {}
    if args.xasr_remap:
        add_xasr_onnx_initializers(
            tensors,
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
        )
        config_out = args.config_out or args.out.parent / "config.json"
        write_json_config(
            config_out,
            xasr_config_from_onnx_metadata(
                encoder=args.encoder,
                decoder=args.decoder,
                joiner=args.joiner,
            ),
        )
        print(f"wrote {config_out} (X-ASR ONNX metadata config)")
    else:
        for path in [args.encoder, args.decoder, args.joiner]:
            for name, value in load_initializers(path).items():
                add_tensor(tensors, name, value)

    write_safetensors(args.out, tensors)
    total_params = sum(int(np.prod(arr.shape)) for arr in tensors.values())
    print(
        f"wrote {args.out} ({len(tensors)} tensors, {total_params / 1e6:.1f}M params, F32)"
    )
    for name in sorted(tensors)[:40]:
        print(f"  {name:60s} {list(tensors[name].shape)}")
    if len(tensors) > 40:
        print(f"  ... (+{len(tensors) - 40} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
