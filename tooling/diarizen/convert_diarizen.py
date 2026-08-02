#!/usr/bin/env python3
"""Convert the pinned DiariZen Base-s80 checkpoint to an OpenASR ``.oasr``.

The target is ``BUT-FIT/diarizen-wavlm-base-s80-md`` at revision
``a9857fc34908197fb5336d9d0562f291834a04b2``.  It is a structured-pruned
WavLM-Base+ encoder followed by a four-layer Conformer and an 11-class
powerset head.  The converter is deliberately architecture-specific: accepting
an arbitrary checkpoint and silently guessing its pruning layout would make the
native runtime contract impossible to audit.

The upstream *weights* are CC BY-NC 4.0.  This converter is tooling only; it
does not grant permission to redistribute a converted pack.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


ARCH = "diarizen-wavlm-conformer-segmentation"
MODEL_ID = "diarizen-wavlm-base-s80-md"
PINNED_REVISION = "a9857fc34908197fb5336d9d0562f291834a04b2"

EXPECTED_MODEL_PATH = "diarizen.models.eend.model_wavlm_conformer.Model"
EXPECTED_MODEL_ARGS = {
    "wavlm_src": "wavlm_base_s80_md",
    "wavlm_layer_num": 13,
    "wavlm_feat_dim": 768,
    "attention_in": 256,
    "ffn_hidden": 1024,
    "num_head": 4,
    "num_layer": 4,
    "kernel_size": 31,
    "dropout": 0.1,
    "use_posi": False,
    "output_activate_function": False,
    "max_speakers_per_chunk": 4,
    "max_speakers_per_frame": 2,
    "chunk_size": 16,
    "selected_channel": 0,
    "sample_rate": 16000,
}

WAVLM_CONV_CHANNELS = [90, 161, 173, 181, 351, 155, 137]
WAVLM_CONV_KERNELS = [10, 3, 3, 3, 3, 2, 2]
WAVLM_CONV_STRIDES = [5, 2, 2, 2, 2, 2, 2]
WAVLM_REMAINING_HEADS = [
    [1, 6],
    [5, 7, 8],
    [0, 3, 9],
    [0, 1, 4, 8, 11],
    [6, 8],
    [0],
    [7, 8, 10, 11],
    [0, 1, 4, 8],
    [],
    [],
    [4, 7],
    [5],
]
WAVLM_FFN_DIMS = [666, 660, 649, 1080, 237, 299, 437, 573, 53, 80, 211, 334]


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorPlan:
    name: str
    values: np.ndarray
    tensor_type: str


POS_CONV_G = (
    "wavlm_model.encoder.transformer.pos_conv_embed.conv."
    "parametrizations.weight.original0"
)
POS_CONV_V = (
    "wavlm_model.encoder.transformer.pos_conv_embed.conv."
    "parametrizations.weight.original1"
)
POS_CONV_WEIGHT = "wavlm_model.encoder.transformer.pos_conv_embed.conv.weight"


def load_and_validate_config(path: Path) -> dict[str, object]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    model = config.get("model")
    if not isinstance(model, dict) or model.get("path") != EXPECTED_MODEL_PATH:
        raise ConversionError(f"unexpected DiariZen model path in {path}")
    args = model.get("args")
    if not isinstance(args, dict):
        raise ConversionError(f"missing model.args in {path}")
    for key, expected in EXPECTED_MODEL_ARGS.items():
        # The upstream TOML omits constructor defaults such as kernel_size and
        # max_speakers_per_frame. Resolve those omissions exactly as Python's
        # constructor does, while still rejecting an explicit drift.
        if args.get(key, expected) != expected:
            raise ConversionError(
                f"unsupported model.args.{key}: expected {expected!r}, got {args.get(key)!r}"
            )
    inference = config.get("inference")
    inference_args = inference.get("args") if isinstance(inference, dict) else None
    if not isinstance(inference_args, dict) or inference_args.get(
        "apply_median_filtering"
    ) is not True:
        raise ConversionError("Base-s80 runtime contract requires median filtering")
    return config


def load_state_dict(path: Path) -> dict[str, np.ndarray]:
    import torch  # Deferred so converter unit tests do not require torch.

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ConversionError("checkpoint does not contain a state dictionary")
    result: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if name.startswith(("loss_func.", "validation_metric.")):
            continue
        if name.endswith("num_batches_tracked"):
            continue
        if not value.is_floating_point():
            raise ConversionError(f"unexpected non-floating tensor {name}: {value.dtype}")
        result[name] = value.detach().to(torch.float32).cpu().numpy()
    validate_state_dict(result)
    return result


def _shape(state: dict[str, np.ndarray], name: str, expected: tuple[int, ...]) -> None:
    value = state.get(name)
    if value is None:
        raise ConversionError(f"missing tensor {name}")
    if tuple(value.shape) != expected:
        raise ConversionError(f"{name} shape {tuple(value.shape)} != {expected}")


def validate_state_dict(state: dict[str, np.ndarray]) -> None:
    _shape(state, "weight_sum.weight", (1, 13))
    _shape(state, "proj.weight", (256, 768))
    _shape(state, "classifier.weight", (11, 256))
    in_channels = 1
    for index, (out_channels, kernel) in enumerate(
        zip(WAVLM_CONV_CHANNELS, WAVLM_CONV_KERNELS, strict=True)
    ):
        _shape(
            state,
            f"wavlm_model.feature_extractor.conv_layers.{index}.conv.weight",
            (out_channels, in_channels, kernel),
        )
        in_channels = out_channels
    for index, (heads, ffn_dim) in enumerate(
        zip(WAVLM_REMAINING_HEADS, WAVLM_FFN_DIMS, strict=True)
    ):
        prefix = f"wavlm_model.encoder.transformer.layers.{index}"
        _shape(state, f"{prefix}.layer_norm.weight", (768,))
        _shape(state, f"{prefix}.final_layer_norm.weight", (768,))
        _shape(state, f"{prefix}.feed_forward.intermediate_dense.weight", (ffn_dim, 768))
        _shape(state, f"{prefix}.feed_forward.output_dense.weight", (768, ffn_dim))
        q_name = f"{prefix}.attention.q_proj.weight"
        if heads:
            _shape(state, q_name, (len(heads) * 64, 768))
        elif q_name in state:
            raise ConversionError(f"pruned attention layer {index} unexpectedly has q_proj")
    for index in range(4):
        prefix = f"conformer.conformer_layer.{index}"
        _shape(state, f"{prefix}.ffn1.w_1.weight", (1024, 256))
        _shape(state, f"{prefix}.mha.mha.linearQ.weight", (256, 256))
        _shape(state, f"{prefix}.conv.depthwise_conv.weight", (256, 1, 31))
    if len(state) != 377:
        raise ConversionError(f"expected 377 runtime tensors, found {len(state)}")


def is_force_f32(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) < 2 or name.endswith(".bias"):
        return True
    if any(
        marker in name
        for marker in (
            "layer_norm",
            "ln_norm",
            "bn_norm",
            "running_mean",
            "running_var",
            "rel_attn_embed",
            "gru_rel_pos",
            "weight_sum",
            "dummy_weight",
        )
    ):
        return True
    return False


def materialize_runtime_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fold weight-normalized positional convolution into one runtime tensor."""
    result = dict(state)
    g = result.pop(POS_CONV_G, None)
    v = result.pop(POS_CONV_V, None)
    if g is None or v is None:
        raise ConversionError("missing positional-convolution weight-norm tensors")
    if tuple(g.shape) != (1, 1, 128) or tuple(v.shape) != (768, 48, 128):
        raise ConversionError(
            f"unexpected positional-convolution weight-norm shapes: {g.shape}, {v.shape}"
        )
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=(0, 1), keepdims=True))
    if np.any(norm == 0.0):
        raise ConversionError("positional-convolution weight norm contains zero")
    result[POS_CONV_WEIGHT] = np.ascontiguousarray(
        (g.astype(np.float64) * v.astype(np.float64) / norm).astype(np.float32)
    )
    return result


def choose_tensor_type(name: str, shape: tuple[int, ...], quant: str) -> str:
    if quant == "f32" or is_force_f32(name, shape):
        return "f32"
    if quant == "f16":
        return "f16"
    # Q8_0 is useful for dense projections. Keep convolutions in f16: ggml's
    # conv kernels consume f16/f32 weights, while quantized matmul is native.
    if name.endswith(".weight") and "conv" not in name and shape[-1] % 32 == 0:
        return "q8_0"
    return "f16"


def build_tensor_plan(state: dict[str, np.ndarray], quant: str) -> list[TensorPlan]:
    validate_state_dict(state)
    state = materialize_runtime_state(state)
    return [
        TensorPlan(
            name=name,
            values=np.ascontiguousarray(state[name]),
            tensor_type=choose_tensor_type(name, tuple(state[name].shape), quant),
        )
        for name in sorted(state)
    ]


def write_pack(
    out_path: Path,
    plan: list[TensorPlan],
    *,
    quant: str,
    model_id: str = MODEL_ID,
) -> None:
    import gguf

    writer = gguf.GGUFWriter(str(out_path), ARCH, use_temp_file=True)
    writer.add_string("openasr.package.version", "1")
    writer.add_string("openasr.model.family", "diarizen-segmentation")
    writer.add_string("openasr.model.architecture", ARCH)
    writer.add_string("openasr.model.id", model_id)
    writer.add_string("openasr.quantization", {"f16": "fp16"}.get(quant, quant))
    writer.add_string("diarizen.upstream_revision", PINNED_REVISION)
    writer.add_uint32("diarizen.sample_rate", 16_000)
    writer.add_uint32("diarizen.window_samples", 16 * 16_000)
    writer.add_uint32("diarizen.window_step_samples", 16 * 1_600)
    writer.add_uint32("diarizen.output_frame_stride_samples", 320)
    writer.add_uint32("diarizen.local_speakers", 4)
    writer.add_uint32("diarizen.max_simultaneous_speakers", 2)
    writer.add_uint32("diarizen.powerset_classes", 11)
    writer.add_uint32("diarizen.median_filter_frames", 11)
    writer.add_string(
        "diarizen.wavlm_config_json",
        json.dumps(
            {
                "conv_channels": WAVLM_CONV_CHANNELS,
                "conv_kernels": WAVLM_CONV_KERNELS,
                "conv_strides": WAVLM_CONV_STRIDES,
                "remaining_heads": WAVLM_REMAINING_HEADS,
                "ffn_dims": WAVLM_FFN_DIMS,
                "hidden_size": 768,
                "total_heads": 12,
                "head_dim": 64,
                "relative_position_buckets": 320,
                "relative_position_max_distance": 800,
            },
            separators=(",", ":"),
        ),
    )

    for tensor in plan:
        if tensor.tensor_type == "q8_0":
            data = gguf.quants.quantize(tensor.values, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(
                tensor.name, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0
            )
        elif tensor.tensor_type == "f16":
            writer.add_tensor(
                tensor.name,
                tensor.values.astype(np.float16),
                raw_dtype=gguf.GGMLQuantizationType.F16,
            )
        else:
            writer.add_tensor(
                tensor.name,
                tensor.values.astype(np.float32),
                raw_dtype=gguf.GGMLQuantizationType.F32,
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def convert(
    checkpoint_path: Path,
    config_path: Path,
    out_path: Path,
    quant: str,
    model_id: str = MODEL_ID,
) -> int:
    load_and_validate_config(config_path)
    state = load_state_dict(checkpoint_path)
    plan = build_tensor_plan(state, quant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_pack(out_path, plan, quant=quant, model_id=model_id)
    counts: dict[str, int] = {}
    for tensor in plan:
        counts[tensor.tensor_type] = counts.get(tensor.tensor_type, 0) + 1
    print(f"wrote {out_path}: {len(plan)} tensors, quant={quant}, types={counts}")
    return len(plan)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quant", choices=("f32", "f16", "q8_0"), default="f16")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args(argv)
    for path in (args.checkpoint, args.config):
        if not path.is_file():
            print(f"error: input not found: {path}", file=sys.stderr)
            return 2
    try:
        convert(args.checkpoint, args.config, args.out, args.quant, args.model_id)
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
