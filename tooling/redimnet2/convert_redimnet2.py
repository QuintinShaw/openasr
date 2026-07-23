#!/usr/bin/env python3
"""Convert a ReDimNet2 speaker-embedder checkpoint (`*.pt`) into an OpenASR
``.oasr`` GGUF pack.

Target: the B6 Chinese-enhanced checkpoint
``b6-vb2+vox2+cnc2_v0-lm.pt`` from PalabraAI/redimnet2 (MIT), a 12.46 M-param
UNet-style "dimension reshaping" speaker net that outputs a 192-d embedding.

Design (see ``docs/design/redimnet2-b6-embedder.md``):

  * ReDimNet2 runs through a **ggml graph** (ggml-only invariant), unlike the
    legacy pure-Rust WeSpeaker ResNet34. So this pack follows the standard ggml
    tensor convention: ``gguf.GGUFWriter`` stores dims in ggml ``ne`` order
    (torch shape reversed) and the flat payload in ggml memory order (ne0
    innermost). The Rust side reads each tensor's flat f32 via the existing
    ``diarize::embed::weights::Weights::from_oasr`` reader and uploads it
    verbatim into a graph tensor created with the same ``ne`` dims -- both sides
    agree on ggml memory order, so no transpose is needed.
  * The ``spec.*`` front-end buffers (DFT kernels, mel matrix, preemph filter)
    are **not** packed: the Rust ``RedimNetFrontend`` recomputes those
    deterministic constants (matching the WeSpeaker ``Fbank`` convention). Only
    the neural weights ship.
  * ``*.num_batches_tracked`` buffers are dropped (they are int64 counters, not
    weights).

Weight typing: norms, biases, the ``weigth1d`` aggregation params, and the ASTP
pool live in f32 for parity; rank>=2 conv/linear/attention ``.weight`` tensors
take the requested ``--quant`` (default f16). BatchNorm is *not* folded here --
the graph applies it explicitly so the pack stays a faithful mirror of the
checkpoint.

Usage::

    python3 convert_redimnet2.py --in b6-....pt --out redimnet2-b6.oasr \
        [--quant f16|f32|q8_0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ARCH = "redimnet2"

# B6 model_config (from the checkpoint; see B6_STRUCTURE_SPEC.md). Baked into
# GGUF metadata so the runtime graph builder reads structural constants from the
# pack instead of hard-assuming B6.
B6_MODEL_CONFIG = {
    "C": 64,
    "F": 72,
    "embed_dim": 192,
    "out_channels": 224,
    "hop_length": 160,
    "n_mels": 72,
    "pooling_func": "ASTP",
    "global_context_att": True,
    "return_2d_output": True,
    "agg_gnorm": True,
    "fm_weigthing_type": "NC",
    "block_1d_type": "conv+att",
    "block_2d_type": "basic_resnet",
    "compress_tconvs": True,
    "group_divisor": 1,
    "dual_agg": False,
    "feat_type": "tf",
    # (freq_stride, time_stride) products of the stages_setup strides.
    "freq_stride": 8,
    "time_stride": 4,
    # (freq_stride, time_stride, num_blocks, conv_exp, att_block_red) per stage.
    # conv_exp kept as-is (may be < 1: stages 4/5 narrow the branch).
    "stages_setup": [
        [1, 1, 3, 3.0, 64],
        [2, 1, 4, 2.0, 64],
        [1, 2, 5, 2.0, 48],
        [2, 1, 5, 1.0, 48],
        [1, 2, 4, 0.75, 32],
        [2, 1, 3, 0.5, 24],
    ],
    "spec_params": {"do_preemph": True, "do_spec_aug": False, "norm_signal": True},
}


class ConversionError(RuntimeError):
    pass


def remap_tensor(name: str) -> Optional[str]:
    """Map a checkpoint tensor name to its GGUF name, or ``None`` to drop it.

    The neural namespace is preserved verbatim (``backbone.*``, ``pool.*``,
    ``bn.*``, ``linear.*``); only the recomputed front end and int counters are
    dropped.
    """
    if name.startswith("spec."):
        return None
    if name.endswith("num_batches_tracked"):
        return None
    return name


def is_force_f32(gguf_name: str, rank: int) -> bool:
    """f32-locked tensors: everything that is not a rank>=2 projection weight.

    Norms/biases, the ``weigth1d`` softmax params (``*.w`` rank-4 but tiny and
    numerically sensitive), BatchNorm running stats, and the ASTP pool all stay
    f32. Quantizing them would move the embedding without shrinking the pack.
    """
    if rank < 2:
        return True
    if gguf_name.endswith(".bias"):
        return True
    if gguf_name.endswith(".w"):  # weigth1d / fin_wght1d aggregation params
        return True
    if gguf_name.startswith("pool."):
        return True
    if ".bn" in gguf_name or gguf_name.endswith("running_mean") or gguf_name.endswith("running_var"):
        return True
    # LayerNorm / GroupNorm affine params are rank-1 -> already caught above.
    return False


def choose_tensor_type(gguf_name: str, shape: tuple[int, ...], quant: str) -> str:
    """Return one of ``f32`` / ``f16`` / ``q8_0`` for this tensor."""
    rank = len(shape)
    if quant == "f32" or is_force_f32(gguf_name, rank):
        return "f32"
    if quant == "q8_0":
        # q8_0 needs the innermost dim divisible by the 32-wide block. 1x1 convs
        # and tiny projections fall back to f16.
        if shape[-1] % 32 == 0:
            return "q8_0"
        return "f16"
    return "f16"


def load_state_dict(path: Path) -> dict[str, "np.ndarray"]:
    import torch  # deferred: only needed for real conversion, not unit tests.

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        out[k] = v.detach().to(torch.float32).cpu().numpy()
    return out


def build_tensor_plan(
    state: dict[str, "np.ndarray"], quant: str
) -> list[tuple[str, "np.ndarray", str]]:
    """`(gguf_name, array, tensor_type)` for every kept tensor, sorted by name."""
    plan: list[tuple[str, np.ndarray, str]] = []
    seen: set[str] = set()
    for name in sorted(state.keys()):
        gguf_name = remap_tensor(name)
        if gguf_name is None:
            continue
        if gguf_name in seen:
            raise ConversionError(f"duplicate GGUF tensor {gguf_name}")
        seen.add(gguf_name)
        arr = np.ascontiguousarray(state[name])
        ttype = choose_tensor_type(gguf_name, tuple(arr.shape), quant)
        plan.append((gguf_name, arr, ttype))
    if not plan:
        raise ConversionError("no tensors survived remap -- wrong checkpoint?")
    return plan


def write_pack(out_path: Path, plan: list[tuple[str, "np.ndarray", str]]) -> None:
    import gguf

    writer = gguf.GGUFWriter(str(out_path), ARCH, use_temp_file=True)

    cfg = B6_MODEL_CONFIG
    writer.add_uint32("redimnet2.embed_dim", cfg["embed_dim"])
    writer.add_uint32("redimnet2.n_mels", cfg["n_mels"])
    writer.add_uint32("redimnet2.channels", cfg["C"])
    writer.add_uint32("redimnet2.freq_bins", cfg["F"])
    writer.add_uint32("redimnet2.out_channels", cfg["out_channels"])
    writer.add_uint32("redimnet2.hop_length", cfg["hop_length"])
    writer.add_uint32("redimnet2.freq_stride", cfg["freq_stride"])
    writer.add_uint32("redimnet2.time_stride", cfg["time_stride"])
    writer.add_bool("redimnet2.global_context_att", cfg["global_context_att"])
    writer.add_bool("redimnet2.return_2d_output", cfg["return_2d_output"])
    writer.add_bool("redimnet2.agg_gnorm", cfg["agg_gnorm"])
    # Full config as JSON for fidelity / auditability.
    writer.add_string("redimnet2.model_config_json", json.dumps(cfg, sort_keys=True))

    for gguf_name, arr, ttype in plan:
        if ttype == "q8_0":
            data = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(gguf_name, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif ttype == "f16":
            writer.add_tensor(
                gguf_name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16
            )
        else:
            writer.add_tensor(
                gguf_name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def convert(in_path: Path, out_path: Path, quant: str) -> int:
    state = load_state_dict(in_path)
    plan = build_tensor_plan(state, quant)
    write_pack(out_path, plan)
    kept = len(plan)
    dropped = len(state) - kept
    print(f"wrote {out_path} : {kept} tensors ({dropped} dropped), quant={quant}")
    return kept


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--quant", choices=["f32", "f16", "q8_0"], default="f16")
    args = ap.parse_args(argv)
    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return 2
    convert(args.in_path, args.out_path, args.quant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
