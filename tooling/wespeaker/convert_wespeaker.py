#!/usr/bin/env python3
"""Convert a WeSpeaker ResNet speaker-embedder checkpoint into an OpenASR
``.oasr`` GGUF pack.

Target: official WeSpeaker ResNet checkpoints (ResNet34 first; 152/221/293
share the same converter and metadata table). Inference runs through a ggml
graph, so this pack follows the standard ggml tensor convention:
``gguf.GGUFWriter`` stores dims in ggml ``ne`` order (torch shape reversed)
and the flat payload in ggml memory order (ne0 innermost).

Dropped tensors:
  * ``projection.*`` (training classification head)
  * ``*.num_batches_tracked`` (int64 BN counters)
  * Identity ``seg_2.*`` / ``seg_bn_1.*`` when present (``two_emb_layer=false``)

Accepted input key styles:
  * official WeSpeaker (no prefix): ``conv1.weight``, ``layer1.0.conv1.weight``
  * pyannote wrapper (``resnet.`` prefix): stripped so the pack always stores
    official names.

Usage::

    python3 convert_wespeaker.py --in avg_model --out wespeaker-resnet34.oasr \\
        --quant f32 --model-id wespeaker-voxceleb-resnet34-lm --depth 34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ARCH = "wespeaker-resnet"
FAMILY = "wespeaker"
LICENSE_NAME = "CC-BY-4.0"
PACKAGE_VERSION = "1"

DEPTH_TABLE = {
    34: {"block_kind": "basic", "num_blocks": [3, 4, 6, 3]},
    152: {"block_kind": "bottleneck", "num_blocks": [3, 8, 36, 3]},
    221: {"block_kind": "bottleneck", "num_blocks": [6, 16, 48, 3]},
    293: {"block_kind": "bottleneck", "num_blocks": [10, 20, 64, 3]},
}

DROP_PREFIXES = ("projection.",)
DROP_EXACT_PREFIXES = ("seg_2.", "seg_bn_1.")
DROP_SUFFIXES = ("num_batches_tracked",)


class ConversionError(RuntimeError):
    pass


def remap_tensor(name: str) -> Optional[str]:
    """Map a checkpoint tensor name to its GGUF name, or ``None`` to drop it.

    Pack tensors always use official WeSpeaker names (no ``resnet.`` prefix).
    """
    if name.startswith("resnet."):
        name = name[len("resnet.") :]
    if name.endswith("num_batches_tracked") or name.endswith(".num_batches_tracked"):
        return None
    if any(name.startswith(prefix) for prefix in DROP_PREFIXES):
        return None
    if any(name.startswith(prefix) for prefix in DROP_EXACT_PREFIXES):
        return None
    if name in ("projection.weight", "projection.bias"):
        return None
    return name


def is_force_f32(gguf_name: str, rank: int) -> bool:
    """f32-locked tensors: BN stats/affine, biases, and rank < 2."""
    if rank < 2:
        return True
    if gguf_name.endswith(".bias"):
        return True
    if (
        ".bn" in gguf_name
        or gguf_name.endswith("running_mean")
        or gguf_name.endswith("running_var")
        or ".shortcut.1." in gguf_name
    ):
        return True
    return False


def choose_tensor_type(gguf_name: str, shape: tuple[int, ...], quant: str) -> str:
    rank = len(shape)
    if quant == "f32" or is_force_f32(gguf_name, rank):
        return "f32"
    return "f16"


def infer_num_blocks(state: dict[str, np.ndarray]) -> list[int]:
    counts: list[int] = []
    for stage in range(1, 5):
        n = 0
        while f"layer{stage}.{n}.conv1.weight" in state:
            n += 1
        counts.append(n)
    return counts


def infer_block_kind(state: dict[str, np.ndarray]) -> str:
    if "layer1.0.conv3.weight" in state:
        return "bottleneck"
    return "basic"


def infer_depth(state: dict[str, np.ndarray]) -> int:
    counts = infer_num_blocks(state)
    kind = infer_block_kind(state)
    for depth, spec in DEPTH_TABLE.items():
        if spec["num_blocks"] == counts and spec["block_kind"] == kind:
            return depth
    raise ConversionError(
        f"unrecognized WeSpeaker ResNet topology: blocks={counts} kind={kind}"
    )


def load_state_dict(path: Path) -> dict[str, np.ndarray]:
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        raise ConversionError(f"unsupported checkpoint type: {type(ckpt)}")
    out: dict[str, np.ndarray] = {}
    for key, value in sd.items():
        if not hasattr(value, "detach"):
            continue
        out[key] = value.detach().to(torch.float32).cpu().numpy()
    if not out:
        raise ConversionError("checkpoint contained no floating-point tensors")
    return out


def canonicalize_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    canonical: dict[str, np.ndarray] = {}
    for name, arr in state.items():
        mapped = remap_tensor(name)
        if mapped is None:
            continue
        if mapped in canonical:
            raise ConversionError(f"duplicate GGUF tensor {mapped} from {name}")
        canonical[mapped] = np.ascontiguousarray(arr)
    if "conv1.weight" not in canonical:
        raise ConversionError("missing conv1.weight after remap -- wrong checkpoint?")
    if "seg_1.weight" not in canonical or "seg_1.bias" not in canonical:
        raise ConversionError("missing seg_1 linear -- two_emb_layer packs are unsupported")
    return canonical


def build_tensor_plan(
    state: dict[str, np.ndarray], quant: str
) -> list[tuple[str, np.ndarray, str]]:
    canonical = canonicalize_state(state)
    plan: list[tuple[str, np.ndarray, str]] = []
    for name in sorted(canonical.keys()):
        arr = canonical[name]
        ttype = choose_tensor_type(name, tuple(arr.shape), quant)
        plan.append((name, arr, ttype))
    if not plan:
        raise ConversionError("no tensors survived remap -- wrong checkpoint?")
    return plan


def write_pack(
    out_path: Path,
    plan: list[tuple[str, np.ndarray, str]],
    *,
    model_id: str,
    quant: str,
    depth: int,
    block_kind: str,
    num_blocks: list[int],
) -> None:
    import gguf

    writer = gguf.GGUFWriter(str(out_path), ARCH, use_temp_file=True)
    writer.add_string("openasr.package.version", PACKAGE_VERSION)
    writer.add_string("openasr.model.family", FAMILY)
    writer.add_string("openasr.model.architecture", ARCH)
    writer.add_string("openasr.model.id", model_id)
    quant_label = {"f16": "fp16", "f32": "f32"}.get(quant, quant)
    writer.add_string("openasr.quantization", quant_label)
    writer.add_string("openasr.license.name", LICENSE_NAME)

    writer.add_uint32("wespeaker.embed_dim", 256)
    writer.add_uint32("wespeaker.n_mels", 80)
    writer.add_uint32("wespeaker.m_channels", 32)
    writer.add_uint32("wespeaker.depth", depth)
    writer.add_string("wespeaker.block_kind", block_kind)
    writer.add_string("wespeaker.num_blocks", json.dumps(num_blocks))
    writer.add_string("wespeaker.pooling", "TSTP")
    writer.add_bool("wespeaker.two_emb_layer", False)

    for gguf_name, arr, ttype in plan:
        if ttype == "f16":
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


def convert(
    in_path: Path,
    out_path: Path,
    quant: str,
    model_id: str,
    depth: Optional[int] = None,
) -> int:
    state = load_state_dict(in_path)
    canonical = canonicalize_state(state)
    inferred = infer_depth(canonical)
    inferred_kind = infer_block_kind(canonical)
    inferred_blocks = infer_num_blocks(canonical)
    if depth is not None and depth != inferred:
        raise ConversionError(
            f"--depth {depth} does not match checkpoint topology "
            f"(inferred {inferred}, blocks={inferred_blocks}, kind={inferred_kind})"
        )
    depth = inferred
    spec = DEPTH_TABLE[depth]
    plan = build_tensor_plan(state, quant)
    write_pack(
        out_path,
        plan,
        model_id=model_id,
        quant=quant,
        depth=depth,
        block_kind=spec["block_kind"],
        num_blocks=spec["num_blocks"],
    )
    kept = len(plan)
    dropped = len(state) - kept
    print(
        f"wrote {out_path} : {kept} tensors ({dropped} dropped), "
        f"quant={quant} depth={depth} kind={spec['block_kind']}"
    )
    return kept


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--quant", choices=["f32", "f16"], default="f32")
    ap.add_argument("--model-id", default="wespeaker-voxceleb-resnet34-lm")
    ap.add_argument("--depth", type=int, choices=sorted(DEPTH_TABLE.keys()))
    args = ap.parse_args(argv)
    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return 2
    try:
        convert(args.in_path, args.out_path, args.quant, args.model_id, args.depth)
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
