#!/usr/bin/env python3
"""Normalize a PyTorch checkpoint (`.pt`/`.ckpt`) into an F32 `.safetensors`
weight pack with clean `state_dict` names, for the Rust
`model-pack import <family>` path.

This is the `.pt` counterpart to ``pyannote_extract.py`` (which sources from
ONNX). Use it for upstreams that ship only PyTorch checkpoints, or whose ONNX
export fuses/renames weights to opaque names. Concrete motivating case:
icefall/k2 streaming zipformer transducers often ship ONNX exports that fold
large parts of the encoder into opaque ``onnx::MatMul_*`` initializers, whereas
the `.pt` keeps clean icefall ``encoder.encoders.{i}...`` names that map 1:1
onto a source-level runtime implementation. Do not assume the `.pt` and ONNX
artifacts are numerically identical; for `GilgameshWind/X-ASR-zh-en`, the public
ONNX deployment weights differ from `streaming_exp/pretrained.pt`, so ONNX is
the quality oracle and `.pt` is only a source-structure reference.

Weights are written as F32 safetensors directly (stdlib ``struct`` + ``json``),
mirroring ``pyannote_extract.write_safetensors`` — so the ONLY dependency is
``torch`` (to read the pickle); no ``safetensors`` package is required.

Example::

    python3 tooling/publish-model/scripts/pt_to_safetensors.py \
        --pt  tmp/xasr-test/model/streaming_exp/pretrained.pt \
        --out tmp/xasr-test/src/model.safetensors

The resulting `.safetensors` (alongside a `config.json` you provide) is then
converted to an `.oasr` runtime pack by the family's
`convert_local_*_source_to_runtime_pack` (Rust). This tool does NO architecture
or `.oasr` logic — it is a dumb, language-neutral weight normalizer.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Common top-level keys under which training checkpoints nest the weights.
CANDIDATE_STATE_DICT_KEYS = ("model", "state_dict", "model_state_dict")


def _tensor_fraction(obj) -> float:
    """Fraction of a dict's values that are torch tensors (0.0 if not a dict)."""
    import torch

    if not isinstance(obj, dict) or not obj:
        return 0.0
    return sum(isinstance(v, torch.Tensor) for v in obj.values()) / len(obj)


def locate_state_dict(checkpoint, override_key: str | None):
    """Find the tensor `state_dict` inside a loaded checkpoint object."""
    if override_key is not None:
        if not isinstance(checkpoint, dict) or override_key not in checkpoint:
            raise SystemExit(
                f"--key {override_key!r} not found; top-level keys: "
                f"{list(checkpoint)[:20] if isinstance(checkpoint, dict) else type(checkpoint)}"
            )
        sd = checkpoint[override_key]
        if _tensor_fraction(sd) < 0.5:
            raise SystemExit(f"--key {override_key!r} does not hold a tensor state_dict")
        return sd

    if _tensor_fraction(checkpoint) >= 0.5:
        return checkpoint

    for key in CANDIDATE_STATE_DICT_KEYS:
        if isinstance(checkpoint, dict) and _tensor_fraction(checkpoint.get(key)) >= 0.5:
            print(f"  using nested state_dict under {key!r}", file=sys.stderr)
            return checkpoint[key]

    # Fallback: pick the top-level value that is most tensor-dense.
    if isinstance(checkpoint, dict):
        best_key, best_frac = None, 0.5
        for key, value in checkpoint.items():
            frac = _tensor_fraction(value)
            if frac > best_frac:
                best_key, best_frac = key, frac
        if best_key is not None:
            print(
                f"  using nested state_dict under {best_key!r} "
                f"(auto-detected, {best_frac:.0%} tensors)",
                file=sys.stderr,
            )
            return checkpoint[best_key]

    raise SystemExit("could not locate a tensor state_dict; pass --key explicitly")


def extract(state_dict, strip_prefix: str) -> dict[str, np.ndarray]:
    import torch

    tensors: dict[str, np.ndarray] = {}
    skipped_nontensor = 0
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            skipped_nontensor += 1
            continue
        clean = name
        if strip_prefix and clean.startswith(strip_prefix):
            clean = clean[len(strip_prefix):]
        arr = value.detach().to("cpu", torch.float32).contiguous().numpy()
        tensors[clean] = np.ascontiguousarray(arr, dtype=np.float32)
    if skipped_nontensor:
        print(f"  (skipped {skipped_nontensor} non-tensor entries)", file=sys.stderr)
    return tensors


def write_safetensors(out_path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write F32 safetensors directly (mirrors pyannote_extract.write_safetensors)."""
    header: dict[str, object] = {}
    blob = bytearray()
    for name in sorted(tensors):
        arr = tensors[name]
        if arr.dtype != np.float32:
            raise SystemExit(f"tensor {name!r} is {arr.dtype}, expected float32")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a PyTorch .pt checkpoint into an F32 safetensors weight pack."
    )
    parser.add_argument("--pt", required=True, type=Path, help="input .pt/.ckpt checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="output .safetensors path")
    parser.add_argument(
        "--key",
        default=None,
        help="explicit top-level checkpoint key holding the state_dict (auto-detected if omitted)",
    )
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="strip this prefix from every tensor name (e.g. 'module.' or 'model.')",
    )
    args = parser.parse_args(argv)

    if not args.pt.is_file():
        raise SystemExit(f".pt not found: {args.pt}")

    import torch

    print(f"loading {args.pt} ...", file=sys.stderr)
    try:
        checkpoint = torch.load(str(args.pt), map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - fall back for checkpoints with metadata
        print(
            f"  weights_only=True failed ({type(error).__name__}); "
            f"retrying weights_only=False (source is operator-trusted)",
            file=sys.stderr,
        )
        checkpoint = torch.load(str(args.pt), map_location="cpu", weights_only=False)

    state_dict = locate_state_dict(checkpoint, args.key)
    tensors = extract(state_dict, args.strip_prefix)
    if not tensors:
        raise SystemExit("no tensors extracted")
    write_safetensors(args.out, tensors)

    total_params = sum(int(np.prod(arr.shape)) for arr in tensors.values())
    print(
        f"wrote {args.out} ({len(tensors)} tensors, {total_params / 1e6:.1f}M params, F32)",
        file=sys.stderr,
    )
    for name in sorted(tensors)[:40]:
        print(f"  {name:60s} {list(tensors[name].shape)}", file=sys.stderr)
    if len(tensors) > 40:
        print(f"  ... (+{len(tensors) - 40} more)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
