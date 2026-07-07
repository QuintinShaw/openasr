#!/usr/bin/env python3
"""Fold an ESPnet `feats_stats.npz` global-MVN into a Dolphin `full.safetensors`
as `encoder.global_cmvn.mean` / `encoder.global_cmvn.istd`.

The cn-dialect Dolphin checkpoints (small.cn, cn-dialect-base) bake their
WeNet-style `global_cmvn` mean/istd directly into the exported state dict, so
`pt_to_safetensors.py` alone is enough for them. The multilingual checkpoints
(dolphin-small, dolphin-base) instead ship an ESPnet-style `feats_stats.npz`
(`count`/`sum`/`sum_square` accumulators) alongside the `.pt`, with no
`encoder.global_cmvn.*` tensors in the state dict at all -- this script derives
the same two tensors `crate::models::dolphin::package_import` requires
(`encoder.global_cmvn.mean`/`.istd`, both `[n_mels]` float32) from those
accumulators and appends them to an existing `full.safetensors` in place.

Formula (identical to WeNet's `wenet/utils/cmvn.py::_load_json_cmvn`, just
sourced from ESPnet's accumulator names instead of `mean_stat`/`var_stat`/
`frame_num`):

    mean = sum / count
    var  = max(sum_square / count - mean**2, eps)
    istd = 1 / sqrt(var)

Example::

    python3 tooling/publish-model/scripts/dolphin_fold_espnet_cmvn.py \\
        --safetensors tmp/publish/dolphin-base/src/full.safetensors \\
        --feats-stats tmp/publish/dolphin-base/src/feats_stats.npz
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# WeNet/ESPnet's shared variance floor, so a near-silent mel bin never blows up
# the derived istd.
VARIANCE_EPS = 1.0e-20


def load_safetensors(path: Path) -> tuple[dict[str, dict], bytes]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    return header, blob


def derive_cmvn(feats_stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(feats_stats_path)
    count = float(stats["count"])
    if count <= 0:
        raise SystemExit(f"feats_stats.npz has non-positive count: {count}")
    total = stats["sum"].astype(np.float64)
    total_sq = stats["sum_square"].astype(np.float64)
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, VARIANCE_EPS)
    istd = 1.0 / np.sqrt(variance)
    return mean.astype(np.float32), istd.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors", required=True, type=Path)
    parser.add_argument("--feats-stats", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.safetensors.is_file():
        raise SystemExit(f"safetensors not found: {args.safetensors}")
    if not args.feats_stats.is_file():
        raise SystemExit(f"feats_stats.npz not found: {args.feats_stats}")

    header, blob = load_safetensors(args.safetensors)
    for existing in ("encoder.global_cmvn.mean", "encoder.global_cmvn.istd"):
        if existing in header:
            raise SystemExit(
                f"{args.safetensors} already has '{existing}'; refusing to duplicate"
            )

    mean, istd = derive_cmvn(args.feats_stats)
    if mean.shape != istd.shape or mean.ndim != 1:
        raise SystemExit(f"unexpected derived CMVN shape: {mean.shape}")

    blob_bytes = bytearray(blob)
    for name, array in (
        ("encoder.global_cmvn.mean", mean),
        ("encoder.global_cmvn.istd", istd),
    ):
        start = len(blob_bytes)
        blob_bytes.extend(array.tobytes(order="C"))
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [start, len(blob_bytes)],
        }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with args.safetensors.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob_bytes)

    print(
        f"folded encoder.global_cmvn.{{mean,istd}} ({mean.shape[0]} mels) into {args.safetensors}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
