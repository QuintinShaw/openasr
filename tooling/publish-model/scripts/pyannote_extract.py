#!/usr/bin/env python3
"""Extract pyannote segmentation-3.0 weights from the ONNX mirror into a
safetensors pack whose tensor names match `crate::diarize::segment`'s loader.

Source (un-gated, MIT): the `onnx-community` pyannote-segmentation-3.0 ONNX mirror.
Pin an exact revision when publishing (see ``--revision`` / ``SOURCE_REVISION``);
download via hf-mirror.com, e.g.::

    huggingface-cli download onnx-community/pyannote-segmentation-3.0 \
        onnx/model.onnx --revision <rev> --local-dir tmp/models/pyannote

    python3 tooling/publish-model/scripts/pyannote_extract.py \
        --onnx tmp/models/pyannote/onnx/model.onnx \
        --out  tmp/models/pyannote/pyannote_seg.safetensors

The resulting `.safetensors` is then converted to a diarization `.oasr` pack by
`convert_local_pyannote_source_to_runtime_pack` (Rust). Most tensors are graph
initializers kept under their exact ONNX names; the single computed tensor the
loader needs — the materialized sinc filter
``/sincnet/conv1d.0/Concat_2_output_0`` ([80, 1, 251]) — is the constant output
of a Concat node, so it is captured by appending it as a graph output and running
one onnxruntime inference on a zero input (the filter is input-independent).

Requires: onnx, onnxruntime, numpy. (No `safetensors` package needed — the file
is written directly.)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

# MIT ONNX mirror; pin a concrete commit when publishing a release pack.
SOURCE_REPO = "onnx-community/pyannote-segmentation-3.0"
# Pinned onnx-community/pyannote-segmentation-3.0 revision the committed
# loader tensor names and parity goldens were validated against.
SOURCE_REVISION = "733a93b6473d019a773298e08cefa686894b1854"

# The one tensor the loader needs that is NOT a graph initializer: the
# materialized SincNet filter, produced by a Concat of constant sinc bands.
MATERIALIZED_SINC = "/sincnet/conv1d.0/Concat_2_output_0"


def extract(onnx_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(onnx_path))
    tensors: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32:
            tensors[init.name] = np.ascontiguousarray(arr, dtype=np.float32)
        else:
            # Loud-but-harmless: the loader only consumes F32 weights, and the
            # pinned mirror is all-F32, so a non-F32 initializer here means the
            # source model changed — surface it rather than dropping silently.
            print(
                f"  (skipped non-F32 initializer {init.name!r}: dtype={arr.dtype})",
                file=sys.stderr,
            )
    tensors[MATERIALIZED_SINC] = materialize_intermediate(onnx_path, MATERIALIZED_SINC)
    return tensors


def materialize_intermediate(onnx_path: Path, name: str) -> np.ndarray:
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    model.graph.output.append(helper.ValueInfoProto(name=name))
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    spec = session.get_inputs()[0]
    rank = len(spec.shape)
    dims = [
        dim if isinstance(dim, int) and dim > 0 else (16_000 if i == rank - 1 else 1)
        for i, dim in enumerate(spec.shape)
    ]
    dummy = np.zeros(dims, dtype=np.float32)
    output = session.run([name], {spec.name: dummy})[0]
    return np.ascontiguousarray(output, dtype=np.float32)


def write_safetensors(out_path: Path, tensors: dict[str, np.ndarray]) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="path to model.onnx")
    parser.add_argument(
        "--out", required=True, type=Path, help="output pyannote_seg.safetensors path"
    )
    parser.add_argument(
        "--revision",
        default=SOURCE_REVISION,
        help=f"source revision (default {SOURCE_REVISION}); recorded for provenance only",
    )
    args = parser.parse_args(argv)

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    tensors = extract(args.onnx)
    write_safetensors(args.out, tensors)
    print(
        f"wrote {args.out} ({len(tensors)} tensors) from "
        f"{SOURCE_REPO}@{args.revision}",
        file=sys.stderr,
    )
    for name in sorted(tensors):
        print(f"  {name:55s} {list(tensors[name].shape)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
