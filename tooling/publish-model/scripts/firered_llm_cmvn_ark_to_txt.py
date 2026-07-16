#!/usr/bin/env python3
"""Convert a FireRedASR2-LLM `cmvn.ark` (Kaldi binary double-matrix CMVN
accumulator) into the Kaldi text-matrix format the existing Rust
`firered_aed::package_import::parse_kaldi_cmvn_stats` (and, by direct code
reuse, `firered_llm::package_import`) already parses.

Why this step exists: `crates/openasr-core/src/models/firered_aed/
package_import.rs::parse_kaldi_cmvn_stats` reads Kaldi's *text* matrix format
(`std::fs::read_to_string` + a `[ ... ]`-delimited, whitespace-separated float
matrix -- the output of Kaldi's `copy-matrix --binary=false`). FireRedASR2-LLM
only ships the *binary* `cmvn.ark`; this script bridges the two without adding
a `kaldiio` dependency or changing the Rust parser.

Byte-level format (reverse-engineered and verified against the real 1311-byte
`cmvn.ark` during stage-1 reconnaissance; see `scratchpad/fr2/T1-findings.md`
S:4 for the full derivation)::

    offset  0: b"\\x00B"        (2 bytes)  Kaldi binary-mode marker
    offset  2: b"DM "           (3 bytes)  token "DM" (double matrix) + space
    offset  5: 0x04              (1 byte)   "next 4 bytes are an int32"
    offset  6: int32 LE rows     (4 bytes)  = 2 (sum row, sum-of-squares row)
    offset 10: 0x04              (1 byte)
    offset 11: int32 LE cols     (4 bytes)  = feature_dim + 1 (frame count column)
    offset 15: rows*cols*8 bytes            row-major float64 payload
                 row 0 = per-dim sum          (+ sums[dim]   = frame count)
                 row 1 = per-dim sum-of-squares (+ sum_sq[dim] = frame count)

Example::

    python3 tooling/publish-model/scripts/firered_llm_cmvn_ark_to_txt.py \\
        --ark tmp-weights/fr2/cmvn.ark \\
        --out tmp-weights/fr2/derived/cmvn.txt
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

BINARY_MARKER = b"\x00B"
MATRIX_TOKEN = b"DM "
INT32_SIZE_MARKER = 0x04


class CmvnArkFormatError(ValueError):
    """Raised when `cmvn.ark` does not match the expected Kaldi binary layout."""


def parse_cmvn_ark(data: bytes) -> list[list[float]]:
    """Parse a Kaldi binary double-matrix `cmvn.ark` into `rows` lists of
    `cols` python floats (row-major), matching the byte layout documented
    above. Raises `CmvnArkFormatError` on any structural mismatch (fail
    closed rather than silently misparsing a differently-shaped upstream
    file)."""
    if data[0:2] != BINARY_MARKER:
        raise CmvnArkFormatError(
            f"cmvn.ark does not start with the Kaldi binary marker {BINARY_MARKER!r}: "
            f"got {data[0:2]!r}"
        )
    if data[2:5] != MATRIX_TOKEN:
        raise CmvnArkFormatError(
            f"cmvn.ark token at offset 2 is not {MATRIX_TOKEN!r} (double matrix): "
            f"got {data[2:5]!r}"
        )
    pos = 5
    if data[pos] != INT32_SIZE_MARKER:
        raise CmvnArkFormatError(
            f"cmvn.ark expected an int32 size marker (0x04) at offset {pos}, got {data[pos]:#x}"
        )
    (rows,) = struct.unpack_from("<i", data, pos + 1)
    pos += 5
    if data[pos] != INT32_SIZE_MARKER:
        raise CmvnArkFormatError(
            f"cmvn.ark expected an int32 size marker (0x04) at offset {pos}, got {data[pos]:#x}"
        )
    (cols,) = struct.unpack_from("<i", data, pos + 1)
    pos += 5
    if rows <= 0 or cols <= 0:
        raise CmvnArkFormatError(f"cmvn.ark declared non-positive shape ({rows}, {cols})")
    payload_bytes = rows * cols * 8
    expected_total = pos + payload_bytes
    if len(data) != expected_total:
        raise CmvnArkFormatError(
            f"cmvn.ark size {len(data)} does not match header-declared "
            f"{rows}x{cols} float64 payload (expected {expected_total} bytes total)"
        )
    values = struct.unpack_from(f"<{rows * cols}d", data, pos)
    return [list(values[r * cols : (r + 1) * cols]) for r in range(rows)]


def format_kaldi_text_matrix(rows: list[list[float]]) -> str:
    """Render `rows` as a Kaldi text matrix: `[ v v ... \\n v v ... ]`, one
    row per line, `%.17g` per value (round-trips a float64 exactly). Matches
    the `[`...`]`-delimited, whitespace-separated shape the existing Rust
    `parse_kaldi_cmvn_stats` text parser expects."""
    lines = []
    for row_index, row in enumerate(rows):
        rendered = " ".join(f"{value:.17g}" for value in row)
        if row_index == 0:
            lines.append(f"[ {rendered}")
        elif row_index == len(rows) - 1:
            lines.append(f"  {rendered} ]")
        else:
            lines.append(f"  {rendered}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Kaldi binary cmvn.ark into Kaldi text-matrix format."
    )
    parser.add_argument("--ark", required=True, type=Path, help="input cmvn.ark (binary)")
    parser.add_argument("--out", required=True, type=Path, help="output cmvn.txt (text)")
    args = parser.parse_args(argv)

    if not args.ark.is_file():
        raise SystemExit(f"cmvn.ark not found: {args.ark}")

    data = args.ark.read_bytes()
    try:
        rows = parse_cmvn_ark(data)
    except CmvnArkFormatError as error:
        raise SystemExit(f"failed to parse {args.ark}: {error}") from error

    if len(rows) != 2:
        raise SystemExit(f"cmvn.ark must have exactly 2 rows (sum, sum-of-squares), got {len(rows)}")

    text = format_kaldi_text_matrix(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)

    dim = len(rows[0]) - 1
    count = rows[0][dim]
    print(
        f"wrote {args.out} ({len(rows)}x{len(rows[0])} matrix, dim={dim}, count={count:.0f})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
