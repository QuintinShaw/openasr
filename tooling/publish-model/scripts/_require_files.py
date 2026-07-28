#!/usr/bin/env python3
"""Fail-closed completeness + integrity gate for staged/converted model files.

  _require_files.py <root> <pattern> [<pattern> ...]

Backs the `require_files` bash helper in lib.sh (used by download.sh and
convert.sh). A glob match alone is not enough: a tokenless fetch of a private
HF repo can land a small captured HTTP response (an error page, a login
redirect body) at exactly the filename the pipeline expected, and a plain
`glob.glob()` truth check reports that as present. This module adds two
independent checks on every matched file:

  1. A per-category minimum size. A weight/checkpoint file (safetensors, .pt,
     .onnx, .gguf) that is not multi-megabyte is never real; a small JSON/text
     config can legitimately be tiny, so its floor stays low.
  2. A content sniff for known captured-failure signatures (HTML error pages,
     JSON error envelopes, auth-rejection text), independent of size, so a
     moderately sized error body does not slip through on size alone.

Exits 0 when every pattern matches at least one file under <root> and every
matched file clears both checks; exits 1 with every problem listed on stderr
otherwise. Never partial -- this is a hard gate, not a warning.
"""
from __future__ import annotations

import glob
import os
import sys

# --- size floors, by category ------------------------------------------
#
# One global magic number cannot serve both classes of required file: a
# weight shard that has shrunk to a few KB is always corrupt (real shards are
# multi-megabyte at minimum), while a legitimate config/tokenizer JSON can be
# genuinely tiny (a minimal special_tokens_map.json can be well under 200
# bytes). Categorizing by extension gives each its own floor sized to what a
# REAL file of that kind looks like, instead of one number that is either too
# loose for weights or too tight for small configs.

# Weight/checkpoint formats the harness imports (safetensors shards, .pt /
# .pth.tar checkpoints, .onnx exports, the one vendored .gguf repackage).
# Every real one is realistically tens of megabytes or larger; 1 MiB only
# excludes literal garbage, so it can never misfire on a legitimate model.
WEIGHT_MIN_BYTES = 1024 * 1024
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".onnx", ".gguf", ".bin")
WEIGHT_COMPOUND_SUFFIXES = (".pth.tar",)

# Numeric feature-normalization archives (Kaldi CMVN stats, npz stat blobs).
# Small next to weights, but a real one always carries a per-dimension
# mean/variance array -- at least a few hundred bytes for any model with a
# non-trivial feature dimension.
STATS_MIN_BYTES = 256
STATS_SUFFIXES = (".npz", ".ark", ".mvn")

# Catch-all for text/JSON/YAML/vocab files (config.json, tokenizer.json,
# *.txt, *.model, *.yaml, a sharded-weights index.json, license text, ...).
# Deliberately low: several of these are legitimately tiny, so this floor
# exists only to catch a captured HTTP error page landing on the expected
# filename (the class of incident this module fixes -- see module docstring
# and the 2026-07 qwen3-forced-aligner q4_k rebuild, where a tokenless fetch
# of a private repo landed a 29-byte error page under a required filename).
#
# Sized against the smallest LEGITIMATE file in this category, not against the
# incident: this repo's own staged inputs include a 21-byte hf_repo.txt and a
# 29-byte one, and 29 is exactly the size of the captured error page that
# started all this. A floor tuned to catch that page by size would reject real
# files of the same length -- it cannot separate them, because length is not
# what distinguishes them. Only emptiness is unambiguous here, so that is all
# this floor claims; THE CONTENT SNIFF BELOW IS THE DEFENSE for this category,
# and it catches the incident body by its text regardless of length.
DEFAULT_MIN_BYTES = 1


def min_bytes_for(path: str) -> int:
    lower = path.lower()
    if lower.endswith(WEIGHT_COMPOUND_SUFFIXES) or lower.endswith(WEIGHT_SUFFIXES):
        return WEIGHT_MIN_BYTES
    if lower.endswith(STATS_SUFFIXES):
        return STATS_MIN_BYTES
    return DEFAULT_MIN_BYTES


# --- content sniff: known captured-failure signatures --------------------
#
# Independent of size: a moderately sized HTML/JSON error body (an
# authenticated-only repo's login page, an API error envelope) can clear the
# byte floor above yet still not be the file the pipeline asked for. Sniffing
# the leading bytes for known error-response shapes catches that case
# regardless of category. This is a targeted denylist for captured failure
# responses, not a general HTML/JSON validator -- the importer and
# `openasr verify` downstream are what validate the file is actually usable.
SNIFF_WINDOW_BYTES = 4096
ERROR_SIGNATURES = (
    b"<!doctype html",
    b"<html",
    b'"error":',
    b"invalid username or password",
    b"repository not found",
    b"401 unauthorized",
    b"403 forbidden",
    b"404 not found",
    b"<error>",  # S3/GCS-style XML error envelopes
    b"access denied",
)


def looks_like_error_page(path: str) -> str | None:
    """Return the matched signature (decoded, for the diagnostic) or None."""
    try:
        with open(path, "rb") as handle:
            window = handle.read(SNIFF_WINDOW_BYTES)
    except OSError:
        return None
    lowered = window.lower()
    for signature in ERROR_SIGNATURES:
        if signature in lowered:
            return signature.decode("ascii", errors="replace")
    return None


def check_required_files(root: str, patterns: list[str]) -> list[str]:
    """Return a list of human-readable problems; empty means everything
    required is present, large enough for its category, and does not look
    like a captured error response."""
    problems: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(root, pattern)))
        if not matches:
            problems.append(f"missing: {pattern}")
            continue
        for match in matches:
            if not os.path.isfile(match):
                continue  # a directory match is not this gate's concern
            rel = os.path.relpath(match, root)
            size = os.path.getsize(match)
            floor = min_bytes_for(match)
            if size < floor:
                problems.append(
                    f"too small: {rel} is {size} byte(s), expected at least {floor} "
                    f"(matched required pattern {pattern!r}) -- looks like a truncated "
                    "download or a captured error response, not the real file"
                )
                continue
            signature = looks_like_error_page(match)
            if signature is not None:
                problems.append(
                    f"looks like an error page: {rel} ({size} byte(s), matched required "
                    f"pattern {pattern!r}) contains the signature {signature!r}"
                )
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("usage: _require_files.py <root> <pattern> [<pattern> ...]\n")
        return 2
    root, patterns = argv[0], argv[1:]
    if not patterns:
        return 0
    problems = check_required_files(root, patterns)
    if problems:
        sys.stderr.write(
            f"required file(s) failed the completeness/integrity gate under {root}:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
