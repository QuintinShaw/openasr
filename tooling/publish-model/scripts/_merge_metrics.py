#!/usr/bin/env python3
"""Merge the CPU and Metal bench-suite baselines into one metrics.json.

  _merge_metrics.py <model-id> <cpu_baseline.json> <metal_baseline.json>

Reads two SuiteBaseline JSON files (each: {entries:[{quant, rtf, peak_rss_bytes,
wer, transcription_text, ...}]}), keyed by the internal quant id, and emits a
compact per-quant metrics object the card renderer consumes. Pack byte size
comes from the on-disk .oasr (the authoritative artifact), not from any subagent
text. JFK ΔWER is measured against this model's fp16 transcript, not an external
human reference.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from _pathlib_helpers import repo_root


# The skill lives OUTSIDE the OpenASR repo, so resolve the repo root from the
# caller's cwd (the skill runs from the repo root) with OPENASR_REPO_ROOT as the
# explicit override -- never from this file's location, which is in the skill dir.
_REPO_ROOT_ENV = os.environ.get("OPENASR_REPO_ROOT")
REPO_ROOT = Path(_REPO_ROOT_ENV) if _REPO_ROOT_ENV else repo_root(Path.cwd())


def load_entries(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {e["quant"]: e for e in data.get("entries", [])}


def pack_path(model: str, quant: str) -> Path:
    return REPO_ROOT / "tmp" / "publish" / model / "packs" / f"{model}-{quant}.oasr"


def pack_size(model: str, quant: str) -> int | None:
    pack = pack_path(model, quant)
    return pack.stat().st_size if pack.exists() else None


def pack_sha256(model: str, quant: str) -> str | None:
    # Binds each quant's bench numbers to the exact pack bytes they were
    # measured on: _manifest.py refuses metrics whose sha256 does not match the
    # upload sidecar, so a re-converted pack forces a re-bench instead of
    # silently signing stale RTF/RAM numbers into the catalog.
    pack = pack_path(model, quant)
    if not pack.exists():
        return None
    digest = hashlib.sha256()
    with pack.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ASCII word-runs count as one token each (word error rate); CJK ideographs
# have no whitespace word boundaries, so each character is its own token
# (character error rate). Mixing both in one pass keeps a code-mixed zh/en
# transcript honest and stops an all-CJK transcript from tokenizing to nothing
# (which would silently drop the drift column to n/a). Punctuation/whitespace
# are separators in both scripts.
_TOKEN_RE = re.compile(r"[0-9a-z]+|[㐀-䶿一-鿿豈-﫿]")


def words(text: str | None) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def levenshtein(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            cost = 0 if left_value == right_value else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def wer_vs_reference(hypothesis: str | None, reference: str | None) -> tuple[float, int, int] | None:
    hyp_words = words(hypothesis)
    ref_words = words(reference)
    if not ref_words:
        return None
    errors = levenshtein(hyp_words, ref_words)
    return errors / len(ref_words), errors, len(ref_words)


def transcript_for_quant(quant: str, cpu: dict[str, dict], metal: dict[str, dict]) -> str | None:
    return (
        cpu.get(quant, {}).get("transcription_text")
        or metal.get(quant, {}).get("transcription_text")
    )


def main(argv: list[str]) -> int:
    model, cpu_path, metal_path = argv[0], argv[1], argv[2]
    cpu = load_entries(cpu_path)
    metal = load_entries(metal_path)
    quants_order = list(cpu) or list(metal)
    fp16_text = transcript_for_quant("fp16", cpu, metal)

    out: dict = {"model": model, "quants": {}}
    for q in quants_order:
        c = cpu.get(q, {})
        m = metal.get(q, {})
        out["quants"][q] = {
            "size_bytes": pack_size(model, q),
            "sha256": pack_sha256(model, q),
            "peak_rss_bytes": c.get("peak_rss_bytes") or m.get("peak_rss_bytes"),
            "rtf_cpu": c.get("rtf"),
            "rtf_metal": m.get("rtf"),
            "wer": c.get("wer") if c.get("wer") is not None else m.get("wer"),
            "audio_seconds": c.get("audio_seconds") or m.get("audio_seconds"),
        }
        jfk_wer = wer_vs_reference(transcript_for_quant(q, cpu, metal), fp16_text)
        if jfk_wer is not None:
            value, errors, ref_words = jfk_wer
            out["quants"][q]["jfk_wer_vs_fp16"] = value
            out["quants"][q]["jfk_wer_errors_vs_fp16"] = errors
            out["quants"][q]["jfk_wer_ref_words"] = ref_words
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
