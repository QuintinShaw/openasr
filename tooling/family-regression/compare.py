#!/usr/bin/env python3
"""Compare a transcript against committed golden variants.

Usage:
    compare.py --transcript <path> --golden-dir <dir> --strategy <exact|normalized|wer:<t>>

The golden dir holds one or more accepted variants (``golden*.txt``), e.g. a
shared ``golden.txt`` plus platform-specific ``golden.linux-x86_64.txt`` when
CPU floating-point differences change the decode. The comparison passes when
the transcript matches ANY variant under the configured strategy.

Strategies (progressively looser; every strategy first tries the stricter
ones, so an exact hit always passes):

- ``exact``       trimmed byte equality
- ``normalized``  casefold, strip punctuation, collapse whitespace
- ``wer:<t>``     word (or CJK char) error rate vs the closest variant <= t
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def tokenize(text: str) -> list[str]:
    """Split normalized text into WER tokens; CJK ideographs count per-char."""
    tokens: list[str] = []
    for word in normalize(text).split():
        run = ""
        for ch in word:
            if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
                if run:
                    tokens.append(run)
                    run = ""
                tokens.append(ch)
            else:
                run += ch
        if run:
            tokens.append(run)
    return tokens


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, start=1):
            cost = 0 if ta == tb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def wer(hyp: str, ref: str) -> float:
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(hyp_tokens, ref_tokens) / len(ref_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()

    strategy = args.strategy
    threshold = None
    if strategy.startswith("wer:"):
        threshold = float(strategy.split(":", 1)[1])
        strategy = "wer"
    elif strategy not in ("exact", "normalized"):
        print(f"unknown strategy: {args.strategy}", file=sys.stderr)
        return 2

    transcript = args.transcript.read_text(encoding="utf-8").strip()
    goldens = sorted(args.golden_dir.glob("golden*.txt"))
    if not goldens:
        print(f"no golden*.txt in {args.golden_dir}", file=sys.stderr)
        return 2

    print(f"transcript: {transcript!r}")
    best: tuple[float, Path] | None = None
    for path in goldens:
        golden = path.read_text(encoding="utf-8").strip()
        if transcript == golden:
            print(f"PASS exact match: {path.name}")
            return 0
        if strategy in ("normalized", "wer") and normalize(transcript) == normalize(golden):
            print(f"PASS normalized match: {path.name}")
            return 0
        if strategy == "wer":
            rate = wer(transcript, golden)
            if best is None or rate < best[0]:
                best = (rate, path)

    if strategy == "wer" and best is not None:
        rate, path = best
        assert threshold is not None
        if rate <= threshold:
            print(f"PASS wer {rate:.4f} <= {threshold} vs {path.name}")
            return 0
        print(f"FAIL wer {rate:.4f} > {threshold} vs {path.name}", file=sys.stderr)
    else:
        print(f"FAIL no {strategy} match among {[p.name for p in goldens]}", file=sys.stderr)
    for path in goldens:
        print(f"  {path.name}: {path.read_text(encoding='utf-8').strip()!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
