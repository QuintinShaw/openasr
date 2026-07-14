#!/usr/bin/env python3
"""Bump Formula/openasr.rb's version and per-target sha256 in place.

Usage:
  update-homebrew-formula.py <formula-path> <version> \\
      --sha256 macos-arm64=<64-hex-char-sha256> \\
      --sha256 linux-x86_64=<64-hex-char-sha256> \\
      --sha256 linux-arm64=<64-hex-char-sha256>

Used by .github/workflows/release-core.yml's `update-homebrew-tap` job, once a
release's full asset matrix (release-binaries.yml) has finished uploading.

Only two kinds of line change:
  - the top `version "X.Y.Z"` line
  - each target's `sha256 "..."` line (the one immediately following the
    `url "...-<target>.tar.gz"` line for that target)
`url` lines interpolate `#{version}` and need no edits.

Fails loudly (nonzero exit) if the formula's shape does not match what this
script expects: a target the formula's url lines reference has no --sha256
given for it (would leave a stale hash for that platform paired with the new
version -- a guaranteed checksum-mismatch install failure on that platform),
or a --sha256 was given for a target no url line references. Silently writing
a formula that does not match the intended release would be worse than a red
release job.
"""
import argparse
import re
import sys

# Matched against each line with its trailing "\n" already stripped (see the
# main loop) -- otherwise a trailing `\s*` would greedily swallow that "\n"
# itself, and re-appending "\n" in the replacement would double it into a
# blank line.
VERSION_LINE_RE = re.compile(r'^(\s*version\s+")([^"]+)("\s*)$')
# The target name sits between the literal `#{version}-` interpolation and
# `.tar.gz"` -- anchoring on that literal (rather than "the last `-`-joined
# segment") avoids ambiguity for multi-hyphen targets like `linux-arm64`.
URL_TARGET_RE = re.compile(r'url\s+".*#\{version\}-([A-Za-z0-9][A-Za-z0-9_.-]*)\.tar\.gz"')
SHA_LINE_RE = re.compile(r'^(\s*sha256\s+")([0-9a-f]{64})("\s*)$')


def parse_sha_arg(value):
    target, _, sha = value.partition("=")
    if not target or len(sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise argparse.ArgumentTypeError(
            f"expected TARGET=<64-hex-char-sha256>, got {value!r}"
        )
    return target, sha


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("formula")
    parser.add_argument("version")
    parser.add_argument(
        "--sha256",
        action="append",
        type=parse_sha_arg,
        required=True,
        metavar="TARGET=SHA256",
        dest="shas",
    )
    args = parser.parse_args()
    shas = dict(args.shas)

    with open(args.formula, encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    pending_target = None
    version_replaced = False
    formula_targets = set()
    replaced_targets = set()

    for raw_line in lines:
        line = raw_line[:-1] if raw_line.endswith("\n") else raw_line

        m = VERSION_LINE_RE.match(line)
        if m and not version_replaced:
            out.append(f"{m.group(1)}{args.version}{m.group(3)}\n")
            version_replaced = True
            continue

        m = URL_TARGET_RE.search(line)
        if m:
            pending_target = m.group(1)
            formula_targets.add(pending_target)
            out.append(raw_line)
            continue

        m = SHA_LINE_RE.match(line)
        if m and pending_target is not None and pending_target in shas:
            out.append(f"{m.group(1)}{shas[pending_target]}{m.group(3)}\n")
            replaced_targets.add(pending_target)
            pending_target = None
            continue

        out.append(raw_line)

    if not version_replaced:
        print(f'error: no `version "..."` line found in {args.formula}', file=sys.stderr)
        return 1

    not_updated = formula_targets - replaced_targets
    if not_updated:
        print(
            f"error: {args.formula} references target(s) with no matching "
            f"--sha256 given, would leave a stale hash paired with the new "
            f"version: {sorted(not_updated)}",
            file=sys.stderr,
        )
        return 1

    extra = set(shas) - formula_targets
    if extra:
        print(
            f"error: --sha256 given for target(s) not referenced by any url "
            f"line in {args.formula}: {sorted(extra)}",
            file=sys.stderr,
        )
        return 1

    with open(args.formula, "w", encoding="utf-8") as f:
        f.writelines(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
