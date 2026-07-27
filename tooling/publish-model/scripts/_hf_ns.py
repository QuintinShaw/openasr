#!/usr/bin/env python3
"""Resolve the Hugging Face namespace to actually push under.

  _hf_ns.py <desired-namespace> [--strict]    # prints the namespace to use (stdout)

The catalog may brand a repo as `openasr/...`, but a push must target a namespace
the $HF_TOKEN actually owns. This returns the canonical-cased match if `desired`
is the token's user or one of its member orgs (case-insensitive); otherwise it
falls back to the token owner's username and warns on stderr. Public releases
must pass --strict so the catalog namespace cannot silently diverge from the
uploaded namespace.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("desired")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    desired = args.desired
    info = HfApi(token=os.environ["HF_TOKEN"]).whoami()
    user = info.get("name", "")
    orgs = [o.get("name", "") for o in info.get("orgs", [])]
    candidates = {c.lower(): c for c in [user, *orgs]}
    match = candidates.get(desired.lower())
    if match:
        print(match)
    else:
        print(
            f"[publish] namespace '{desired}' not owned by this token "
            f"(owner='{user}', orgs={orgs}); falling back to '{user}'.",
            file=sys.stderr,
        )
        if args.strict:
            raise SystemExit("public releases cannot fall back to the token owner namespace")
        print(user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
