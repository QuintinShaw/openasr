#!/usr/bin/env python3
"""Print the current immutable commit sha for a Hugging Face model repo."""
from __future__ import annotations

import os
import re
import sys

from huggingface_hub import HfApi


def main(argv: list[str]) -> int:
    repo_id = argv[0]
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(repo_id)
    sha = getattr(info, "sha", "") or ""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise SystemExit(f"could not resolve immutable commit sha for {repo_id}")
    print(sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
