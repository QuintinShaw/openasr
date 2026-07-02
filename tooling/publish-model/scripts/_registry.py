#!/usr/bin/env python3
"""Write/refresh the local model-registry card for a published model.

  _registry.py <model-id>

Produces a schema-valid model-registry/models/<registry_id>.toml (matching the
Rust ModelCard struct) from the catalog + measured metrics. The card carries a
single default variant (format=oasr, quantization=recommended); per-quant packs
live in the HF repo, not as separate registry cards.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _catalog import REGISTRY_CARD_DEFAULTS, languages_for_model, load as load_publish_catalog
from _file_loaders import atomic_write_text
from _pathlib_helpers import repo_root


PUB = Path(__file__).resolve().parent
REPO_ROOT = repo_root(PUB)


def q(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'


def toml_str_array(items: list[str]) -> str:
    return "[" + ", ".join(q(item) for item in items) + "]"


def main(argv: list[str]) -> int:
    model = argv[0]
    catalog = load_publish_catalog()[model]
    rid = catalog["registry_id"]

    published_repo_path = REPO_ROOT / "tmp" / "publish" / model / "hf_repo.txt"
    hf_repo = catalog["hf_repo"]
    if published_repo_path.exists():
        published_repo = published_repo_path.read_text().strip()
        if published_repo:
            hf_repo = published_repo

    lines = [
        f"id = {q(rid)}",
        f"default_variant = {q(REGISTRY_CARD_DEFAULTS['default_variant'])}",
        f"display_name = {q(catalog['display_name'] + ' (OpenASR pack)')}",
        f"languages = {toml_str_array(languages_for_model(catalog))}",
        f"size = {q(catalog['size'])}",
        f"license = {q(catalog['license_name'])}",
        f"source = {q('Published OpenASR packs: https://huggingface.co/' + hf_repo)}",
        "",
        "[variant]",
        f"quantization = {q(catalog['recommended_quant'])}",
        "",
    ]
    out = REPO_ROOT / "model-registry" / "models" / f"{rid}.toml"
    atomic_write_text(out, "\n".join(lines))
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
