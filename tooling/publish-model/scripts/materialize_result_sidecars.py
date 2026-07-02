#!/usr/bin/env python3
"""Materialize publish result sidecars from existing .oasr packs.

This is a recovery/bridging tool for packs that already exist under
tmp/publish/<model>/packs/ but were produced before convert.sh emitted
<model>.<quant>.result.json. It is deliberately fail-closed: missing packs,
non-.oasr pack paths, invalid existing sidecars, and existing sidecars whose
contents differ from the observed pack all fail without rewriting them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

from _catalog import CATALOG_CORE, QUANT_METADATA, load as load_publish_catalog
from _file_loaders import atomic_write_json, load_json, load_toml
from _pathlib_helpers import repo_root

SCRIPT_DIR = Path(__file__).resolve().parent
CATALOG = CATALOG_CORE
QWEN3_ASR_EXPECTED_GENERAL_ARCHITECTURE = b"qwen3-asr"
QWEN3_ASR_LEGACY_GENERAL_ARCHITECTURE = b"qwen3asr"
HYMT2_EXPECTED_GENERAL_ARCHITECTURE = b"hunyuan-dense"
HYMT2_REQUIRED_HEADER_MARKERS = (
    b"openasr.model.kind",
    b"translation-model",
    b"openasr.translation.source_langs",
    b"openasr.translation.target_langs",
    b"openasr.upstream.base_revision",
    b"9a341cd1b679d3efd23b46e847b01745a71ed792",
    b"openasr.upstream.gguf_revision",
    b"1cd5208700acedef4ef93019b6cfc148b8522d45",
    b"openasr.license.files",
    b"LICENSE.txt",
    b"NOTICE.openasr.txt",
)
GGUF_GENERAL_ARCHITECTURE_KEY = b"general.architecture"


class ResultSidecarError(RuntimeError):
    """Raised when a sidecar cannot be materialized safely."""


def load_catalog(path: Path) -> dict:
    if path == CATALOG:
        return load_publish_catalog()
    return load_toml(path)


def catalog_quants(catalog_path: Path, model: str) -> list[str]:
    catalog = load_catalog(catalog_path)
    try:
        quants = catalog[model]["quants"]
    except KeyError as error:
        raise ResultSidecarError(f"unknown model in publish catalog: {model}") from error
    if not isinstance(quants, list) or not all(isinstance(quant, str) for quant in quants):
        raise ResultSidecarError(f"invalid quant list for model: {model}")
    return quants


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_general_architecture(model: str) -> bytes | None:
    if model.startswith("qwen3-asr-"):
        return QWEN3_ASR_EXPECTED_GENERAL_ARCHITECTURE
    if model == "hymt2-1.8b":
        return HYMT2_EXPECTED_GENERAL_ARCHITECTURE
    return None


def validate_pack_runtime_metadata(model: str, pack: Path) -> None:
    expected = expected_general_architecture(model)
    if expected is None:
        return
    with pack.open("rb") as handle:
        header = handle.read(4 * 1024 * 1024)
    key_index = header.find(GGUF_GENERAL_ARCHITECTURE_KEY)
    if key_index < 0:
        raise ResultSidecarError(f"pack missing general.architecture metadata: {pack}")
    window = header[key_index : key_index + 1024]
    if QWEN3_ASR_LEGACY_GENERAL_ARCHITECTURE in window:
        raise ResultSidecarError(
            f"pack uses legacy qwen general.architecture 'qwen3asr': {pack}"
        )
    if expected not in window:
        raise ResultSidecarError(
            f"pack general.architecture mismatch for {model}: expected {expected.decode()}: {pack}"
        )
    if model == "hymt2-1.8b":
        for marker in HYMT2_REQUIRED_HEADER_MARKERS:
            if marker not in header:
                raise ResultSidecarError(
                    f"pack missing Hy-MT2 required metadata marker {marker.decode(errors='replace')}: {pack}"
                )


def pack_path(repo_root: Path, model: str, quant: str) -> Path:
    return repo_root / "tmp" / "publish" / model / "packs" / f"{model}-{quant}.oasr"


def result_path(repo_root: Path, model: str, quant: str) -> Path:
    return repo_root / "tmp" / "publish" / model / "packs" / f"{model}.{quant}.result.json"


def build_sidecar(model: str, quant: str, pack: Path) -> dict:
    if quant not in QUANT_METADATA:
        raise ResultSidecarError(f"unsupported quant id: {quant}")
    if pack.suffix != ".oasr":
        raise ResultSidecarError(f"pack must use .oasr extension: {pack}")
    if not pack.exists():
        raise ResultSidecarError(f"pack missing: {pack}")
    if not pack.is_file():
        raise ResultSidecarError(f"pack is not a file: {pack}")
    size = pack.stat().st_size
    if size <= 0:
        raise ResultSidecarError(f"pack is empty: {pack}")
    validate_pack_runtime_metadata(model, pack)
    return {
        "model": model,
        "quant": quant,
        "cli_token": QUANT_METADATA[quant].cli_token,
        "pack": str(pack),
        "size_bytes": size,
        "sha256": sha256_file(pack),
    }


def read_existing(path: Path) -> dict:
    try:
        data = load_json(path)
    except json.JSONDecodeError as error:
        raise ResultSidecarError(f"invalid existing result sidecar: {path}: {error}") from error
    if not isinstance(data, dict):
        raise ResultSidecarError(f"existing result sidecar is not an object: {path}")
    return data


def write_or_verify_result(result: Path, expected: dict) -> None:
    if result.exists():
        existing = read_existing(result)
        if existing != expected:
            relocated = {**existing, "pack": expected["pack"]}
            if relocated == expected:
                atomic_write_json(result, expected, compact=True)
                return
            raise ResultSidecarError(f"existing result sidecar mismatch: {result}")
        return
    atomic_write_json(result, expected, compact=True)


def materialize_quant(repo_root: Path, model: str, quant: str) -> Path:
    pack = pack_path(repo_root, model, quant)
    result = result_path(repo_root, model, quant)
    sidecar = build_sidecar(model, quant, pack)
    result.parent.mkdir(parents=True, exist_ok=True)
    write_or_verify_result(result, sidecar)
    return result


def materialize_model(
    *,
    repo_root: Path,
    catalog_path: Path,
    model: str,
    quants: Iterable[str] | None = None,
) -> list[Path]:
    selected = list(quants) if quants is not None else catalog_quants(catalog_path, model)
    known = set(catalog_quants(catalog_path, model))
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ResultSidecarError(f"quant not in publish catalog for {model}: {', '.join(unknown)}")
    return [materialize_quant(repo_root, model, quant) for quant in selected]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--repo-root", type=Path, default=repo_root(SCRIPT_DIR))
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--quant", action="append", dest="quants")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        results = materialize_model(
            repo_root=args.repo_root.resolve(),
            catalog_path=args.catalog,
            model=args.model,
            quants=args.quants,
        )
    except ResultSidecarError as error:
        raise SystemExit(str(error)) from None
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
