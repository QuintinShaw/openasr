#!/usr/bin/env python3
"""Publish OpenASR .oasr packs to Hugging Face.

Default scope is intentionally narrow for the public release lane:
qwen3-asr-0.6b with fp16/q8_0/q4_k. The script writes immutable revision
sidecars under tmp/publish/<model>/ so _manifest.py can generate signed catalogs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _catalog import load as load_publish_catalog
from _file_loaders import atomic_write_text, load_required_json
from _pathlib_helpers import repo_root

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = repo_root(SCRIPT_DIR)
DEFAULT_MODEL = "qwen3-asr-0.6b"
DEFAULT_QUANTS = ("fp16", "q8_0", "q4_k")
ALL_TARGETS = ("hf",)
DEFAULT_TARGETS = ("hf",)
HF_TOKEN_ENV = "HF_TOKEN"
QWEN3_ASR_EXPECTED_GENERAL_ARCHITECTURE = b"qwen3-asr"
QWEN3_ASR_LEGACY_GENERAL_ARCHITECTURE = b"qwen3asr"
XASR_ZIPFORMER_EXPECTED_GENERAL_ARCHITECTURE = b"xasr-zipformer-transducer"
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
# Models cleared for the public release lane.
RELEASE_LANE_MODELS = (
    DEFAULT_MODEL,
    "qwen3-asr-1.7b",
    "moonshine-tiny",
    "xasr-zh-en",
    "cohere-transcribe-03-2026",
    "dolphin-cn-dialect-small",
    "sensevoice-small",
    "whisper-small",
    "whisper-large-v3-turbo",
    "whisper-tiny",
    "whisper-base",
    "whisper-medium",
    "whisper-large-v3",
    "whisper-tiny.en",
    "whisper-base.en",
    "whisper-small.en",
    "whisper-medium.en",
    "wespeaker-voxceleb-resnet34-lm",
    "pyannote-segmentation-3.0",
    "hymt2-1.8b",
    "dolphin-cn-dialect-base",
    "dolphin-small",
    "dolphin-base",
    "parakeet-tdt-0.6b-v3",
)
GGUF_GENERAL_ARCHITECTURE_KEY = b"general.architecture"


def run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        rendered = " ".join(args[:3])
        raise SystemExit(
            f"command failed ({result.returncode}): {rendered}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def work_root(model: str) -> Path:
    return REPO_ROOT / "tmp" / "publish" / model


def expected_general_architecture(model: str) -> bytes | None:
    if model.startswith("qwen3-asr-"):
        return QWEN3_ASR_EXPECTED_GENERAL_ARCHITECTURE
    if model == "xasr-zh-en":
        return XASR_ZIPFORMER_EXPECTED_GENERAL_ARCHITECTURE
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
        raise SystemExit(f"pack missing general.architecture metadata: {pack}")
    window = header[key_index : key_index + 1024]
    if QWEN3_ASR_LEGACY_GENERAL_ARCHITECTURE in window:
        raise SystemExit(f"pack uses legacy qwen general.architecture 'qwen3asr': {pack}")
    if expected not in window:
        raise SystemExit(
            f"pack general.architecture mismatch for {model}: expected {expected.decode()}: {pack}"
        )
    if model == "hymt2-1.8b":
        for marker in HYMT2_REQUIRED_HEADER_MARKERS:
            if marker not in header:
                raise SystemExit(
                    f"pack missing Hy-MT2 required metadata marker {marker.decode(errors='replace')}: {pack}"
                )


def pack_result(model: str, quant: str) -> dict:
    result = load_required_json(work_root(model) / "packs" / f"{model}.{quant}.result.json")
    pack = Path(result["pack"])
    if not pack.exists():
        local_pack = work_root(model) / "packs" / f"{model}-{quant}.oasr"
        pack = local_pack if local_pack.exists() else pack
    if not pack.exists():
        raise SystemExit(f"pack file missing for {model}:{quant}: {pack}")
    if pack.stat().st_size != result["size_bytes"]:
        raise SystemExit(f"pack size mismatch for {model}:{quant}: {pack}")
    validate_pack_runtime_metadata(model, pack)
    return {**result, "pack_path": pack}


def validate_scope(model: str, quants: list[str], catalog_quants: list[str]) -> None:
    if model not in RELEASE_LANE_MODELS:
        raise SystemExit(
            f"this release lane only publishes {', '.join(RELEASE_LANE_MODELS)}, got {model}"
        )
    # A release must carry the model's full catalog-declared quant set — no
    # partial publishes — and nothing the catalog does not declare.
    unknown = sorted(set(quants) - set(catalog_quants))
    missing = sorted(set(catalog_quants) - set(quants))
    if unknown or missing:
        raise SystemExit(
            f"{model} release quants must be exactly {', '.join(catalog_quants)}"
        )


def copy_stage(model: str, quants: list[str], readme: str, stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "README.md").write_text(readme, encoding="utf-8")
    (stage / ".gitattributes").write_text("*.oasr filter=lfs diff=lfs merge=lfs -text\n")
    for quant in quants:
        result = pack_result(model, quant)
        shutil.copy2(result["pack_path"], stage / Path(result["pack_path"]).name)


def hf_readme(model: str) -> str:
    path = work_root(model) / "repo" / "README.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return run([sys.executable, str(SCRIPT_DIR / "render_card.py"), model])


def commit_stage(stage: Path, message: str, *, use_lfs: bool) -> str:
    run(["git", "init", "-b", "main"], cwd=stage)
    if use_lfs:
        run(["git", "lfs", "install", "--local"], cwd=stage)
    run(["git", "add", "."], cwd=stage)
    # Committer identity is env-configurable so a publisher can attribute uploads to
    # their own git/Hugging Face identity without hardcoding a personal address in the
    # open-core repo. Defaults to the project release bot identity.
    committer_name = os.environ.get("OPENASR_PUBLISH_COMMITTER_NAME", "OpenASR Release")
    committer_email = os.environ.get("OPENASR_PUBLISH_COMMITTER_EMAIL", "release@openasr.org")
    git_config = [
        "-c",
        f"user.name={committer_name}",
        "-c",
        f"user.email={committer_email}",
    ]
    commit_cmd = ["commit", "-m", message]
    # Optional OpenPGP signing: set OPENASR_PUBLISH_SIGNING_KEY to a key id to sign
    # the upload commit (e.g. a hardware-token-backed key) so Hugging Face marks it
    # Verified. No key material lives in the repo; only the local env opts in.
    signing_key = os.environ.get("OPENASR_PUBLISH_SIGNING_KEY")
    if signing_key:
        git_config += ["-c", f"user.signingkey={signing_key}", "-c", "commit.gpgsign=true"]
        commit_cmd = ["commit", "-S", "-m", message]
    run(["git", *git_config, *commit_cmd], cwd=stage)
    return run(["git", "rev-parse", "HEAD"], cwd=stage)


def ensure_hf_repo(repo: str, token: str, dry_run: bool) -> None:
    """Create (or reuse) the HF repo, always **private** at creation time.

    Publish never flips a repo public on its own -- `release_public` in
    models-core.toml only gates whether `_manifest.py --public` may list the
    model in the *catalog*; it says nothing about Hugging Face repo
    visibility. Making an HF repo public is a separate, deliberate step taken
    manually (or via a dedicated script) after the catalog-listing gate has
    already passed, so a model can never go public on HF purely because its
    catalog metadata flipped a bit.
    """
    if dry_run:
        return
    args = [
        "hf", "repo", "create", repo, "--type", "model", "--exist-ok", "--token", token,
        "--private",
    ]
    run(args)


def push_git(stage: Path, remote: str, dry_run: bool, branch: str = "main") -> str:
    revision = run(["git", "rev-parse", "HEAD"], cwd=stage)
    if dry_run:
        return revision
    run(["git", "remote", "add", "origin", remote], cwd=stage)
    run(["git", "push", "--force", "origin", f"HEAD:{branch}"], cwd=stage)
    return revision


def hf_remote(repo: str, token: str) -> str:
    return f"https://oauth2:{token}@huggingface.co/{repo}.git"


def publish_hf(model: str, entry: dict, quants: list[str], dry_run: bool) -> str:
    token = os.environ.get(HF_TOKEN_ENV)
    if not token and not dry_run:
        raise SystemExit(f"{HF_TOKEN_ENV} is required to publish Hugging Face artifacts")
    repo = entry["hf_repo"]
    with tempfile.TemporaryDirectory(prefix=f"openasr-hf-{model}.") as tmp:
        stage = Path(tmp)
        copy_stage(model, quants, hf_readme(model), stage)
        commit_stage(stage, f"publish {model} OpenASR packs", use_lfs=not dry_run)
        ensure_hf_repo(repo, token or "", dry_run)
        revision = push_git(stage, hf_remote(repo, token or "DRY_RUN_TOKEN"), dry_run)
    if not dry_run:
        atomic_write_text(work_root(model) / "hf_repo.txt", repo + "\n")
        atomic_write_text(work_root(model) / "hf_revision.txt", revision + "\n")
    return revision


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant", action="append", dest="quants")
    parser.add_argument("--target", action="append", choices=ALL_TARGETS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    catalog = load_publish_catalog()
    if args.model not in catalog:
        raise SystemExit(f"unknown model: {args.model}")
    entry = catalog[args.model]
    quants = args.quants or list(entry["quants"])
    validate_scope(args.model, quants, list(entry["quants"]))
    targets = args.target or list(DEFAULT_TARGETS)
    if "hf" in targets:
        revision = publish_hf(args.model, entry, quants, args.dry_run)
        print(f"hf {entry['hf_repo']} {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
