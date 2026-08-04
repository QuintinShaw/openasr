#!/usr/bin/env python3
"""Publish OpenASR .oasr packs to Hugging Face.

Default scope is intentionally narrow for the public release lane:
qwen3-asr-0.6b with fp16/q8_0/q4_k. The script writes immutable revision
sidecars under tmp/publish/<model>/ so _manifest.py can generate signed catalogs.

Before any upload, every pack is staged by the client's install-time preflight
(`openasr model-pack preflight`, fail-closed). The CLI verifies the source,
creates the destination, and seals it read-only; this publisher only validates
the receipt and never copies pack bytes itself.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
PREFLIGHT_RECEIPT_SCHEMA = "openasr.model-pack-preflight.v1"
BUILD_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# Models cleared for this pack-publish lane. Repository visibility and public
# catalog listing remain separately gated; DiariZen stays `release_public=false`.
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
    "redimnet2-b6-cn",
    "pyannote-segmentation-3.0",
    "diarizen-large-s80-v2",
    "hymt2-1.8b",
    "dolphin-cn-dialect-base",
    "dolphin-small",
    "dolphin-base",
    "parakeet-tdt-0.6b-v3",
    "firered-aed-l-v2",
    "qwen3-forced-aligner-0.6b",
    "firered-punc",
    "firered2-llm",
    "mimo-v2.5-asr",
    "moss-transcribe-diarize",
    "funasr-nano",
    "granite-speech-4.1-2b",
)


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
    return {**result, "pack_path": pack}


def openasr_release_binary() -> Path:
    override = os.environ.get("OPENASR_PUBLISH_OPENASR_BIN")
    if override:
        path = Path(override)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(
                f"OPENASR_PUBLISH_OPENASR_BIN does not point at an executable binary: {override}"
            )
        return path
    path = REPO_ROOT / "target" / "release" / "openasr"
    if not path.is_file() or not os.access(path, os.X_OK):
        # Fail closed: a missing binary is a toolchain fault, never a reason to
        # skip the receipt/staging contract.
        raise SystemExit(
            f"release binary missing: {path}; build it with "
            "`cargo build --release -p openasr-cli` before publishing "
            "(the preflight staging gate refuses to run without it)"
        )
    return path


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


def reject_staged_pack(destination: Path, message: str) -> None:
    """Remove a CLI-verified stage that fails release-identity binding."""
    if destination.exists():
        try:
            destination.chmod(0o600)
        except OSError:
            pass
        try:
            destination.unlink()
        except OSError as error:
            raise SystemExit(f"{message}; could not remove rejected stage: {error}") from error
    raise SystemExit(message)


def copy_stage(model: str, entry: dict, quants: list[str], readme: str, stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "README.md").write_text(readme, encoding="utf-8")
    (stage / ".gitattributes").write_text("*.oasr filter=lfs diff=lfs merge=lfs -text\n")
    binary = openasr_release_binary()
    for quant in quants:
        result = pack_result(model, quant)
        source = result["pack_path"]
        destination = stage / source.name
        receipt_output = run(
            [
                str(binary),
                "model-pack",
                "preflight",
                str(source),
                "--stage",
                str(destination),
                "--json",
            ]
        )
        try:
            receipt = json.loads(receipt_output)
        except json.JSONDecodeError as error:
            reject_staged_pack(
                destination,
                f"preflight receipt is not one JSON object for {source}: {error}"
            )
        if not isinstance(receipt, dict):
            reject_staged_pack(
                destination, f"preflight receipt is not a JSON object for {source}"
            )
        if receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA:
            reject_staged_pack(
                destination,
                f"preflight receipt schema mismatch for {source}: "
                f"expected {PREFLIGHT_RECEIPT_SCHEMA}, got {receipt.get('schema')!r}"
            )
        expected_route = "asr" if entry.get("kind", "asr-model") == "asr-model" else "aux"
        if receipt.get("route") != expected_route:
            reject_staged_pack(
                destination,
                f"preflight receipt route mismatch for {source}: "
                f"expected {expected_route}, got {receipt.get('route')!r}"
            )
        if receipt.get("catalog_family_id") != entry["family"]:
            reject_staged_pack(
                destination,
                f"preflight receipt family mismatch for {source}: "
                f"expected {entry['family']}, got {receipt.get('catalog_family_id')!r}"
            )
        build_commit = receipt.get("build_commit")
        if not isinstance(build_commit, str) or BUILD_COMMIT_RE.fullmatch(build_commit) is None:
            reject_staged_pack(
                destination,
                f"preflight receipt has no pinned 40-hex build_commit for {source}: "
                f"{build_commit!r}; rebuild through convert.sh with OPENASR_BUILD_COMMIT"
            )
        if receipt.get("size_bytes") != result["size_bytes"]:
            reject_staged_pack(
                destination,
                f"preflight receipt size mismatch for {source}: "
                f"expected {result['size_bytes']}, got {receipt.get('size_bytes')!r}"
            )
        expected_content_id = f"sha256:{result['sha256']}"
        if receipt.get("content_id") != expected_content_id:
            reject_staged_pack(
                destination,
                f"preflight receipt content_id mismatch for {source}: "
                f"expected {expected_content_id}, got {receipt.get('content_id')!r}"
            )


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
        copy_stage(model, entry, quants, hf_readme(model), stage)
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
