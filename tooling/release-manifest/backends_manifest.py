#!/usr/bin/env python3
"""Generate the UNSIGNED `backends-manifest.json` release asset.

  backends_manifest.py generate \\
      --version <semver> --source-commit <40-hex-sha> --dist-dir <dir> \\
      --out <backends-manifest.json>

`backends-manifest.json` is the per-release index the desktop app reads to
download a switchable Windows GPU-kernel sidecar (vulkan / cuda / hip) at
runtime, without shipping every backend in the base install. This script only
ASSEMBLES the manifest from already-built release archives (reads bytes,
computes sha256 + size) -- it never touches signing key material. Signing is a
separate, LOCAL-ONLY step (never run in CI), exactly like
`tooling/publish-model/scripts/publish_catalog.sh` signs the model catalog:

  cd <repo root>
  OPENASR_CATALOG_SIGNING_KEY_SEED_HEX=<real production seed> \\
    cargo run --quiet -p openasr-cli -- __openasr-sign-backends-manifest \\
      backends-manifest.json --out backends-manifest.signature.json \\
      --manifest-url https://dl.openasr.org/core/v<version>/backends-manifest.json

This generation step, by contrast, has no secret involved and is safe to run
in CI (release-binaries.yml's `checksums` job, after the Windows GPU archives
are staged) so the maintainer only has to sign+upload, not rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 1
PLATFORM = "windows-x86_64"
DEFAULT_BASE_URL = "https://dl.openasr.org/core"
DEFAULT_REPO = "QuintinShaw/openasr"
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")

# backend key (matches the Cargo feature name, e.g. `hip`) -> the friendly
# asset-name suffix release-binaries.yml packages it under (matches the
# `asset:` field in that workflow's matrix -- `hip`'s asset is named `rocm`
# for cross-platform consistency; see that file's matrix comment) and the PE
# import-table substring(s) the desktop app's runtime backend probe looks for
# (case-insensitive prefix match against openasr.exe's import table -- any
# one hit is sufficient).
BACKENDS: dict[str, dict[str, object]] = {
    "vulkan": {
        "asset_suffix": "vulkan",
        "pe_import_markers": ["vulkan-1.dll"],
    },
    "cuda": {
        "asset_suffix": "cuda",
        "pe_import_markers": ["cublas64_"],
    },
    "hip": {
        "asset_suffix": "rocm",
        "pe_import_markers": ["amdhip64_"],
    },
}


class BackendsManifestError(Exception):
    pass


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def asset_filename(version: str, asset_suffix: str) -> str:
    return f"openasr-{version}-{PLATFORM}-{asset_suffix}.zip"


def backend_entry(
    version: str, backend: str, dist_dir: Path, base_url: str, repo: str
) -> dict[str, object]:
    spec = BACKENDS[backend]
    asset_suffix = spec["asset_suffix"]
    filename = asset_filename(version, asset_suffix)  # type: ignore[arg-type]
    path = dist_dir / filename
    if not path.is_file():
        raise BackendsManifestError(
            f"backend '{backend}': expected archive not found: {path}"
        )
    sha256, size_bytes = sha256_and_size(path)
    if size_bytes == 0:
        raise BackendsManifestError(f"backend '{backend}': archive is empty: {path}")
    return {
        "asset": filename,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "urls": [
            f"{base_url}/v{version}/{filename}",
            f"https://github.com/{repo}/releases/download/v{version}/{filename}",
        ],
        "pe_import_markers": list(spec["pe_import_markers"]),  # type: ignore[arg-type]
    }


def build_manifest(
    version: str,
    source_commit: str,
    dist_dir: Path,
    base_url: str = DEFAULT_BASE_URL,
    repo: str = DEFAULT_REPO,
    backends: list[str] | None = None,
) -> dict[str, object]:
    if not VERSION_RE.fullmatch(version):
        raise BackendsManifestError(f"--version must be a semver string, got: {version!r}")
    if not COMMIT_RE.fullmatch(source_commit):
        raise BackendsManifestError(
            f"--source-commit must be a 40-hex-char commit sha, got: {source_commit!r}"
        )
    if not dist_dir.is_dir():
        raise BackendsManifestError(f"--dist-dir is not a directory: {dist_dir}")

    selected = backends if backends is not None else sorted(BACKENDS)
    unknown = [name for name in selected if name not in BACKENDS]
    if unknown:
        raise BackendsManifestError(
            f"unknown backend(s) {unknown}; known backends: {sorted(BACKENDS)}"
        )
    if not selected:
        raise BackendsManifestError("at least one backend is required")

    entries = {
        backend: backend_entry(version, backend, dist_dir, base_url, repo)
        for backend in selected
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "core_version": version,
        "source_commit": source_commit.lower(),
        "platforms": {
            PLATFORM: {
                "backends": {backend: entries[backend] for backend in selected},
            },
        },
    }


def manifest_url_for(version: str, base_url: str = DEFAULT_BASE_URL) -> str:
    return f"{base_url}/v{version}/backends-manifest.json"


def write_manifest(manifest: dict[str, object], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    out.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Assemble the unsigned backends-manifest.json")
    generate.add_argument("--version", required=True, help="Release semver, e.g. 0.1.10")
    generate.add_argument("--source-commit", required=True, help="Full 40-hex release commit sha")
    generate.add_argument(
        "--dist-dir",
        required=True,
        type=Path,
        help="Directory containing the built openasr-<version>-windows-x86_64-{vulkan,cuda,rocm}.zip archives",
    )
    generate.add_argument("--out", required=True, type=Path, help="Output backends-manifest.json path")
    generate.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Primary CDN base URL (default: %(default)s)")
    generate.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo for the fallback URL (default: %(default)s)")
    generate.add_argument(
        "--backend",
        dest="backends",
        action="append",
        choices=sorted(BACKENDS),
        help=f"Restrict to specific backend(s) (repeatable). Default: all of {sorted(BACKENDS)}.",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        try:
            manifest = build_manifest(
                version=args.version,
                source_commit=args.source_commit,
                dist_dir=args.dist_dir,
                base_url=args.base_url,
                repo=args.repo,
                backends=args.backends,
            )
        except BackendsManifestError as error:
            print(f"backends_manifest.py: {error}", file=sys.stderr)
            return 1
        write_manifest(manifest, args.out)
        print(args.out)
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
