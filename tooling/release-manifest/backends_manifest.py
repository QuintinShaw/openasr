#!/usr/bin/env python3
"""Generate the UNSIGNED `backends-manifest.json` release asset.

  backends_manifest.py generate \\
      --version <semver> --source-commit <40-hex-sha> --dist-dir <dir> \\
      --out <backends-manifest.json>

`backends-manifest.json` is the per-release index the desktop app reads to
download a switchable Windows GPU-kernel sidecar (vulkan / cuda / hip) at
runtime, without shipping every backend in the base install. Produces
`schema_version: 2` (see `crates/openasr-core/src/backend_manifest.rs`'s
module doc and `docs/backend-kernels.md`): `cuda`/`hip` ship as a small
per-release "sidecar" archive (just `openasr.exe` + its own build artifacts)
that references a separate, large, content-addressed `vendor_layers` entry
(the NVIDIA/AMD redistributable runtime DLLs), which release-binaries.yml
stages independently and this script only locates + hashes -- it never builds
either archive itself. `vulkan` stays self-contained (no vendor layer,
identical shape to a v1 manifest's `vulkan` entry).

This script only ASSEMBLES the manifest from already-built release archives
(reads bytes, computes sha256 + size) -- it never touches signing key
material. Signing is a separate, LOCAL-ONLY step (never run in CI), exactly
like `tooling/publish-model/scripts/publish_catalog.sh` signs the model
catalog:

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

SCHEMA_VERSION = 2
PLATFORM = "windows-x86_64"
DEFAULT_BASE_URL = "https://dl.openasr.org/core"
DEFAULT_REPO = "QuintinShaw/openasr"
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")

# backend key (matches the Cargo feature name, e.g. `hip`) -> the friendly
# asset-name suffix release-binaries.yml packages it under (matches the
# `asset:` field in that workflow's matrix -- `hip`'s asset is named `rocm`
# for cross-platform consistency; see that file's matrix comment), the PE
# import-table substring(s) the desktop app's runtime backend probe looks for
# (case-insensitive prefix match against openasr.exe's import table -- any
# one hit is sufficient), and (v2) the `vendor_layer` key this backend's
# sidecar archive depends on -- `None` for a self-contained backend (vulkan
# stays self-contained: the Vulkan loader redistributable is small enough to
# ship inline, unlike NVIDIA's/AMD's multi-hundred-MB runtimes).
BACKENDS: dict[str, dict[str, object]] = {
    "vulkan": {
        "asset_suffix": "vulkan",
        "pe_import_markers": ["vulkan-1.dll"],
        "vendor_layer": None,
    },
    "cuda": {
        "asset_suffix": "cuda",
        "pe_import_markers": ["cublas64_"],
        "vendor_layer": "cuda-runtime",
    },
    "hip": {
        "asset_suffix": "rocm",
        "pe_import_markers": ["amdhip64_"],
        "vendor_layer": "rocm-runtime",
    },
}

# vendor_layer key -> human-readable build toolchain identifier carried in
# the manifest for traceability only (not used in any verification
# decision). Keyed independently of BACKENDS since a vendor layer is a
# core-version- (and, in principle, backend-count-) independent concept; see
# `VendorLayer` in `crates/openasr-core/src/backend_manifest.rs`.
VENDOR_LAYER_TOOLCHAINS: dict[str, str] = {
    "cuda-runtime": "cuda-13.0",
    "rocm-runtime": "rocm-7.2",
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


def asset_filename(version: str, backend: str) -> str:
    """The sidecar (small, per-release) archive filename for `backend`. A
    backend with a `vendor_layer` (cuda/hip) gets a `-sidecar` suffix so it is
    visually distinct from the (large, content-addressed) vendor archive its
    manifest entry points at; a self-contained backend (vulkan) keeps the
    plain v1-shaped name."""
    spec = BACKENDS[backend]
    asset_suffix = spec["asset_suffix"]
    if spec.get("vendor_layer"):
        return f"openasr-{version}-{PLATFORM}-{asset_suffix}-sidecar.zip"
    return f"openasr-{version}-{PLATFORM}-{asset_suffix}.zip"


def backend_entry(
    version: str, backend: str, dist_dir: Path, base_url: str, repo: str
) -> dict[str, object]:
    spec = BACKENDS[backend]
    filename = asset_filename(version, backend)
    path = dist_dir / filename
    if not path.is_file():
        raise BackendsManifestError(
            f"backend '{backend}': expected archive not found: {path}"
        )
    sha256, size_bytes = sha256_and_size(path)
    if size_bytes == 0:
        raise BackendsManifestError(f"backend '{backend}': archive is empty: {path}")
    entry: dict[str, object] = {
        "asset": filename,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "urls": [
            f"{base_url}/v{version}/{filename}",
            f"https://github.com/{repo}/releases/download/v{version}/{filename}",
        ],
        "pe_import_markers": list(spec["pe_import_markers"]),  # type: ignore[arg-type]
    }
    vendor_layer = spec.get("vendor_layer")
    if vendor_layer:
        entry["vendor_layer"] = vendor_layer
    return entry


def vendor_layer_entry(
    vendor_key: str, version: str, dist_dir: Path, base_url: str, repo: str
) -> dict[str, object]:
    """Locate the already-built, content-addressed vendor archive for
    `vendor_key` (e.g. `"cuda-runtime"`) under `dist_dir` -- release-binaries.yml
    stages it as `openasr-vendor-<vendor_key>-<sha12>.zip`, where `<sha12>` is
    the first 12 hex chars of the archive's own sha256 (computed once the
    deterministic zip is built; see that workflow's "Stage archive contents"
    step) -- hash it, and assemble its manifest entry. Fails loudly if the
    archive is missing, ambiguous, or its filename's short hash does not
    actually prefix the freshly computed sha256 (a staging bug that would
    otherwise silently ship a mislabeled/stale vendor archive under a
    content-addressed name other code trusts at face value)."""
    matches = sorted(dist_dir.glob(f"openasr-vendor-{vendor_key}-*.zip"))
    if not matches:
        raise BackendsManifestError(
            f"vendor layer '{vendor_key}': expected archive not found under {dist_dir} "
            f"(glob 'openasr-vendor-{vendor_key}-*.zip')"
        )
    if len(matches) > 1:
        raise BackendsManifestError(
            f"vendor layer '{vendor_key}': multiple candidate archives found: "
            f"{[path.name for path in matches]}"
        )
    path = matches[0]
    sha256, size_bytes = sha256_and_size(path)
    if size_bytes == 0:
        raise BackendsManifestError(f"vendor layer '{vendor_key}': archive is empty: {path}")

    short_hash = path.name.removeprefix(f"openasr-vendor-{vendor_key}-").removesuffix(".zip")
    if not short_hash or not sha256.startswith(short_hash.lower()):
        raise BackendsManifestError(
            f"vendor layer '{vendor_key}': filename {path.name} embeds short hash "
            f"'{short_hash}' that does not prefix-match its actual sha256 {sha256} -- "
            "refusing a mislabeled vendor archive (content-addressed dedup depends on "
            "this being honest)"
        )

    return {
        "sha256": sha256,
        "asset": path.name,
        "size_bytes": size_bytes,
        "urls": [
            f"{base_url}/vendor/{sha256}/{path.name}",
            f"https://github.com/{repo}/releases/download/v{version}/{path.name}",
        ],
        "toolchain": VENDOR_LAYER_TOOLCHAINS[vendor_key],
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

    # Only assemble a vendor_layers entry for a layer some SELECTED backend
    # actually references (e.g. `--backend vulkan` alone needs none at all;
    # a single-leg `only_target` dry-run should not demand a vendor archive
    # for a backend that was not built this run).
    needed_vendor_layers = sorted(
        {
            spec["vendor_layer"]  # type: ignore[index]
            for backend, spec in BACKENDS.items()
            if backend in selected and spec.get("vendor_layer")
        }
    )
    vendor_layers = {
        key: vendor_layer_entry(key, version, dist_dir, base_url, repo)
        for key in needed_vendor_layers
    }

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "core_version": version,
        "source_commit": source_commit.lower(),
    }
    if vendor_layers:
        manifest["vendor_layers"] = vendor_layers
    manifest["platforms"] = {
        PLATFORM: {
            "backends": {backend: entries[backend] for backend in selected},
        },
    }
    return manifest


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
        help=(
            "Directory containing the built openasr-<version>-windows-x86_64-vulkan.zip, "
            "-{cuda,rocm}-sidecar.zip, and openasr-vendor-{cuda,rocm}-runtime-<sha12>.zip archives"
        ),
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
