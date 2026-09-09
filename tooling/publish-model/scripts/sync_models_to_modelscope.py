#!/usr/bin/env python3
"""Idempotent sync of signed catalog packs onto ModelScope.

Reads catalog.public.json (HF identity URLs + sha256). Uploads the same bytes
to `openasr/<repo>` (ModelScope org is lowercase). Never writes ModelScope
URLs back into the catalog.

Skip (exit 0) when no token is set:
  MODELSCOPE_SDK_TOKEN or MS_TOKEN

Cache:
  OPENASR_MODELSCOPE_CACHE  override download cache (default ~/.cache/openasr-modelscope-sync)

Usage:
  tooling/publish-model/scripts/sync_models_to_modelscope.py [--catalog PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELSCOPE_OWNER = "openasr"
MODELSCOPE_ORIGIN = "https://www.modelscope.cn"
# ModelScope Hub rejects commits to Hugging Face git SHAs
# (`commit rejected by repository policy`). Default branch is the only
# writable ref; signed catalog sha256 is the pin.
MODELSCOPE_UPLOAD_REVISION = "master"
HF_ORIGIN = "https://huggingface.co/"


def token() -> str:
    return (
        os.environ.get("MODELSCOPE_SDK_TOKEN", "").strip()
        or os.environ.get("MS_TOKEN", "").strip()
    )


def modelscope_resolve_url(hf_url: str) -> str | None:
    rest = hf_url[len(HF_ORIGIN) :] if hf_url.startswith(HF_ORIGIN) else None
    if rest is None:
        return None
    parts = rest.split("/")
    # OpenASR/<repo>/resolve/<rev>/<file>
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    repo, rev, filename = parts[1], parts[3], "/".join(parts[4:])
    if not repo or not rev or not filename or ".." in filename or "\\" in filename:
        return None
    return (
        f"{MODELSCOPE_ORIGIN}/models/{MODELSCOPE_OWNER}/{repo}/resolve/"
        f"{MODELSCOPE_UPLOAD_REVISION}/{filename}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_packs(catalog: dict) -> list[dict]:
    packs = []
    for model in catalog.get("models") or []:
        if model.get("public") is not True:
            continue
        hf_repo = str(model.get("hf_repo") or "")
        if not hf_repo.lower().startswith("openasr/"):
            continue
        repo = hf_repo.split("/", 1)[1]
        for quant in model.get("quants") or []:
            url = str(quant.get("url") or "")
            sha = str(quant.get("sha256") or "").lower()
            filename = str(quant.get("filename") or "")
            size = int(quant.get("size_bytes") or 0)
            rev = str(quant.get("hf_revision") or model.get("hf_revision") or "")
            if not url or not sha or not filename or size <= 0:
                continue
            packs.append(
                {
                    "model_id": model["id"],
                    "repo": repo,
                    "revision": rev,
                    "filename": filename,
                    "url": url,
                    "sha256": sha,
                    "size_bytes": size,
                    "ms_url": modelscope_resolve_url(url),
                }
            )
    return packs


def ssl_context():
    """macOS python.org builds ship without a default CA bundle."""
    import ssl

    try:
        import certifi

        cafile = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", cafile)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
        return ssl.create_default_context(cafile=cafile)
    except ImportError:
        return ssl.create_default_context()


def remote_size_bytes(url: str) -> int | None:
    """Total object size if the URL exists, else None.

    ModelScope `/resolve/master/<file>` HEAD is 200 with no Content-Length
    even for objects that are already on the LFS CDN. A 1-byte Range GET
    follows the 302 and returns `Content-Range: bytes 0-0/<total>`.
    """
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "openasr-modelscope-sync",
            "Range": "bytes=0-0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30, context=ssl_context()) as response:
            response.read(8)
            content_range = response.headers.get("Content-Range") or ""
            if "/" in content_range:
                total = content_range.rsplit("/", 1)[-1]
                if total.isdigit():
                    return int(total)
            length = response.headers.get("Content-Length")
            if length and int(length) > 1:
                return int(length)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None
    return None


def remote_sha256(url: str, expected_size: int) -> str | None:
    """Treat a same-size remote object as already present; else None."""
    size = remote_size_bytes(url)
    if size == expected_size:
        return "size-match"
    return None


def part_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".part")


def cache_root() -> Path:
    raw = os.environ.get("OPENASR_MODELSCOPE_CACHE", "").strip()
    path = Path(raw) if raw else Path.home() / ".cache" / "openasr-modelscope-sync"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = part_path(dest)
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_size:
        partial.unlink()
        existing = 0
    if existing == expected_size:
        partial.replace(dest)
        return

    headers = {"User-Agent": "openasr-modelscope-sync"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    # Multi-GB packs stall if we use a short socket timeout; 10 minutes
    # applies per blocking read, not the whole transfer.
    with urllib.request.urlopen(request, timeout=600, context=ssl_context()) as response:
        status = getattr(response, "status", None) or response.getcode()
        length_header = response.headers.get("Content-Length")
        length = int(length_header) if length_header else None
        if existing and status == 206:
            remaining = expected_size - existing
            if length is not None and length != remaining:
                raise RuntimeError(
                    f"Content-Length {length} != remaining {remaining} for {expected_size}"
                )
            mode = "ab"
            written = existing
        else:
            if existing:
                partial.unlink(missing_ok=True)
                existing = 0
            if length is not None and length != expected_size:
                raise RuntimeError(
                    f"Content-Length {length} != catalog size_bytes {expected_size}"
                )
            mode = "wb"
            written = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
    if written != expected_size:
        # Keep `.part` so a later attempt can resume; only drop a truncated
        # full GET (mode was wb from byte 0). Resume leftovers stay.
        if written == 0:
            partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {written} bytes, expected {expected_size}")
    partial.replace(dest)


def ensure_repo(api, repo: str) -> None:
    model_id = f"{MODELSCOPE_OWNER}/{repo}"
    try:
        api.get_model(model_id)
        return
    except Exception:
        pass
    visibility = getattr(api, "visibility", None)
    kwargs = {"model_id": model_id, "visibility": 5}
    for method_name in ("create_model", "create_repo"):
        method = getattr(api, method_name, None)
        if method is None:
            continue
        try:
            method(**kwargs)
            print(f"created ModelScope repo {model_id}", file=sys.stderr)
            return
        except TypeError:
            try:
                method(model_id)
                return
            except Exception as error:
                print(f"create {model_id} via {method_name}: {error}", file=sys.stderr)
        except Exception as error:
            print(f"create {model_id} via {method_name}: {error}", file=sys.stderr)
    print(f"warning: could not ensure repo {model_id}; upload may still work", file=sys.stderr)


def upload_file(api, repo: str, local_path: Path, filename: str) -> None:
    model_id = f"{MODELSCOPE_OWNER}/{repo}"
    errors = []
    if hasattr(api, "upload_file"):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=model_id,
                revision=MODELSCOPE_UPLOAD_REVISION,
                commit_message=f"sync {filename} from signed catalog",
            )
            return
        except TypeError as error:
            errors.append(str(error))
            try:
                api.upload_file(str(local_path), filename, model_id)
                return
            except Exception as inner:
                errors.append(str(inner))
        except Exception as error:
            errors.append(str(error))
    raise RuntimeError(f"upload {model_id}/{filename} failed: {errors or 'no upload_file API'}")


def prefetch_sidecar(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".prefetch")


def adopt_prefetch(dest: Path, expected_sha: str, wait_seconds: int = 3600) -> bool:
    """Wait for a live `--prefetch-only` sibling and take its completed file.

    Returns True when `dest` now holds the expected sha, so the uploader
    must not start a second Hugging Face GET.
    """
    prefetch = prefetch_sidecar(dest)
    prefetch_part = part_path(prefetch)
    if not prefetch.exists() and not prefetch_part.exists():
        return False
    deadline = time.time() + wait_seconds
    while prefetch_part.exists() and time.time() < deadline:
        print(f"waiting for prefetch {prefetch_part.name}", file=sys.stderr)
        time.sleep(5)
    if dest.exists() and sha256_file(dest).lower() == expected_sha:
        prefetch.unlink(missing_ok=True)
        return True
    if prefetch.exists() and sha256_file(prefetch).lower() == expected_sha:
        if dest.exists() or part_path(dest).exists():
            prefetch.unlink(missing_ok=True)
            return dest.exists() and sha256_file(dest).lower() == expected_sha
        prefetch.replace(dest)
        return True
    prefetch.unlink(missing_ok=True)
    prefetch_part.unlink(missing_ok=True)
    return False


def prefetch_pack(pack: dict, dest: Path) -> str:
    """Fill `dest` from the HF identity URL without racing the uploader.

    Returns skip-local / skip-remote / skip-busy / ok / fail.
    The live uploader writes `dest.part`. This path writes `dest.prefetch`
    and only renames onto `dest` when that part file is absent.
    """
    if dest.exists() and sha256_file(dest).lower() == pack["sha256"]:
        return "skip-local"
    if part_path(dest).exists():
        return "skip-busy"
    if pack["ms_url"] and remote_sha256(pack["ms_url"], pack["size_bytes"]) == "size-match":
        return "skip-remote"
    prefetch = dest.with_suffix(dest.suffix + ".prefetch")
    try:
        download(pack["url"], prefetch, pack["size_bytes"])
        if sha256_file(prefetch).lower() != pack["sha256"]:
            prefetch.unlink(missing_ok=True)
            return "fail"
        if dest.exists() or part_path(dest).exists():
            prefetch.unlink(missing_ok=True)
            return "skip-busy"
        prefetch.replace(dest)
        return "ok"
    except Exception:
        prefetch.unlink(missing_ok=True)
        part_path(prefetch).unlink(missing_ok=True)
        return "fail"


def prefetch_missing(packs: list[dict], cache: Path) -> int:
    failures = 0
    for pack in packs:
        if pack["ms_url"] is None:
            continue
        dest = cache / pack["repo"] / pack["filename"]
        result = prefetch_pack(pack, dest)
        print(f"prefetch {result} {pack['repo']}/{pack['filename']}", file=sys.stderr)
        if result == "fail":
            failures += 1
    if failures:
        print(f"{failures} prefetch(s) failed", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="sync at most N packs (0 = all)")
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="download missing packs into the cache; skip files the uploader holds as .part",
    )
    args = parser.parse_args(argv)

    ssl_context()

    repo_root = Path(__file__).resolve().parents[3]
    catalog_path = args.catalog or (repo_root / "model-registry" / "catalog.public.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    packs = iter_packs(catalog)
    if args.limit:
        packs = packs[: args.limit]
    print(f"{len(packs)} catalog pack(s) to consider", file=sys.stderr)

    if args.dry_run:
        for pack in packs:
            print(f"DRY {pack['model_id']} {pack['filename']} -> {pack['ms_url']}")
        return 0

    cache = cache_root()
    if args.prefetch_only:
        return prefetch_missing(packs, cache)

    tok = token()
    if not tok:
        print("ModelScope sync skipped (no MODELSCOPE_SDK_TOKEN / MS_TOKEN)", file=sys.stderr)
        return 0

    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("modelscope SDK not installed; pip install modelscope", file=sys.stderr)
        return 1

    api = HubApi()
    api.login(tok)

    failures = 0
    seen_repos: set[str] = set()
    for pack in packs:
        if pack["ms_url"] is None:
            print(f"skip (not an HF resolve URL): {pack['url']}", file=sys.stderr)
            continue
        if pack["repo"] not in seen_repos:
            ensure_repo(api, pack["repo"])
            seen_repos.add(pack["repo"])
        dest = cache / pack["repo"] / pack["filename"]
        if remote_sha256(pack["ms_url"], pack["size_bytes"]) == "size-match":
            print(
                f"skip (ModelScope already has {pack['filename']} at {pack['size_bytes']} bytes)",
                file=sys.stderr,
            )
            continue
        if dest.exists() and sha256_file(dest).lower() == pack["sha256"]:
            local = dest
        elif adopt_prefetch(dest, pack["sha256"]):
            print(f"adopted prefetch {pack['repo']}/{pack['filename']}", file=sys.stderr)
            local = dest
        else:
            print(f"downloading {pack['filename']} from HF identity URL", file=sys.stderr)
            local = None
            for attempt in range(1, 4):
                try:
                    download(pack["url"], dest, pack["size_bytes"])
                except Exception as error:
                    print(
                        f"download failed {pack['url']} (attempt {attempt}/3): {error}",
                        file=sys.stderr,
                    )
                    dest.unlink(missing_ok=True)
                    part_path(dest).unlink(missing_ok=True)
                    continue
                actual = sha256_file(dest).lower()
                if actual == pack["sha256"]:
                    local = dest
                    break
                print(
                    f"sha mismatch after HF download {pack['filename']} "
                    f"(attempt {attempt}/3): {actual} != {pack['sha256']}",
                    file=sys.stderr,
                )
                dest.unlink(missing_ok=True)
            if local is None:
                failures += 1
                continue
        try:
            upload_file(api, pack["repo"], local, pack["filename"])
            print(f"ok {MODELSCOPE_OWNER}/{pack['repo']}/{pack['filename']}")
        except Exception as error:
            print(f"upload failed {pack['filename']}: {error}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"{failures} pack(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
