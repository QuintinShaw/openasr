#!/usr/bin/env python3
"""Retry GitHub Release and CDN downloads with stall detection.

``gh release download`` hangs or EOFs on draft assets, especially glob
patterns against a large draft. Every release reader must go through this
helper: resolve a named asset via the GitHub JSON API, then curl the
octet-stream URL over HTTP/1.1 with a stall abort and retries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ATTEMPTS = 6
DEFAULT_TIMEOUT_SECONDS = 1800
STALL_SECONDS = 30
STALL_BYTES_PER_SEC = 1024
USER_AGENT = "openasr-release-fetch"


class DownloadError(RuntimeError):
    pass


def _safe_name(name: str) -> str:
    if Path(name).name != name or not name or name in {".", ".."}:
        raise ValueError(f"unsafe release asset name: {name!r}")
    return name


def _repo_args(repository: str | None) -> list[str]:
    return ["--repo", repository] if repository else []


def _auth_token() -> str:
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    completed = subprocess.run(
        ["gh", "auth", "token"],
        check=True,
        capture_output=True,
        text=True,
    )
    token = completed.stdout.strip()
    if not token:
        raise DownloadError("no GitHub token for release asset download")
    return token


def list_assets(tag: str, repository: str | None = None) -> list[dict[str, Any]]:
    payload = json.loads(
        subprocess.check_output(
            ["gh", "release", "view", tag, "--json", "assets", *_repo_args(repository)],
            text=True,
        )
    )
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise DownloadError(f"release {tag} asset list is missing")
    return [asset for asset in assets if isinstance(asset, dict)]


def list_asset_names(tag: str, repository: str | None = None) -> list[str]:
    names: list[str] = []
    for asset in list_assets(tag, repository):
        name = asset.get("name")
        if isinstance(name, str) and Path(name).name == name and name not in {".", ".."}:
            names.append(name)
    return names


def _asset_api_url(tag: str, name: str, repository: str | None) -> str:
    name = _safe_name(name)
    for asset in list_assets(tag, repository):
        if asset.get("name") != name:
            continue
        api_url = asset.get("apiUrl")
        if isinstance(api_url, str) and api_url.startswith("https://api.github.com/"):
            return api_url
        raise DownloadError(f"release {tag} asset {name} has no GitHub API URL")
    raise DownloadError(f"release {tag} has no asset {name}")


def curl_download(
    url: str,
    dest: Path,
    *,
    headers: list[tuple[str, str]] | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download URL: {url!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-fL",
        "--http1.1",
        "--connect-timeout",
        "20",
        "--max-time",
        str(timeout_seconds),
        "--speed-time",
        str(STALL_SECONDS),
        "--speed-limit",
        str(STALL_BYTES_PER_SEC),
        "-A",
        USER_AGENT,
        "-o",
        str(dest),
    ]
    for key, value in headers or []:
        command.extend(["-H", f"{key}: {value}"])
    command.append(url)
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        print(
            f"download {url} -> {dest.name} (attempt {attempt}/{attempts})",
            file=sys.stderr,
            flush=True,
        )
        dest.unlink(missing_ok=True)
        try:
            subprocess.run(command, check=True, timeout=timeout_seconds + 60)
            if dest.is_file() and dest.stat().st_size > 0:
                return
            last_error = DownloadError(f"empty download: {url}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            last_error = error
        if attempt == attempts:
            break
        time.sleep(min(2**attempt, 16))
    assert last_error is not None
    raise last_error


def download_url(
    url: str,
    dest: Path,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    curl_download(url, dest, attempts=attempts, timeout_seconds=timeout_seconds)


def _github_headers(token: str) -> list[tuple[str, str]]:
    return [
        ("Authorization", f"Bearer {token}"),
        ("Accept", "application/octet-stream"),
        ("X-GitHub-Api-Version", "2022-11-28"),
    ]


def download_asset(
    tag: str,
    name: str,
    dest_dir: Path,
    *,
    repository: str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    name = _safe_name(name)
    dest = Path(dest_dir) / name
    curl_download(
        _asset_api_url(tag, name, repository),
        dest,
        headers=_github_headers(_auth_token()),
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )


def download_assets(
    tag: str,
    names: list[str],
    dest_dir: Path,
    *,
    repository: str | None = None,
) -> None:
    catalog = {asset.get("name"): asset for asset in list_assets(tag, repository)}
    token = _auth_token()
    seen: set[str] = set()
    for name in names:
        name = _safe_name(name)
        if name in seen:
            continue
        seen.add(name)
        asset = catalog.get(name)
        if not isinstance(asset, dict):
            raise DownloadError(f"release {tag} has no asset {name}")
        api_url = asset.get("apiUrl")
        if not isinstance(api_url, str) or not api_url.startswith("https://api.github.com/"):
            raise DownloadError(f"release {tag} asset {name} has no GitHub API URL")
        curl_download(
            api_url,
            Path(dest_dir) / name,
            headers=_github_headers(token),
        )


def _dispatch(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("tag")
    download.add_argument("dest", type=Path)
    download.add_argument("names", nargs="+")
    download.add_argument("--repo")

    url = subparsers.add_parser("download-url")
    url.add_argument("url")
    url.add_argument("dest", type=Path)

    packs = subparsers.add_parser("download-packs")
    packs.add_argument("tag")
    packs.add_argument("dest", type=Path)
    packs.add_argument("--repo")

    args = parser.parse_args(argv)
    if args.command == "download":
        download_assets(args.tag, args.names, args.dest, repository=args.repo)
        return
    if args.command == "download-url":
        download_url(args.url, args.dest)
        return

    import release_completeness

    download_assets(
        args.tag,
        release_completeness.backend_pack_names(release_completeness.load_matrix()),
        args.dest,
        repository=args.repo,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        _dispatch(sys.argv[1:] if argv is None else argv)
    except (DownloadError, ValueError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
