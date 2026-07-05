#!/usr/bin/env python3
"""Fail-closed public-listing gate for OpenASR model catalog entries.

The gate intentionally uses unauthenticated Hugging Face HTTP requests. A model
is allowed to become `public:true` only when the same anonymous surface used by
end users can see the repo and every advertised pack URL.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import socket
import ssl
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from _pathlib_helpers import repo_root

DEFAULT_PUBLIC_NAMESPACE = "OpenASR"
HF_BASE_URL = "https://huggingface.co"
USER_AGENT = "openasr-public-gate/1"
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
HF_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
GA_VERSION_RE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
TRANSIENT_HTTP_ERRORS = (
    urllib.error.URLError,
    ssl.SSLError,
    ssl.SSLEOFError,
    http.client.HTTPException,
    ConnectionError,
    socket.timeout,
    TimeoutError,
)


class PublicGateError(RuntimeError):
    """Raised when a catalog entry is not safe to list publicly."""


@dataclass(frozen=True)
class HeadResult:
    status: int
    headers: Mapping[str, str]
    final_url: str


class PublicProbe:
    """Anonymous Hugging Face probe used by the gate."""

    def __init__(self, timeout: float = 60.0, head_attempts: int = 3) -> None:
        self.timeout = timeout
        self.head_attempts = head_attempts
        self.ssl_context = _ssl_context()

    def model_info(self, repo_id: str) -> Mapping[str, Any]:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        url = f"{HF_BASE_URL}/api/models/{encoded_repo}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        attempts = max(1, self.head_attempts)
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self._urlopen(request) as response:
                    payload = response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code in TRANSIENT_HTTP_STATUSES and attempt + 1 < attempts:
                    last_error = error
                    self._sleep(min(0.25 * (2**attempt), 2.0))
                    continue
                raise PublicGateError(
                    f"anonymous model_info for {repo_id} returned HTTP {error.code}"
                ) from error
            except TRANSIENT_HTTP_ERRORS as error:
                if attempt + 1 < attempts:
                    last_error = error
                    self._sleep(min(0.25 * (2**attempt), 2.0))
                    continue
                reason = _transport_error_reason(error)
                if last_error is not None:
                    reason = f"{_transport_error_reason(last_error)}; final attempt failed: {reason}"
                raise PublicGateError(f"anonymous model_info for {repo_id} failed: {reason}") from error
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as error:
            raise PublicGateError(f"anonymous model_info for {repo_id} returned invalid JSON") from error
        if not isinstance(data, dict):
            raise PublicGateError(f"anonymous model_info for {repo_id} returned a non-object payload")
        return data

    def head(self, url: str) -> HeadResult:
        last_error: BaseException | None = None
        for attempt in range(max(1, self.head_attempts)):
            try:
                result = self._request_metadata(url, method="HEAD")
                if result.status in TRANSIENT_HTTP_STATUSES and attempt + 1 < max(1, self.head_attempts):
                    last_error = PublicGateError(f"HTTP {result.status}")
                    self._sleep(min(0.25 * (2**attempt), 2.0))
                    continue
                return result
            except TRANSIENT_HTTP_ERRORS as error:
                last_error = error
                if attempt + 1 < max(1, self.head_attempts):
                    self._sleep(min(0.25 * (2**attempt), 2.0))
        try:
            return self._request_metadata(
                url,
                method="GET",
                request_headers={"Range": "bytes=0-0"},
                read_one_byte=True,
            )
        except TRANSIENT_HTTP_ERRORS as error:
            reason = _transport_error_reason(error)
            if last_error is not None:
                reason = f"{_transport_error_reason(last_error)}; ranged GET fallback failed: {reason}"
            raise PublicGateError(f"anonymous HEAD transient transport failure for {url}: {reason}") from error

    def _request_metadata(
        self,
        url: str,
        *,
        method: str,
        request_headers: Mapping[str, str] | None = None,
        read_one_byte: bool = False,
    ) -> HeadResult:
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self.ssl_context),
            _NoRedirectHandler(),
        )
        current_url = url
        headers: dict[str, str] = {}
        for _ in range(8):
            request = urllib.request.Request(
                current_url,
                headers={"User-Agent": USER_AGENT, **(request_headers or {})},
                method=method,
            )
            try:
                with self._open(opener, request) as response:
                    if read_one_byte:
                        response.read(1)
                    headers.update({key.lower(): value for key, value in response.headers.items()})
                    return HeadResult(status=response.status, headers=headers, final_url=response.geturl())
            except urllib.error.HTTPError as error:
                headers.update({key.lower(): value for key, value in error.headers.items()})
                if error.code not in REDIRECT_STATUSES:
                    return HeadResult(status=error.code, headers=headers, final_url=error.url)
                location = error.headers.get("Location")
                if not location:
                    raise PublicGateError(f"anonymous {method} redirect missing Location for {current_url}") from error
                current_url = urllib.parse.urljoin(current_url, location)
        raise PublicGateError(f"anonymous {method} followed too many redirects for {url}")

    def _open(self, opener: urllib.request.OpenerDirector, request: urllib.request.Request):  # type: ignore[no-untyped-def]
        return opener.open(request, timeout=self.timeout)

    def _urlopen(self, request: urllib.request.Request):  # type: ignore[no-untyped-def]
        return urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context)

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def sha256(self, url: str) -> tuple[str, int]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
        except urllib.error.HTTPError as error:
            raise PublicGateError(f"anonymous GET for sha256 returned HTTP {error.code}: {url}") from error
        except urllib.error.URLError as error:
            raise PublicGateError(f"anonymous GET for sha256 failed for {url}: {error.reason}") from error
        return digest.hexdigest(), size


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lower = name.lower()
    for key, value in headers.items():
        if key.lower() == lower:
            return value.strip()
    return None


def _head_sha256(headers: Mapping[str, str]) -> str | None:
    value = _header(headers, "x-linked-etag")
    if value is None:
        return None
    value = value.strip('"').lower()
    if SHA256_RE.fullmatch(value):
        return value
    return None


def _transport_error_reason(error: BaseException) -> str:
    reason = getattr(error, "reason", None)
    return str(reason if reason is not None else error)


def _head_size(headers: Mapping[str, str]) -> int | None:
    content_range = _header(headers, "content-range")
    if content_range is not None:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total and total != "*":
            return int(total)
    linked_size = _header(headers, "x-linked-size")
    if linked_size is not None:
        return int(linked_size)
    content_length = _header(headers, "content-length")
    if content_length is not None:
        return int(content_length)
    return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicGateError(message)


def _pack_url(repo_id: str, revision: str, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    _require(parsed.scheme == "https", f"pack URL must use HTTPS: {url}")
    expected_prefix = f"/{repo_id}/resolve/{revision}/"
    _require(
        parsed.netloc == "huggingface.co" and parsed.path.startswith(expected_prefix),
        f"pack URL must be pinned to https://huggingface.co/{repo_id}/resolve/{revision}/: {url}",
    )


def _validate_min_cli(value: Any) -> None:
    _require(isinstance(value, str) and bool(GA_VERSION_RE.fullmatch(value)), "min_cli_version must be a GA semver like 0.1.0")


_SEMVER_TUPLE_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _semver_tuple(value: str, *, label: str) -> tuple[int, int, int]:
    match = _SEMVER_TUPLE_RE.fullmatch(value.strip())
    _require(match is not None, f"{label} must be a plain major.minor.patch semver, got {value!r}")
    assert match is not None  # narrows for type checkers; _require already raised otherwise
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def _workspace_version() -> str:
    root = repo_root(Path(__file__))
    cargo_toml = root / "Cargo.toml"
    try:
        data = tomllib.loads(cargo_toml.read_text())
    except FileNotFoundError as error:
        raise PublicGateError(f"workspace Cargo.toml not found at {cargo_toml}") from error
    version = data.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str):
        raise PublicGateError(f"workspace Cargo.toml missing [workspace.package].version at {cargo_toml}")
    return version


def _validate_version_floor_consistency(model_id: str, model: Mapping[str, Any]) -> None:
    """A public catalog entry must not advertise CLI/core compatibility floors
    that are lower than reality: `min_cli_version` must be >= any declared
    `min_core_version` (the CLI floor cannot be more permissive than the core
    runtime floor the model actually needs), and neither floor may exceed the
    workspace's current release version (both are meant to describe *already
    shipped* builds, never a future one).
    """
    min_cli_version = model.get("min_cli_version")
    min_cli_tuple = _semver_tuple(str(min_cli_version), label=f"{model_id}: min_cli_version")

    min_core_version = model.get("min_core_version")
    min_core_tuple: tuple[int, int, int] | None = None
    if min_core_version is not None:
        min_core_tuple = _semver_tuple(str(min_core_version), label=f"{model_id}: min_core_version")
        _require(
            min_cli_tuple >= min_core_tuple,
            f"{model_id}: min_cli_version {min_cli_version} must be >= min_core_version "
            f"{min_core_version} (a public entry cannot claim CLI compatibility older than "
            "the core runtime it requires)",
        )

    workspace_version = _workspace_version()
    workspace_tuple = _semver_tuple(workspace_version, label="workspace Cargo.toml version")
    _require(
        min_cli_tuple <= workspace_tuple,
        f"{model_id}: min_cli_version {min_cli_version} exceeds the workspace version {workspace_version}",
    )
    if min_core_tuple is not None:
        _require(
            min_core_tuple <= workspace_tuple,
            f"{model_id}: min_core_version {min_core_version} exceeds the workspace version {workspace_version}",
        )


def validate_public_model(
    model: Mapping[str, Any],
    *,
    expected_namespace: str = DEFAULT_PUBLIC_NAMESPACE,
    probe: Any | None = None,
    timeout: float = 60.0,
) -> None:
    """Validate that a generated catalog model is safe to expose publicly."""

    probe = probe or PublicProbe(timeout=timeout)

    model_id = str(model.get("id") or "<unknown>")
    _require(model.get("public") is True, f"{model_id}: public gate called for a non-public entry")

    repo_id = model.get("hf_repo")
    _require(isinstance(repo_id, str) and "/" in repo_id, f"{model_id}: hf_repo must be '<owner>/<name>'")
    namespace, _repo_name = repo_id.split("/", 1)
    _require(
        namespace == expected_namespace,
        f"{model_id}: public releases must use canonical namespace '{expected_namespace}', got '{namespace}'",
    )

    revision = model.get("hf_revision")
    _require(isinstance(revision, str) and bool(HF_REVISION_RE.fullmatch(revision)), f"{model_id}: hf_revision must be a 40-hex commit sha")
    _validate_min_cli(model.get("min_cli_version"))
    _validate_version_floor_consistency(model_id, model)

    quants = model.get("quants")
    _require(isinstance(quants, list) and len(quants) > 0, f"{model_id}: public entry must include at least one quant")

    info = probe.model_info(repo_id)
    _require(info.get("private") is not True, f"{model_id}: anonymous model_info reports {repo_id} as private")

    for quant in quants:
        _require(isinstance(quant, dict), f"{model_id}: quant entries must be objects")
        quant_name = str(quant.get("quant") or "<unknown>")
        url = quant.get("url")
        _require(isinstance(url, str) and url, f"{model_id}:{quant_name}: missing pack URL")
        _pack_url(repo_id, revision, url)
        filename = str(quant.get("filename") or "")
        _require(filename.endswith(".oasr"), f"{model_id}:{quant_name}: filename must be a .oasr basename")
        _require(
            not quant.get("mirrors"),
            f"{model_id}:{quant_name}: mirror sources are not supported",
        )

        expected_sha = quant.get("sha256")
        _require(isinstance(expected_sha, str) and bool(SHA256_RE.fullmatch(expected_sha)), f"{model_id}:{quant_name}: sha256 must be 64 hex characters")
        expected_sha = expected_sha.lower()
        expected_size = quant.get("size_bytes")
        _require(isinstance(expected_size, int) and expected_size > 0, f"{model_id}:{quant_name}: size_bytes must be positive")

        perf = quant.get("perf")
        _require(isinstance(perf, dict), f"{model_id}:{quant_name}: perf metrics are required")
        _require(
            any(perf.get(key) is not None for key in ("rtf_cpu", "rtf_metal", "peak_rss_bytes", "jfk_wer_vs_fp16")),
            f"{model_id}:{quant_name}: perf metrics must contain at least one measured value",
        )

        head = probe.head(url)
        _require(
            head.status in {200, 206},
            f"{model_id}:{quant_name}: anonymous pack probe returned HTTP {head.status}",
        )

        try:
            head_size = _head_size(head.headers)
        except ValueError as error:
            raise PublicGateError(f"{model_id}:{quant_name}: invalid remote size header") from error
        if head_size is not None:
            _require(head_size == expected_size, f"{model_id}:{quant_name}: remote size {head_size} does not match local size {expected_size}")

        resolved_revision = _header(head.headers, "x-repo-commit")
        if resolved_revision is not None:
            _require(
                resolved_revision.lower() == revision.lower(),
                f"{model_id}:{quant_name}: HEAD resolved revision {resolved_revision} does not match catalog revision {revision}",
            )

        head_sha = _head_sha256(head.headers)
        if head_sha is not None:
            _require(head_sha == expected_sha, f"{model_id}:{quant_name}: HF sha256 {head_sha} does not match local sha256 {expected_sha}")
        else:
            observed_sha, observed_size = probe.sha256(url)
            _require(observed_sha.lower() == expected_sha, f"{model_id}:{quant_name}: HF sha256 {observed_sha} does not match local sha256 {expected_sha}")
            _require(observed_size == expected_size, f"{model_id}:{quant_name}: downloaded size {observed_size} does not match local size {expected_size}")


def _load_catalog_model(path: Path, model_id: str) -> Mapping[str, Any]:
    data = json.loads(path.read_text())
    for model in data.get("models", []):
        if model.get("id") == model_id:
            return model
    raise SystemExit(f"catalog model not found: {model_id}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--catalog", type=Path, default=Path("model-registry/catalog.json"))
    parser.add_argument("--public-namespace", default=DEFAULT_PUBLIC_NAMESPACE)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model = _load_catalog_model(args.catalog, args.model)
    validate_public_model(model, expected_namespace=args.public_namespace, timeout=args.timeout)
    print(f"public gate passed: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
