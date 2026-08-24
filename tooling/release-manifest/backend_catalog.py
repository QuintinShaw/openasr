#!/usr/bin/env python3
"""Compile verified backend build artifacts into signed-catalog entries.

This tool never signs or downloads payload bytes. It derives every byte identity
from the staged release files. Merging preserves prior ABI-scoped entries so
older neutral hosts can keep resolving their compatible pack, and replaces
same-ABI slots in place when a later plugin build reuses the stable target id.
verify-cdn HEADs the signed file URLs so a release cannot go public while the
canonical CDN prefix is empty.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


class BackendCatalogError(ValueError):
    pass


def _write_utf8_lf(path: Path, value: object) -> None:
    """Write stable catalog JSON bytes on every host OS."""

    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def sha256_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def pe_image_identity(path: Path) -> tuple[str, int]:
    """Hash the executable image while excluding Authenticode-only mutations."""

    data = path.read_bytes()
    if data[:2] != b"MZ" or len(data) < 0x40:
        raise BackendCatalogError(f"bundled DLL '{path.name}' is not a PE image")
    pe_offset = _unpack_from("<I", data, 0x3C, path)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise BackendCatalogError(f"bundled DLL '{path.name}' has no PE signature")
    coff = pe_offset + 4
    optional_size = _unpack_from("<H", data, coff + 16, path)[0]
    optional = coff + 20
    optional_end = optional + optional_size
    if optional_end > len(data):
        raise BackendCatalogError(f"bundled DLL '{path.name}' has a truncated optional header")
    magic = _unpack_from("<H", data, optional, path)[0]
    if magic == 0x10B:
        directory_count_offset, directory_offset = 92, 96
    elif magic == 0x20B:
        directory_count_offset, directory_offset = 108, 112
    else:
        raise BackendCatalogError(f"bundled DLL '{path.name}' has unsupported PE magic")
    checksum = optional + 64
    directory_count = _unpack_from("<I", data, optional + directory_count_offset, path)[0]
    if directory_count < 5:
        raise BackendCatalogError(f"bundled DLL '{path.name}' has no security directory")
    security = optional + directory_offset + 4 * 8
    if checksum + 4 > optional_end or security + 8 > optional_end or checksum >= security:
        raise BackendCatalogError(f"bundled DLL '{path.name}' has invalid PE security metadata")
    certificate_offset, certificate_size = _unpack_from("<II", data, security, path)
    if certificate_offset == 0 and certificate_size == 0:
        certificate_start = certificate_end = len(data)
    else:
        certificate_start = certificate_offset
        certificate_end = certificate_offset + certificate_size
        if certificate_start < optional_end or certificate_end > len(data):
            raise BackendCatalogError(f"bundled DLL '{path.name}' has an invalid certificate table")

    digest = hashlib.sha256()
    digest.update(data[:checksum])
    digest.update(b"\0" * 4)
    digest.update(data[checksum + 4 : security])
    digest.update(b"\0" * 8)
    digest.update(data[security + 8 : certificate_start])
    digest.update(data[certificate_end:])
    return digest.hexdigest(), len(data) - (certificate_end - certificate_start)


def _unpack_from(fmt: str, data: bytes, offset: int, path: Path) -> tuple[Any, ...]:
    try:
        return struct.unpack_from(fmt, data, offset)
    except (struct.error, OverflowError) as error:
        raise BackendCatalogError(
            f"bundled DLL '{path.name}' has a truncated PE structure"
        ) from error


def bundle_contract_sha256(host_abi_fingerprint: str, files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()

    def field(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)

    field("openasr-bundle-contract-v1")
    field(host_abi_fingerprint)
    contract_files = sorted(
        (entry for entry in files if entry["provider"] != "dependency"),
        key=lambda entry: (entry["filename"].lower(), entry["provider"]),
    )
    for entry in contract_files:
        field(entry["filename"].lower())
        field(entry["provider"].lower())
        field(entry["image_sha256"])
        digest.update(struct.pack("<Q", entry["image_size_bytes"]))
    return digest.hexdigest()


def provider_bundle_contract_sha256(
    host_abi_fingerprint: str, files: list[dict[str, Any]], provider: str
) -> str:
    """Bind the common host DLLs plus exactly one bundled provider rail."""

    selected = [
        entry
        for entry in files
        if entry["provider"] == "host" or entry["provider"] == provider
    ]
    return bundle_contract_sha256(host_abi_fingerprint, selected)


def compile_bundled_manifest(
    directory: Path,
    host_abi_path: Path,
    out: Path,
) -> None:
    """Bind the neutral host's final, post-signing DLL bytes.

    build.rs emits the same schema for local Cargo runs. Release signing changes
    PE bytes, however, so the release workflow regenerates this manifest after
    Authenticode and before the archive smoke test.
    """

    host_abi = _read_json(host_abi_path)
    fingerprint = str(host_abi.get("fingerprint", "")).lower()
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        raise BackendCatalogError("host ABI manifest has no lowercase SHA-256 fingerprint")

    paths: dict[str, Path] = {}
    roles: dict[str, str] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".dll":
            continue
        lower = path.name.lower()
        provider: str | None = None
        if lower in {"ggml.dll", "ggml-base.dll"}:
            provider = "host"
        elif lower.startswith("ggml-cpu"):
            provider = "cpu"
        elif lower in {"ggml-vulkan.dll", "vulkan-1.dll"}:
            raise BackendCatalogError(
                f"optional Vulkan file '{path.name}' leaked into the neutral CPU bundle"
            )
        if provider is not None:
            if lower in roles:
                raise BackendCatalogError(f"duplicate bundled DLL name '{path.name}'")
            roles[lower] = provider
            paths[lower] = path

    required = {"ggml.dll", "ggml-base.dll"}
    missing = sorted(required - roles.keys())
    if missing:
        raise BackendCatalogError(f"neutral bundle is missing required DLLs: {missing}")
    if not any(name.startswith("ggml-cpu") for name in roles):
        raise BackendCatalogError("neutral bundle has no CPU backend module")
    files: list[dict[str, Any]] = []
    for name in sorted(roles):
        path = paths[name]
        sha256, size = sha256_size(path)
        image_sha256, image_size = pe_image_identity(path)
        if size <= 0:
            raise BackendCatalogError(f"bundled DLL '{path.name}' is empty")
        files.append(
            {
                "filename": path.name,
                "provider": roles[name],
                "sha256": sha256,
                "size_bytes": size,
                "image_sha256": image_sha256,
                "image_size_bytes": image_size,
            }
        )

    result = {
        "schema_version": 4,
        "host_abi_fingerprint": fingerprint,
        "bundle_contract_sha256": bundle_contract_sha256(fingerprint, files),
        "cpu_contract_sha256": provider_bundle_contract_sha256(
            fingerprint, files, "cpu"
        ),
        "files": files,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_utf8_lf(out, result)


def materialized_tree_sha256(root: Path, extract_subdir: str) -> str:
    files: list[tuple[str, int, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        relative = f"{extract_subdir}/{relative}" if extract_subdir else relative
        sha256, size = sha256_size(path)
        files.append((relative, size, sha256))
    if not files:
        raise BackendCatalogError("vendor runtime tree is empty")
    files.sort(key=lambda item: item[0])
    digest = hashlib.sha256(b"openasr-backend-tree-v1\0")
    for relative, size, sha256 in files:
        encoded = relative.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", size))
        digest.update(sha256.encode("ascii"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BackendCatalogError(f"{path} must contain a JSON object")
    return value


def compile_entry(args: argparse.Namespace) -> dict[str, Any]:
    build = _read_json(args.build_manifest)
    if build.get("schema_version") != 1 or build.get("topology") != "neutral-backend-dl":
        raise BackendCatalogError("build manifest is not a v1 neutral BACKEND_DL build")
    build_flags = build.get("build_flags")
    if not isinstance(build_flags, dict) or any(
        build_flags.get(field) is not True
        for field in ("backend_dl", "shared", "verified_backend_loading_only")
    ):
        raise BackendCatalogError(
            "build manifest does not enforce the verified-only shared BACKEND_DL host contract"
        )
    cmake_contract = build.get("cmake_contract")
    cmake_contract_sha256 = str(build.get("cmake_contract_sha256", ""))
    if not isinstance(cmake_contract, dict) or cmake_contract.get("schema_version") != 1:
        raise BackendCatalogError("build manifest has no complete CMake contract")
    canonical_cmake = json.dumps(
        cmake_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(canonical_cmake).hexdigest() != cmake_contract_sha256:
        raise BackendCatalogError("build manifest CMake contract digest does not verify")
    cmake_entries = cmake_contract.get("entries")
    compilers = cmake_contract.get("compilers")
    if not isinstance(cmake_entries, dict) or not isinstance(compilers, dict):
        raise BackendCatalogError("build manifest CMake contract is incomplete")
    for field, expected in {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "ON",
        "GGML_BACKEND_DL": "ON",
        "OPENASR_VERIFIED_BACKEND_LOADING_ONLY": "ON",
        "GGML_NATIVE": "OFF",
    }.items():
        if str(cmake_entries.get(field, "")).upper() != expected.upper():
            raise BackendCatalogError(f"CMake contract field '{field}' is not '{expected}'")
    required_compilers = ["c", "cxx"] + (["cuda"] if args.provider == "cuda" else [])
    for role in required_compilers:
        identity = compilers.get(role)
        if (
            not isinstance(identity, dict)
            or len(str(identity.get("sha256", ""))) != 64
            or int(identity.get("size_bytes", 0)) <= 0
        ):
            raise BackendCatalogError(f"CMake contract has no concrete '{role}' compiler identity")
    provider = args.provider
    providers = build.get("providers")
    if not isinstance(providers, dict) or providers.get(provider) is not True:
        raise BackendCatalogError(f"build manifest did not enable provider '{provider}'")
    host_abi = build.get("host_abi")
    if not isinstance(host_abi, dict) or len(str(host_abi.get("fingerprint", ""))) != 64:
        raise BackendCatalogError("build manifest has no complete host ABI")
    targets = build.get("backend_targets", {}).get(provider)
    if not isinstance(targets, list) or not all(isinstance(v, str) and v for v in targets):
        raise BackendCatalogError(f"build manifest has no '{provider}' targets")
    if provider in {"cuda", "hip"} and not targets:
        raise BackendCatalogError(f"build manifest has no '{provider}' targets")
    if provider == "vulkan" and targets:
        raise BackendCatalogError("Vulkan artifact must not encode a physical device target")

    release_provider = "rocm" if provider == "hip" else provider
    expected_plugin_names = {
        f"ggml-{provider}.dll",
        f"openasr-{args.version}-windows-x86_64-{release_provider}-plugin.dll",
    }
    plugin_sha, plugin_size = sha256_size(args.plugin)
    fingerprint = str(host_abi["fingerprint"])
    require_single_target = bool(getattr(args, "require_single_target", False))
    if require_single_target and provider in {"cuda", "hip"} and len(targets) != 1:
        raise BackendCatalogError(
            f"{provider} target-scoped pack must declare exactly one target, got {targets!r}"
        )
    if len(targets) == 1:
        target_label = targets[0]
        if not re.fullmatch(r"[A-Za-z0-9_]+", target_label):
            raise BackendCatalogError(
                f"{provider} target '{target_label}' is not safe for a release asset name"
            )
    else:
        target_label = "generic" if provider == "vulkan" else "fat"
    target_asset_name = (
        f"openasr-{args.version}-windows-x86_64-{release_provider}-{target_label}-plugin.dll"
    )
    expected_plugin_names.add(target_asset_name)
    if args.plugin.name.lower() not in {name.lower() for name in expected_plugin_names}:
        raise BackendCatalogError(
            f"plugin filename must identify exactly one {provider} module and its declared target"
        )
    backend_id = args.backend_id or f"{provider}-windows-x86_64-{fingerprint[:12]}-{target_label}"
    files: list[dict[str, Any]] = [
        {
            "filename": args.plugin.name,
            "url": f"{args.base_url.rstrip('/')}/{args.plugin.name}",
            "mirrors": _mirrors(args.mirror_base_url, args.plugin.name),
            "sha256": plugin_sha,
            "size_bytes": plugin_size,
            "role": "plugin",
        }
    ]
    if args.vendor_archive is not None:
        if args.vendor_tree is None:
            raise BackendCatalogError("--vendor-tree is required with --vendor-archive")
        vendor_sha, vendor_size = sha256_size(args.vendor_archive)
        files.append(
            {
                "filename": args.vendor_archive.name,
                "url": f"{args.base_url.rstrip('/')}/{args.vendor_archive.name}",
                "mirrors": _mirrors(args.mirror_base_url, args.vendor_archive.name),
                "sha256": vendor_sha,
                "size_bytes": vendor_size,
                "role": "archive",
                "extract_subdir": args.vendor_extract_subdir,
                "extracted_tree_sha256": materialized_tree_sha256(
                    args.vendor_tree, args.vendor_extract_subdir
                ),
            }
        )
    elif provider in {"cuda", "hip", "vulkan"}:
        raise BackendCatalogError(f"{provider} pack requires a vendor runtime archive")

    return {
        "id": backend_id,
        "vendor": provider,
        "version": args.version,
        "display_name": args.display_name or f"OpenASR {provider.upper()} backend",
        "description": "Optional verified GPU backend for the neutral Windows host.",
        "targets": targets,
        "min_driver_api": args.minimum_driver_api,
        "min_cli_version": args.minimum_cli_version,
        "activation": {"state": "published-inert"},
        "host_abi": host_abi,
        "files": files,
    }


def _mirrors(base: str | None, filename: str) -> list[dict[str, str]]:
    if not base:
        return []
    return [{"source": "github", "url": f"{base.rstrip('/')}/{filename}"}]


def merge_catalog(catalog_path: Path, entry_paths: list[Path], out: Path) -> None:
    catalog = _read_json(catalog_path)
    existing = catalog.get("backends", [])
    if not isinstance(existing, list):
        raise BackendCatalogError("catalog.backends must be an array")
    incoming: dict[str, dict[str, Any]] = {}
    for entry in [_read_json(path) for path in entry_paths]:
        backend_id = entry.get("id")
        if not isinstance(backend_id, str) or not backend_id:
            raise BackendCatalogError("every backend entry needs a non-empty id")
        if backend_id in incoming and incoming[backend_id] != entry:
            raise BackendCatalogError(f"backend id '{backend_id}' has conflicting entries")
        incoming[backend_id] = entry
    by_id: dict[str, dict[str, Any]] = {}
    for entry in existing:
        backend_id = entry.get("id")
        if not isinstance(backend_id, str) or not backend_id:
            raise BackendCatalogError("every backend entry needs a non-empty id")
        by_id[backend_id] = entry
    # Same host ABI keeps a stable id. A later release therefore replaces the
    # slot's plugin bytes in place instead of colliding with the prior version.
    by_id.update(incoming)

    identities: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for backend_id, entry in by_id.items():
        entry.setdefault("activation", {"state": "published-inert"})
        key = (
            str(entry.get("vendor")),
            str(entry.get("host_abi", {}).get("fingerprint")),
            tuple(entry.get("targets", [])),
        )
        prior = identities.get(key)
        if prior is not None and prior != backend_id:
            raise BackendCatalogError(
                f"'{prior}' and '{backend_id}' have an ambiguous provider/ABI/target identity"
            )
        identities[key] = backend_id
    catalog["backends"] = [by_id[key] for key in sorted(by_id)]
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_utf8_lf(out, catalog)


def verify_release_assets(
    entry_paths: list[Path], asset_directory: Path, expected_version: str
) -> dict[str, object]:
    """Bind signed-catalog metadata to the exact release bytes on disk.

    The release workflow emits backend entries and payloads independently.  A
    maintainer must never sign/serve an entry merely because the JSON parses:
    every declared file has to be present under its declared basename and its
    byte size and SHA-256 must match.  This deliberately accepts a directory of
    already-downloaded release assets so the caller owns network/auth policy.
    """

    providers: dict[str, list[str]] = {}
    host_abi: str | None = None
    identities: set[tuple[str, str]] = set()
    verified_files = 0
    verified_bytes = 0
    for entry_path in entry_paths:
        entry = _read_json(entry_path)
        provider = str(entry.get("vendor", ""))
        if provider not in {"cuda", "hip", "vulkan"}:
            raise BackendCatalogError("release verification accepts only GPU provider entries")
        if str(entry.get("version", "")) != expected_version:
            raise BackendCatalogError(
                f"backend '{provider}' version does not match release {expected_version}"
            )
        fingerprint = str(entry.get("host_abi", {}).get("fingerprint", ""))
        if len(fingerprint) != 64:
            raise BackendCatalogError(f"backend '{provider}' has no complete host ABI")
        if host_abi is not None and host_abi != fingerprint:
            raise BackendCatalogError("GPU release entries do not share one host ABI")
        host_abi = fingerprint
        targets = entry.get("targets")
        if not isinstance(targets, list) or (
            provider in {"cuda", "hip"}
            and (len(targets) != 1 or not isinstance(targets[0], str))
        ) or (provider == "vulkan" and targets):
            raise BackendCatalogError(
                f"backend '{provider}' release entry has invalid artifact targets"
            )
        target_identity = targets[0] if targets else "generic"
        identity = (provider, target_identity)
        if identity in identities:
            raise BackendCatalogError(
                f"release contains duplicate {provider} target entry '{target_identity}'"
            )
        identities.add(identity)
        backend_id = str(entry.get("id", ""))
        if not backend_id:
            raise BackendCatalogError(f"backend '{provider}' entry has no id")
        providers.setdefault(provider, []).append(backend_id)

        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise BackendCatalogError(f"backend '{provider}' has no release files")
        archive_files = [file for file in files if isinstance(file, dict) and file.get("role") == "archive"]
        if len(archive_files) != 1:
            raise BackendCatalogError(
                f"backend '{provider}' target pack must declare exactly one vendor archive"
            )
        archive = archive_files[0]
        for file in files:
            if not isinstance(file, dict):
                raise BackendCatalogError(f"backend '{provider}' has an invalid file record")
            filename = str(file.get("filename", ""))
            if (
                not filename
                or Path(filename).name != filename
                or filename in {".", ".."}
                or "/" in filename
                or "\\" in filename
            ):
                raise BackendCatalogError(
                    f"backend '{provider}' has an unsafe release filename '{filename}'"
                )
            asset = asset_directory / filename
            if not asset.is_file():
                raise BackendCatalogError(
                    f"backend '{provider}' release asset is missing: {filename}"
                )
            actual_sha, actual_size = sha256_size(asset)
            expected_sha = str(file.get("sha256", "")).lower()
            expected_size = int(file.get("size_bytes", 0))
            if actual_sha != expected_sha or actual_size != expected_size:
                raise BackendCatalogError(
                    f"backend '{provider}' release asset does not match its catalog entry: {filename}"
                )
            url = str(file.get("url", ""))
            canonical_url = (
                f"https://dl.openasr.org/core/v{expected_version}/{filename}"
            )
            if url != canonical_url:
                raise BackendCatalogError(
                    f"backend '{provider}' release URL is not canonical for {filename}"
                )
            expected_mirror = (
                "https://github.com/QuintinShaw/openasr/releases/download/"
                f"v{expected_version}/{filename}"
            )
            mirrors = file.get("mirrors", [])
            if not isinstance(mirrors, list) or not any(
                isinstance(mirror, dict) and mirror.get("url") == expected_mirror
                for mirror in mirrors
            ):
                raise BackendCatalogError(
                    f"backend '{provider}' has no canonical GitHub release mirror for {filename}"
                )
            verified_files += 1
            verified_bytes += actual_size

    if set(providers) != {"cuda", "hip", "vulkan"}:
        raise BackendCatalogError("release verification requires Vulkan, CUDA, and HIP packs")
    return {
        "schema_version": 1,
        "version": expected_version,
        "host_abi_fingerprint": host_abi,
        "providers": providers,
        "verified_files": verified_files,
        "verified_bytes": verified_bytes,
    }


def verify_catalog_entries(catalog_path: Path, entry_paths: list[Path]) -> dict[str, object]:
    """Require a catalog to contain the release entries byte-for-byte."""

    catalog = _read_json(catalog_path)
    catalog_entries = catalog.get("backends", [])
    if not isinstance(catalog_entries, list):
        raise BackendCatalogError("catalog.backends must be an array")
    by_id = {
        str(entry.get("id")): entry
        for entry in catalog_entries
        if isinstance(entry, dict) and entry.get("id")
    }
    verified: list[str] = []
    for entry_path in entry_paths:
        expected = _read_json(entry_path)
        backend_id = str(expected.get("id", ""))
        if not backend_id or by_id.get(backend_id) != expected:
            raise BackendCatalogError(
                f"catalog does not contain the exact release backend entry '{backend_id}'"
            )
        verified.append(backend_id)
    return {
        "schema_version": 1,
        "catalog": str(catalog_path),
        "verified_backend_ids": sorted(verified),
    }


def head_cdn_url(url: str) -> tuple[int, int | None]:
    # Cloudflare on dl.openasr.org rejects Python-urllib's default User-Agent
    # with HTTP 403. curl is the maintainer-host probe used everywhere else.
    completed = subprocess.run(
        [
            "curl",
            "-sS",
            "-I",
            "--connect-timeout",
            "5",
            "--max-time",
            "20",
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "--retry-all-errors",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BackendCatalogError(
            f"CDN HEAD failed for {url}: {completed.stderr.strip() or completed.returncode}"
        )
    status: int | None = None
    length: int | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])
        elif line.lower().startswith("content-length:"):
            value = line.split(":", 1)[1].strip()
            if value.isdigit():
                length = int(value)
    if status is None:
        raise BackendCatalogError(f"CDN HEAD failed for {url}: no HTTP status")
    return status, length


def verify_catalog_cdn(
    catalog_path: Path,
    version: str | None = None,
    head: Callable[[str], tuple[int, int | None]] | None = None,
) -> dict[str, object]:
    """Require signed GPU provider file URLs to be live before deployment."""

    catalog = _read_json(catalog_path)
    head_fn = head or head_cdn_url
    seen: dict[str, tuple[str, str, int, str]] = {}
    checked_versions: set[str] = set()
    for entry in catalog.get("backends", []):
        if not isinstance(entry, dict):
            continue
        entry_version = str(entry.get("version", ""))
        if entry.get("vendor") not in {"cuda", "hip", "vulkan"} or (
            version is not None and entry_version != version
        ):
            continue
        for file in entry.get("files", []):
            if not isinstance(file, dict):
                raise BackendCatalogError("backend file records must be objects")
            filename = str(file.get("filename", ""))
            url = str(file.get("url", ""))
            size = int(file.get("size_bytes", 0))
            sha256 = str(file.get("sha256", "")).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise BackendCatalogError(
                    f"backend file '{filename}' has an invalid signed sha256"
                )
            canonical_url = f"https://dl.openasr.org/core/v{entry_version}/{filename}"
            if not entry_version or url != canonical_url:
                raise BackendCatalogError(
                    f"backend file '{filename}' is not the canonical CDN URL for {entry_version or '<missing version>'}"
                )
            identity = (entry_version, filename, size, sha256)
            if url in seen:
                if seen[url] != identity:
                    raise BackendCatalogError(
                        f"conflicting signed metadata for duplicate CDN URL '{url}': "
                        f"first={seen[url]!r} later={identity!r}"
                    )
                continue
            seen[url] = identity
            checked_versions.add(entry_version)
    if not seen:
        scope = f"version {version}" if version is not None else "the catalog"
        raise BackendCatalogError(
            f"catalog has no GPU backend files for {scope} to verify on CDN"
        )

    def probe(item: tuple[str, tuple[str, str, int, str]]) -> str:
        url, (entry_version, filename, size, _sha256) = item
        status, length = head_fn(url)
        if status != 200 or length != size:
            raise BackendCatalogError(
                f"CDN object missing or size-mismatched for '{filename}': "
                f"HEAD {url} -> {status} content-length={length} signed_size={size}. "
                f"Run scripts/sync-windows-backend-cdn.sh v{entry_version} before catalog deployment."
            )
        return url

    items = sorted(seen.items())
    if head is None and len(items) > 1:
        # Network HEADs are independent. Bound parallelism so a catalog with
        # many target packs does not turn deploy/finalize into a long serial
        # TLS loop, while curl's timeout/retry bounds every worker.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
            checked = list(pool.map(probe, items))
    else:
        # Injected probes stay deterministic for unit tests and local callers.
        checked = [probe(item) for item in items]
    return {
        "schema_version": 1,
        "versions": sorted(checked_versions),
        "verified_urls": sorted(checked),
    }


def artifact_fingerprint(entry: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    role_tags = {"runtime": 0, "plugin": 1, "archive": 2}
    for value in (
        entry.get("id", ""),
        entry.get("vendor", ""),
        entry.get("version", ""),
        entry.get("host_abi", {}).get("fingerprint", ""),
        entry.get("min_driver_api", ""),
    ):
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    for target in entry.get("targets", []):
        encoded = str(target).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    for file in entry.get("files", []):
        for value in (
            file.get("filename", ""),
            file.get("sha256", ""),
            file.get("extract_subdir", ""),
            file.get("extracted_tree_sha256", ""),
        ):
            encoded = str(value).encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        digest.update(struct.pack("<Q", int(file.get("size_bytes", 0))))
        role = str(file.get("role", ""))
        if role not in role_tags:
            raise BackendCatalogError(f"unknown backend file role '{role}'")
        digest.update(bytes([role_tags[role]]))
    return digest.hexdigest()


def compile_update_hints(entry_paths: list[Path], out: Path) -> None:
    entries = [_read_json(path) for path in entry_paths]
    providers: dict[str, dict[str, dict[str, Any]]] = {}
    vendor_identities: dict[str, dict[str, Any] | None] = {}
    host_abi: str | None = None
    core_versions: set[str] = set()
    ggml_revisions: set[str] = set()
    for entry in entries:
        provider = str(entry.get("vendor", ""))
        if provider not in {"cuda", "hip", "vulkan"}:
            raise BackendCatalogError("update hints accept only GPU provider entries")
        fingerprint = str(entry.get("host_abi", {}).get("fingerprint", ""))
        if len(fingerprint) != 64:
            raise BackendCatalogError(f"backend '{provider}' has no complete host ABI")
        if host_abi is not None and host_abi != fingerprint:
            raise BackendCatalogError("GPU update hints do not share one host ABI")
        host_abi = fingerprint
        core_versions.add(str(entry.get("version", "")).split("+", 1)[0])
        ggml_revisions.add(str(entry.get("host_abi", {}).get("ggml_revision", "")))
        targets = entry.get("targets")
        if not isinstance(targets, list) or (
            provider in {"cuda", "hip"}
            and (len(targets) != 1 or not isinstance(targets[0], str))
        ) or (provider == "vulkan" and targets):
            raise BackendCatalogError(
                f"update hint '{provider}' has invalid artifact targets"
            )
        target = targets[0] if targets else "generic"
        files = entry.get("files")
        archive_files = [file for file in files if isinstance(file, dict) and file.get("role") == "archive"] \
            if isinstance(files, list) else []
        if len(archive_files) != 1:
            raise BackendCatalogError(
                f"update hint '{provider}' target '{target}' has no unique vendor archive"
            )
        archive = archive_files[0]
        vendor_identity = {
            "filename": archive.get("filename"),
            "sha256": archive.get("sha256"),
            "size_bytes": archive.get("size_bytes"),
            "extracted_tree_sha256": archive.get("extracted_tree_sha256"),
        }
        existing_vendor = vendor_identities.get(provider)
        if provider not in vendor_identities:
            vendor_identities[provider] = vendor_identity
        elif existing_vendor is not None and existing_vendor != vendor_identity:
            vendor_identities[provider] = None
        candidates = providers.setdefault(provider, {})
        if target in candidates:
            raise BackendCatalogError(f"update hints contain duplicate {provider} target '{target}'")
        candidates[target] = {
            "backend_id": entry.get("id"),
            "artifact_fingerprint": artifact_fingerprint(entry),
            "size_bytes": sum(int(file.get("size_bytes", 0)) for file in entry.get("files", [])),
            "vendor": vendor_identity,
        }
    if set(providers) != {"cuda", "hip", "vulkan"}:
        raise BackendCatalogError("update hints require Vulkan, CUDA, and HIP packs")
    if len(core_versions) != 1 or "" in core_versions:
        raise BackendCatalogError("update hints do not share one core version")
    if len(ggml_revisions) != 1 or "" in ggml_revisions:
        raise BackendCatalogError("update hints do not share one ggml revision")
    providers = {
        provider: {
            **(
                {"vendor": vendor_identities[provider]}
                if vendor_identities.get(provider) is not None
                else {}
            ),
            "targets": {target: candidates[target] for target in sorted(candidates)},
        }
        for provider, candidates in sorted(providers.items())
    }
    result = {
        "windows-x86_64": {
            "core_version": next(iter(core_versions)),
            "host_abi_fingerprint": host_abi,
            "ggml_revision": next(iter(ggml_revisions)),
            "catalog_entries_sha256": hashlib.sha256(
                json.dumps(
                    sorted(entries, key=lambda entry: str(entry.get("id", ""))),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "providers": providers,
        }
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_utf8_lf(out, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--build-manifest", type=Path, required=True)
    compile_parser.add_argument(
        "--provider", choices=("cuda", "hip", "vulkan"), required=True
    )
    compile_parser.add_argument("--plugin", type=Path, required=True)
    compile_parser.add_argument("--vendor-archive", type=Path)
    compile_parser.add_argument("--vendor-tree", type=Path)
    compile_parser.add_argument("--vendor-extract-subdir", default="vendor")
    compile_parser.add_argument("--version", required=True)
    compile_parser.add_argument("--minimum-cli-version", required=True)
    compile_parser.add_argument("--minimum-driver-api")
    compile_parser.add_argument("--base-url", required=True)
    compile_parser.add_argument("--mirror-base-url")
    compile_parser.add_argument("--backend-id")
    compile_parser.add_argument("--display-name")
    compile_parser.add_argument("--require-single-target", action="store_true")
    compile_parser.add_argument("--out", type=Path, required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--catalog", type=Path, required=True)
    merge_parser.add_argument("--entry", type=Path, action="append", required=True)
    merge_parser.add_argument("--out", type=Path, required=True)
    verify_assets_parser = subparsers.add_parser("verify-assets")
    verify_assets_parser.add_argument("--entry", type=Path, action="append", required=True)
    verify_assets_parser.add_argument("--asset-directory", type=Path, required=True)
    verify_assets_parser.add_argument("--version", required=True)
    verify_catalog_parser = subparsers.add_parser("verify-catalog")
    verify_catalog_parser.add_argument("--catalog", type=Path, required=True)
    verify_catalog_parser.add_argument("--entry", type=Path, action="append", required=True)
    verify_cdn_parser = subparsers.add_parser("verify-cdn")
    verify_cdn_parser.add_argument("--catalog", type=Path, required=True)
    verify_cdn_parser.add_argument(
        "--version",
        help="limit the gate to one X.Y.Z release; omit to verify every signed backend URL",
    )
    hints_parser = subparsers.add_parser("hints")
    hints_parser.add_argument("--entry", type=Path, action="append", required=True)
    hints_parser.add_argument("--out", type=Path, required=True)
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--directory", type=Path, required=True)
    bundle_parser.add_argument("--host-abi", type=Path, required=True)
    bundle_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "compile":
            entry = compile_entry(args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            _write_utf8_lf(args.out, entry)
        elif args.command == "merge":
            merge_catalog(args.catalog, args.entry, args.out)
        elif args.command == "verify-assets":
            print(
                json.dumps(
                    verify_release_assets(args.entry, args.asset_directory, args.version),
                    sort_keys=True,
                )
            )
        elif args.command == "verify-catalog":
            print(
                json.dumps(
                    verify_catalog_entries(args.catalog, args.entry), sort_keys=True
                )
            )
        elif args.command == "verify-cdn":
            print(
                json.dumps(
                    verify_catalog_cdn(args.catalog, args.version), sort_keys=True
                )
            )
        elif args.command == "hints":
            compile_update_hints(args.entry, args.out)
        else:
            compile_bundled_manifest(args.directory, args.host_abi, args.out)
    except (BackendCatalogError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
