#!/usr/bin/env python3
"""Compile verified backend build artifacts into signed-catalog entries.

This tool never signs or downloads. It derives every byte identity from the
staged release files and preserves prior ABI-scoped entries when merging, so
older neutral hosts can continue resolving their compatible pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


class BackendCatalogError(ValueError):
    pass


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


def compile_bundled_manifest(
    directory: Path,
    host_abi_path: Path,
    out: Path,
    require_vulkan_loader: bool,
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
        elif lower == "ggml-vulkan.dll":
            provider = "vulkan"
        elif lower == "vulkan-1.dll":
            provider = "dependency"
        if provider is not None:
            if lower in roles:
                raise BackendCatalogError(f"duplicate bundled DLL name '{path.name}'")
            roles[lower] = provider
            paths[lower] = path

    required = {"ggml.dll", "ggml-base.dll", "ggml-vulkan.dll"}
    missing = sorted(required - roles.keys())
    if missing:
        raise BackendCatalogError(f"neutral bundle is missing required DLLs: {missing}")
    if not any(name.startswith("ggml-cpu") for name in roles):
        raise BackendCatalogError("neutral bundle has no CPU backend module")
    if require_vulkan_loader and "vulkan-1.dll" not in roles:
        raise BackendCatalogError("neutral release bundle has no pinned Vulkan loader")

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
        "schema_version": 2,
        "host_abi_fingerprint": fingerprint,
        "bundle_contract_sha256": bundle_contract_sha256(fingerprint, files),
        "files": files,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def materialized_tree_sha256(root: Path, extract_subdir: str) -> str:
    files: list[tuple[str, int, str]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        relative = f"{extract_subdir}/{relative}" if extract_subdir else relative
        sha256, size = sha256_size(path)
        files.append((relative, size, sha256))
    if not files:
        raise BackendCatalogError("vendor runtime tree is empty")
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
    if not isinstance(targets, list) or not targets or not all(isinstance(v, str) and v for v in targets):
        raise BackendCatalogError(f"build manifest has no '{provider}' targets")

    release_provider = "rocm" if provider == "hip" else provider
    expected_plugin_names = {
        f"ggml-{provider}.dll",
        f"openasr-{args.version}-windows-x86_64-{release_provider}-plugin.dll",
    }
    if args.plugin.name.lower() not in expected_plugin_names:
        raise BackendCatalogError(
            f"plugin filename must identify exactly one {provider} module"
        )
    plugin_sha, plugin_size = sha256_size(args.plugin)
    fingerprint = str(host_abi["fingerprint"])
    backend_id = args.backend_id or f"{provider}-windows-x86_64-{fingerprint[:12]}-fat"
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
    elif provider in {"cuda", "hip"}:
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
    entries = [_read_json(path) for path in entry_paths]
    by_id: dict[str, dict[str, Any]] = {}
    for entry in [*existing, *entries]:
        backend_id = entry.get("id")
        if not isinstance(backend_id, str) or not backend_id:
            raise BackendCatalogError("every backend entry needs a non-empty id")
        if backend_id in by_id and by_id[backend_id] != entry:
            raise BackendCatalogError(f"backend id '{backend_id}' has conflicting entries")
        by_id[backend_id] = entry

    identities: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for backend_id, entry in by_id.items():
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
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    providers: dict[str, dict[str, Any]] = {}
    host_abi: str | None = None
    for entry in entries:
        provider = str(entry.get("vendor", ""))
        if provider not in {"cuda", "hip"} or provider in providers:
            raise BackendCatalogError("update hints require one CUDA and one HIP entry")
        fingerprint = str(entry.get("host_abi", {}).get("fingerprint", ""))
        if len(fingerprint) != 64:
            raise BackendCatalogError(f"backend '{provider}' has no complete host ABI")
        if host_abi is not None and host_abi != fingerprint:
            raise BackendCatalogError("CUDA and HIP update hints do not share one host ABI")
        host_abi = fingerprint
        providers[provider] = {
            "backend_id": entry.get("id"),
            "artifact_fingerprint": artifact_fingerprint(entry),
            "size_bytes": sum(int(file.get("size_bytes", 0)) for file in entry.get("files", [])),
        }
    if set(providers) != {"cuda", "hip"}:
        raise BackendCatalogError("update hints require both CUDA and HIP entries")
    result = {
        "windows-x86_64": {
            "host_abi_fingerprint": host_abi,
            "providers": providers,
        }
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--build-manifest", type=Path, required=True)
    compile_parser.add_argument("--provider", choices=("cuda", "hip"), required=True)
    compile_parser.add_argument("--plugin", type=Path, required=True)
    compile_parser.add_argument("--vendor-archive", type=Path)
    compile_parser.add_argument("--vendor-tree", type=Path)
    compile_parser.add_argument("--vendor-extract-subdir", default="vendor")
    compile_parser.add_argument("--version", required=True)
    compile_parser.add_argument("--minimum-cli-version", required=True)
    compile_parser.add_argument("--minimum-driver-api", required=True)
    compile_parser.add_argument("--base-url", required=True)
    compile_parser.add_argument("--mirror-base-url")
    compile_parser.add_argument("--backend-id")
    compile_parser.add_argument("--display-name")
    compile_parser.add_argument("--out", type=Path, required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--catalog", type=Path, required=True)
    merge_parser.add_argument("--entry", type=Path, action="append", required=True)
    merge_parser.add_argument("--out", type=Path, required=True)
    hints_parser = subparsers.add_parser("hints")
    hints_parser.add_argument("--entry", type=Path, action="append", required=True)
    hints_parser.add_argument("--out", type=Path, required=True)
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--directory", type=Path, required=True)
    bundle_parser.add_argument("--host-abi", type=Path, required=True)
    bundle_parser.add_argument("--out", type=Path, required=True)
    bundle_parser.add_argument("--require-vulkan-loader", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "compile":
            entry = compile_entry(args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        elif args.command == "merge":
            merge_catalog(args.catalog, args.entry, args.out)
        elif args.command == "hints":
            compile_update_hints(args.entry, args.out)
        else:
            compile_bundled_manifest(
                args.directory,
                args.host_abi,
                args.out,
                args.require_vulkan_loader,
            )
    except (BackendCatalogError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
