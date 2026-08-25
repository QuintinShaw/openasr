#!/usr/bin/env python3
"""Compile inert backend qualification artifact manifests.

The compiler joins three independent release facts without creating a runtime
activation authority:

* a neutral Windows host archive supplies the exact executable and host ABI;
* one backend-pack candidate supplies an exact CUDA/HIP plugin target or the
  generic Vulkan plugin, together with its vendor archives;
* one Sigstore bundle supplies provenance for every referenced release subject.

The resulting JSON intentionally contains no model, activation mode, or
candidate-generation policy. It must still be signed locally with the
qualification-specific Ed25519 domain before the qualification runner accepts
it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import backend_catalog


SCHEMA_VERSION = 2
HOST_ABI_SCHEMA_VERSION = 3
HOST_ABI_MEMBER = "openasr-backend-host-abi-v1.json"
BINARY_MEMBER = "openasr.exe"
ATTESTATION_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
ATTESTATION_REPOSITORY = "QuintinShaw/openasr"
ATTESTATION_SIGNER_WORKFLOW = (
    "QuintinShaw/openasr/.github/workflows/release-binaries.yml"
)
HOST_ABI_FIELDS = {
    "schema_version",
    "fingerprint",
    "target",
    "crt",
    "toolchain",
    "compile_flags_sha256",
    "ggml_backend_api_version",
    "ggml_revision",
    "ggml_headers_sha256",
    "openasr_ffi_sha256",
    "openasr_extension_sha256",
}
MAX_CAPTURE_BYTES = 1024 * 1024
MAX_ZIP_ENTRIES = 50_000
MAX_ZIP_ENTRY_BYTES = 2 * 1024 * 1024 * 1024
MAX_ZIP_UNPACKED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 500
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
RELEASE_SUBJECT = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)\Z")
CUDA_TARGET = re.compile(r"sm_[0-9]{2,3}\Z")
HIP_TARGET = re.compile(r"gfx[0-9a-f]{3,8}\Z")
PLUGIN_RELEASE_PREFIX = {
    "cuda": "cuda",
    "hip": "rocm",
    "vulkan": "vulkan",
}
VENDOR_LAYER_KEY = {
    "cuda": "cuda-runtime",
    "hip": "rocm-runtime",
    "vulkan": "vulkan-loader",
}


class QualificationManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ZipTree:
    rows: tuple[tuple[str, int, str], ...]
    unpacked_size_bytes: int
    captured: dict[str, bytes]

    def digest(self, extract_subdir: str = "") -> str:
        prefix = f"{extract_subdir}/" if extract_subdir else ""
        return backend_catalog.materialized_tree_sha256_rows(
            [(f"{prefix}{name}", size, digest) for name, size, digest in self.rows]
        )


def _sha256_size(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise QualificationManifestError(f"release artifact is not a regular file: {path}")
    return backend_catalog.sha256_size(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationManifestError(f"could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationManifestError(f"{path} must contain a JSON object")
    return value


def _windows_safe_component(component: str) -> bool:
    if (
        not component
        or not component.isascii()
        or component.endswith((".", " "))
        or ":" in component
        or any(character in '<>"|?*' for character in component)
        or any(ord(character) < 32 for character in component)
    ):
        return False
    stem = component.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL"}:
        return False
    return not (
        (stem.startswith("COM") or stem.startswith("LPT"))
        and stem[3:] in {str(number) for number in range(1, 10)}
    )


def _safe_zip_relative_path(raw_name: str, *, is_directory: bool) -> str:
    if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
        raise QualificationManifestError(f"ZIP entry has an unsafe path: {raw_name!r}")
    name = raw_name[:-1] if is_directory and raw_name.endswith("/") else raw_name
    if not name or name.endswith("/"):
        raise QualificationManifestError(f"ZIP entry has an unsafe path: {raw_name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} or not _windows_safe_component(part) for part in parts):
        raise QualificationManifestError(
            f"ZIP entry is not a portable Windows relative path: {raw_name!r}"
        )
    return "/".join(parts)


def inspect_zip(path: Path, capture_basenames: set[str] | None = None) -> ZipTree:
    _sha256_size(path)
    capture_basenames = {name.lower() for name in (capture_basenames or set())}
    rows: list[tuple[str, int, str]] = []
    captured: dict[str, bytes] = {}
    seen_paths: set[str] = set()
    file_paths: set[str] = set()
    total = 0
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise QualificationManifestError(f"could not open ZIP archive {path}: {error}") from error
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise QualificationManifestError(
                f"ZIP archive {path} exceeds {MAX_ZIP_ENTRIES} entries"
            )
        declared_total = 0
        for info in entries:
            if info.file_size > MAX_ZIP_ENTRY_BYTES:
                raise QualificationManifestError(
                    f"ZIP entry {info.filename!r} exceeds {MAX_ZIP_ENTRY_BYTES} bytes"
                )
            declared_total += info.file_size
            if declared_total > MAX_ZIP_UNPACKED_BYTES:
                raise QualificationManifestError(
                    f"ZIP archive {path} exceeds {MAX_ZIP_UNPACKED_BYTES} unpacked bytes"
                )
            if info.file_size > max(info.compress_size, 1) * MAX_ZIP_COMPRESSION_RATIO:
                raise QualificationManifestError(
                    f"ZIP entry {info.filename!r} exceeds compression ratio limit"
                )
        for info in entries:
            is_directory = info.is_dir()
            relative = _safe_zip_relative_path(info.filename, is_directory=is_directory)
            folded = relative.lower()
            if folded in seen_paths:
                raise QualificationManifestError(
                    f"ZIP archive {path} contains case-colliding entry {relative!r}"
                )
            seen_paths.add(folded)
            if info.flag_bits & 0x1:
                raise QualificationManifestError(
                    f"ZIP archive {path} contains encrypted entry {relative!r}"
                )
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            file_kind = stat.S_IFMT(unix_mode)
            allowed_kind = stat.S_IFDIR if is_directory else stat.S_IFREG
            if file_kind not in {0, allowed_kind}:
                raise QualificationManifestError(
                    f"ZIP archive {path} contains non-regular entry {relative!r}"
                )
            if is_directory:
                continue
            parts = relative.split("/")
            if any("/".join(parts[:index]).lower() in file_paths for index in range(1, len(parts))):
                raise QualificationManifestError(
                    f"ZIP archive {path} nests {relative!r} below a file entry"
                )
            file_paths.add(folded)
            digest = hashlib.sha256()
            size = 0
            capture = PurePosixPath(relative).name.lower() in capture_basenames
            if capture and info.file_size > MAX_CAPTURE_BYTES:
                raise QualificationManifestError(
                    f"captured ZIP metadata entry {relative!r} exceeds {MAX_CAPTURE_BYTES} bytes"
                )
            payload = bytearray() if capture else None
            try:
                source = archive.open(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise QualificationManifestError(
                    f"could not read ZIP entry {relative!r} from {path}: {error}"
                ) from error
            with source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > MAX_ZIP_ENTRY_BYTES or total + size > MAX_ZIP_UNPACKED_BYTES:
                        raise QualificationManifestError(
                            f"ZIP entry {relative!r} exceeded the unpacked byte limit while reading"
                        )
                    if payload is not None:
                        payload.extend(chunk)
            if size != info.file_size:
                raise QualificationManifestError(
                    f"ZIP entry {relative!r} size changed while reading: {size} != {info.file_size}"
                )
            total += size
            if total > (1 << 64) - 1:
                raise QualificationManifestError(f"ZIP archive {path} unpacked size exceeds u64")
            rows.append((relative, size, digest.hexdigest()))
            if payload is not None:
                captured[relative] = bytes(payload)
    if not rows:
        raise QualificationManifestError(f"ZIP archive is empty: {path}")
    ordered_paths = sorted(seen_paths)
    for index, relative in enumerate(ordered_paths[:-1]):
        if relative in file_paths and ordered_paths[index + 1].startswith(f"{relative}/"):
            raise QualificationManifestError(
                f"ZIP archive {path} nests {relative!r} below a file entry"
            )
    rows.sort(key=lambda item: item[0])
    return ZipTree(tuple(rows), total, captured)


def _require_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or not (HEX_40 if length == 40 else HEX_64).fullmatch(value):
        raise QualificationManifestError(f"{field} must be {length} lowercase hex characters")
    return value


def _safe_basename(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or PurePosixPath(value).name != value:
        raise QualificationManifestError(f"{field} must be a safe basename")
    if not _windows_safe_component(value) or any(
        not (character.isascii() and (character.isalnum() or character in "._-"))
        for character in value
    ):
        raise QualificationManifestError(f"{field} must be an ASCII portable basename")
    return value


def _base_url(value: str, field: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise QualificationManifestError(f"{field} must be a credential-free HTTPS base URL")
    return value.rstrip("/")


def _urls(base_url: str, mirror_base_url: str | None, file_name: str) -> list[str]:
    values = [f"{_base_url(base_url, 'base_url')}/{file_name}"]
    if mirror_base_url:
        values.append(f"{_base_url(mirror_base_url, 'mirror_base_url')}/{file_name}")
    if len(set(values)) != len(values):
        raise QualificationManifestError("primary and mirror artifact URLs must be distinct")
    return values


def _artifact(
    path: Path,
    artifact_format: str,
    base_url: str,
    mirror_base_url: str | None,
    *,
    zip_tree: ZipTree | None = None,
) -> dict[str, Any]:
    file_name = _safe_basename(path.name, "artifact.file_name")
    sha256, size_bytes = _sha256_size(path)
    value: dict[str, Any] = {
        "file_name": file_name,
        "format": artifact_format,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "urls": _urls(base_url, mirror_base_url, file_name),
    }
    if artifact_format == "zip_archive":
        if zip_tree is None:
            zip_tree = inspect_zip(path)
        value["unpacked_size_bytes"] = zip_tree.unpacked_size_bytes
        value["unpacked_tree_sha256"] = zip_tree.digest()
    elif zip_tree is not None:
        raise QualificationManifestError("non-ZIP artifact cannot carry a ZIP tree")
    return value


def _host_archive_identity(path: Path) -> tuple[ZipTree, dict[str, Any], tuple[str, int, str]]:
    tree = inspect_zip(path, {HOST_ABI_MEMBER})
    binaries = [row for row in tree.rows if PurePosixPath(row[0]).name.lower() == BINARY_MEMBER]
    if len(binaries) != 1:
        raise QualificationManifestError(
            f"neutral host archive must contain exactly one {BINARY_MEMBER}; found {len(binaries)}"
        )
    binary = binaries[0]
    prefix = PurePosixPath(binary[0]).parent
    for relative, _size, _sha256 in tree.rows:
        try:
            PurePosixPath(relative).relative_to(prefix)
        except ValueError as error:
            raise QualificationManifestError(
                "neutral archive contains files outside the executable bundle root"
            ) from error
    abi_path = str(prefix / HOST_ABI_MEMBER)
    abi_payload = tree.captured.get(abi_path)
    if abi_payload is None:
        raise QualificationManifestError(
            f"neutral host archive is missing {HOST_ABI_MEMBER} beside {BINARY_MEMBER}"
        )
    try:
        host_abi = json.loads(abi_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationManifestError(
            f"neutral host archive contains an invalid {HOST_ABI_MEMBER}: {error}"
        ) from error
    if not isinstance(host_abi, dict):
        raise QualificationManifestError(f"{HOST_ABI_MEMBER} must contain a JSON object")
    _validate_host_abi(host_abi)
    return tree, host_abi, binary


def _validate_host_abi(host_abi: dict[str, Any]) -> None:
    if set(host_abi) != HOST_ABI_FIELDS:
        raise QualificationManifestError(
            f"host ABI fields differ from qualification schema v1: {sorted(host_abi)}"
        )
    if host_abi.get("schema_version") != HOST_ABI_SCHEMA_VERSION:
        raise QualificationManifestError(
            f"host ABI schema must be {HOST_ABI_SCHEMA_VERSION}"
        )
    for field in (
        "fingerprint",
        "compile_flags_sha256",
        "ggml_headers_sha256",
        "openasr_ffi_sha256",
        "openasr_extension_sha256",
    ):
        _require_hex(host_abi.get(field), 64, f"host_abi.{field}")
    _require_hex(host_abi.get("ggml_revision"), 40, "host_abi.ggml_revision")
    if host_abi.get("target") != "x86_64-pc-windows-msvc":
        raise QualificationManifestError("qualification v1 requires x86_64-pc-windows-msvc")
    toolchain = host_abi.get("toolchain")
    if (
        host_abi.get("crt") != "msvc-md"
        or not isinstance(toolchain, str)
        or not toolchain
        or not all(
            character.isascii() and (character.isalnum() or character in "._-:")
            for character in toolchain
        )
    ):
        raise QualificationManifestError("host ABI must declare the MSVC dynamic CRT/toolchain")
    backend_api = host_abi.get("ggml_backend_api_version")
    if isinstance(backend_api, bool) or not isinstance(backend_api, int) or backend_api <= 0:
        raise QualificationManifestError("host ABI backend API identity is incomplete")


def _attestation_subjects(bundle_path: Path) -> tuple[str, dict[str, str]]:
    bundle = _read_json_object(bundle_path)
    envelope = bundle.get("dsseEnvelope")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), str):
        raise QualificationManifestError("attestation bundle has no DSSE payload")
    try:
        statement_bytes = base64.b64decode(envelope["payload"], validate=True)
        statement = json.loads(statement_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationManifestError(f"attestation DSSE payload is invalid: {error}") from error
    if not isinstance(statement, dict):
        raise QualificationManifestError("attestation DSSE payload must be a JSON object")
    predicate_type = statement.get("predicateType")
    if predicate_type != ATTESTATION_PREDICATE_TYPE:
        raise QualificationManifestError(
            f"attestation predicate must be {ATTESTATION_PREDICATE_TYPE}"
        )
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or not subjects:
        raise QualificationManifestError("attestation statement has no subjects")
    resolved: dict[str, str] = {}
    folded_names: set[str] = set()
    for subject in subjects:
        if not isinstance(subject, dict):
            raise QualificationManifestError("attestation subject must be an object")
        name = _safe_basename(subject.get("name"), "attestation subject name")
        digests = subject.get("digest")
        if not isinstance(digests, dict):
            raise QualificationManifestError(f"attestation subject {name!r} has no digest")
        digest = _require_hex(digests.get("sha256"), 64, f"attestation subject {name}")
        if name.lower() in folded_names:
            raise QualificationManifestError(f"attestation has duplicate subject {name!r}")
        folded_names.add(name.lower())
        resolved[name] = digest
    return predicate_type, resolved


def _require_attested(subjects: dict[str, str], path: Path) -> None:
    digest, _size = _sha256_size(path)
    actual = subjects.get(path.name)
    if actual != digest:
        raise QualificationManifestError(
            f"attestation does not bind exact release subject {path.name!r}"
        )


def _candidate_urls(file: dict[str, Any]) -> list[str]:
    url = file.get("url")
    mirrors = file.get("mirrors", [])
    if not isinstance(url, str) or not isinstance(mirrors, list):
        raise QualificationManifestError("backend candidate file URLs are malformed")
    values = [url]
    for mirror in mirrors:
        if not isinstance(mirror, dict) or not isinstance(mirror.get("url"), str):
            raise QualificationManifestError("backend candidate mirror URL is malformed")
        values.append(mirror["url"])
    return values


def artifact_cell(entry: dict[str, Any]) -> tuple[str, str]:
    """Return the immutable qualification artifact cell for one backend pack.

    CUDA/HIP cells are the compiled target. Vulkan cells are always
    ``("vulkan", "generic")``; a live ``vk_caps_*`` identity is never a
    release-artifact fact.
    """

    provider = entry.get("vendor")
    targets = entry.get("targets")
    if provider == "vulkan":
        if targets != []:
            raise QualificationManifestError(
                "generic Vulkan qualification artifact must not encode a physical device target"
            )
        return "vulkan", "generic"
    if provider not in {"cuda", "hip"}:
        raise QualificationManifestError(
            "backend qualification entry must be CUDA, HIP, or Vulkan"
        )
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], str):
        raise QualificationManifestError("CUDA/HIP qualification entry must declare one target")
    target = targets[0]
    pattern = CUDA_TARGET if provider == "cuda" else HIP_TARGET
    if not pattern.fullmatch(target):
        raise QualificationManifestError(
            f"backend target {target!r} is not canonical for {provider}"
        )
    return provider, target


def expected_artifact_cells(
    matrix: list[dict[str, Any]],
    *,
    promoted_cuda_targets: set[str] | None = None,
) -> set[tuple[str, str]]:
    """Return inert qualification cells implied by the release matrix.

    CUDA/HIP cells use the compiled target. Vulkan is always the generic
    artifact cell; a live ``vk_caps_*`` identity is never a matrix fact.
    Experimental CUDA rows enter only when explicitly promoted.
    """

    promoted = promoted_cuda_targets or set()
    cells: set[tuple[str, str]] = set()
    for row in matrix:
        if not isinstance(row, dict):
            continue
        provider = row.get("provider")
        experimental = bool(row.get("experimental", False))
        if provider == "cuda":
            target = row.get("cuda_gpu_target")
            if target is None:
                continue
            token = str(target)
            if not experimental or token in promoted:
                cells.add(("cuda", f"sm_{token}"))
        elif provider == "hip" and not experimental:
            target = row.get("hip_gpu_target")
            if target is None:
                continue
            cells.add(("hip", str(target)))
        elif provider == "vulkan" and not experimental:
            cells.add(("vulkan", "generic"))
    return cells


def _backend_artifacts(
    entry_path: Path,
    asset_directory: Path,
    release_version: str,
    host_abi: dict[str, Any],
    base_url: str,
    mirror_base_url: str | None,
) -> tuple[str, str, dict[str, Any], list[Path]]:
    entry = _read_json_object(entry_path)
    provider, target = artifact_cell(entry)
    if entry.get("version") != release_version:
        raise QualificationManifestError("backend entry version differs from release subject")
    entry_abi = entry.get("host_abi")
    if not isinstance(entry_abi, dict):
        raise QualificationManifestError("backend entry has no host ABI object")
    _validate_host_abi(entry_abi)
    if entry_abi != host_abi:
        raise QualificationManifestError("backend entry host ABI differs from neutral host archive")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise QualificationManifestError("backend entry has no release files")
    plugin_files = [file for file in files if isinstance(file, dict) and file.get("role") == "plugin"]
    archives = [file for file in files if isinstance(file, dict) and file.get("role") == "archive"]
    if len(plugin_files) != 1 or not archives or len(plugin_files) + len(archives) != len(files):
        raise QualificationManifestError(
            "backend entry must contain exactly one plugin and one or more vendor archives"
        )
    referenced: list[Path] = []
    plugin_file = plugin_files[0]
    plugin_name = _safe_basename(plugin_file.get("filename"), "backend plugin filename")
    if not plugin_name.lower().endswith(".dll"):
        raise QualificationManifestError("backend plugin must be a Windows DLL")
    release_prefix = PLUGIN_RELEASE_PREFIX[provider]
    expected_plugin_name = (
        f"openasr-{release_version}-windows-x86_64-{release_prefix}-{target}-plugin.dll"
    )
    if plugin_name != expected_plugin_name:
        raise QualificationManifestError(
            f"backend plugin filename must bind exact provider/target {expected_plugin_name!r}"
        )
    plugin_path = asset_directory / plugin_name
    plugin_sha256, plugin_size = _sha256_size(plugin_path)
    if plugin_file.get("sha256") != plugin_sha256 or plugin_file.get("size_bytes") != plugin_size:
        raise QualificationManifestError("backend plugin bytes differ from candidate metadata")
    expected_plugin_urls = _urls(base_url, mirror_base_url, plugin_name)
    if _candidate_urls(plugin_file) != expected_plugin_urls:
        raise QualificationManifestError("backend plugin URLs differ from qualification release URLs")
    plugin = _artifact(plugin_path, "native_library", base_url, mirror_base_url)
    referenced.append(plugin_path)
    vendors: list[dict[str, Any]] = []
    for archive_file in archives:
        archive_name = _safe_basename(
            archive_file.get("filename"), "backend vendor archive filename"
        )
        if not archive_name.lower().endswith(".zip"):
            raise QualificationManifestError("backend vendor artifact must be a ZIP archive")
        vendor_layer = VENDOR_LAYER_KEY[provider]
        expected_vendor = re.compile(
            rf"openasr-vendor-{re.escape(vendor_layer)}-[0-9a-f]{{12}}\.zip\Z"
        )
        if not expected_vendor.fullmatch(archive_name):
            raise QualificationManifestError(
                f"backend vendor filename does not bind vendor layer {vendor_layer!r}"
            )
        archive_path = asset_directory / archive_name
        archive_sha256, archive_size = _sha256_size(archive_path)
        if (
            archive_file.get("sha256") != archive_sha256
            or archive_file.get("size_bytes") != archive_size
        ):
            raise QualificationManifestError(
                f"backend vendor archive {archive_name!r} differs from candidate metadata"
            )
        if not archive_name.endswith(f"-{archive_sha256[:12]}.zip"):
            raise QualificationManifestError(
                f"backend vendor archive {archive_name!r} is not named by its sha256 prefix"
            )
        if _candidate_urls(archive_file) != _urls(base_url, mirror_base_url, archive_name):
            raise QualificationManifestError(
                f"backend vendor archive {archive_name!r} URLs differ from release URLs"
            )
        extract_subdir = archive_file.get("extract_subdir")
        if not isinstance(extract_subdir, str) or not extract_subdir:
            raise QualificationManifestError(
                f"backend vendor archive {archive_name!r} has no extraction root"
            )
        _safe_zip_relative_path(extract_subdir, is_directory=False)
        archive_tree = inspect_zip(archive_path)
        if archive_file.get("extracted_tree_sha256") != archive_tree.digest(extract_subdir):
            raise QualificationManifestError(
                f"backend vendor archive {archive_name!r} tree differs from candidate metadata"
            )
        vendors.append(
            _artifact(
                archive_path,
                "zip_archive",
                base_url,
                mirror_base_url,
                zip_tree=archive_tree,
            )
        )
        referenced.append(archive_path)
    vendors.sort(key=lambda artifact: artifact["file_name"])
    return provider, target, {"plugin": plugin, "vendor": vendors}, referenced


def manifest_asset_name(version: str, provider: str, target: str) -> str:
    return f"openasr-{version}-qualification-{provider}-{target}.json"


def compile_manifest(args: argparse.Namespace) -> dict[str, Any]:
    release_match = RELEASE_SUBJECT.fullmatch(args.release_subject)
    if release_match is None:
        raise QualificationManifestError("release_subject must be vX.Y.Z")
    release_version = release_match.group(1)
    source_digest = _require_hex(args.source_digest, 40, "source_digest")
    asset_directory = args.asset_directory.resolve()
    neutral_archive = args.neutral_archive.resolve()
    attestation_bundle = args.attestation_bundle.resolve()
    expected_neutral_name = f"openasr-{release_version}-windows-x86_64-neutral.zip"
    if neutral_archive.name != expected_neutral_name:
        raise QualificationManifestError(
            f"neutral archive must be named {expected_neutral_name!r}"
        )
    neutral_tree, host_abi, binary = _host_archive_identity(neutral_archive)
    predicate_type, attested_subjects = _attestation_subjects(attestation_bundle)
    _require_attested(attested_subjects, neutral_archive)
    binary_bundle = _artifact(
        neutral_archive,
        "zip_archive",
        args.base_url,
        args.mirror_base_url,
        zip_tree=neutral_tree,
    )
    artifacts: dict[str, Any] = {
        "binary": {
            "file_name": BINARY_MEMBER,
            "sha256": binary[2],
            "size_bytes": binary[1],
            "bundle": binary_bundle,
        }
    }
    provider, target, optional, referenced = _backend_artifacts(
        args.backend_entry.resolve(),
        asset_directory,
        release_version,
        host_abi,
        args.base_url,
        args.mirror_base_url,
    )
    artifacts.update(optional)
    for path in referenced:
        _require_attested(attested_subjects, path)
    expected_output = manifest_asset_name(release_version, provider, target)
    if args.out.name != expected_output:
        raise QualificationManifestError(
            f"qualification manifest output must use exact-cell basename {expected_output!r}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "release_subject": args.release_subject,
        "host_abi": host_abi,
        "provider_target": {"provider": provider, "target": target},
        "artifacts": artifacts,
        "attestation": {
            "predicate_type": predicate_type,
            "repository": ATTESTATION_REPOSITORY,
            "signer_workflow": ATTESTATION_SIGNER_WORKFLOW,
            "source_digest": source_digest,
            "deny_self_hosted_runners": True,
            "bundle": _artifact(
                attestation_bundle,
                "attestation_bundle",
                args.base_url,
                args.mirror_base_url,
            ),
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral-archive", type=Path, required=True)
    parser.add_argument("--backend-entry", type=Path, required=True)
    parser.add_argument("--asset-directory", type=Path, required=True)
    parser.add_argument("--attestation-bundle", type=Path, required=True)
    parser.add_argument("--release-subject", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mirror-base-url")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = compile_manifest(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        backend_catalog._write_utf8_lf(args.out, manifest)
    except (QualificationManifestError, backend_catalog.BackendCatalogError, OSError) as error:
        print(f"qualification manifest generation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
