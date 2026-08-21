#!/usr/bin/env python3
"""Generate auditable Windows GPU release evidence from fresh processes.

The published summary stays schema-v2 compatible with the validator embedded
in the release tag. A separate raw audit asset binds the attested release
subjects, exact activated backend status before and after every child, unique
nonce-scoped receipts, and actual graph placement. The summary's
``evidence_sha256`` is the canonical SHA-256 of that raw audit document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import backend_hardware_evidence as gate


RAW_SCHEMA = "openasr.backend-hardware-audit.v1"
RUN_SCOPE = "backend-hardware-evidence"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise gate.EvidenceError(f"{path} must contain a JSON object")
    return value


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2:
            raise gate.EvidenceError(f"{path}:{line_number} is not a checksum record")
        digest, filename = parts
        filename = filename.lstrip("*")
        gate._lower_hex(digest, 64, f"{path}:{line_number} digest")
        if not filename or filename in checksums:
            raise gate.EvidenceError(f"{path}:{line_number} has a duplicate/empty filename")
        checksums[filename] = digest
    return checksums


def _verify_attestation(
    path: Path,
    *,
    repo: str,
    signer_workflow: str,
    source_digest: str,
) -> dict[str, str]:
    completed = subprocess.run(
        [
            "gh",
            "attestation",
            "verify",
            str(path),
            "--repo",
            repo,
            "--signer-workflow",
            signer_workflow,
            "--source-digest",
            source_digest,
            "--format=json",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise gate.EvidenceError(
            f"GitHub attestation failed for {path}: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise gate.EvidenceError(f"GitHub attestation output for {path} is not JSON") from error
    if not isinstance(result, list) or not result:
        raise gate.EvidenceError(f"GitHub attestation output for {path} is empty")
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "verification_sha256": _sha256_bytes(completed.stdout),
    }


def _verify_release_subjects(
    paths: list[Path],
    *,
    checksums_path: Path,
    repo: str,
    signer_workflow: str,
    source_digest: str,
) -> list[dict[str, str]]:
    checksums = _read_checksums(checksums_path)
    seen: set[str] = set()
    verified = []
    for path in paths:
        if path.name in seen:
            raise gate.EvidenceError(f"duplicate release subject filename: {path.name}")
        seen.add(path.name)
        expected = checksums.get(path.name)
        if expected is None or _sha256_file(path) != expected:
            raise gate.EvidenceError(f"{path} does not match SHA256SUMS")
        verified.append(
            _verify_attestation(
                path,
                repo=repo,
                signer_workflow=signer_workflow,
                source_digest=source_digest,
            )
        )
    return sorted(verified, key=lambda item: item["filename"])


def _verify_neutral_extraction(archive: Path, binary: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(archive) as bundle:
            candidates = [
                info
                for info in bundle.infolist()
                if not info.is_dir()
                and info.filename.replace("\\", "/").lower().endswith(
                    "/openasr.exe"
                )
            ]
            if len(candidates) != 1:
                raise gate.EvidenceError(
                    f"{archive} must contain exactly one top-level openasr.exe"
                )
            executable_name = candidates[0].filename.replace("\\", "/")
            prefix = executable_name.rsplit("/", 1)[0] + "/"
            expected: dict[str, str] = {}
            expected_casefold: set[str] = set()
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                name = info.filename.replace("\\", "/")
                if not name.startswith(prefix):
                    raise gate.EvidenceError(
                        f"{archive} contains a file outside its neutral bundle root"
                    )
                relative = PurePosixPath(name[len(prefix) :])
                if (
                    not relative.parts
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise gate.EvidenceError(f"{archive} contains an unsafe member path")
                relative_name = relative.as_posix()
                folded = relative_name.casefold()
                if folded in expected_casefold:
                    raise gate.EvidenceError(
                        f"{archive} contains duplicate case-insensitive member paths"
                    )
                expected_casefold.add(folded)
                digest = hashlib.sha256()
                with bundle.open(info) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                expected[relative_name] = digest.hexdigest()
    except zipfile.BadZipFile as error:
        raise gate.EvidenceError(f"{archive} is not a valid neutral host ZIP") from error

    root = binary.parent
    if binary.resolve() != (root / "openasr.exe").resolve():
        raise gate.EvidenceError("--binary must be openasr.exe in the extracted bundle root")
    actual: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in actual:
            raise gate.EvidenceError(
                "extracted bundle contains duplicate case-insensitive paths"
            )
        actual[folded] = path
    if set(actual) != expected_casefold:
        raise gate.EvidenceError(
            "extracted neutral bundle file set does not match the release ZIP"
        )
    for relative, expected_sha in expected.items():
        if _sha256_file(actual[relative.casefold()]) != expected_sha:
            raise gate.EvidenceError(
                f"extracted neutral bundle file differs from release ZIP: {relative}"
            )
    return _sha256_file(binary), _canonical_sha256(expected)


def _status(binary: Path, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), "__openasr-backend-plugin", "status"],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise gate.EvidenceError(
            "backend status failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise gate.EvidenceError("backend status did not emit one JSON object") from error
    if not isinstance(value, dict):
        raise gate.EvidenceError("backend status did not emit one JSON object")
    return value


def _release_entries(
    paths: list[Path], provider: str, target: str
) -> tuple[list[gate.EntryIdentity], gate.EntryIdentity]:
    identities = [gate._entry_identity(path)[1] for path in paths]
    selected = [identity for identity in identities if identity.provider == provider]
    if not selected:
        raise gate.EvidenceError(f"no {provider} release entries were supplied")
    versions = {identity.version for identity in selected}
    if len(versions) != 1:
        raise gate.EvidenceError(f"{provider} release entries span multiple versions")
    targets = [identity.target for identity in selected]
    if len(targets) != len(set(targets)):
        raise gate.EvidenceError(f"duplicate {provider} release targets")
    tested = [identity for identity in selected if identity.target == target]
    if len(tested) != 1:
        raise gate.EvidenceError(
            f"expected exactly one {provider} release entry for {target}"
        )
    return selected, tested[0]


def _verify_local_backend_payloads(
    tested: gate.EntryIdentity,
    plugin: Path,
    vendor_archive: Path,
) -> tuple[str, str]:
    entry = _load_json(tested.path)
    plugin_files = [file for file in entry.get("files", []) if file.get("role") == "plugin"]
    archive_files = [file for file in entry.get("files", []) if file.get("role") == "archive"]
    if len(plugin_files) != 1 or len(archive_files) != 1:
        raise gate.EvidenceError("tested entry must contain one plugin and one vendor archive")
    plugin_sha = _sha256_file(plugin)
    archive_sha = _sha256_file(vendor_archive)
    for path, actual_sha, metadata in (
        (plugin, plugin_sha, plugin_files[0]),
        (vendor_archive, archive_sha, archive_files[0]),
    ):
        if actual_sha != metadata.get("sha256") or path.stat().st_size != metadata.get(
            "size_bytes"
        ):
            raise gate.EvidenceError(f"{path} does not match the tested release entry")
    return plugin_sha, archive_sha


def _verify_preview_catalog(
    *,
    candidate_path: Path,
    preview_path: Path,
    signature_path: Path,
    catalog_url: str,
    tested: gate.EntryIdentity,
    plugin: Path,
    vendor_archive: Path,
    model: str,
    model_pack: Path,
) -> tuple[str, str, str]:
    expected_url = preview_path.resolve().as_uri()
    if catalog_url != expected_url:
        raise gate.EvidenceError(
            f"--catalog-url must equal the canonical preview URI: {expected_url}"
        )
    if signature_path.resolve() != preview_path.with_name(
        "catalog.signature.json"
    ).resolve():
        raise gate.EvidenceError(
            "--catalog-signature must be catalog.signature.json beside --catalog"
        )

    candidate = _load_json(candidate_path)
    preview = _load_json(preview_path)
    expected_preview = json.loads(json.dumps(candidate))
    matching_backends = [
        backend
        for backend in expected_preview.get("backends", [])
        if isinstance(backend, dict) and backend.get("id") == tested.backend_id
    ]
    if len(matching_backends) != 1:
        raise gate.EvidenceError(
            "candidate catalog does not contain exactly one tested backend"
        )
    role_paths = {
        "plugin": _core_payload_file_url(plugin),
        "archive": _core_payload_file_url(vendor_archive),
    }
    seen_roles: set[str] = set()
    for file in matching_backends[0].get("files", []):
        if not isinstance(file, dict) or file.get("role") not in role_paths:
            raise gate.EvidenceError("tested candidate backend has an unexpected file role")
        role = file["role"]
        if role in seen_roles:
            raise gate.EvidenceError("tested candidate backend has a duplicate file role")
        seen_roles.add(role)
        file["url"] = role_paths[role]
        file["mirrors"] = []
    if seen_roles != set(role_paths):
        raise gate.EvidenceError("tested candidate backend lacks plugin/vendor payloads")
    if preview != expected_preview:
        raise gate.EvidenceError(
            "preview catalog differs from the attested candidate outside exact local URLs"
        )

    model_matches: list[dict[str, Any]] = []
    for model_entry in candidate.get("models", []):
        if not isinstance(model_entry, dict):
            continue
        for quant in model_entry.get("quants", []):
            if isinstance(quant, dict) and quant.get("pull") == model:
                model_matches.append(quant)
    if len(model_matches) != 1:
        raise gate.EvidenceError(
            "candidate catalog does not contain exactly one requested model pull"
        )
    model_pack_sha = _sha256_file(model_pack)
    model_metadata = model_matches[0]
    if (
        model_metadata.get("sha256") != model_pack_sha
        or model_metadata.get("size_bytes") != model_pack.stat().st_size
    ):
        raise gate.EvidenceError(
            "model pack does not match the attested candidate catalog"
        )

    signature = _load_json(signature_path)
    signature_value = signature.get("signature")
    if (
        signature.get("schema_version") != 1
        or signature.get("catalog_url") != catalog_url
        or signature.get("catalog_sha256") != _sha256_file(preview_path)
        or type(signature.get("catalog_epoch")) is not int
        or signature["catalog_epoch"] < 1
        or not isinstance(signature_value, dict)
        or signature_value.get("algorithm") != "ed25519"
        or signature_value.get("key_id") != "openasr-catalog-local-dev-v1"
    ):
        raise gate.EvidenceError("preview catalog signature metadata is invalid")
    gate._lower_hex(signature_value.get("value"), 128, "preview catalog signature")
    return (
        _sha256_file(candidate_path),
        _sha256_file(preview_path),
        _sha256_file(signature_path),
    )


def _activation_matches(
    status: object,
    tested: gate.EntryIdentity,
) -> None:
    if not isinstance(status, dict) or status.get("host_mode") != "neutral_dynamic":
        raise gate.EvidenceError("backend status is not a neutral dynamic host")
    host_abi = status.get("host_abi")
    activated = status.get("activated")
    if not isinstance(host_abi, dict) or not isinstance(activated, dict):
        raise gate.EvidenceError("backend status lacks host ABI or active backend")
    entry = _load_json(tested.path)
    host_fingerprint = entry.get("host_abi", {}).get("fingerprint")
    expected = {
        "backend_id": tested.backend_id,
        "vendor": tested.provider,
        "version": tested.version,
        "artifact_fingerprint": tested.artifact_fingerprint,
        "host_abi_fingerprint": host_fingerprint,
        "device_target": tested.target,
    }
    if host_abi.get("fingerprint") != host_fingerprint:
        raise gate.EvidenceError("active host ABI does not match the tested release entry")
    for field, value in expected.items():
        if activated.get(field) != value:
            raise gate.EvidenceError(f"active backend {field} does not match the release entry")
    if not isinstance(activated.get("driver_version"), str) or not activated.get(
        "driver_version"
    ):
        raise gate.EvidenceError("active backend has no live driver version")


def _provider_backend_name(provider: str, backend_name: str) -> bool:
    normalized = backend_name.strip().lower()
    if provider == "hip":
        return normalized.startswith("hip") or normalized.startswith("rocm")
    return normalized.startswith("cuda") or "nvidia" in normalized


def _validate_receipt(
    receipt: object,
    *,
    tested: gate.EntryIdentity,
    nonce: str,
    scope: str,
    core_commit: str,
    workload_sha: str,
    model_pack_sha: str,
) -> str:
    if not isinstance(receipt, dict):
        raise gate.EvidenceError("fresh process receipt is not a JSON object")
    if receipt.get("schema") != "openasr.short-audio-receipt.v0":
        raise gate.EvidenceError("fresh process receipt has the wrong schema")
    if receipt.get("core_commit") != core_commit or receipt.get("scope") != f"{scope}/{nonce}":
        raise gate.EvidenceError("fresh process receipt is not commit/nonce bound")
    pack = receipt.get("pack")
    audio = receipt.get("audio")
    run = receipt.get("run")
    if not isinstance(pack, dict) or pack.get("content_sha256") != model_pack_sha:
        raise gate.EvidenceError("fresh process receipt has the wrong model pack")
    if not isinstance(audio, dict) or audio.get("sha256") != workload_sha:
        raise gate.EvidenceError("fresh process receipt has the wrong workload")
    if not isinstance(run, dict) or any(
        (
            run.get("backend") != "native",
            run.get("device") != tested.provider,
            run.get("os") != "windows",
            receipt.get("placement") != tested.provider,
        )
    ):
        raise gate.EvidenceError("fresh process receipt has the wrong run binding")
    env = run.get("env_allowlist")
    if (
        not isinstance(env, dict)
        or env.get("OPENASR_GGML_BACKEND") != tested.provider
        or env.get("OPENASR_OFFLINE") != "1"
    ):
        raise gate.EvidenceError("fresh process receipt lacks offline provider confinement")
    observed = receipt.get("observed_placement")
    if not isinstance(observed, dict):
        raise gate.EvidenceError("fresh process receipt lacks observed placement")
    direct = observed.get("direct_graph_computes")
    scheduler = observed.get("scheduler_graph_computes")
    if type(direct) is not int or direct <= 0 or type(scheduler) is not int or scheduler != 0:
        raise gate.EvidenceError("fresh process receipt is not FullDevice")
    compute = observed.get("observed_compute_nodes_by_backend")
    if (
        not isinstance(compute, dict)
        or not compute
        or any(type(nodes) is not int or nodes <= 0 for nodes in compute.values())
        or any(not _provider_backend_name(tested.provider, str(name)) for name in compute)
    ):
        raise gate.EvidenceError("fresh process receipt has missing/cross-provider compute")
    if observed.get("fallback_node_samples_by_backend") not in (None, {}):
        raise gate.EvidenceError("fresh process receipt contains fallback samples")
    transcript = receipt.get("transcript")
    if not isinstance(transcript, dict):
        raise gate.EvidenceError("fresh process receipt has no transcript")
    return gate._lower_hex(
        transcript.get("text_sha256"), 64, "receipt transcript.text_sha256"
    )


def _run_receipt(
    *,
    binary: Path,
    model: str,
    model_pack: Path,
    audio: Path,
    tested: gate.EntryIdentity,
    core_commit: str,
    base_scope: str,
    workload_sha: str,
    model_pack_sha: str,
    env: dict[str, str],
    output: Path,
    home: Path,
    preview_catalog_sha: str,
    preview_signature_sha: str,
) -> tuple[dict[str, Any], str]:
    nonce = uuid.uuid4().hex
    scope = f"{base_scope}/{nonce}"
    _catalog_cache_matches(home, preview_catalog_sha, preview_signature_sha)
    activation_before = _status(binary, env)
    _activation_matches(activation_before, tested)
    command = [
        str(binary),
        "bench-receipt",
        "short-audio",
        "--model",
        model,
        "--model-pack",
        str(model_pack),
        "--audio",
        str(audio),
        "--backend",
        "native",
        "--device",
        tested.provider,
        "--out",
        str(output),
        "--runs",
        "1",
        "--warmup-runs",
        "0",
        "--core-commit",
        core_commit,
        "--scope",
        scope,
    ]
    started_at = _utc_now()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    ended_at = _utc_now()
    activation_after = _status(binary, env)
    _activation_matches(activation_after, tested)
    _catalog_cache_matches(home, preview_catalog_sha, preview_signature_sha)
    if process.returncode != 0:
        raise gate.EvidenceError(
            f"fresh process failed with exit {process.returncode}: "
            + stderr.decode("utf-8", errors="replace").strip()
        )
    if not output.is_file():
        raise gate.EvidenceError("fresh process succeeded without writing its receipt")
    receipt = _load_json(output)
    transcript_sha = _validate_receipt(
        receipt,
        tested=tested,
        nonce=nonce,
        scope=base_scope,
        core_commit=core_commit,
        workload_sha=workload_sha,
        model_pack_sha=model_pack_sha,
    )
    return (
        {
            "nonce": nonce,
            "process_id": process.pid,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "exit_code": process.returncode,
            "activation_before": activation_before,
            "activation_after": activation_after,
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": _sha256_bytes(stderr),
            "receipt_sha256": _canonical_sha256(receipt),
            "receipt": receipt,
        },
        transcript_sha,
    )


def _core_payload_file_url(path: Path) -> str:
    """Return the literal file:// path syntax understood by the v0.1.36 core."""
    return "file://" + str(path.resolve()).replace("\\", "/")


def _evidence_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("OPENASR_CATALOG_SIGNING_KEY_SEED_HEX", None)
    env.update(
        {
            "OPENASR_HOME": str(args.home),
            "OPENASR_CATALOG_URL": args.catalog_url,
            "OPENASR_CATALOG_FILE": str(args.catalog.resolve()),
            "OPENASR_CATALOG_IDENTITY": args.catalog_url,
            "OPENASR_GGML_BACKEND": args.provider,
            "OPENASR_OFFLINE": "1",
        }
    )
    return env


def _catalog_cache_matches(
    home: Path,
    preview_catalog_sha: str,
    preview_signature_sha: str,
) -> None:
    cached_catalog = home / "catalog.json"
    cached_signature = home / "catalog.signature.json"
    if (
        not cached_catalog.is_file()
        or not cached_signature.is_file()
        or _sha256_file(cached_catalog) != preview_catalog_sha
        or _sha256_file(cached_signature) != preview_signature_sha
    ):
        raise gate.EvidenceError(
            "OPENASR_HOME does not cache the exact preview catalog/signature pair"
        )


def _verify_catalog_signature_with_fresh_home(
    *,
    binary: Path,
    env: dict[str, str],
    preview_catalog_sha: str,
    preview_signature_sha: str,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="openasr-catalog-preflight-") as temp:
        fresh_home = Path(temp)
        preflight_env = env.copy()
        preflight_env["OPENASR_HOME"] = str(fresh_home)
        completed = subprocess.run(
            [str(binary), "doctor"],
            env=preflight_env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise gate.EvidenceError(
                "fresh-home catalog signature preflight failed: "
                + completed.stderr.decode("utf-8", errors="replace").strip()
            )
        _catalog_cache_matches(
            fresh_home, preview_catalog_sha, preview_signature_sha
        )
        return {
            "stdout_sha256": _sha256_bytes(completed.stdout),
            "stderr_sha256": _sha256_bytes(completed.stderr),
            "cached_catalog_sha256": _sha256_file(fresh_home / "catalog.json"),
            "cached_signature_sha256": _sha256_file(
                fresh_home / "catalog.signature.json"
            ),
        }


def generate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    required_files = [
        args.binary,
        args.neutral_archive,
        args.plugin,
        args.vendor_archive,
        args.catalog_candidate,
        args.catalog,
        args.catalog_signature,
        args.model_pack,
        args.audio,
        args.checksums,
        *args.entry,
    ]
    for path in required_files:
        if not path.is_file():
            raise gate.EvidenceError(f"required file is missing: {path}")
    if not args.home.is_dir():
        raise gate.EvidenceError(f"evidence home is missing: {args.home}")
    if args.fresh_process_runs < 5:
        raise gate.EvidenceError("--fresh-process-runs must be at least 5")
    gate._lower_hex(args.core_commit, 40, "core_commit")

    release_subjects = [
        args.neutral_archive,
        args.plugin,
        args.vendor_archive,
        args.catalog_candidate,
        *args.entry,
    ]
    attestations = _verify_release_subjects(
        release_subjects,
        checksums_path=args.checksums,
        repo=args.repo,
        signer_workflow=args.signer_workflow,
        source_digest=args.core_commit,
    )
    attested_sha_by_name = {
        subject["filename"]: subject["sha256"] for subject in attestations
    }
    provider_entries, tested = _release_entries(
        args.entry, args.provider, args.device_target
    )
    plugin_sha, vendor_archive_sha = _verify_local_backend_payloads(
        tested, args.plugin, args.vendor_archive
    )
    catalog_candidate_sha, preview_catalog_sha, preview_signature_sha = (
        _verify_preview_catalog(
            candidate_path=args.catalog_candidate,
            preview_path=args.catalog,
            signature_path=args.catalog_signature,
            catalog_url=args.catalog_url,
            tested=tested,
            plugin=args.plugin,
            vendor_archive=args.vendor_archive,
            model=args.model,
            model_pack=args.model_pack,
        )
    )
    if attested_sha_by_name.get(args.catalog_candidate.name) != catalog_candidate_sha:
        raise gate.EvidenceError(
            "candidate catalog changed between preview validation and provenance verification"
        )
    binary_sha, neutral_tree_sha = _verify_neutral_extraction(
        args.neutral_archive, args.binary
    )
    neutral_archive_sha = _sha256_file(args.neutral_archive)
    for path, actual_sha in (
        (args.neutral_archive, neutral_archive_sha),
        (args.plugin, plugin_sha),
        (args.vendor_archive, vendor_archive_sha),
    ):
        if attested_sha_by_name.get(path.name) != actual_sha:
            raise gate.EvidenceError(
                f"release subject changed after provenance verification: {path}"
            )
    workload_sha = _sha256_file(args.audio)
    model_pack_sha = _sha256_file(args.model_pack)
    generator_sha = _sha256_file(Path(__file__))

    env = _evidence_environment(args)
    catalog_signature_preflight = _verify_catalog_signature_with_fresh_home(
        binary=args.binary,
        env=env,
        preview_catalog_sha=preview_catalog_sha,
        preview_signature_sha=preview_signature_sha,
    )
    _catalog_cache_matches(args.home, preview_catalog_sha, preview_signature_sha)
    scope = f"{RUN_SCOPE}-v{tested.version}-{args.provider}"
    runs: list[dict[str, Any]] = []
    transcript_hashes: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="openasr-hardware-evidence-") as temp:
        root = Path(temp)
        for index in range(args.fresh_process_runs):
            run, transcript_sha = _run_receipt(
                binary=args.binary,
                model=args.model,
                model_pack=args.model_pack,
                audio=args.audio,
                tested=tested,
                core_commit=args.core_commit,
                base_scope=scope,
                workload_sha=workload_sha,
                model_pack_sha=model_pack_sha,
                env=env,
                output=root / f"receipt-{index + 1}.json",
                home=args.home,
                preview_catalog_sha=preview_catalog_sha,
                preview_signature_sha=preview_signature_sha,
            )
            runs.append(run)
            transcript_hashes.add(transcript_sha)
    if len({run["process_id"] for run in runs}) != len(runs):
        raise gate.EvidenceError("fresh child processes did not have unique process ids")
    if len({run["nonce"] for run in runs}) != len(runs):
        raise gate.EvidenceError("fresh child processes did not have unique nonces")
    if len(transcript_hashes) != 1:
        raise gate.EvidenceError("fresh child processes did not produce one stable transcript")
    _catalog_cache_matches(args.home, preview_catalog_sha, preview_signature_sha)

    unchanged_inputs = {
        **{
            path: attested_sha_by_name[path.name]
            for path in release_subjects
        },
        args.catalog: preview_catalog_sha,
        args.catalog_signature: preview_signature_sha,
        args.audio: workload_sha,
        args.model_pack: model_pack_sha,
    }
    for path, expected_sha in unchanged_inputs.items():
        if _sha256_file(path) != expected_sha:
            raise gate.EvidenceError(f"evidence input changed while processes ran: {path}")
    final_binary_sha, final_neutral_tree_sha = _verify_neutral_extraction(
        args.neutral_archive, args.binary
    )
    if (final_binary_sha, final_neutral_tree_sha) != (binary_sha, neutral_tree_sha):
        raise gate.EvidenceError(
            "extracted neutral bundle changed while evidence processes ran"
        )

    raw_audit: dict[str, Any] = {
        "schema": RAW_SCHEMA,
        "generator_sha256": generator_sha,
        "repository": args.repo,
        "signer_workflow": args.signer_workflow,
        "source_digest": args.core_commit,
        "scope": scope,
        "binary_sha256": binary_sha,
        "neutral_archive_sha256": neutral_archive_sha,
        "neutral_extracted_tree_sha256": neutral_tree_sha,
        "plugin_sha256": plugin_sha,
        "vendor_archive_sha256": vendor_archive_sha,
        "catalog_candidate_sha256": catalog_candidate_sha,
        "preview_catalog_url": args.catalog_url,
        "preview_catalog_sha256": preview_catalog_sha,
        "preview_catalog_signature_sha256": preview_signature_sha,
        "catalog_signature_preflight": catalog_signature_preflight,
        "workload_sha256": workload_sha,
        "model_pack_sha256": model_pack_sha,
        "checksums_sha256": _sha256_file(args.checksums),
        "attested_release_subjects": attestations,
        "runs": runs,
    }
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "scope": "provider_matrix",
        "result": "pass",
        "provider": tested.provider,
        "device_target": tested.target,
        "backend_id": tested.backend_id,
        "release_version": tested.version,
        "artifact_fingerprint": tested.artifact_fingerprint,
        "plugin_sha256": tested.plugin_sha256,
        "binary_sha256": binary_sha,
        "workload_sha256": workload_sha,
        "model_pack_sha256": model_pack_sha,
        "evidence_sha256": _canonical_sha256(raw_audit),
        "fresh_process_runs": len(runs),
        "placement": "full_device",
        "cpu_fallback": False,
        "approved_targets": sorted(identity.target for identity in provider_entries),
        "provider_matrix_sha256": gate.provider_matrix_sha256(provider_entries),
    }
    return evidence, raw_audit


def _write_validated_outputs(
    *,
    evidence: dict[str, Any],
    raw_audit: dict[str, Any],
    entry_paths: list[Path],
    output: Path,
    raw_output: Path,
) -> None:
    if output.resolve() == raw_output.resolve():
        raise gate.EvidenceError("summary and raw audit outputs must be different files")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_encoded = json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    raw_encoded = json.dumps(raw_audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary_paths: list[Path] = []
    committed_links: list[tuple[Path, Path]] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            handle.write(evidence_encoded)
            evidence_temp = Path(handle.name)
            temporary_paths.append(evidence_temp)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=raw_output.parent, delete=False
        ) as handle:
            handle.write(raw_encoded)
            raw_temp = Path(handle.name)
            temporary_paths.append(raw_temp)
        gate.approved_entry_paths(entry_paths, [evidence_temp])
        if _canonical_sha256(_load_json(raw_temp)) != evidence["evidence_sha256"]:
            raise gate.EvidenceError("raw audit does not match evidence_sha256")
        for temporary, destination in (
            (raw_temp, raw_output),
            (evidence_temp, output),
        ):
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise gate.EvidenceError(
                    f"refusing to overwrite existing output: {destination}"
                ) from error
            committed_links.append((destination, temporary))
        # Both durable names now exist. Temporary-link cleanup must not turn a
        # complete pair into a rollback or report a false publication failure.
        committed_links.clear()
    finally:
        for destination, temporary in reversed(committed_links):
            try:
                if destination.exists() and destination.samefile(temporary):
                    destination.unlink()
            except FileNotFoundError:
                pass
        for path in temporary_paths:
            _unlink_best_effort(path)


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_output_paths(output: Path, raw_output: Path) -> None:
    if output.resolve() == raw_output.resolve():
        raise gate.EvidenceError("summary and raw audit outputs must be different files")
    if not (
        output.name.startswith("backend-hardware-evidence-")
        and output.name.endswith(".json")
    ):
        raise gate.EvidenceError(
            "--output must be named backend-hardware-evidence-*.json"
        )
    if not (
        raw_output.name.startswith("backend-hardware-audit-")
        and raw_output.name.endswith(".json")
    ):
        raise gate.EvidenceError(
            "--raw-output must be named backend-hardware-audit-*.json"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "hip"), required=True)
    parser.add_argument("--device-target", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--neutral-archive", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--vendor-archive", type=Path, required=True)
    parser.add_argument("--catalog-candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-signature", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--signer-workflow", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-pack", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--catalog-url", required=True)
    parser.add_argument("--core-commit", required=True)
    parser.add_argument("--fresh-process-runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    _validate_output_paths(args.output, args.raw_output)
    for path in (args.output, args.raw_output):
        if path.exists():
            raise gate.EvidenceError(f"refusing to overwrite existing output: {path}")
    evidence, raw_audit = generate(args)
    _write_validated_outputs(
        evidence=evidence,
        raw_audit=raw_audit,
        entry_paths=args.entry,
        output=args.output,
        raw_output=args.raw_output,
    )
    print(args.output)
    print(args.raw_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
