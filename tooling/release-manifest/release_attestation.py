"""Exact-byte GitHub attestation verification shared by release gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AttestationError(ValueError):
    pass


def verify_paths(
    paths: list[Path],
    *,
    repository: str,
    signer_workflow: str,
    source_digest: str,
    label: str,
) -> None:
    if (
        len(source_digest) != 40
        or any(char not in "0123456789abcdef" for char in source_digest)
    ):
        raise AttestationError("source_digest must be lowercase 40-hex")
    if not paths:
        raise AttestationError(f"no {label} paths were supplied for attestation")
    resolved: set[Path] = set()
    for path in paths:
        canonical = path.resolve()
        if canonical in resolved:
            raise AttestationError(f"duplicate {label} path: {path}")
        resolved.add(canonical)
        completed = subprocess.run(
            [
                "gh",
                "attestation",
                "verify",
                str(path),
                "--repo",
                repository,
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
            raise AttestationError(
                f"GitHub attestation failed for {label} {path}: "
                + completed.stderr.decode("utf-8", errors="replace").strip()
            )
        try:
            attestation = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise AttestationError(
                f"GitHub attestation output for {label} {path} is not JSON"
            ) from error
        if not isinstance(attestation, list) or not attestation:
            raise AttestationError(
                f"GitHub attestation output for {label} {path} is empty"
            )
