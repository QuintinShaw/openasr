#!/usr/bin/env python3
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / ".github/ci/linux-build-env.lock"
IMAGE_PATTERN = re.compile(
    r"ghcr\.io/quintinshaw/openasr-ci-linux@sha256:[0-9a-f]{64}"
)
CONSUMERS = {
    ".github/workflows/ci.yml": 2,
    ".github/workflows/family-regression.yml": 1,
    ".github/workflows/public-hf-e2e.yml": 1,
    ".github/workflows/release-core.yml": 1,
    ".github/workflows/serve-batch-parity.yml": 1,
}

APTGET_FORBIDDEN = {
    ".github/workflows/ci.yml",
    ".github/workflows/family-regression.yml",
    ".github/workflows/public-hf-e2e.yml",
    ".github/workflows/release-core.yml",
    ".github/workflows/serve-batch-parity.yml",
}

LINUX_CI_MATRIX = ROOT / "tooling/release-manifest/release_binaries_matrix.json"
LINUX_CI_MATRIX_TARGETS = ("x86_64-unknown-linux-gnu",)
GPU_LOCKS = (
    (
        "x86_64-unknown-linux-gnu-cuda",
        ROOT / ".github/ci/linux-cuda.lock",
        re.compile(
            r"ghcr\.io/quintinshaw/openasr-ci-linux-cuda@sha256:[0-9a-f]{64}"
        ),
    ),
    (
        "x86_64-unknown-linux-gnu-rocm",
        ROOT / ".github/ci/linux-rocm.lock",
        re.compile(
            r"ghcr\.io/quintinshaw/openasr-ci-linux-rocm@sha256:[0-9a-f]{64}"
        ),
    ),
)


def main() -> None:
    expected = LOCK.read_text(encoding="utf-8").strip()
    if IMAGE_PATTERN.fullmatch(expected) is None:
        raise SystemExit(f"invalid Linux CI image lock: {expected!r}")

    failures: list[str] = []
    for relative, expected_count in CONSUMERS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        references = IMAGE_PATTERN.findall(text)
        if references != [expected] * expected_count:
            failures.append(
                f"{relative}: expected {expected_count} reference(s) to {expected}, "
                f"found {references}"
            )
        if relative in APTGET_FORBIDDEN and "apt-get" in text:
            failures.append(f"{relative}: routine consumer must not run apt-get")

    matrix = json.loads(LINUX_CI_MATRIX.read_text(encoding="utf-8"))
    if not isinstance(matrix, list):
        failures.append(f"{LINUX_CI_MATRIX}: expected a JSON array")
    else:
        by_target = {
            row.get("target"): row
            for row in matrix
            if isinstance(row, dict) and isinstance(row.get("target"), str)
        }
        for target in LINUX_CI_MATRIX_TARGETS:
            row = by_target.get(target)
            if row is None:
                failures.append(f"{LINUX_CI_MATRIX}: missing release leg {target}")
                continue
            container = row.get("container")
            if container != expected:
                failures.append(
                    f"{LINUX_CI_MATRIX}: {target} container must be {expected}, "
                    f"found {container!r}"
                )
        for target, lock_path, pattern in GPU_LOCKS:
            pinned = lock_path.read_text(encoding="utf-8").strip()
            if pattern.fullmatch(pinned) is None:
                failures.append(f"invalid GPU image lock {lock_path}: {pinned!r}")
                continue
            row = by_target.get(target)
            if row is None:
                failures.append(f"{LINUX_CI_MATRIX}: missing release leg {target}")
                continue
            if row.get("container") != pinned:
                failures.append(
                    f"{LINUX_CI_MATRIX}: {target} container must be {pinned}, "
                    f"found {row.get('container')!r}"
                )

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Linux CI image consumers match {expected}")


if __name__ == "__main__":
    main()
